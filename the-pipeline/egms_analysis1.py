# pipeline/egms_analysis.py
# Step E: EGMS Ortho (Sentinel-1, Dec 2020 - 2025) at a site.
# - reads the big tile CSVs from the zips in chunks, keeps the site box, caches it
# - coverage: share of ground measured by EGMS vs NISAR, in forest and open ground
# - distance bands (same definition as Step D) for mean velocity and seasonality
# - motion over time near vs far from the wind farm, with monthly rainfall
# - the 2020 slide area after the slide; east-west sanity check
# Data: European Union, Copernicus Land Monitoring Service, EEA.
# Usage:
#     python pipeline\egms_analysis.py meenbog


import sys
import zipfile
from datetime import datetime

import geopandas as gpd
import matplotlib
matplotlib.use("Agg") #ensures that you can save as image -> using past data on the EGMS service data as well

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.ops import transform as shapely_transform

from config1 import RAW_DIR, SITES, WORK_DIR
from coherence_check1 import MARKERS
from distance_bands1 import (ACCESS_TYPES, ACCESS_WITHIN_M, BANDS, CLASSES, MIN_FROM_INFRA_M, MIN_FROM_PUBLIC_M,
                            PUBLIC_TYPES, REFERENCE, SLIDE_EXCLUDE_M, TURBINE_RADIUS_M, burn, grid_transform, union)
from rainfall_check import combine, load_rain

EGMS_CRS = "EPSG:3035"
MIN_POINTS = 20
N_BOOT = 1000
SLIDE_AREA_M = 300
RING_M = (500, 1500)
RAIN_FILES = [RAW_DIR / "weather" / "met_eireann_ardnawark_daily.csv",
              RAW_DIR / "weather" / "met_eireann_finner_daily.csv"]


def read_component(site, component, bounds):
    """1. Points inside the bounds from the EGMS zip for one component (U or E), cached after the first run."""
    cache = WORK_DIR / f"{site}_egms_{component}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    zips = sorted((RAW_DIR / "egms").glob(f"*_{component}_*.zip"))
    if not zips:
        raise FileNotFoundError(f"No EGMS zip with _{component}_ in its name in {RAW_DIR / 'egms'}")
    xmin, ymin, xmax, ymax = bounds
    parts = []
    print(f"Reading {zips[0].name} in chunks (first run only, can take a few minutes)...")
    with zipfile.ZipFile(zips[0]) as z:
        name = next(m for m in z.namelist() if m.endswith(".csv"))
        with z.open(name) as f:
            for chunk in pd.read_csv(f, chunksize=100_000):
                keep = chunk[(chunk["easting"] >= xmin) & (chunk["easting"] <= xmax) &
                             (chunk["northing"] >= ymin) & (chunk["northing"] <= ymax)]
                if len(keep):
                    parts.append(keep)
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    table.to_csv(cache, index=False)
    print(f"  kept {len(table)} points, cached as {cache.name}")
    return table


def band_difference(values, band, ref, rng):
    """Median in the band minus median in the reference, with a bootstrap uncertainty."""
    if band.sum() < MIN_POINTS or ref.sum() < MIN_POINTS:
        return None
    a, b = values[band], values[ref]
    diff = float(np.median(a) - np.median(b))
    boots = [np.median(rng.choice(a, a.size)) - np.median(rng.choice(b, b.size)) for _ in range(N_BOOT)]
    return diff, float(np.std(boots))

ELEV_STEP = 25   # height groups (m) for the matched comparison


def adjusted_effect(values, height, band, ref):
    """Fit value = a + b*height + c*(in band) on band + reference points. c = effect after allowing for height."""
    use = band | ref
    X = np.column_stack([np.ones(use.sum()), height[use], band[use].astype(float)])
    y = values[use]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ coef
    variance = residual @ residual / max(len(y) - 3, 1)
    cov = variance * np.linalg.inv(X.T @ X)
    return float(coef[2]), float(np.sqrt(cov[2, 2])), float(coef[1])


def matched_difference(values, height, band, ref):
    """Near minus reference within each height group, averaged by the band's point count."""
    lo, hi = np.floor(height.min() / ELEV_STEP) * ELEV_STEP, np.ceil(height.max() / ELEV_STEP) * ELEV_STEP
    diffs, weights = [], []
    for start in np.arange(lo, hi, ELEV_STEP):
        in_group = (height >= start) & (height < start + ELEV_STEP)
        a, r = values[band & in_group], values[ref & in_group]
        if a.size >= 5 and r.size >= 5:
            diffs.append(np.median(a) - np.median(r))
            weights.append(a.size)
    if not weights:
        return None, 0.0
    return float(np.average(diffs, weights=weights)), float(sum(weights) / band.sum())


def elevation_check(site, up, d_infra, cover):
    """Is the near-wind-farm difference really about height (e.g. atmosphere), not the wind farm?"""
    height = up["height_ortho"].values.astype(float)

    # 1. Average height per band.
    print("\nElevation check (EGMS height_ortho)")
    print("  1. Average height per band:")
    for cls in CLASSES:
        ref = (d_infra >= REFERENCE[0]) & (d_infra < REFERENCE[1]) & cover[cls]
        parts = []
        for lo, hi in BANDS:
            band = (d_infra >= lo) & (d_infra < hi) & cover[cls]
            if band.sum():
                parts.append(f"{lo}-{hi} m: {height[band].mean():.0f} m")
        if ref.sum():
            parts.append(f"reference: {height[ref].mean():.0f} m")
        print(f"     {cls:7s} " + " | ".join(parts))

    # 2. Seasonality vs height, using reference points only (away from the wind farm).
    ref_open = (d_infra >= REFERENCE[0]) & (d_infra < REFERENCE[1]) & cover["open"]
    season = up["seasonality"].values.astype(float)
    print("  2. Seasonality vs height, open ground 1-3 km from the wind farm:")
    trend_x, trend_y = [], []
    lo, hi = np.floor(height.min() / ELEV_STEP) * ELEV_STEP, np.ceil(height.max() / ELEV_STEP) * ELEV_STEP
    for start in np.arange(lo, hi, ELEV_STEP):
        group = ref_open & (height >= start) & (height < start + ELEV_STEP)
        if group.sum() >= 10:
            trend_x.append(start + ELEV_STEP / 2)
            trend_y.append(float(np.median(season[group])))
            print(f"     {start:4.0f}-{start + ELEV_STEP:4.0f} m: {int(group.sum()):4d} points, "
                  f"median seasonality {trend_y[-1]:.2f} mm")
    if ref_open.sum() > 10:
        r = np.corrcoef(height[ref_open], season[ref_open])[0, 1]
        print(f"     correlation of seasonality with height (reference points): {r:+.2f}")

    # 3. Raw vs matched vs adjusted, for both measures.
    print("  3. Difference from reference: raw / matched heights / adjusted for height")
    print("     measure          class    band (m)    raw      matched (share matched)   adjusted +/- uncertainty")
    bars = []
    for column, unit in [("seasonality", "mm"), ("mean_velocity", "mm/yr")]:
        values = up[column].values.astype(float)
        for cls in CLASSES:
            ref = (d_infra >= REFERENCE[0]) & (d_infra < REFERENCE[1]) & cover[cls]
            for lo_b, hi_b in BANDS:
                band = (d_infra >= lo_b) & (d_infra < hi_b) & cover[cls]
                if band.sum() < MIN_POINTS or ref.sum() < MIN_POINTS:
                    continue
                raw = float(np.median(values[band]) - np.median(values[ref]))
                matched, share = matched_difference(values, height, band, ref)
                effect, se, slope = adjusted_effect(values, height, band, ref)
                matched_text = f"{matched:+6.2f} ({share * 100:3.0f}%)" if matched is not None else "  n/a (  0%)"
                print(f"     {column:15s}  {cls:7s}  {lo_b:4d}-{hi_b:<5d} {raw:+6.2f}   {matched_text:22s}"
                      f"   {effect:+6.2f} +/- {se:4.2f} {unit}")
                if column == "seasonality":
                    bars.append((f"{cls}\n{lo_b}-{hi_b}", raw, matched if matched is not None else np.nan, effect, se))
    print("     (if 'matched' and 'adjusted' shrink towards 0, height explains the raw difference)")

    # Picture: seasonality vs height, and raw vs matched vs adjusted.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(height[ref_open], season[ref_open], s=4, color="lightgray", label="reference (1-3 km)")
    for (lo_b, hi_b), colour in zip(BANDS, ["#d7301f", "#fc8d59", "#fdcc8a"]):
        band = (d_infra >= lo_b) & (d_infra < hi_b) & cover["open"]
        ax1.scatter(height[band], season[band], s=6, color=colour, label=f"{lo_b}-{hi_b} m")
    if trend_x:
        ax1.plot(trend_x, trend_y, color="black", linewidth=2, label="reference median by height")
    ax1.set_xlabel("Height (m)")
    ax1.set_ylabel("Seasonality (mm)")
    ax1.set_title("Open ground: seasonal cycle vs height")
    ax1.legend(fontsize=8)
    if bars:
        labels, raws, matches, effects, ses = zip(*bars)
        idx = np.arange(len(labels))
        ax2.bar(idx - 0.27, raws, width=0.27, label="raw")
        ax2.bar(idx, matches, width=0.27, label="matched heights")
        ax2.bar(idx + 0.27, effects, width=0.27, yerr=np.array(ses) * 2, capsize=3, label="adjusted for height")
        ax2.set_xticks(idx)
        ax2.set_xticklabels(labels, fontsize=8)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("Seasonality difference from reference (mm)")
        ax2.set_title("Does the near-wind-farm difference survive allowing for height?")
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(WORK_DIR / f"{site}_egms_elevation_check.png", dpi=110)
    plt.close(fig)
    print(f"  Saved picture: {site}_egms_elevation_check.png")
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\egms_analysis.py <site>")
        return
    site = sys.argv[1]
    rng = np.random.default_rng(0)

    # 1. Read the points inside the site box.
    box = shapely_wkt.loads(SITES[site]["wkt"])
    to_egms = Transformer.from_crs("EPSG:4326", EGMS_CRS, always_xy=True)
    box_egms = shapely_transform(lambda x, y, z=None: to_egms.transform(x, y), box)
    up = read_component(site, "U", box_egms.buffer(200).bounds)
    east = read_component(site, "E", box_egms.buffer(200).bounds)
    date_cols = [c for c in up.columns if str(c).isdigit() and len(str(c)) == 8]
    dates = [datetime.strptime(str(c), "%Y%m%d") for c in date_cols]
    points_egms = gpd.GeoSeries(gpd.points_from_xy(up["easting"], up["northing"]), crs=EGMS_CRS)
    inside = points_egms.within(box_egms).values
    up, points_egms = up[inside].reset_index(drop=True), points_egms[inside].reset_index(drop=True)
    print(f"\nEGMS points inside the box: {len(up)} | dates {dates[0]:%d %b %Y} to {dates[-1]:%d %b %Y} "
          f"({len(dates)} dates)")

    # 2. Same projection as the NISAR work.
    nisar_files = sorted(WORK_DIR.glob(f"{site}_*_timeseries.npz"))
    nisar = np.load(nisar_files[-1])
    epsg = int(nisar["epsg"])
    pts = points_egms.to_crs(epsg)
    px, py = pts.x.values, pts.y.values

    # Infrastructure exactly as in Step D.
    turbines = gpd.read_file(WORK_DIR / f"{site}_turbines.geojson").to_crs(epsg)
    roads = gpd.read_file(WORK_DIR / f"{site}_roads.geojson").to_crs(epsg)
    forest = gpd.read_file(WORK_DIR / f"{site}_forest.geojson").to_crs(epsg)
    near_turbine = roads.distance(union(turbines.geometry)) <= ACCESS_WITHIN_M
    access = roads[roads["highway"].isin(ACCESS_TYPES) & near_turbine]
    public = roads[roads["highway"].isin(PUBLIC_TYPES)]
    infra = union(gpd.GeoSeries(list(turbines.buffer(TURBINE_RADIUS_M)) + list(access.geometry), crs=epsg))
    forest_union = union(forest.geometry)
    d_infra = pts.distance(infra).values
    d_public = pts.distance(union(public.geometry)).values
    in_forest = pts.within(forest_union).values
    sx, sy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(*MARKERS[site])
    d_slide = np.hypot(px - sx, py - sy)
    usable = (d_infra >= MIN_FROM_INFRA_M) & (d_public >= MIN_FROM_PUBLIC_M) & (d_slide > SLIDE_EXCLUDE_M)
    cover = {"open": usable & ~in_forest, "forest": usable & in_forest}

    # 3. Coverage: share of ground measured, EGMS vs NISAR.
    box_utm = gpd.GeoSeries([box], crs="EPSG:4326").to_crs(epsg).iloc[0]
    area_total = box_utm.area / 1e6
    area_forest = box_utm.intersection(forest_union).area / 1e6
    area_open = area_total - area_forest
    egms_forest = in_forest.sum() * 0.01 / area_forest * 100      # each EGMS point = 100 m x 100 m
    egms_open = (~in_forest).sum() * 0.01 / area_open * 100
    shape = nisar["vertical_mm"].shape[1:]
    transform, _ = grid_transform(nisar["x"], nisar["y"])
    valid = np.all(np.isfinite(nisar["vertical_mm"]), axis=0)
    forest_grid = burn(forest.geometry, shape, transform, all_touched=False)
    nisar_forest = (valid & forest_grid).sum() * 0.0016 / area_forest * 100   # each NISAR pixel = 40 m x 40 m
    nisar_open = (valid & ~forest_grid).sum() * 0.0016 / area_open * 100
    name_nisar = nisar_files[-1].stem.replace("_timeseries", "")
    print(f"\nShare of ground measured (box: {area_forest:.0f} km² forest, {area_open:.0f} km² open):")
    print(f"  EGMS (Sentinel-1 C-band, 100 m):     forest {egms_forest:5.1f}%   open {egms_open:5.1f}%")
    print(f"  NISAR {name_nisar} (L-band, 40 m):   forest {nisar_forest:5.1f}%   open {nisar_open:5.1f}%")

    # 4. Distance bands for velocity and seasonality.
    rows = []
    print("\nDistance bands (same definition as Step D), relative to the 1-3 km reference:")
    print("  measure          class    band (m)     points   difference         verdict")
    profile = {}
    for column, unit in [("mean_velocity", "mm/yr"), ("seasonality", "mm")]:
        values = up[column].values.astype(float)
        for cls in CLASSES:
            ref = (d_infra >= REFERENCE[0]) & (d_infra < REFERENCE[1]) & cover[cls]
            for lo, hi in BANDS:
                band = (d_infra >= lo) & (d_infra < hi) & cover[cls]
                result = band_difference(values, band, ref, rng)
                if result is None:
                    print(f"  {column:15s}  {cls:7s}  {lo:4d}-{hi:<5d}  {int(band.sum()):6d}   not enough points")
                    continue
                diff, unc = result
                if abs(diff) > 2 * unc:
                    text = f"differs ({'lower' if diff < 0 else 'higher'} near the wind farm)"
                else:
                    text = f"no difference beyond noise (limit about +/-{2 * unc:.1f} {unit})"
                print(f"  {column:15s}  {cls:7s}  {lo:4d}-{hi:<5d}  {int(band.sum()):6d}   "
                      f"{diff:+6.2f} +/- {unc:4.2f} {unit:5s}  {text}")
                rows.append({"measure": column, "unit": unit, "class": cls, "band_m": f"{lo}-{hi}",
                             "points": int(band.sum()), "difference": round(diff, 3),
                             "uncertainty": round(unc, 3), "verdict": text})
                profile.setdefault((column, cls), []).append(((lo + hi) / 2, diff, unc))
    print("  (neighbouring EGMS points are not fully independent, so treat borderline results cautiously)")

    # 6. The slide area after the slide.
    slide_area = d_slide <= SLIDE_AREA_M
    ring = (d_slide >= RING_M[0]) & (d_slide <= RING_M[1])
    print(f"\nThe 2020 slide area after the slide (within {SLIDE_AREA_M} m vs {RING_M[0]}-{RING_M[1]} m around):")
    for label, mask in [("slide area", slide_area), ("surroundings", ring)]:
        if mask.sum():
            print(f"  {label:12s}: {int(mask.sum()):4d} points, velocity {np.median(up['mean_velocity'][mask]):+.1f} mm/yr, "
                  f"seasonality {np.median(up['seasonality'][mask]):.1f} mm")
        else:
            print(f"  {label:12s}: no EGMS points")

    # 7. East-west sanity check.
    merged = up[["easting", "northing", "mean_velocity"]].merge(
        east[["easting", "northing", "mean_velocity"]], on=["easting", "northing"], how="left", suffixes=("_u", "_e"))
    ve = merged["mean_velocity_e"].values
    print(f"\nEast-west velocity: median {np.nanmedian(ve):+.2f} mm/yr, middle 90% between "
          f"{np.nanpercentile(ve, 5):+.2f} and {np.nanpercentile(ve, 95):+.2f} mm/yr "
          f"(vertical: {np.percentile(up['mean_velocity'], 5):+.2f} to {np.percentile(up['mean_velocity'], 95):+.2f})")

    # Outputs: summary CSV and points file for the viewer.
    pd.DataFrame(rows).to_csv(WORK_DIR / f"{site}_egms_bands.csv", index=False)
    lonlat = points_egms.to_crs("EPSG:4326")
    pd.DataFrame({"lon": lonlat.x.round(6), "lat": lonlat.y.round(6),
                  "velocity_up_mm_yr": up["mean_velocity"], "velocity_east_mm_yr": np.round(ve, 2),
                  "seasonality_mm": up["seasonality"], "forest": in_forest,
                  "distance_to_windfarm_m": np.round(d_infra).astype(int)}).to_csv(
        WORK_DIR / f"{site}_egms_points.csv", index=False)

    # Map of vertical velocity.
    vel = up["mean_velocity"].values
    limit = max(2.0, float(np.percentile(np.abs(vel), 98)))
    fig, ax = plt.subplots(figsize=(11, 10))
    forest.boundary.plot(ax=ax, color="forestgreen", linewidth=0.5)
    public.plot(ax=ax, color="gray", linewidth=0.6)
    access.plot(ax=ax, color="saddlebrown", linewidth=1)
    sc = ax.scatter(px, py, c=vel, cmap="RdBu", vmin=-limit, vmax=limit, s=6)
    turbines.plot(ax=ax, color="black", marker="^", markersize=30)
    ax.plot(sx, sy, marker="x", color="black", markersize=12, mew=3)
    minx, miny, maxx, maxy = box_utm.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Vertical velocity (mm/year, blue = rising)")
    ax.set_title(f"EGMS vertical velocity, {dates[0]:%b %Y} to {dates[-1]:%b %Y}\n"
                 "green = forest edge, brown = wind farm tracks, triangles = turbines")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(WORK_DIR / f"{site}_egms_velocity_map.png", dpi=110)
    plt.close(fig)

    # Band profiles.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (column, unit) in zip(axes, [("mean_velocity", "mm/yr"), ("seasonality", "mm")]):
        for cls, colour in zip(CLASSES, ["darkgoldenrod", "forestgreen"]):
            if (column, cls) in profile:
                mids, diffs, uncs = zip(*profile[(column, cls)])
                ax.errorbar(mids, diffs, yerr=np.array(uncs) * 2, marker="o", capsize=4, color=colour, label=cls)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Distance from wind farm infrastructure (m, band centre)")
        ax.set_ylabel(f"Difference from 1-3 km reference ({unit})")
        ax.set_title(f"{column} (error bars = 2 x uncertainty)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(WORK_DIR / f"{site}_egms_distance_profile.png", dpi=110)
    plt.close(fig)

    # 5. Over time, with monthly rainfall.
    series = up[[str(c) for c in date_cols]].values.astype(float)
    rain = None
    if all(p.exists() for p in RAIN_FILES):
        _, main_rain = load_rain(RAIN_FILES[0])
        _, fill_rain = load_rain(RAIN_FILES[1])
        daily, _ = combine(main_rain, fill_rain)
        rain = daily[(daily.index >= dates[0]) & (daily.index <= dates[-1])].resample("MS").sum()
    fig, axes = plt.subplots(len(CLASSES) + 1, 1, figsize=(13, 4 * (len(CLASSES) + 1)), sharex=True)
    for ax, cls in zip(axes[:-1], CLASSES):
        ref = (d_infra >= REFERENCE[0]) & (d_infra < REFERENCE[1]) & cover[cls]
        if ref.sum() < MIN_POINTS:
            ax.set_title(f"{cls}: not enough reference points")
            continue
        ref_series = np.nanmedian(series[ref], axis=0)
        for lo, hi in BANDS:
            band = (d_infra >= lo) & (d_infra < hi) & cover[cls]
            if band.sum() >= MIN_POINTS:
                ax.plot(dates, np.nanmedian(series[band], axis=0) - ref_series, linewidth=1, label=f"{lo}-{hi} m")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("vs 1-3 km reference (mm)")
        ax.set_title(f"{cls}: vertical motion near the wind farm relative to the reference")
        ax.legend(fontsize=8)
    ax = axes[-1]
    if slide_area.sum() and ring.sum():
        ax.plot(dates, np.nanmedian(series[slide_area], axis=0) - np.nanmedian(series[ring], axis=0),
                color="black", linewidth=1, label="slide area minus surroundings")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("mm")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_title("2020 slide area after the slide" + ("" if rain is None else ", with monthly rainfall (bars)"))
    if rain is not None:
        bars = ax.twinx()
        bars.bar(rain.index, rain.values, width=20, color="steelblue", alpha=0.3)
        bars.set_ylabel("Monthly rain (mm)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(WORK_DIR / f"{site}_egms_over_time.png", dpi=110)
    plt.close(fig)

    elevation_check(site, up, d_infra, cover)
    print(f"\nSaved: {site}_egms_bands.csv, {site}_egms_points.csv, {site}_egms_velocity_map.png, "
          f"{site}_egms_distance_profile.png, {site}_egms_over_time.png")


if __name__ == "__main__":
    main()