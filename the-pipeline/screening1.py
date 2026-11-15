# pipeline/screening.py
# "Where not to build": a transparent damage-risk screening map, plus a hindcast against recorded landslides.
# Score per 40 m cell: slope class (0-3) + peat soil (+2) + near a river/stream (+1)
#   + seasonal movement above normal for its height (+1, EGMS) + sinking faster than 4 mm/yr (+1, EGMS).
# Protected areas are recorded as a constraint, not scored. 0-2 lower, 3-4 moderate, 5+ higher.
# The hindcast uses the terrain part only (slope, peat, streams) over a wider region.
# Usage:
#     python pipeline\screening.py meenbog

import json
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely import wkt as shapely_wkt
from shapely.geometry import box as make_box

from config1 import RAW_DIR, SITES, WORK_DIR
from coherence_check1 import MARKERS
from distance_bands1 import burn

REGION = {"meenbog": (-8.45, 54.55, -7.75, 54.95)}   # west, south, east, north (Donegal uplands)
SCREEN_DIR = RAW_DIR / "screening"
SLOPE_STEPS = [5, 10, 15]          # degrees: <5 = 0, 5-10 = 1, 10-15 = 2, >15 = 3
PEAT_POINTS = 2
STREAM_M = 100
SEASON_EXCESS_MM = 1.0             # seasonal cycle this much above same-height ground
SINKING_MM_YR = -4.0
ELEV_STEP = 25
CLASS_EDGES = (2, 4)               # <=2 lower, 3-4 moderate, >=5 higher
CLASS_NAMES = ["Lower", "Moderate", "Higher"]

ox.settings.use_cache = True
ox.settings.cache_folder = str(WORK_DIR / "osm_cache")
ox.settings.requests_timeout = 300


def unzip_all():
    """Unzip each download once and sort its shapefiles by what they are."""
    groups = {"landslides": [], "peat": [], "protected": []}
    for item in sorted(SCREEN_DIR.iterdir()):
        if item.suffix.lower() == ".zip":
            target = SCREEN_DIR / item.stem
            if not target.exists():
                with zipfile.ZipFile(item) as z:
                    z.extractall(target)
        elif item.is_dir():
            target = item
        else:
            continue
        name = target.name.lower()
        if name.startswith(("sac", "spa", "nha")):
            key = "protected"
        elif "landslide" in name:
            key = "landslides"
        elif "peat" in name:
            key = "peat"
        else:
            continue
        for shp in sorted(target.rglob("*.shp")):
            name_lower = shp.name.lower()
            if key == "landslides" and any(word in name_lower for word in ("suscept", "class", "extent")):
                continue   # use the landslide Locations layer only
            if key == "peat" and name_lower != "ipsm_mc28.shp":
                continue   # use the final Irish Peat Soils Map layer only
            if shp not in groups[key]:
                groups[key].append(shp)
    return groups


def read_layers(paths, region_ll, crs, label):
    """Read only the features inside the region, in the working projection."""
    area = gpd.GeoSeries([make_box(*region_ll)], crs="EPSG:4326")
    parts = []
    for path in paths:
        try:
            gdf = gpd.read_file(path, bbox=area)
        except Exception as error:
            print(f"  could not read {path.name}: {error}")
            continue
        if len(gdf) and gdf.crs is None:
            print(f"  skipping {path.name}: no projection information")
            continue
        if len(gdf):
            parts.append(gdf.to_crs(crs))
            print(f"    {path.name}: {len(gdf)} features")
    if not parts:
        print(f"  {label}: nothing found in the region")
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs)
    print(f"  {label}: {len(out)} features from {len(parts)} file(s)")
    return out


def load_dem(site, region_ll):
    """The elevation model inside the HyP3 zips (Copernicus DEM resampled to the product grid)."""
    layer_dir = WORK_DIR / f"{site}_s1_event_layers"
    layer_dir.mkdir(parents=True, exist_ok=True)
    dem_path = next(iter(sorted(layer_dir.glob("*_dem.tif"))), None)
    if dem_path is None:
        zips = sorted((RAW_DIR / f"{site}_s1_event").glob("*.zip"))
        with zipfile.ZipFile(zips[0]) as z:
            member = next(m for m in z.namelist() if m.endswith("_dem.tif"))
            dem_path = layer_dir / Path(member).name
            dem_path.write_bytes(z.read(member))
    with rasterio.open(dem_path) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *region_ll)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        fill = src.nodata if src.nodata is not None else -9999
        dem = src.read(1, window=window, boundless=True, fill_value=fill).astype("float32")
        transform, crs = src.window_transform(window), src.crs
    dem[(dem == fill) | (dem < -100)] = np.nan
    return dem, transform, crs


def stream_mask(region_ll, shape, transform, crs):
    """Cells within STREAM_M of a river or stream (OpenStreetMap)."""
    try:
        water = ox.features_from_polygon(make_box(*region_ll), {"waterway": ["river", "stream"]})
        water = water[water.geom_type.isin(["LineString", "MultiLineString"])].to_crs(crs)
        lines = burn(water.geometry, shape, transform)
        dist = ndimage.distance_transform_edt(~lines, sampling=(abs(transform.e), abs(transform.a)))
        print(f"  rivers and streams: {len(water)} features")
        return dist <= STREAM_M
    except Exception as error:
        print(f"  rivers and streams unavailable ({type(error).__name__}); scored as 0")
        return np.zeros(shape, dtype=bool)


def motion_flags(site, shape, transform, crs):
    """EGMS: seasonal cycle above normal for the same height, and fast sinking. Only where EGMS has points."""
    cache = WORK_DIR / f"{site}_egms_U.csv.gz"
    empty = np.zeros(shape, dtype=bool)
    if not cache.exists():
        print("  EGMS cache not found; motion factors scored as 0")
        return empty, empty, empty
    t = pd.read_csv(cache, usecols=["easting", "northing", "height_ortho", "mean_velocity", "seasonality"])
    groups = np.floor(t["height_ortho"] / ELEV_STEP)
    excess = (t["seasonality"] - t.groupby(groups)["seasonality"].transform("median")).values
    x, y = Transformer.from_crs("EPSG:3035", crs, always_xy=True).transform(t["easting"].values, t["northing"].values)
    tree = cKDTree(np.column_stack([x, y]))
    cols, rows = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    cx = transform.c + (cols + 0.5) * transform.a
    cy = transform.f + (rows + 0.5) * transform.e
    dist, idx = tree.query(np.column_stack([cx.ravel(), cy.ravel()]), distance_upper_bound=71)
    found = np.isfinite(dist)
    season, sink = np.zeros(found.size, dtype=bool), np.zeros(found.size, dtype=bool)
    season[found] = excess[idx[found]] >= SEASON_EXCESS_MM
    sink[found] = t["mean_velocity"].values[idx[found]] <= SINKING_MM_YR
    print(f"  EGMS motion: {int(found.sum())} cells covered")
    return season.reshape(shape), sink.reshape(shape), found.reshape(shape)


def classify(score):
    return np.where(score <= CLASS_EDGES[0], 0, np.where(score <= CLASS_EDGES[1], 1, 2))


def auc(scores_all, scores_pos):
    """Chance that a landslide cell scores higher than a random cell (ties count half)."""
    values = np.arange(0, 8)
    neg = np.array([(scores_all == v).sum() for v in values], dtype=float)
    pos = np.array([(scores_pos == v).sum() for v in values], dtype=float)
    below = np.concatenate([[0], np.cumsum(neg)[:-1]])
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum())) if pos.sum() else float("nan")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\screening.py <site>")
        return
    site = sys.argv[1]
    region = REGION.get(site, shapely_wkt.loads(SITES[site]["wkt"]).buffer(0.15).bounds)
    print(f"Screening region {region} (west, south, east, north)")

    dem, transform, crs = load_dem(site, region)
    shape = dem.shape
    print(f"  elevation: {shape[1]} x {shape[0]} cells of {abs(transform.a):.0f} m")
    groups = unzip_all()
    peat = read_layers(groups["peat"], region, crs, "peat soils")
    if "gridcode" in peat.columns:
        peat = peat[peat["gridcode"] == 1]   # 1 = peat, 2 = not peat
        print(f"  peat soils kept (gridcode 1 = peat): {len(peat)} shapes")
    if "gridcode" in peat.columns:
        peat = peat[peat["gridcode"] == 1]   # 1 = peat, 2 = not peat
        print(f"  peat soils kept (gridcode 1 = peat): {len(peat)} shapes")
    slides = read_layers(groups["landslides"], region, crs, "recorded landslides")
    protected = read_layers(groups["protected"], region, crs, "protected areas")

    # Factors on the grid.
    valid = np.isfinite(dem)
    gy, gx = np.gradient(dem, abs(transform.e), abs(transform.a))
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    valid &= np.isfinite(slope)
    slope_pts = np.digitize(np.nan_to_num(slope), SLOPE_STEPS).astype("int16")
    peat_mask = burn(peat.geometry, shape, transform, all_touched=False)
    near_stream = stream_mask(region, shape, transform, crs)
    protected_mask = burn(protected.geometry, shape, transform, all_touched=False)
    season, sink, covered = motion_flags(site, shape, transform, crs)

    terrain = slope_pts + PEAT_POINTS * peat_mask + near_stream.astype("int16")
    score = (terrain + season + sink).astype("float32")
    reasons = (slope_pts | (peat_mask << 2) | (near_stream << 3) | (season << 4) | (sink << 5)
               | (protected_mask << 6)).astype("float32")
    score[~valid] = np.nan
    reasons[~valid] = np.nan

    # Landslide points (polygons become an inside point; near-duplicates removed).
    pts = slides.copy()
    if len(pts):
        pts["geometry"] = slides.geometry.representative_point()
        grid_xy = np.round(np.column_stack([pts.geometry.x, pts.geometry.y]) / 25)
        pts = pts[~pd.DataFrame(grid_xy).duplicated().values].copy()
        text = pts.drop(columns="geometry").select_dtypes(include="object")
        pts["peat"] = text.apply(lambda c: c.astype(str).str.contains("peat", case=False, na=False)).any(axis=1) \
            if len(text.columns) else False
    else:
        pts["peat"] = []

    # Hindcast on the terrain part only.
    r, c = rowcol(transform, pts.geometry.x.values, pts.geometry.y.values) if len(pts) else ([], [])
    r, c = np.asarray(r, dtype=int), np.asarray(c, dtype=int)
    inside = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    inside[inside] = valid[r[inside], c[inside]]
    ls_scores = terrain[r[inside], c[inside]]
    all_scores = terrain[valid]
    peat_flags = pts["peat"].values[inside].astype(bool)
    terrain_class = classify(terrain)
    print(f"\nHindcast (terrain only: slope, peat, streams) against {int(inside.sum())} recorded landslides "
          f"({int(peat_flags.sum())} mention peat):")
    print("  Rating      share of land   share of landslides   landslides vs average")
    rows_out = []
    for k, name in enumerate(CLASS_NAMES):
        land = float((terrain_class[valid] == k).mean() * 100)
        slid = float((classify(ls_scores) == k).mean() * 100) if len(ls_scores) else float("nan")
        ratio = slid / land if land else float("nan")
        rows_out.append({"rating": name, "land_pct": round(land, 1), "landslide_pct": round(slid, 1),
                         "ratio": round(ratio, 2)})
        print(f"  {name:9s}   {land:10.1f}%   {slid:14.1f}%      {ratio:8.2f}x")
    auc_all = auc(all_scores, ls_scores)
    auc_peat = auc(all_scores, ls_scores[peat_flags]) if peat_flags.any() else float("nan")
    print(f"  AUC, all landslides: {auc_all:.2f} | peat landslides only: {auc_peat:.2f} (0.5 = chance, 1 = perfect)")

    sx, sy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*MARKERS[site])
    sr, sc = rowcol(transform, sx, sy)
    slide_score = float(score[sr, sc]) if 0 <= sr < shape[0] and 0 <= sc < shape[1] else float("nan")
    slide_terrain = float(terrain[sr, sc]) if np.isfinite(slide_score) else float("nan")
    slide_name = CLASS_NAMES[int(classify(np.array(slide_score)))] if np.isfinite(slide_score) else "outside"
    print(f"  2020 Meenbog slide location: score {slide_score:.0f} ({slide_name}), terrain part {slide_terrain:.0f}")

    hindcast = {"region": list(region), "landslides": int(inside.sum()), "peat_landslides": int(peat_flags.sum()),
                "auc_all": round(auc_all, 2), "auc_peat": None if np.isnan(auc_peat) else round(auc_peat, 2),
                "classes": rows_out, "slide": {"score": slide_score, "rating": slide_name}}
    (WORK_DIR / f"{site}_screening_hindcast.json").write_text(json.dumps(hindcast, indent=1))

    # Save the grids, plus map files for the app.
    np.savez_compressed(WORK_DIR / f"{site}_screening.npz", score=score, reasons=reasons,
                        transform=np.array(transform)[:6], crs=crs.to_wkt(), region=np.array(region))
    if len(pts):
        out_pts = pts[["peat", "geometry"]].to_crs("EPSG:4326")
        out_pts.to_file(WORK_DIR / f"{site}_landslides.geojson", driver="GeoJSON")
    if len(protected):
        name_col = next((c for c in protected.columns if "name" in c.lower()), None)
        keep = (protected[[name_col, "geometry"]] if name_col else protected[["geometry"]]).copy()
        region_box = gpd.GeoSeries([make_box(*region)], crs="EPSG:4326").to_crs(crs).iloc[0]
        keep = gpd.clip(keep, region_box)                          # only the part inside the region
        keep["geometry"] = keep.geometry.simplify(25).boundary     # outlines only, much lighter to draw
        keep = keep[~keep.geometry.is_empty]
        keep.to_crs("EPSG:4326").to_file(WORK_DIR / f"{site}_protected.geojson", driver="GeoJSON")

    # Check picture.
    fig, ax = plt.subplots(figsize=(11, 10))
    extent = (transform.c, transform.c + transform.a * shape[1], transform.f + transform.e * shape[0], transform.f)
    shown = np.where(valid, classify(np.nan_to_num(score)), np.nan)
    ax.imshow(shown, extent=extent, cmap=ListedColormap(["#3a9d5d", "#e0a526", "#d64541"]), vmin=0, vmax=2,
              interpolation="nearest")
    if len(pts):
        ax.scatter(pts.geometry.x, pts.geometry.y, s=10, c="black", label="recorded landslide")
    site_box = gpd.GeoSeries([shapely_wkt.loads(SITES[site]["wkt"])], crs="EPSG:4326").to_crs(crs)
    site_box.boundary.plot(ax=ax, color="white", linewidth=1.5)
    ax.plot(sx, sy, marker="x", color="white", markersize=12, mew=3)
    ax.set_title(f"Screening: green lower, amber moderate, red higher | AUC {auc_all:.2f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(WORK_DIR / f"{site}_screening_map.png", dpi=100)
    plt.close(fig)
    print(f"\nSaved: {site}_screening.npz, {site}_screening_hindcast.json, {site}_screening_map.png "
          f"(+ landslides and protected areas as GeoJSON)")


if __name__ == "__main__":
    main()