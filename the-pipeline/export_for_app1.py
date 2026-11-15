# pipeline/export_for_app.py
# Turns analysis results into web files for the BogWatch dashboard (app/public/data):
# - NISAR motion frames and 2020 slide replay: see-through PNGs + the real values (int16 binary) for click-to-inspect
# - EGMS points (JSON) + their full time series (int16 binary)
# - daily rainfall (JSON), infrastructure and forest (GeoJSON), findings tables
# - manifest.json: the data contract (files, bounds, dates, colour scales, value encodings, credits)
# Usage:
#     python pipeline\export_for_app.py meenbog

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import shapely
from matplotlib import colormaps
from matplotlib.image import imsave
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely import wkt as shapely_wkt

from config1 import RAW_DIR, SITES, WORK_DIR
from coherence_check1 import MARKERS
from distance_bands1 import (ACCESS_TYPES, ACCESS_WITHIN_M, MIN_FROM_INFRA_M, TURBINE_RADIUS_M, grid_transform, union)
from rainfall_check1 import combine, load_rain

OUT = Path(__file__).resolve().parent.parent / "app" / "public" / "data"
LONLAT = CRS.from_epsg(4326)
NODATA = -32768                    # "no value" in the int16 files
MOTION_LIMIT_MM = 20               # NISAR colour scale: -20 to +20 mm
EGMS_LIMIT = 8                     # EGMS colour scale: -8 to +8 mm/year
RAIN_FILES = [RAW_DIR / "weather" / "met_eireann_ardnawark_daily.csv",
              RAW_DIR / "weather" / "met_eireann_finner_daily.csv"]
CREDITS = [
    "NISAR data: NASA/JPL-Caltech and ISRO, via the Alaska Satellite Facility (ASF DAAC)",
    "Sentinel-1 slide replay: contains modified Copernicus Sentinel data (2020), processed by ASF HyP3",
    "EGMS: European Union, Copernicus Land Monitoring Service, European Environment Agency (EEA)",
    "Rainfall: Met Eireann (CC BY 4.0), Ardnawark Barnesmore with gaps filled from Finner",
    "Turbines, tracks, forest: (c) OpenStreetMap contributors (ODbL)",
    "Terrain and imagery in the viewer: Cesium ion (incl. Sentinel-2 imagery, Copernicus)",
    "Recorded landslides: Geological Survey Ireland (CC BY 4.0)",
    "Peat soils: Irish Peat Soils Map, Gilet et al. 2024, Geoderma 447 (CC BY 4.0)",
    "Protected areas: National Parks and Wildlife Service (SAC, SPA, NHA)",
]


def north_up(array, transform):
    """Make row 0 the northern edge (some grids store south first)."""
    if transform.e > 0:
        h = array.shape[-2]
        array = array[..., ::-1, :]
        transform = Affine(transform.a, 0, transform.c, 0, -transform.e, transform.f + transform.e * h)
    return array, transform


def lonlat_grid(shape, transform, crs):
    """A latitude/longitude grid covering the same area, plus its bounds [west, south, east, north]."""
    h, w = shape
    dst_transform, dw, dh = calculate_default_transform(crs, LONLAT, w, h, *array_bounds(h, w, transform))
    west, north = dst_transform.c, dst_transform.f
    east, south = west + dst_transform.a * dw, north + dst_transform.e * dh
    return dst_transform, (dh, dw), [round(west, 6), round(south, 6), round(east, 6), round(north, 6)]


def warp(array, transform, crs, grid):
    """Reproject one 2D picture onto the latitude/longitude grid. No-data stays NaN."""
    dst_transform, shape, _ = grid
    out = np.full(shape, np.nan, dtype="float32")
    reproject(source=np.ascontiguousarray(array, dtype="float32"), destination=out,
              src_transform=transform, src_crs=crs, src_nodata=np.nan,
              dst_transform=dst_transform, dst_crs=LONLAT, dst_nodata=np.nan,
              resampling=Resampling.nearest)
    return out


def save_png(array, path, cmap, vmin, vmax, hide=None, alpha=0.85):
    """Colour a picture; no-data (and anything in 'hide') becomes transparent."""
    scaled = np.clip((array - vmin) / (vmax - vmin), 0, 1)
    rgba = colormaps[cmap](np.nan_to_num(scaled))
    visible = np.isfinite(array) if hide is None else np.isfinite(array) & ~hide
    rgba[..., 3] = np.where(visible, alpha, 0)
    imsave(path, (rgba * 255).astype(np.uint8))


def save_int16(name, stack, scale):
    """Store values as int16 (value x scale), NaN as NODATA. Row-major, little-endian."""
    stack = np.asarray(stack, dtype="float64")
    packed = np.where(np.isfinite(stack), np.clip(np.round(stack * scale), -32767, 32767), NODATA).astype("<i2")
    packed.tofile(OUT / name)
    return {"file": name, "scale": scale, "nodata": NODATA}


def export_motion(key, cube, x, y, epsg, dates, title, description, note):
    """One PNG per date (movement since the first date) + the values for click-to-inspect."""
    transform, _ = grid_transform(x, y)
    crs = CRS.from_epsg(epsg)
    cube, transform = north_up(cube, transform)
    grid = lonlat_grid(cube.shape[1:], transform, crs)
    warped = [warp(cube[t], transform, crs, grid) for t in range(len(dates))]
    frames = []
    for d, picture in zip(dates, warped):
        name = f"{key}_{d}.png"
        save_png(picture, OUT / name, "RdBu", -MOTION_LIMIT_MM, MOTION_LIMIT_MM)
        frames.append({"date": d, "file": name})
    values = save_int16(f"{key}_values.bin", warped, 10)
    values.update({"width": int(grid[1][1]), "height": int(grid[1][0])})
    print(f"  {key}: {len(frames)} frames, {grid[1][1]} x {grid[1][0]} pixels")
    return {"id": key, "type": "image-frames", "title": title, "description": description, "note": note,
            "bounds": grid[2], "frames": frames, "units": "mm", "values": values,
            "colourScale": {"colormap": "RdBu", "min": -MOTION_LIMIT_MM, "max": MOTION_LIMIT_MM,
                            "lowLabel": "down", "highLabel": "up"}}


def export_event(site):
    """2020 slide replay: before / across / after coherence and the coherence drop, per gap length."""
    path = WORK_DIR / f"{site}_event_replay.npz"
    if not path.exists():
        return []
    ev = np.load(path)
    crs = CRS.from_string(str(ev["crs"]))
    base = Affine(*[float(v) for v in ev["transform"][:6]])
    layers = []
    for span in ("6d", "12d"):
        if f"coh_before_{span}" not in ev.files:
            continue
        raw = {k: ev[f"coh_{k}_{span}"] for k in ("before", "across", "after")}
        raw["drop"] = raw["before"] - raw["across"]
        oriented = {k: north_up(a, base) for k, a in raw.items()}
        transform = oriented["before"][1]
        grid = lonlat_grid(oriented["before"][0].shape, transform, crs)
        warped = {k: warp(a, transform, crs, grid) for k, (a, _) in oriented.items()}
        frames = []
        for k in ("before", "across", "after"):
            name = f"event_{span}_{k}.png"
            save_png(warped[k], OUT / name, "gray", 0, 1, alpha=0.9)
            frames.append({"date": k, "file": name})
        name = f"event_{span}_drop.png"
        save_png(warped["drop"], OUT / name, "magma", 0, 0.8, hide=~(warped["drop"] > 0.15), alpha=0.9)
        frames.append({"date": "drop", "file": name})
        values = save_int16(f"event_{span}_values.bin", [warped[k] for k in ("before", "across", "after", "drop")], 1000)
        values.update({"width": int(grid[1][1]), "height": int(grid[1][0])})
        gap = span.replace("d", "-day")
        layers.append({"id": f"event_{span}", "type": "image-frames", "title": f"Slide replay, {gap} pairs",
                       "description": "Sentinel-1 coherence before, across and after the 12 Nov 2020 Meenbog "
                                      "slide (white = unchanged ground), then where coherence dropped across it.",
                       "note": "Coherence near 1 means the ground did not change between images; a slide destroys it.",
                       "bounds": grid[2], "frames": frames, "units": "coherence (0-1)", "values": values,
                       "colourScale": {"colormap": "gray", "min": 0, "max": 1,
                                       "lowLabel": "changed", "highLabel": "unchanged"}})
        print(f"  event_{span}: before / across / after / drop")
    return layers


def export_egms(site):
    """EGMS points inside the site box (JSON) and their full time series (int16, mm x 10)."""
    cache = WORK_DIR / f"{site}_egms_U.csv.gz"
    if not cache.exists():
        print("  egms: no cached EGMS data (run egms_analysis.py first)")
        return None
    table = pd.read_csv(cache)
    date_cols = [c for c in table.columns if str(c).isdigit() and len(str(c)) == 8]
    lon, lat = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True).transform(
        table["easting"].values, table["northing"].values)
    west, south, east, north = shapely_wkt.loads(SITES[site]["wkt"]).bounds
    inside = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    points = np.column_stack([np.round(lon[inside], 6), np.round(lat[inside], 6),
                              np.round(table["mean_velocity"].values[inside], 2),
                              np.round(table["seasonality"].values[inside], 2)])
    (OUT / "egms_points.json").write_text(json.dumps(
        {"columns": ["lon", "lat", "velocity_up_mm_yr", "seasonality_mm"], "points": points.tolist()}))
    series = save_int16("egms_series.bin", table[date_cols].values[inside], 10)
    series["dates"] = [str(c) for c in date_cols]
    print(f"  egms: {int(inside.sum())} points x {len(date_cols)} dates")
    return {"id": "egms", "type": "points", "title": "EGMS vertical velocity, Dec 2020 to 2025",
            "description": "Sentinel-1 ground motion (EGMS Ortho, 100 m cells). Dot colour = average vertical "
                           "velocity; blue = rising, red = sinking.",
            "file": "egms_points.json", "units": "mm/year", "series": series,
            "colourScale": {"colormap": "RdBu", "min": -EGMS_LIMIT, "max": EGMS_LIMIT,
                            "lowLabel": "sinking", "highLabel": "rising"}}


def export_rain():
    """Daily rainfall from 2019: Ardnawark, gaps filled from Finner, with the source of each day."""
    if not RAIN_FILES[0].exists():
        return None
    _, main_rain = load_rain(RAIN_FILES[0])
    fill = load_rain(RAIN_FILES[1])[1] if RAIN_FILES[1].exists() else None
    daily, source = combine(main_rain, fill)
    keep = daily.index >= "2019-01-01"
    daily, source = daily[keep], source[keep]
    (OUT / "rainfall.json").write_text(json.dumps({
        "start": daily.index[0].strftime("%Y-%m-%d"),
        "mm": [None if not np.isfinite(v) else round(float(v), 1) for v in daily.values],
        "source": [{"local": "l", "filled": "f"}.get(s) for s in source.values],
        "stations": {"l": "Ardnawark Barnesmore", "f": "Finner"}}))
    print(f"  rainfall: {len(daily)} days from {daily.index[0]:%d %b %Y}")
    return "rainfall.json"

CLUSTER_RADIUS_M = 3000   # turbines within this distance of the slide = this site's wind farm


def export_carbon(site, epsg):
    """Areas around this wind farm's infrastructure, for the carbon calculator."""
    turbines = gpd.read_file(WORK_DIR / f"{site}_turbines.geojson").to_crs(epsg)
    roads = gpd.read_file(WORK_DIR / f"{site}_roads.geojson").to_crs(epsg)
    sx, sy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(*MARKERS[site])
    farm = turbines[turbines.distance(shapely.Point(sx, sy)) <= CLUSTER_RADIUS_M]
    access = roads[roads["highway"].isin(ACCESS_TYPES) & (roads.distance(union(farm.geometry)) <= ACCESS_WITHIN_M)]
    infra = union(gpd.GeoSeries(list(farm.buffer(TURBINE_RADIUS_M)) + list(access.geometry), crs=f"EPSG:{epsg}"))
    box = gpd.GeoSeries([shapely_wkt.loads(SITES[site]["wkt"])], crs="EPSG:4326").to_crs(epsg).iloc[0]
    footprint = infra.buffer(MIN_FROM_INFRA_M)
    areas = {str(d): round(infra.buffer(d).difference(footprint).intersection(box).area / 1e4, 1)
             for d in (250, 500, 1000)}
    carbon = {"turbines": int(len(farm)), "track_km": round(access.length.sum() / 1000, 1),
              "footprint_ha": round(footprint.intersection(box).area / 1e4, 1), "areas_ha": areas}
    print(f"  carbon: {carbon['turbines']} turbines, {carbon['track_km']} km of tracks, areas (ha) {areas}")
    return carbon

SCREEN_COLOURS = [(58, 157, 93), (224, 165, 38), (214, 69, 65)]   # lower, moderate, higher


def export_screening(site):
    """The screening map as a coloured picture + score and reasons for click-to-inspect."""
    path = WORK_DIR / f"{site}_screening.npz"
    if not path.exists():
        return None, {}
    s = np.load(path)
    crs = CRS.from_wkt(str(s["crs"]))
    base = Affine(*[float(v) for v in s["transform"][:6]])
    score, transform = north_up(s["score"], base)
    reasons, _ = north_up(s["reasons"], base)
    grid = lonlat_grid(score.shape, transform, crs)
    score_ll = warp(score, transform, crs, grid)
    reasons_ll = warp(reasons, transform, crs, grid)
    classes = np.where(score_ll <= 2, 0, np.where(score_ll <= 4, 1, 2))
    rgba = np.zeros(score_ll.shape + (4,), dtype=np.uint8)
    for k, (r, g, b) in enumerate(SCREEN_COLOURS):
        rgba[(classes == k) & np.isfinite(score_ll)] = (r, g, b, 150)
    imsave(OUT / "screening.png", rgba)
    size = {"width": int(grid[1][1]), "height": int(grid[1][0])}
    values = save_int16("screening_values.bin", [score_ll], 1)
    values.update(size)
    reasons_meta = save_int16("screening_reasons.bin", [reasons_ll], 1)
    reasons_meta.update(size)
    hindcast_path = WORK_DIR / f"{site}_screening_hindcast.json"
    hindcast = json.loads(hindcast_path.read_text()) if hindcast_path.exists() else None
    files = {}
    for key in ("landslides", "protected"):
        src = WORK_DIR / f"{site}_{key}.geojson"
        if src.exists():
            (OUT / f"{key}.geojson").write_bytes(src.read_bytes())
            files[key] = f"{key}.geojson"
    auc_text = f" Hindcast AUC {hindcast['auc_all']:.2f} against {hindcast['landslides']} recorded landslides." \
        if hindcast else ""
    print(f"  screening: {size['width']} x {size['height']} cells{auc_text}")
    layer = {"id": "screening", "type": "image-frames", "title": "Where not to build: damage-risk screening",
             "description": "Each 40 m patch scored from slope, peat soil, nearness to rivers and streams, and EGMS "
                            "ground movement. Click a patch for its reasons." + auc_text,
             "note": "A screening tool showing where site investigation is needed first, not a planning decision "
                     "and not a prediction of collapse.",
             "bounds": grid[2], "frames": [{"date": "screening", "file": "screening.png"}], "units": "risk score",
             "values": values, "reasons": reasons_meta, "hindcast": hindcast,
             "colourScale": {"type": "classes", "classes": [
                 {"label": "Lower (score 0–2)", "colour": "#3a9d5d"},
                 {"label": "Moderate (3–4)", "colour": "#e0a526"},
                 {"label": "Higher (5+)", "colour": "#d64541"}]}}
    return layer, files

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\export_for_app.py <site>")
        return
    site = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {OUT}")
    layers = []

    # NISAR: combined true vertical, then the descending track (more coverage).
    combined = np.load(WORK_DIR / f"{site}_combined.npz")
    layers.append(export_motion(
        "nisar_up", combined["up_mm"], combined["x"], combined["y"], int(combined["epsg"]),
        [str(d) for d in combined["dates"]], "NISAR vertical motion (ascending + descending)",
        "True vertical movement since the first shared date, from NISAR tracks 58 (ascending) and 9 "
        "(descending) combined. Blue = up, red = down.",
        "Single pixels are noisy (about +/-15 mm); trust patterns covering many pixels."))
    desc_id = next((k for k in SITES if k.startswith(f"{site}_") and SITES[k].get("direction") == "DESCENDING"
                    and (WORK_DIR / f"{k}_timeseries.npz").exists()), None)
    if desc_id:
        d = np.load(WORK_DIR / f"{desc_id}_timeseries.npz")
        layers.append(export_motion(
            "nisar_desc", d["vertical_mm"], d["x"], d["y"], int(d["epsg"]), [str(v) for v in d["dates"]],
            "NISAR descending track (more coverage)",
            "Movement from one NISAR pass (track 9), converted to vertical assuming no sideways motion. "
            "Fuller coverage than the combined layer, especially under forest.",
            "Assumes vertical movement; single pixels are noisy (about +/-15 mm)."))

    layers.extend(export_event(site))
    egms = export_egms(site)
    if egms:
        layers.append(egms)
    rainfall = export_rain()
    screening_layer, screening_files = export_screening(site)
    if screening_layer:
        layers.append(screening_layer)

    # Infrastructure and forest (simplified, rounded to ~0.1 m).
    epsg = int(combined["epsg"])
    turbines = gpd.read_file(WORK_DIR / f"{site}_turbines.geojson")
    roads = gpd.read_file(WORK_DIR / f"{site}_roads.geojson").to_crs(epsg)
    near = roads.distance(union(turbines.to_crs(epsg).geometry)) <= ACCESS_WITHIN_M
    tracks = roads[roads["highway"].isin(ACCESS_TYPES) & near].to_crs("EPSG:4326")
    forest = gpd.read_file(WORK_DIR / f"{site}_forest.geojson").to_crs(epsg)
    forest["geometry"] = forest.geometry.simplify(10)
    forest = forest.to_crs("EPSG:4326")
    for gdf, name, columns in [(turbines, "turbines", ["name"]), (tracks, "tracks", ["highway"]),
                               (forest, "forest", [])]:
        keep = gdf[[c for c in columns if c in gdf.columns] + ["geometry"]].copy()
        keep["geometry"] = shapely.set_precision(keep.geometry.values, 1e-6)
        keep.to_file(OUT / f"{name}.geojson", driver="GeoJSON")
    layers.append({"id": "infrastructure", "type": "geojson", "title": "Wind farm infrastructure and forest",
                    "files": {"turbines": "turbines.geojson", "tracks": "tracks.geojson", "forest": "forest.geojson",**screening_files},
                   "description": "Turbines and access tracks (within 600 m of a turbine), and forest, "
                                  "from OpenStreetMap."})
    print(f"  infrastructure: {len(turbines)} turbines, {len(tracks)} track segments, {len(forest)} forest areas")

    # Findings tables.
    findings = {}
    for key, fname in [("nisar_distance_bands", f"{site}_distance_bands.csv"),
                       ("egms_distance_bands", f"{site}_egms_bands.csv")]:
        path = WORK_DIR / fname
        if path.exists():
            findings[key] = pd.read_csv(path).to_dict(orient="records")
    key_path = WORK_DIR / f"{site}_key_findings.json"
    if key_path.exists():
        findings["key"] = json.loads(key_path.read_text(encoding="utf-8"))        
    (OUT / "findings.json").write_text(json.dumps(findings, indent=1))
    
    carbon = export_carbon(site, epsg)

    # Manifest: the data contract.
    box = shapely_wkt.loads(SITES[site]["wkt"])
    lon, lat = MARKERS[site]
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site": {"id": site, "name": SITES[site]["name"], "bounds": [round(v, 6) for v in box.bounds],
                 "slide": {"lon": lon, "lat": lat, "date": "2020-11-12", "label": "2020 peat slide"}},
        "layers": layers, "findings": "findings.json", "rainfall": rainfall, "carbon": carbon, "credits": CREDITS,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) / 1e6
    print(f"Saved manifest.json ({len(layers)} layers); total size {total:.1f} MB")


if __name__ == "__main__":
    main()