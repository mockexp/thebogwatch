# pipeline/event_replay.py
# Step C: tracks the November 2020 Meenbog peat slide in Sentinel-1 HyP3 InSAR products.
# Compares coherence before / across / after the slide, at the slide versus surrounding bog,
# and makes coherence-drop and brightness-change maps.
# Usage:
#     python pipeline\event_replay.py meenbog

import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from shapely import wkt as shapely_wkt

from config1 import RAW_DIR, SITES, WORK_DIR
from coherence_check1 import MARKERS

SLIDE_DATE = datetime(2020, 11, 12)   # earliest reported date of the slide
CORE_M = 150                          # "slide core": within this distance of the slide centre
RING_M = (500, 1500)                  # "surrounding bog": between these distances
ZOOM_M = 2000                         # half-width of the zoomed pictures
DROP = 0.3                            # coherence drop counted as "flagged"
LAYERS = {"corr": "_corr.tif", "amp": "_amp.tif"}


def pair_dates(name):
    """Reference and secondary dates from a HyP3 product name."""
    m = re.search(r"(\d{8})T\d{6}_(\d{8})T\d{6}", name)
    return (datetime.strptime(m.group(1), "%Y%m%d"), datetime.strptime(m.group(2), "%Y%m%d")) if m else (None, None)


def extract_layers(zip_path, out_dir):
    """2. Copy just the coherence and brightness files out of a zip (the zip itself is untouched)."""
    found = {}
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            for key, ending in LAYERS.items():
                if member.endswith(ending):
                    target = out_dir / Path(member).name
                    if not target.exists():
                        target.write_bytes(z.read(member))
                    found[key] = target
    return found


def reference_grid(path, site):
    """3. A grid covering the site box, in the product's own map projection."""
    lon_min, lat_min, lon_max, lat_max = shapely_wkt.loads(site["wkt"]).bounds
    with rasterio.open(path) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, lon_min, lat_min, lon_max, lat_max)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        return src.crs, src.window_transform(window), (int(window.height), int(window.width))


def read_onto(path, crs, transform, shape):
    """3. Read a layer and resample it onto the common grid. Zeros are treated as 'no data'."""
    out = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    out[out == 0] = np.nan
    return out


def classify(ref, sec):
    """4. Where a pair sits relative to the slide."""
    if sec < SLIDE_DATE:
        return "before"
    if ref < SLIDE_DATE <= sec:
        return "across"
    return "after"


def brightness_change_db(amp_before, amp_after):
    """7. Brightness change in decibels (handles files already in dB)."""
    if np.nanmin(amp_before) < 0 or np.nanmin(amp_after) < 0:
        return amp_after - amp_before
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20 * np.log10(amp_after / amp_before)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\event_replay.py <site>")
        return
    site_id = sys.argv[1]
    site = SITES[site_id]
    zips = sorted((RAW_DIR / f"{site_id}_s1_event").glob("*.zip"))
    if not zips:
        print(f"No zips found in {RAW_DIR / f'{site_id}_s1_event'}")
        return
    layer_dir = WORK_DIR / f"{site_id}_s1_event_layers"
    layer_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. Dates and layers for every pair.
    pairs = []
    for z in zips:
        ref, sec = pair_dates(z.name)
        files = extract_layers(z, layer_dir)
        if ref is None or "corr" not in files:
            print(f"  Skipping {z.name}: couldn't read dates or find a coherence file.")
            continue
        pairs.append({"name": z.stem, "ref": ref, "sec": sec, "days": (sec - ref).days,
                      "group": classify(ref, sec), **files})
    pairs.sort(key=lambda p: (p["ref"], p["sec"]))
    print(f"Found {len(pairs)} pairs.\n")

    # 3. Common grid, and every pair resampled onto it.
    crs, transform, shape = reference_grid(pairs[0]["corr"], site)
    for p in pairs:
        p["coh"] = read_onto(p["corr"], crs, transform, shape)
        p["bright"] = read_onto(p["amp"], crs, transform, shape) if "amp" in p else None

    # Distance of every pixel from the slide centre.
    to_map = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    sx, sy = to_map.transform(*MARKERS[site_id])
    cols, rows = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    px = transform.c + (cols + 0.5) * transform.a
    py = transform.f + (rows + 0.5) * transform.e
    dist = np.hypot(px - sx, py - sy)
    core = dist <= CORE_M
    ring = (dist >= RING_M[0]) & (dist <= RING_M[1])

    # 6. Coherence at the slide core vs surrounding bog, for every pair.
    print("Pair                     days  group    coherence: slide core   surroundings")
    for p in pairs:
        p["core"] = float(np.nanmean(p["coh"][core]))
        p["ring"] = float(np.nanmean(p["coh"][ring]))
        print(f"{p['ref']:%d %b} -> {p['sec']:%d %b %Y}  {p['days']:4d}  {p['group']:7s}  "
              f"{p['core']:20.2f}   {p['ring']:12.2f}")

    # 5. Like-with-like sets: latest before, closest across, earliest after, per gap length.
    sets = {}
    for span in sorted({p["days"] for p in pairs}):
        group = [p for p in pairs if p["days"] == span]
        before = max((p for p in group if p["group"] == "before"), key=lambda p: p["sec"], default=None)
        across = min((p for p in group if p["group"] == "across"), key=lambda p: p["sec"], default=None)
        after = min((p for p in group if p["group"] == "after"), key=lambda p: p["ref"], default=None)
        if before and across and after:
            sets[span] = (before, across, after)
    if not sets:
        print("\nNo complete before/across/after set found.")
        return

    # 7-8. Change maps and how much ground is flagged.
    zoom = dist <= ZOOM_M * 1.42
    zr, zc = np.where(zoom)
    r0, r1, c0, c1 = zr.min(), zr.max() + 1, zc.min(), zc.max() + 1
    fig, axes = plt.subplots(len(sets), 5, figsize=(24, 5 * len(sets)), squeeze=False)
    saved = {}
    print("\nDetection (coherence dropped by more than "
          f"{DROP} from 'before' to 'across'):")
    for row, (span, (before, across, after)) in enumerate(sets.items()):
        drop = before["coh"] - across["coh"]
        flagged = drop > DROP
        core_share = 100 * float(np.nanmean(flagged[core]))
        ring_share = 100 * float(np.nanmean(flagged[ring]))
        print(f"  {span}-day pairs: flagged {core_share:.0f}% of the slide core vs "
              f"{ring_share:.0f}% of the surrounding bog")

        bright = None
        if before["bright"] is not None and after["bright"] is not None:
            bright = brightness_change_db(before["bright"], after["bright"])

        panels = [
            (before["coh"], f"Before: {before['ref']:%d %b}-{before['sec']:%d %b}", "gray", 0, 1),
            (across["coh"], f"Across: {across['ref']:%d %b}-{across['sec']:%d %b}", "gray", 0, 1),
            (after["coh"], f"After: {after['ref']:%d %b}-{after['sec']:%d %b}", "gray", 0, 1),
            (drop, "Coherence drop (before - across)", "magma", 0, 0.8),
            (bright, "Brightness change after vs before (dB)", "RdBu_r", -6, 6),
        ]
        for col, (data, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row][col]
            if data is None:
                ax.axis("off")
                continue
            ax.imshow(data[r0:r1, c0:c1], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            cy, cx = (np.array(np.unravel_index(np.argmin(dist), dist.shape)) - [r0, c0])
            pix_m = abs(transform.a)
            for radius, style in [(CORE_M, "-"), (RING_M[0], "--"), (RING_M[1], "--")]:
                circle = plt.Circle((cx, cy), radius / pix_m, fill=False, color="cyan", linestyle=style, linewidth=1)
                ax.add_patch(circle)
            ax.set_title(f"{span}-day | {title}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        saved[f"coh_before_{span}d"] = before["coh"]
        saved[f"coh_across_{span}d"] = across["coh"]
        saved[f"coh_after_{span}d"] = after["coh"]
        if bright is not None:
            saved[f"bright_change_{span}d"] = bright

    fig.suptitle(f"{site['name']}: 2020 slide replay (solid circle = slide core, dashed = surrounding bog)", fontsize=13)
    fig.tight_layout()
    out_png = WORK_DIR / f"{site_id}_event_replay.png"
    fig.savefig(out_png, dpi=100)
    plt.close(fig)

    # 9. Save the crops for the 3D viewer later.
    np.savez_compressed(WORK_DIR / f"{site_id}_event_replay.npz", transform=np.array(transform)[:6],
                        crs=str(crs), slide_xy=np.array([sx, sy]), **saved)
    print(f"\nSaved picture: {out_png.name}")
    print(f"Saved data for the viewer: {site_id}_event_replay.npz")


if __name__ == "__main__":
    main()