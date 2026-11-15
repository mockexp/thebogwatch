# pipeline/coherence_check.py
# The data test: crops every downloaded GUNW file to the site box, measures coherence,
# and saves a 3-panel picture (coherence, unwrapped phase, connected components) per file.
# Usage (Command Prompt, inside bogwatch, with (.venv) showing):
#     python pipeline\coherence_check.py viking

import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")  # draw straight to image files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt as shapely_wkt

from config1 import SITES, WORK_DIR, raw_dir_for

# Where the layers live inside every GUNW file (found with inspect_file.py).
GRID = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
COHERENCE = GRID + "/HH/coherenceMagnitude"
PHASE = GRID + "/HH/unwrappedPhase"
COMPONENTS = GRID + "/HH/connectedComponents"

MEENBOG_SLIDE = (-7.8733, 54.7192)  # centre of the 2020 slide (Islam et al. 2022)
# Rough centre of Viking Wind Farm (longitude, latitude), marked on the pictures.
MARKERS = {
    "viking": (-1.2612, 60.2866),
    "galway": (-9.3676, 53.3662),
    "meenbog": MEENBOG_SLIDE,
    "meenbog_d9": MEENBOG_SLIDE,
    "meenbog_d110": MEENBOG_SLIDE,
    "meenbog_a58": MEENBOG_SLIDE,
    "meenbog_a130": MEENBOG_SLIDE,
}


def find_dataset(f, name, prefer=GRID):
    """Find a dataset by its final name anywhere in the file,
    preferring matches inside the unwrapped interferogram folder."""
    hits = []

    def visit(path, item):
        if isinstance(item, h5py.Dataset) and path.split("/")[-1] == name:
            hits.append("/" + path)
        return None  # returning anything else would stop the search early

    f.visititems(visit)
    hits.sort(key=lambda p: (not p.startswith(prefer), len(p)))
    return hits[0] if hits else None


def get_epsg(f):
    """The map projection code (e.g. 32630 = UTM zone 30 north), stored inside the file."""
    path = find_dataset(f, "projection")
    if path is None:
        return None
    ds = f[path]
    for key in ("epsg_code", "epsg"):
        if key in ds.attrs:
            return int(ds.attrs[key])
    try:
        value = int(ds[()])
        return value if value > 1000 else None
    except (TypeError, ValueError):
        return None


def read_crop(ds, rows, cols):
    """Read just the cropped part of a layer, turning 'no data' values into blanks (NaN)."""
    data = ds[rows[0]:rows[1], cols[0]:cols[1]].astype("float64")
    fill = ds.attrs.get("_FillValue")
    if fill is not None:
        data[data == float(np.asarray(fill).ravel()[0])] = np.nan
    return data


def pair_dates(filename):
    """Pull the reference and secondary dates out of the file name."""
    stamps = re.findall(r"(\d{8})T\d{6}", filename)
    if len(stamps) >= 3:
        return stamps[0], stamps[2]
    return "unknown", "unknown"


def check_file(path, site_id, site):
    with h5py.File(path, "r") as f:
        # 2. Coordinates and projection.
        x_path, y_path = find_dataset(f, "xCoordinates"), find_dataset(f, "yCoordinates")
        epsg = get_epsg(f)
        if x_path is None or y_path is None or epsg is None:
            print(f"  Could not find coordinates or projection (x={x_path}, y={y_path}, epsg={epsg}).")
            return None
        x = f[x_path][:]
        y = f[y_path][:]

        # 3. Convert the site box from longitude/latitude to the file's metres.
        to_map = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        lon_min, lat_min, lon_max, lat_max = shapely_wkt.loads(site["wkt"]).bounds
        corner_x, corner_y = to_map.transform(
            [lon_min, lon_max, lon_max, lon_min], [lat_min, lat_min, lat_max, lat_max]
        )

        # 4. Which rows and columns fall inside the box?
        col_idx = np.where((x >= min(corner_x)) & (x <= max(corner_x)))[0]
        row_idx = np.where((y >= min(corner_y)) & (y <= max(corner_y)))[0]
        if len(col_idx) == 0 or len(row_idx) == 0:
            print("  The site box does not overlap this file.")
            return None
        rows = (row_idx.min(), row_idx.max() + 1)
        cols = (col_idx.min(), col_idx.max() + 1)

        # 5. Read only the crop of each layer.
        coherence = read_crop(f[COHERENCE], rows, cols)
        phase = read_crop(f[PHASE], rows, cols)
        components = read_crop(f[COMPONENTS], rows, cols)

    # Coherence of exactly 0 means no data (sea, outside the swath).
    coherence[coherence <= 0] = np.nan
    valid = np.isfinite(coherence)

    # 6. Statistics.
    coh = coherence[valid]
    unreliable = np.isnan(components[valid]) | (components[valid] == 0)
    stats = {
        "reference": None, "secondary": None,
        "pixels": int(valid.sum()),
        "mean_coherence": round(float(coh.mean()), 3) if coh.size else np.nan,
        "pct_good_ge_0.5": round(100 * float((coh >= 0.5).mean()), 1) if coh.size else np.nan,
        "pct_usable_0.3_0.5": round(100 * float(((coh >= 0.3) & (coh < 0.5)).mean()), 1) if coh.size else np.nan,
        "pct_poor_lt_0.3": round(100 * float((coh < 0.3).mean()), 1) if coh.size else np.nan,
        "pct_unreliable_components": round(100 * float(unreliable.mean()), 1) if coh.size else np.nan,
    }
    ref, sec = pair_dates(path.name)
    stats["reference"], stats["secondary"] = ref, sec

    # Map extent for the pictures, with north at the top.
    x_crop = x[cols[0]:cols[1]]
    y_crop = y[rows[0]:rows[1]]
    if y_crop[0] < y_crop[-1]:  # rows run south to north: flip so north is up
        coherence, phase, components = np.flipud(coherence), np.flipud(phase), np.flipud(components)
    extent = (x_crop.min(), x_crop.max(), y_crop.min(), y_crop.max())

    # 7. Three-panel picture.
    phase_shown = phase - np.nanmedian(phase)          # only differences matter
    limit = np.nanpercentile(np.abs(phase_shown), 98)  # ignore extreme outliers for the colour scale
    components_shown = np.where(components == 0, np.nan, components)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        (coherence, "Coherence (white = trustworthy)", "gray", 0, 1),
        (phase_shown, "Unwrapped phase, radians (relative)", "RdBu", -limit, limit),
        (components_shown, "Connected components (blank = unreliable)", "tab20", None, None),
    ]
    marker = MARKERS.get(site_id)
    for ax, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        img = ax.imshow(data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        fig.colorbar(img, ax=ax, shrink=0.7)
        if marker:
            mx, my = to_map.transform(*marker)
            ax.plot(mx, my, marker="x", color="yellow", markersize=12, mew=3)

    fig.suptitle(f"{site['name']}  |  {ref} to {sec}  |  mean coherence {stats['mean_coherence']}")
    fig.tight_layout()

    # 8. Save the picture.
    out_png = WORK_DIR / f"{site_id}_check_{ref}_{sec}.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  Saved picture: {out_png.name}  (projection EPSG:{epsg}, crop {coherence.shape[0]} x {coherence.shape[1]} pixels)")
    return stats


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\coherence_check.py <site>")
        return

    site_id = sys.argv[1]
    site = SITES[site_id]

    # 1. Every GUNW file for this site, oldest first.
    files = sorted(raw_dir_for(site_id).glob("*GUNW*.h5"))
    if not files:
        print("No GUNW files found. Run download.py first.")
        return

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for path in files:
        print(f"Checking {path.name[:60]}...")
        stats = check_file(path, site_id, site)
        if stats:
            all_stats.append(stats)

    # 8. Summary table comparing all pairs.
    if all_stats:
        table = pd.DataFrame(all_stats)
        out_csv = WORK_DIR / f"{site_id}_coherence_summary.csv"
        table.to_csv(out_csv, index=False)
        print("\nCoherence summary:\n")
        print(table.to_string(index=False))
        print(f"\nSaved summary to: {out_csv}")


if __name__ == "__main__":
    main()