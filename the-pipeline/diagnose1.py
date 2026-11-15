# pipeline/diagnose.py
# Prints the raw numbers inside the site crop for coherence, phase and connected components,
# so we can see why a panel looks blank.
# Usage:
#     python pipeline\diagnose.py viking

import sys

import h5py
import numpy as np
from pyproj import Transformer
from shapely import wkt as shapely_wkt

from config1 import SITES, raw_dir_for
from coherence_check1 import COHERENCE, PHASE, COMPONENTS, find_dataset, get_epsg


def describe(label, ds, rows, cols):
    """Print the facts about one layer inside the crop, without changing any values."""
    raw = ds[rows[0]:rows[1], cols[0]:cols[1]].astype("float64")
    print(f"\n=== {label} ===")
    print(f"  stored as: {ds.dtype}")

    # 1. Attributes (skipping the long description text).
    for key in ds.attrs:
        if key != "description":
            print(f"  attribute {key}: {ds.attrs[key]}")

    # 2. Blank pixels.
    print(f"  pixels in crop: {raw.size}, blank (NaN): {int(np.isnan(raw).sum())}")

    # 3. Basic numbers.
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        print("  every pixel is blank")
        return
    print(f"  min {finite.min():.4f} | max {finite.max():.4f} | mean {finite.mean():.4f} | median {np.median(finite):.4f}")

    # 4. Distinct values and how often each appears.
    values, counts = np.unique(finite, return_counts=True)
    if len(values) <= 20:
        print("  distinct values (value: count):")
        for v, c in zip(values, counts):
            print(f"    {v:g}: {c}")
    else:
        print(f"  {len(values)} distinct values; the 5 most common:")
        for i in np.argsort(counts)[::-1][:5]:
            print(f"    {values[i]:g}: {counts[i]}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\diagnose.py <site>")
        return

    site_id = sys.argv[1]
    site = SITES[site_id]
    path = sorted(raw_dir_for(site_id).glob("*GUNW*.h5"))[0]
    print(f"File: {path.name}")

    with h5py.File(path, "r") as f:
        # Same crop as coherence_check.py.
        x = f[find_dataset(f, "xCoordinates")][:]
        y = f[find_dataset(f, "yCoordinates")][:]
        epsg = get_epsg(f)
        to_map = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        lon_min, lat_min, lon_max, lat_max = shapely_wkt.loads(site["wkt"]).bounds
        cx, cy = to_map.transform([lon_min, lon_max, lon_max, lon_min], [lat_min, lat_min, lat_max, lat_max])
        col_idx = np.where((x >= min(cx)) & (x <= max(cx)))[0]
        row_idx = np.where((y >= min(cy)) & (y <= max(cy)))[0]
        rows = (row_idx.min(), row_idx.max() + 1)
        cols = (col_idx.min(), col_idx.max() + 1)
        print(f"Projection EPSG:{epsg}, crop rows {rows}, cols {cols}")

        describe("COHERENCE", f[COHERENCE], rows, cols)
        describe("UNWRAPPED PHASE", f[PHASE], rows, cols)
        describe("CONNECTED COMPONENTS", f[COMPONENTS], rows, cols)


if __name__ == "__main__":
    main()