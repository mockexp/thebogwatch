# pipeline/pair_grid.py
# Test A: shows every 12-day pair side by side, processed exactly like the time series,
# to see whether the "near the wind farm" signal is consistent (likely real ground motion)
# or comes from one or two pairs (likely atmosphere).
# Usage:
#     python pipeline\pair_grid.py viking

import math
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

from config1 import SITES, WORK_DIR, raw_dir_for
from coherence_check1 import MARKERS
from frame_overview1 import box_outline
from build_timeseries1 import (load_pairs, largest_patch, remove_plane, get_wavelength,
                              get_mean_incidence, DEFAULT_INCIDENCE_DEG, NEAR_KM, FAR_KM)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\pair_grid.py <site>")
        return

    site_id = sys.argv[1]
    site = SITES[site_id]
    pairs = load_pairs(site_id)
    if len(pairs) < 2:
        print("Need at least two unwrapped pairs.")
        return

    # 1. Same processing as build_timeseries.py.
    x, y, epsg = pairs[0]["x"], pairs[0]["y"], int(pairs[0]["epsg"])
    valid_everywhere = np.all([np.isfinite(p["unwrapped"]) for p in pairs], axis=0)
    region = largest_patch(valid_everywhere)

    h5_path = sorted(raw_dir_for(site_id).glob("*GUNW*.h5"))[0]
    wavelength, _ = get_wavelength(h5_path)
    incidence = get_mean_incidence(h5_path, site, epsg) or DEFAULT_INCIDENCE_DEG
    to_mm_vertical = -wavelength / (4 * np.pi) * 1000 / np.cos(np.radians(incidence))

    xx, yy = np.meshgrid((x - x.mean()) / 1000, (y - y.mean()) / 1000)
    maps = []
    for p in pairs:
        phase = np.where(region, p["unwrapped"], np.nan)
        flat, _ = remove_plane(phase, xx, yy, region)
        flat = flat - np.nanmedian(flat[region])
        mm = flat * to_mm_vertical
        mm[~region] = np.nan
        maps.append(mm)
    total = np.sum(maps, axis=0)

    # Near and far areas, as in the time series.
    to_map = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    mx, my = to_map.transform(*MARKERS[site_id])
    gx, gy = np.meshgrid(x, y)
    distance_km = np.hypot(gx - mx, gy - my) / 1000
    near = region & (distance_km <= NEAR_KM)
    far = region & (distance_km >= FAR_KM)

    # 3. Two numbers per pair: near - far, and how much the pair looks like the total.
    in_region = region & np.isfinite(total)
    rows = []
    for p, mm in zip(pairs, maps):
        near_minus_far = float(np.nanmedian(mm[near]) - np.nanmedian(mm[far]))
        match = float(np.corrcoef(mm[in_region], total[in_region])[0, 1])
        days = (datetime.strptime(str(p["secondary"]), "%Y%m%d")
                - datetime.strptime(str(p["reference"]), "%Y%m%d")).days
        rows.append((str(p["reference"]), str(p["secondary"]), days, near_minus_far, match))

    # 2. The 3 x 3 grid, one shared colour scale.
    all_values = np.concatenate([mm[region] for mm in maps])
    limit = float(np.nanpercentile(np.abs(all_values), 98))
    flip = y[0] < y[-1]
    extent = (x.min(), x.max(), y.min(), y.max())
    bx, by = box_outline(site, epsg)
    circle_angle = np.linspace(0, 2 * np.pi, 100)

    n_cols = 3
    n_rows = math.ceil(len(maps) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    img = None
    for ax, mm, (ref, sec, days, nmf, match) in zip(axes, maps, rows):
        img = ax.imshow(np.flipud(mm) if flip else mm, extent=extent, cmap="RdBu",
                        vmin=-limit, vmax=limit, interpolation="nearest")
        ax.plot(bx, by, color="black", linewidth=0.8)
        ax.plot(mx + NEAR_KM * 1000 * np.cos(circle_angle), my + NEAR_KM * 1000 * np.sin(circle_angle),
                color="black", linewidth=1, linestyle="--")
        ax.plot(mx, my, marker="x", color="black", markersize=9, mew=2)
        ax.set_title(f"{ref} to {sec} ({days} days)\nnear - far {nmf:+.1f} mm | match {match:+.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(maps):]:
        ax.axis("off")
    fig.colorbar(img, ax=axes.tolist(), shrink=0.6, label="vertical mm in that pair (blue = up, red = down, sign to be confirmed)")
    fig.suptitle(f"{site['name']}: each 12-day pair after slope removal, same colour scale", fontsize=13)
    out_png = WORK_DIR / f"{site_id}_pair_grid.png"
    fig.savefig(out_png, dpi=100)
    plt.close(fig)

    # 4. Table and a plain-English hint.
    print("Pair                    days   near - far (mm)   match with total")
    for ref, sec, days, nmf, match in rows:
        print(f"{ref} to {sec}   {days:4d}   {nmf:+15.1f}   {match:+16.2f}")

    differences = np.array([r[3] for r in rows])
    total_difference = differences.sum()
    total_size = np.abs(differences).sum()
    biggest_share = np.abs(differences).max() / total_size if total_size > 0 else 0
    negative_count = int((differences < -1).sum())
    noise_guess = np.std(differences) * np.sqrt(len(differences))  # rough size of noise for a sum
    print(f"\nTotal near - far over all pairs: {total_difference:+.1f} mm")
    print(f"Pairs where near moved down by more than 1 mm relative to far: {negative_count} of {len(rows)}")
    print(f"Biggest single pair's share of the total size: {100 * biggest_share:.0f}%")
    print(f"Rough noise level for a total like this: about +/-{noise_guess:.0f} mm")
    if biggest_share > 0.35:
        print("Hint: one pair dominates. Check that pair's map before trusting the total.")
    elif abs(total_difference) < noise_guess:
        print("Hint: the total is within the noise. No clear signal yet.")
    elif negative_count >= 0.6 * len(rows):
        print("Hint: a consistent difference across most pairs, larger than the noise. Worth testing further (elevation check next).")
    else:
        print("Hint: mixed picture. Look at the maps to judge.")
    print(f"\nSaved picture: {out_png}")


if __name__ == "__main__":
    main()