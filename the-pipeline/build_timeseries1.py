# pipeline/build_timeseries.py
# Turns the unwrapped pairs into one vertical ground-motion time series in millimetres.
# Steps: keep the main land patch, remove the atmospheric slope (a plane) from each pair,
# zero each pair's median, convert radians -> mm -> vertical, then add the pairs in date order.
# Usage:
#     python pipeline\build_timeseries.py viking

#time series is essentially , 
import sys
from datetime import datetime

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from scipy import ndimage

from config1 import SITES, WORK_DIR, raw_dir_for
from coherence_check1 import find_dataset, MARKERS
from frame_overview1 import box_outline, index_range

SPEED_OF_LIGHT = 299_792_458.0     # metres per second
DEFAULT_WAVELENGTH_M = 0.2384      # NISAR L-band (about 1.257 GHz), used if the file doesn't say
DEFAULT_INCIDENCE_DEG = 40.0       # used only if the file's incidence angle can't be read
RADAR_GRID = "/science/LSAR/GUNW/metadata/radarGrid"
NEAR_KM, FAR_KM = 2.0, 6.0         # rough first look: near the wind farm vs far from it


def load_pairs(site_id):
    """Load every unwrapped pair, oldest first. Uses corrected (_unwfix_) versions where they
    exist, and skips any pair listed under exclude_pairs in config.py."""
    excluded = SITES[site_id].get("exclude_pairs", {})
    pairs = []
    used_fixed = 0
    for path in sorted(WORK_DIR.glob(f"{site_id}_unw_*.npz")):
        pair_name = path.stem.replace(f"{site_id}_unw_", "")  # e.g. 20261005_20261017
        if pair_name in excluded:
            print(f"Skipping {pair_name}: {excluded[pair_name]}")
            continue
        fixed = path.with_name(path.name.replace("_unw_", "_unwfix_"))
        if fixed.exists():
            path = fixed
            used_fixed += 1
        data = np.load(path)
        pairs.append({key: data[key] for key in data.files})
    pairs.sort(key=lambda p: str(p["reference"]))
    print(f"Loaded {len(pairs)} pairs ({used_fixed} corrected versions)")
    return pairs


def get_wavelength(h5_path):
    """Find the radar wavelength in the file (from frequency if needed), else use the default."""
    with h5py.File(h5_path, "r") as f:
        for name in ("centerFrequency", "processedCenterFrequency"):
            path = find_dataset(f, name, prefer="/science/LSAR/GUNW")
            if path:
                value = float(np.asarray(f[path][()]).ravel()[0])
                if 1.0e9 < value < 1.5e9:  # sanity check: L-band is about 1.2-1.3 GHz
                    return SPEED_OF_LIGHT / value, f"{name} = {value / 1e9:.4f} GHz, found at {path}"
        path = find_dataset(f, "wavelength", prefer="/science/LSAR/GUNW")
        if path:
            value = float(np.asarray(f[path][()]).ravel()[0])
            if 0.2 < value < 0.3:
                return value, f"wavelength = {value} m, found at {path}"
    return DEFAULT_WAVELENGTH_M, "not found in the file, using the NISAR L-band default"


def get_mean_incidence(h5_path, site, epsg):
    """Average incidence angle over the site, from the file's coarse 3D radar grid,
    at the height level closest to 100 m (typical of the hills around Viking)."""
    with h5py.File(h5_path, "r") as f:
        cube = f[RADAR_GRID + "/incidenceAngle"]
        x = f[find_dataset(f, "xCoordinates", prefer=RADAR_GRID)][:]
        y = f[find_dataset(f, "yCoordinates", prefer=RADAR_GRID)][:]
        if len(x) != cube.shape[2] or len(y) != cube.shape[1]:
            return None
        h_path = find_dataset(f, "heightAboveEllipsoid", prefer=RADAR_GRID)
        level = int(np.argmin(np.abs(f[h_path][:] - 100.0))) if h_path else 0
        bx, by = box_outline(site, epsg)
        rows, cols = index_range(y, by.min(), by.max()), index_range(x, bx.min(), bx.max())
        if rows is None or cols is None:
            return None
        values = cube[level, rows[0]:rows[1], cols[0]:cols[1]].astype("float64")
    values = values[np.isfinite(values) & (values > 0)]
    return float(values.mean()) if values.size else None


def largest_patch(mask):
    """Keep only the biggest connected area of True pixels."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    return labels == (np.argmax(sizes) + 1)


def remove_plane(values, xx, yy, region):
    """Fit a tilted flat plane to the values inside the region and subtract it everywhere."""
    design = np.column_stack([np.ones(region.sum()), xx[region], yy[region]])
    coef, *_ = np.linalg.lstsq(design, values[region], rcond=None)
    plane = coef[0] + coef[1] * xx + coef[2] * yy
    return values - plane, plane


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\build_timeseries.py <site>")
        return

    site_id = sys.argv[1]
    site = SITES[site_id]
    pairs = load_pairs(site_id)
    if len(pairs) < 2:
        print("Need at least two unwrapped pairs. Run unwrap_site.py first.")
        return

    # Every pair must be on exactly the same grid.
    shape = pairs[0]["unwrapped"].shape
    if any(p["unwrapped"].shape != shape for p in pairs):
        print("The pairs are not all the same size; they can't be stacked.")
        return
    x, y, epsg = pairs[0]["x"], pairs[0]["y"], int(pairs[0]["epsg"])

    # Check the chain: each pair should start where the previous one ended.
    for a, b in zip(pairs, pairs[1:]):
        if str(a["secondary"]) != str(b["reference"]):
            print(f"WARNING: gap in the chain between {a['secondary']} and {b['reference']}")

    # 1. The largest patch of land that is reliable in every pair.
    valid_everywhere = np.all([np.isfinite(p["unwrapped"]) for p in pairs], axis=0)
    region = largest_patch(valid_everywhere)
    print(f"Main land patch: {region.sum()} pixels ({region.sum() * 0.04 * 0.04:.0f} km²)")

    # Wavelength and incidence angle from the first raw file.
    h5_path = sorted(raw_dir_for(site_id).glob("*GUNW*.h5"))[0]
    wavelength, how = get_wavelength(h5_path)
    print(f"Wavelength: {wavelength * 100:.2f} cm ({how})")
    incidence = get_mean_incidence(h5_path, site, epsg)
    if incidence is None:
        incidence = DEFAULT_INCIDENCE_DEG
        print(f"WARNING: could not read the incidence angle; using {incidence} degrees")
    else:
        print(f"Average incidence angle over the site: {incidence:.1f} degrees")

    # Coordinates in kilometres from the centre (keeps the plane fit numerically tidy).
    xx, yy = np.meshgrid((x - x.mean()) / 1000, (y - y.mean()) / 1000)
    rad_to_mm_los = -wavelength / (4 * np.pi) * 1000
    to_vertical = 1 / np.cos(np.radians(incidence))

    pair_mm = []
    print("\nPer pair: size of the removed atmospheric slope (mm, line of sight, across the site)")
    for p in pairs:
        phase = np.where(region, p["unwrapped"], np.nan)
        # 2. Remove the atmospheric slope.
        flat, plane = remove_plane(phase, xx, yy, region)
        # 3. Zero the median.
        flat = flat - np.nanmedian(flat[region])
        # 4. Radians -> mm along the line of sight -> vertical.
        vertical_mm = flat * rad_to_mm_los * to_vertical
        vertical_mm[~region] = np.nan
        pair_mm.append(vertical_mm)
        slope_size = float(np.ptp(plane[region]) * abs(rad_to_mm_los))
        print(f"  {p['reference']} to {p['secondary']}: {slope_size:.0f} mm")

    # 5. Add the pairs up in date order.
    dates = [str(pairs[0]["reference"])] + [str(p["secondary"]) for p in pairs]
    cumulative = np.zeros((len(dates),) + shape, dtype="float32")
    cumulative[0][~region] = np.nan
    for i, step in enumerate(pair_mm, start=1):
        cumulative[i] = cumulative[i - 1] + step

    out_npz = WORK_DIR / f"{site_id}_timeseries.npz"
    np.savez_compressed(out_npz, dates=np.array(dates), vertical_mm=cumulative, x=x, y=y,
                        epsg=epsg, region=region, wavelength_m=wavelength, incidence_deg=incidence)
    print(f"\nSaved time series: {out_npz}")

    # --- Picture 1: total movement from the first to the last date ---
    flip = y[0] < y[-1]
    extent = (x.min(), x.max(), y.min(), y.max())
    total = cumulative[-1]
    limit = np.nanpercentile(np.abs(total), 98)
    fig, ax = plt.subplots(figsize=(9, 8))
    img = ax.imshow(np.flipud(total) if flip else total, extent=extent, cmap="RdBu",
                    vmin=-limit, vmax=limit, interpolation="nearest")
    bx, by = box_outline(site, epsg)
    ax.plot(bx, by, color="black", linewidth=1)
    to_map = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    mx, my = to_map.transform(*MARKERS[site_id])
    ax.plot(mx, my, marker="x", color="black", markersize=12, mew=3)
    ax.set_title(f"{site['name']}: vertical movement {dates[0]} to {dates[-1]} (mm)\n"
                 "blue = up, red = down, relative to the site median (sign convention to be confirmed)")
    fig.colorbar(img, ax=ax, shrink=0.8, label="mm")
    fig.tight_layout()
    fig.savefig(WORK_DIR / f"{site_id}_total_movement.png", dpi=110)
    plt.close(fig)

    # --- Picture 2: movement over time, near vs far from the wind farm, plus sample spots ---
    distance_km = np.hypot(np.meshgrid(x, y)[0] - mx, np.meshgrid(x, y)[1] - my) / 1000
    near = region & (distance_km <= NEAR_KM)
    far = region & (distance_km >= FAR_KM)
    when = [datetime.strptime(d, "%Y%m%d") for d in dates]
    near_series = [float(np.nanmedian(c[near])) if near.any() else np.nan for c in cumulative]
    far_series = [float(np.nanmedian(c[far])) if far.any() else np.nan for c in cumulative]

    rng = np.random.default_rng(42)  # fixed seed: the same sample spots every run
    rows_idx, cols_idx = np.where(region)
    picks = rng.choice(len(rows_idx), size=min(5, len(rows_idx)), replace=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    for k in picks:
        ax.plot(when, cumulative[:, rows_idx[k], cols_idx[k]], color="grey", alpha=0.5, linewidth=1)
    ax.plot(when, near_series, color="red", linewidth=3, marker="o", label=f"Bog within {NEAR_KM:g} km of the wind farm centre")
    ax.plot(when, far_series, color="blue", linewidth=3, marker="o", label=f"Bog more than {FAR_KM:g} km away")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Vertical movement since first date (mm)")
    ax.set_title("Rough first look (grey = random sample spots). Not the final wind farm analysis.")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(WORK_DIR / f"{site_id}_movement_over_time.png", dpi=110)
    plt.close(fig)

    print("\nDate       near (mm)   far (mm)")
    for d, n, fr in zip(dates, near_series, far_series):
        print(f"{d}   {n:8.1f}   {fr:8.1f}")
    print(f"\nSaved pictures: {site_id}_total_movement.png and {site_id}_movement_over_time.png")


if __name__ == "__main__":
    main()