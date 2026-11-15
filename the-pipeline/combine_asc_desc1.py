# pipeline/combine_asc_desc.py
# Step A: combines an ascending and a descending NISAR track into true vertical (up)
# and east-west motion. Uses each track's time series from build_timeseries.py.
# Usage:
#     python pipeline\combine_asc_desc.py meenbog_a58 meenbog_d9

import sys
from datetime import datetime

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from config1 import SITES, WORK_DIR, raw_dir_for
from coherence_check1 import find_dataset, get_epsg, MARKERS
from frame_overview1 import box_outline, index_range

RADAR_GRID = "/science/LSAR/GUNW/metadata/radarGrid"
SITE_HEIGHT_M = 250.0          # typical height of the hills around Meenbog
NEAR_KM, FAR_KM = 2.0, 5.0     # rough look: near the slide vs further away
PIXEL_KM2 = 0.04 * 0.04        # area of one 40 m pixel


def load_track(site_id):
    """1. Load a track's time series and turn 'assumed vertical' back into line of sight."""
    data = np.load(WORK_DIR / f"{site_id}_timeseries.npz")
    track = {key: data[key] for key in data.files}
    track["dates"] = [str(d) for d in track["dates"]]
    incidence = float(track["incidence_deg"])
    track["los_mm"] = track["vertical_mm"] * np.cos(np.radians(incidence))  # towards the satellite
    return track


def read_los_vector(site_id):
    """2. East and north parts of the line-of-sight direction, averaged over the site."""
    site = SITES[site_id]
    h5_path = sorted(raw_dir_for(site_id).glob("*GUNW*.h5"))[0]
    with h5py.File(h5_path, "r") as f:
        epsg = get_epsg(f)
        x = f[find_dataset(f, "xCoordinates", prefer=RADAR_GRID)][:]
        y = f[find_dataset(f, "yCoordinates", prefer=RADAR_GRID)][:]
        heights = f[find_dataset(f, "heightAboveEllipsoid", prefer=RADAR_GRID)][:]
        level = int(np.argmin(np.abs(heights - SITE_HEIGHT_M)))
        east_ds = f[RADAR_GRID + "/losUnitVectorX"]
        north_ds = f[RADAR_GRID + "/losUnitVectorY"]
        if len(x) != east_ds.shape[2] or len(y) != east_ds.shape[1]:
            raise RuntimeError("Radar grid coordinates don't match the line-of-sight cube.")
        bx, by = box_outline(site, epsg)
        rows, cols = index_range(y, by.min(), by.max()), index_range(x, bx.min(), bx.max())
        if rows is None or cols is None:
            raise RuntimeError("The site box doesn't overlap the radar grid.")
        east = east_ds[level, rows[0]:rows[1], cols[0]:cols[1]].astype("float64")
        north = north_ds[level, rows[0]:rows[1], cols[0]:cols[1]].astype("float64")
        description = east_ds.attrs.get("description", b"")
    if isinstance(description, bytes):
        description = description.decode("utf-8", errors="replace")
    e, n = float(np.nanmean(east)), float(np.nanmean(north))
    u = float(np.sqrt(max(0.0, 1.0 - e * e - n * n)))  # up part, so the direction has length 1
    print(f"  {site_id}: east {e:+.3f}, north {n:+.3f}, up {u:.3f}  (file says: {description[:90]})")
    return e, n, u


def regrid(values, src_x, src_y, src_epsg, dst_x, dst_y, dst_epsg):
    """4. Move a (dates, rows, cols) stack from one map grid onto another."""
    xs, ys, v = src_x, src_y, values
    if xs[0] > xs[-1]:
        xs, v = xs[::-1], v[:, :, ::-1]
    if ys[0] > ys[-1]:
        ys, v = ys[::-1], v[:, ::-1, :]
    gy, gx = np.meshgrid(dst_y, dst_x, indexing="ij")
    px, py = gx.ravel(), gy.ravel()
    if src_epsg != dst_epsg:  # different map projections: convert the target points first
        to_src = Transformer.from_crs(f"EPSG:{dst_epsg}", f"EPSG:{src_epsg}", always_xy=True)
        px, py = to_src.transform(px, py)
    points = np.column_stack([py, px])
    out = np.full((v.shape[0],) + gy.shape, np.nan, dtype="float32")
    for i in range(v.shape[0]):
        interp = RegularGridInterpolator((ys, xs), v[i], bounds_error=False, fill_value=np.nan)
        out[i] = interp(points).reshape(gy.shape)
    return out


def to_days(dates):
    base = datetime(2026, 1, 1)
    return np.array([(datetime.strptime(d, "%Y%m%d") - base).days for d in dates], dtype="float64")


def interp_time(cube, src_days, tgt_days):
    """5. Estimate values on target days by straight-line interpolation between observations."""
    idx = np.clip(np.searchsorted(src_days, tgt_days, side="right") - 1, 0, len(src_days) - 2)
    weight = (tgt_days - src_days[idx]) / (src_days[idx + 1] - src_days[idx])
    return cube[idx] * (1 - weight)[:, None, None] + cube[idx + 1] * weight[:, None, None]


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in SITES or sys.argv[2] not in SITES:
        print("Usage: python pipeline\\combine_asc_desc.py <ascending site> <descending site>")
        return

    first, second = sys.argv[1], sys.argv[2]
    by_direction = {SITES[first]["direction"]: first, SITES[second]["direction"]: second}
    if set(by_direction) != {"ASCENDING", "DESCENDING"}:
        print("Need one ASCENDING and one DESCENDING site.")
        return
    asc_id, desc_id = by_direction["ASCENDING"], by_direction["DESCENDING"]
    base_name = desc_id.split("_")[0]
    print(f"Combining ascending {asc_id} with descending {desc_id}\n")

    # 1. Load both tracks.
    asc, desc = load_track(asc_id), load_track(desc_id)

    # 2-3. Viewing directions, with the sanity check on which way they point.
    print("Line-of-sight directions (averaged over the site):")
    e_a, n_a, u_a = read_los_vector(asc_id)
    e_d, n_d, u_d = read_los_vector(desc_id)
    if e_a > 0 > e_d:
        print("  Directions point from ground to satellite, as expected (ascending east, descending west).")
    elif e_a < 0 < e_d:
        e_a, n_a, e_d, n_d = -e_a, -n_a, -e_d, -n_d
        print("  Directions pointed from satellite to ground; flipped them to ground-to-satellite.")
    else:
        print("  WARNING: unexpected directions (both east or both west). Results may be unreliable.")
    print(f"  North parts are {n_a:+.3f} and {n_d:+.3f}: small, so ignoring north is reasonable.\n")

    # 4. Ascending onto the descending grid.
    asc_on_desc = regrid(asc["los_mm"], asc["x"], asc["y"], int(asc["epsg"]),
                         desc["x"], desc["y"], int(desc["epsg"]))

    # 5. Common dates: descending dates inside the ascending period.
    asc_days, desc_days = to_days(asc["dates"]), to_days(desc["dates"])
    keep = (desc_days >= asc_days.min()) & (desc_days <= asc_days.max())
    common_days = desc_days[keep]
    common_dates = [d for d, k in zip(desc["dates"], keep) if k]
    if len(common_dates) < 3:
        print("Fewer than 3 shared dates; can't combine.")
        return
    A = interp_time(asc_on_desc, asc_days, common_days)
    D = desc["los_mm"][keep].astype("float32")
    A, D = A - A[0], D - D[0]  # both start at zero on the first shared date
    print(f"Shared dates: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} dates)")

    # 6. Common reference: zero median over the area valid in both tracks.
    shared = np.all(np.isfinite(A), axis=0) & np.all(np.isfinite(D), axis=0)
    print(f"Area valid in both tracks: {shared.sum()} pixels ({shared.sum() * PIXEL_KM2:.0f} km²)\n")
    for t in range(len(common_dates)):
        A[t] -= np.nanmedian(A[t][shared])
        D[t] -= np.nanmedian(D[t][shared])
    A[:, ~shared] = np.nan
    D[:, ~shared] = np.nan

    # 7. Solve the two equations for east and up at every pixel and date.
    det = e_a * u_d - e_d * u_a
    east = (u_d * A - u_a * D) / det
    up = (e_a * D - e_d * A) / det

    out = WORK_DIR / f"{base_name}_combined.npz"
    np.savez_compressed(out, dates=np.array(common_dates), up_mm=up.astype("float32"),
                        east_mm=east.astype("float32"), x=desc["x"], y=desc["y"],
                        epsg=int(desc["epsg"]), shared=shared,
                        los_asc=np.array([e_a, n_a, u_a]), los_desc=np.array([e_d, n_d, u_d]))
    print(f"Saved: {out.name}")

    # 8. Sanity check and pictures.
    final_up, final_east = up[-1], east[-1]
    print(f"\nSanity check over {common_dates[0]} to {common_dates[-1]}:")
    print(f"  Up:   median {np.nanmedian(final_up):+.1f} mm, middle 90% between "
          f"{np.nanpercentile(final_up, 5):+.1f} and {np.nanpercentile(final_up, 95):+.1f} mm")
    print(f"  East: median {np.nanmedian(final_east):+.1f} mm, middle 90% between "
          f"{np.nanpercentile(final_east, 5):+.1f} and {np.nanpercentile(final_east, 95):+.1f} mm")

    epsg = int(desc["epsg"])
    to_map = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    mx, my = to_map.transform(*MARKERS[desc_id])
    gx, gy = np.meshgrid(desc["x"], desc["y"])
    distance_km = np.hypot(gx - mx, gy - my) / 1000
    near, far = shared & (distance_km <= NEAR_KM), shared & (distance_km >= FAR_KM)

    print("\nDate       up near   up far   east near   east far   (mm)")
    rows = []
    for t, d in enumerate(common_dates):
        r = [float(np.nanmedian(up[t][near])), float(np.nanmedian(up[t][far])),
             float(np.nanmedian(east[t][near])), float(np.nanmedian(east[t][far]))]
        rows.append(r)
        print(f"{d}   {r[0]:+6.1f}   {r[1]:+6.1f}    {r[2]:+6.1f}     {r[3]:+6.1f}")

    flip = desc["y"][0] < desc["y"][-1]
    extent = (desc["x"].min(), desc["x"].max(), desc["y"].min(), desc["y"].max())
    bx, by = box_outline(SITES[desc_id], epsg)
    limit = float(np.nanpercentile(np.abs(np.concatenate([final_up[shared], final_east[shared]])), 98))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, data, title in [(axes[0], final_up, "Vertical (up)"), (axes[1], final_east, "East-west (east positive)")]:
        img = ax.imshow(np.flipud(data) if flip else data, extent=extent, cmap="RdBu",
                        vmin=-limit, vmax=limit, interpolation="nearest")
        ax.plot(bx, by, color="black", linewidth=0.8)
        ax.plot(mx, my, marker="x", color="black", markersize=10, mew=2)
        ax.set_title(f"{title}, {common_dates[0]} to {common_dates[-1]} (mm)")
        ax.tick_params(labelsize=7)
        fig.colorbar(img, ax=ax, shrink=0.8)
    fig.suptitle(f"{base_name}: ascending + descending combined")
    fig.tight_layout()
    fig.savefig(WORK_DIR / f"{base_name}_combined_maps.png", dpi=110)
    plt.close(fig)

    when = [datetime.strptime(d, "%Y%m%d") for d in common_dates]
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(when, rows[:, 0], color="red", marker="o", linewidth=3, label=f"Up, within {NEAR_KM:g} km of the slide")
    ax.plot(when, rows[:, 1], color="blue", marker="o", linewidth=3, label=f"Up, more than {FAR_KM:g} km away")
    ax.plot(when, rows[:, 2], color="red", linestyle="--", label="East, near")
    ax.plot(when, rows[:, 3], color="blue", linestyle="--", label="East, far")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Movement since first shared date (mm)")
    ax.set_title("Combined ascending + descending (rough near/far look)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(WORK_DIR / f"{base_name}_combined_over_time.png", dpi=110)
    plt.close(fig)
    print(f"\nSaved pictures: {base_name}_combined_maps.png and {base_name}_combined_over_time.png")


if __name__ == "__main__":
    main()