# pipeline/frame_overview.py
# Whole-frame overview of every downloaded GUNW file, with the site box drawn on top,
# plus the 20 m wrapped interferogram over the site. Used to find out where the
# unwrapped phase is zero and whether the raw (wrapped) signal is real.
# Usage:
#     python pipeline\frame_overview.py viking

import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt as shapely_wkt

from config1 import SITES, WORK_DIR, raw_dir_for
from coherence_check1 import GRID, COHERENCE, PHASE, COMPONENTS, find_dataset, get_epsg, pair_dates

WRAPPED_GRID = "/science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram"
WRAPPED = WRAPPED_GRID + "/HH/wrappedInterferogram"
MASK = GRID + "/mask"
STEP = 8  # read every 8th pixel for the whole-frame view


def box_outline(site, epsg):
    """The site box traced with many points, converted to the file's map projection."""
    to_map = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    lon_min, lat_min, lon_max, lat_max = shapely_wkt.loads(site["wkt"]).bounds
    n = 50
    lons = np.concatenate([np.linspace(lon_min, lon_max, n), np.full(n, lon_max),
                           np.linspace(lon_max, lon_min, n), np.full(n, lon_min)])
    lats = np.concatenate([np.full(n, lat_min), np.linspace(lat_min, lat_max, n),
                           np.full(n, lat_max), np.linspace(lat_max, lat_min, n)])
    bx, by = to_map.transform(lons, lats)
    return np.asarray(bx), np.asarray(by)


def index_range(coords, low, high):
    """First and last+1 positions whose coordinates fall between low and high."""
    idx = np.where((coords >= low) & (coords <= high))[0]
    return (idx.min(), idx.max() + 1) if idx.size else None


def show(ax, data, xs, ys, title, cmap, vmin=None, vmax=None, outline=None):
    """Draw one panel with north at the top, optionally with the site box on top."""
    if ys[0] < ys[-1]:
        data = np.flipud(data)
    extent = (xs.min(), xs.max(), ys.min(), ys.max())
    img = ax.imshow(data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if outline is not None:
        ax.plot(outline[0], outline[1], color="red", linewidth=1.5)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    plt.colorbar(img, ax=ax, shrink=0.6)


def overview(path, site_id, site):
    ref, sec = pair_dates(path.name)
    with h5py.File(path, "r") as f:
        epsg = get_epsg(f)
        x = f[find_dataset(f, "xCoordinates")][:]
        y = f[find_dataset(f, "yCoordinates")][:]
        bx, by = box_outline(site, epsg)

        # 1-2. Whole frame, every 8th pixel.
        coh = f[COHERENCE][::STEP, ::STEP].astype("float64")
        phase = f[PHASE][::STEP, ::STEP].astype("float64")
        comps = f[COMPONENTS][::STEP, ::STEP].astype("float64")
        mask = f[MASK][::STEP, ::STEP].astype("float64")
        xs, ys = x[::STEP], y[::STEP]

        # The site crop at full 80 m detail, for the percentages.
        rows, cols = index_range(y, by.min(), by.max()), index_range(x, bx.min(), bx.max())
        crop_coh = f[COHERENCE][rows[0]:rows[1], cols[0]:cols[1]].astype("float64")
        crop_phase = f[PHASE][rows[0]:rows[1], cols[0]:cols[1]].astype("float64")
        crop_mask = f[MASK][rows[0]:rows[1], cols[0]:cols[1]]

        # 4. The 20 m wrapped interferogram over the site only.
        wrapped_angle, wxs, wys = None, None, None
        wx_path = find_dataset(f, "xCoordinates", prefer=WRAPPED_GRID)
        wy_path = find_dataset(f, "yCoordinates", prefer=WRAPPED_GRID)
        wx, wy = f[wx_path][:], f[wy_path][:]
        if len(wx) == f[WRAPPED].shape[1] and len(wy) == f[WRAPPED].shape[0]:
            wr, wc = index_range(wy, by.min(), by.max()), index_range(wx, bx.min(), bx.max())
            if wr and wc:
                wrapped = f[WRAPPED][wr[0]:wr[1], wc[0]:wc[1]]
                wrapped_angle = np.angle(wrapped)  # the phase of each complex number, -pi to +pi
                wrapped_angle[np.abs(wrapped) == 0] = np.nan
                wxs, wys = wx[wc[0]:wc[1]], wy[wr[0]:wr[1]]
        else:
            print("  Could not match coordinates to the wrapped layer; skipping that panel.")

    # 5. Percentages of usable pixels whose phase is exactly zero.
    valid_crop = np.isfinite(crop_coh) & (crop_coh > 0)
    valid_frame = np.isfinite(coh) & (coh > 0)
    pct_zero_site = 100 * float((crop_phase[valid_crop] == 0).mean()) if valid_crop.any() else np.nan
    pct_zero_frame = 100 * float((phase[valid_frame] == 0).mean()) if valid_frame.any() else np.nan
    mask_values = np.unique(crop_mask)
    mask_text = ", ".join(str(v) for v in mask_values[:10]) + (" ..." if len(mask_values) > 10 else "")

    # Prepare the panels: hide no-data and exact zeros so real values stand out.
    coh[coh <= 0] = np.nan
    phase_shown = np.where((phase == 0) | ~valid_frame, np.nan, phase)
    finite_phase = phase_shown[np.isfinite(phase_shown)]
    limit = np.percentile(np.abs(finite_phase), 98) if finite_phase.size else 1
    comps_shown = np.where((comps == 0) | (comps == 65535), np.nan, comps)

    n_panels = 5 if wrapped_angle is not None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5))
    outline = (bx, by)
    show(axes[0], coh, xs, ys, "Coherence (whole frame)", "gray", 0, 1, outline)
    show(axes[1], phase_shown, xs, ys, "Unwrapped phase, zeros hidden", "RdBu", -limit, limit, outline)
    show(axes[2], comps_shown, xs, ys, "Connected components", "tab20", None, None, outline)
    show(axes[3], mask, xs, ys, "Mask layer (raw values)", "viridis", None, None, outline)
    if wrapped_angle is not None:
        show(axes[4], wrapped_angle, wxs, wys, "Wrapped phase over site, 20 m", "twilight", -np.pi, np.pi)

    fig.suptitle(f"{site['name']}  |  {ref} to {sec}  |  zero phase: site {pct_zero_site:.1f}%, frame {pct_zero_frame:.1f}%")
    fig.tight_layout()
    out_png = WORK_DIR / f"{site_id}_overview_{ref}_{sec}.png"
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    print(f"  Saved picture: {out_png.name}")

    return {
        "reference": ref, "secondary": sec,
        "pct_zero_phase_site": round(pct_zero_site, 1),
        "pct_zero_phase_frame": round(pct_zero_frame, 1),
        "mask_values_in_site": mask_text,
    }


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\frame_overview.py <site>")
        return

    site_id = sys.argv[1]
    site = SITES[site_id]
    files = sorted(raw_dir_for(site_id).glob("*GUNW*.h5"))
    if not files:
        print("No GUNW files found.")
        return

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        print(f"Overview of {path.name[:60]}...")
        rows.append(overview(path, site_id, site))

    # 6. Summary table.
    table = pd.DataFrame(rows)
    print("\nSummary:\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()