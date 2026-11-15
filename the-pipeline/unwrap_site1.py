# pipeline/unwrap_site.py
# NISAR's own unwrapped phase is empty (all zeros) over Viking, but the 20 m wrapped
# interferogram is clean. This script crops the wrapped phase to the site, averages
# 2x2 pixels (20 m -> 40 m), hides unreliable pixels, and unwraps it ourselves.
# Usage:
#     python pipeline\unwrap_site.py viking

import sys
import warnings

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.restoration import unwrap_phase

from config1 import SITES, WORK_DIR, raw_dir_for
from coherence_check1 import find_dataset, get_epsg, pair_dates
from frame_overview1 import WRAPPED_GRID, WRAPPED, box_outline, index_range

WRAPPED_COHERENCE = WRAPPED_GRID + "/HH/coherenceMagnitude"
LOOKS = 2              # average 2x2 pixels: 20 m -> 40 m
MIN_COHERENCE = 0.3    # hide pixels less trustworthy than this
MIN_REGION_PIXELS = 400  # hide isolated patches smaller than this (400 pixels at 40 m = about 0.6 km²)

def multilook(array, n):
    """Average each n x n block of pixels into one (drops leftover edge pixels).
    For complex numbers this cancels noise while real signal adds up."""
    rows = (array.shape[0] // n) * n
    cols = (array.shape[1] // n) * n
    blocks = array[:rows, :cols].reshape(rows // n, n, cols // n, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # blocks that are all blank
        return np.nanmean(blocks, axis=(1, 3))


def multilook_coords(coords, n):
    """Centre coordinate of each block of n pixels."""
    usable = (len(coords) // n) * n
    return coords[:usable].reshape(-1, n).mean(axis=1)


def unwrap_file(path, site_id, site):
    ref, sec = pair_dates(path.name)
    with h5py.File(path, "r") as f:
        epsg = get_epsg(f)
        bx, by = box_outline(site, epsg)

        # 1. Crop the 20 m wrapped interferogram and its coherence to the site.
        wx = f[find_dataset(f, "xCoordinates", prefer=WRAPPED_GRID)][:]
        wy = f[find_dataset(f, "yCoordinates", prefer=WRAPPED_GRID)][:]
        rows, cols = index_range(wy, by.min(), by.max()), index_range(wx, bx.min(), bx.max())
        ifg = f[WRAPPED][rows[0]:rows[1], cols[0]:cols[1]]
        coh = f[WRAPPED_COHERENCE][rows[0]:rows[1], cols[0]:cols[1]].astype("float64")
        xs, ys = wx[cols[0]:cols[1]], wy[rows[0]:rows[1]]

    # No-data pixels: treat as blank before averaging.
    ifg = ifg.astype("complex128")
    no_data = (np.abs(ifg) == 0) | ~np.isfinite(coh) | (coh <= 0)
    ifg[no_data] = np.nan
    coh[no_data] = np.nan

    # 2. Average 2x2 blocks (multilooking).
    ifg_ml = multilook(ifg, LOOKS)
    coh_ml = multilook(coh, LOOKS)
    xs_ml, ys_ml = multilook_coords(xs, LOOKS), multilook_coords(ys, LOOKS)

    # 3. The phase (clock-hand position) of each averaged pixel.
    wrapped_phase = np.angle(ifg_ml)

    # 4. Hide unreliable pixels.
    hide = ~np.isfinite(ifg_ml) | ~np.isfinite(coh_ml) | (coh_ml < MIN_COHERENCE)
    wrapped_phase[hide] = 0.0  # placeholder; hidden pixels are ignored by the unwrapper
        # 4b. Also hide small isolated patches (noise specks over sea), keeping big connected land.
    labels, n_patches = ndimage.label(~hide)
    if n_patches > 0:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n_patches + 1))
        small_ids = np.where(sizes < MIN_REGION_PIXELS)[0] + 1
        hide = hide | np.isin(labels, small_ids)

    # 5. Unwrap only the reliable pixels.
    unwrapped = unwrap_phase(np.ma.masked_array(wrapped_phase, mask=hide))
    unwrapped = unwrapped.filled(np.nan)

    # 6. Save the result for later stages.
    out_npz = WORK_DIR / f"{site_id}_unw_{ref}_{sec}.npz"
    np.savez_compressed(out_npz, unwrapped=unwrapped, coherence=coh_ml,
                        x=xs_ml, y=ys_ml, epsg=epsg, reference=ref, secondary=sec)

    # 7. Picture: before, coherence, after (north at the top).
    kept = np.isfinite(unwrapped)
    shown = unwrapped - np.nanmedian(unwrapped)
    limit = np.nanpercentile(np.abs(shown), 98) if kept.any() else 1
    flip = ys_ml[0] < ys_ml[-1]
    extent = (xs_ml.min(), xs_ml.max(), ys_ml.min(), ys_ml.max())

    def prep(a):
        return np.flipud(a) if flip else a

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    before = np.where(hide, np.nan, np.angle(ifg_ml))
    panels = [
        (before, "Wrapped phase, 40 m (unreliable hidden)", "twilight", -np.pi, np.pi),
        (coh_ml, "Coherence, 40 m", "gray", 0, 1),
        (shown, "Our unwrapped phase, radians (relative)", "RdBu", -limit, limit),
    ]
    for ax, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        img = ax.imshow(prep(data), extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.plot(bx, by, color="red", linewidth=1)
        ax.set_title(title)
        ax.tick_params(labelsize=7)
        fig.colorbar(img, ax=ax, shrink=0.7)
    fig.suptitle(f"{site['name']}  |  {ref} to {sec}  |  pixels kept {100 * kept.mean():.0f}%")
    fig.tight_layout()
    out_png = WORK_DIR / f"{site_id}_unw_{ref}_{sec}.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    # 8. Summary numbers.
    span = float(np.nanpercentile(unwrapped, 99) - np.nanpercentile(unwrapped, 1)) if kept.any() else np.nan
    print(f"  {ref} to {sec}: kept {100 * kept.mean():.0f}% of pixels, "
          f"phase spans {span:.1f} radians (about {span / (2 * np.pi):.1f} cycles). "
          f"Saved {out_npz.name} and {out_png.name}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\unwrap_site.py <site>")
        return

    site_id = sys.argv[1]
    site = SITES[site_id]
    files = sorted(raw_dir_for(site_id).glob("*GUNW*.h5"))
    if not files:
        print("No GUNW files found.")
        return

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for path in files:
        print(f"Unwrapping {path.name[:60]}...")
        unwrap_file(path, site_id, site)


if __name__ == "__main__":
    main()