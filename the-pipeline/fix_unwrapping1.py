# pipeline/fix_unwrapping.py
# Finds unwrapping errors (neighbouring pixels differing by more than half a cycle),
# cuts each map along those jump lines, then shifts every piece by whole cycles so it
# lines up with the main area. Saves corrected pairs as new files and a before/after
# picture per pair. Originals are not changed.
# Usage:
#     python pipeline\fix_unwrapping.py viking

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from config1 import SITES, WORK_DIR

TWO_PI = 2 * np.pi
MIN_PIECE = 50     # pieces smaller than this (50 pixels at 40 m = 0.08 km²) are dropped
REACH = 3          # how many pixels to look across a jump line when comparing pieces
PASSES = 3         # repeat so pieces touching other pieces also get lined up


def find_jumps(unw):
    """1. Mark every pixel that differs from a neighbour by more than half a cycle (pi).
    Returns a map of those pixels and the number of jumps."""
    across = np.abs(np.diff(unw, axis=1)) > np.pi  # left-right neighbours (NaN comparisons are False)
    down = np.abs(np.diff(unw, axis=0)) > np.pi    # up-down neighbours
    pixels = np.zeros(unw.shape, dtype=bool)
    pixels[:, :-1] |= across
    pixels[:, 1:] |= across
    pixels[:-1, :] |= down
    pixels[1:, :] |= down
    return pixels, int(across.sum() + down.sum())


def fix_pair(unw):
    unw = unw.copy()
    valid = np.isfinite(unw)
    jump_pixels, jumps_before = find_jumps(unw)

    # 2. Cut along the jump lines: label the smooth pieces that remain.
    pieces, n_pieces = ndimage.label(valid & ~jump_pixels)
    if n_pieces == 0:
        return unw, jumps_before, jumps_before, 0, 0
    sizes = ndimage.sum(np.ones_like(pieces), pieces, index=np.arange(1, n_pieces + 1))
    order = np.argsort(sizes)[::-1] + 1  # piece numbers, biggest first

    # 3. The biggest piece is the trusted main area.
    main = pieces == order[0]
    shifted_pixels, cycles_used = 0, 0

    # 4-5. Line up the other pieces with the main area, a few passes.
    for _ in range(PASSES):
        joined_any = False
        for piece_id in order[1:]:
            piece = pieces == piece_id
            if piece.sum() < MIN_PIECE or (piece & main).any():
                continue  # too small, or already joined
            # Main-area pixels just across the jump line from this piece...
            ring = ndimage.binary_dilation(piece, iterations=REACH) & main
            if ring.sum() < 5:
                continue  # doesn't touch the main area yet; try again next pass
            # ...and this piece's own pixels next to them.
            edge = ndimage.binary_dilation(ring, iterations=REACH) & piece
            cycles = np.round((np.median(unw[ring]) - np.median(unw[edge])) / TWO_PI)
            if cycles != 0:
                unw[piece] += cycles * TWO_PI
                shifted_pixels += int(piece.sum())
                cycles_used += int(abs(cycles))
            main |= piece
            joined_any = True
        if not joined_any:
            break

    # 6. Keep only the joined main area (drops jump lines and tiny or unconnected pieces).
    unw[~main] = np.nan

    # 7. Count jumps again.
    _, jumps_after = find_jumps(unw)
    return unw, jumps_before, jumps_after, shifted_pixels, cycles_used


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\fix_unwrapping.py <site>")
        return

    site_id = sys.argv[1]
    files = sorted(WORK_DIR.glob(f"{site_id}_unw_*.npz"))
    if not files:
        print("No unwrapped pairs found. Run unwrap_site.py first.")
        return

    print("Pair                    jumps before   jumps after   pixels shifted   cycles   kept")
    for path in files:
        data = np.load(path)
        saved = {key: data[key] for key in data.files}
        before = saved["unwrapped"]
        after, n_before, n_after, n_shifted, n_cycles = fix_pair(before)

        kept = 100 * np.isfinite(after).sum() / max(np.isfinite(before).sum(), 1)
        ref, sec = str(saved["reference"]), str(saved["secondary"])
        print(f"{ref} to {sec}   {n_before:12d}   {n_after:11d}   {n_shifted:14d}   {n_cycles:6d}   {kept:4.0f}%")

        # Save the corrected pair as a new file.
        saved["unwrapped"] = after
        np.savez_compressed(WORK_DIR / path.name.replace("_unw_", "_unwfix_"), **saved)

        # Before / jump lines / after picture.
        y = saved["y"]
        flip = y[0] < y[-1]
        extent = (saved["x"].min(), saved["x"].max(), y.min(), y.max())
        jump_pixels, _ = find_jumps(before)

        def prep(a):
            return np.flipud(a) if flip else a

        b_shown = before - np.nanmedian(before)
        a_shown = after - np.nanmedian(after)
        limit = np.nanpercentile(np.abs(b_shown), 98)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(prep(b_shown), extent=extent, cmap="RdBu", vmin=-limit, vmax=limit, interpolation="nearest")
        axes[0].set_title("Before (radians, relative)")
        axes[1].imshow(prep(np.where(jump_pixels, 1.0, np.where(np.isfinite(before), 0.0, np.nan))),
                       extent=extent, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title(f"Jump lines in black ({n_before} jumps)")
        axes[2].imshow(prep(a_shown), extent=extent, cmap="RdBu", vmin=-limit, vmax=limit, interpolation="nearest")
        axes[2].set_title(f"After ({n_after} jumps left, {n_cycles} cycle shifts)")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{SITES[site_id]['name']}  |  {ref} to {sec}")
        fig.tight_layout()
        fig.savefig(WORK_DIR / f"{site_id}_fix_{ref}_{sec}.png", dpi=100)
        plt.close(fig)

    print("\nCorrected pairs saved as *_unwfix_*.npz, pictures as *_fix_*.png in data\\work")


if __name__ == "__main__":
    main()