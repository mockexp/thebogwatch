# pipeline/rainfall_check.py
# Step B: compares NISAR bog motion with Met Eireann daily rainfall.
# - uses a main (local) station, filling its gaps from a second station
# - total rain in each pair vs near/far movement (the early-July question)
# - a per-pixel "rain response" map: where ground rises more (or less) than average when it rains
# - rainfall before the November 2020 slide
# Pairs with missing rain days are left out of correlations and maps (never treated as zero).
# Usage:
#     python pipeline\rainfall_check.py meenbog data\raw\weather\met_eireann_ardnawark_daily.csv data\raw\weather\met_eireann_finner_daily.csv
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from config1 import SITES, WORK_DIR
from coherence_check1 import MARKERS

NEAR_KM, FAR_KM = 2.0, 5.0
LAG_DAYS = 7                      # also count rain in the week before each pair
SLIDE_DATE = datetime(2020, 11, 12)
COLOURS = {"local": "steelblue", "filled": "darkorange"}


DATE_LINE = re.compile(r"^\s*\d{1,2}-[A-Za-z]{3}-\d{4},")   # a data line, e.g. 01-jan-2019,...
LEGEND_LINE = re.compile(r"^\s*([A-Za-z_]+)\s*:?\s*-\s")     # a legend line, e.g. "  rain  - Precipitation"


def load_rain(path):
    """Daily rainfall (mm) from a Met Eireann CSV.
    Works whether or not the file has a header row: if there's none, the column order
    is taken from the legend lines at the top."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    station = next((l.split(":", 1)[1].strip() for l in lines if "station name" in l.lower()), Path(path).stem)

    header = next((i for i, l in enumerate(lines)
                   if l.strip().lower().replace('"', "").startswith("date,") and "rain" in l.lower()), None)
    if header is not None:
        table = pd.read_csv(path, skiprows=header, low_memory=False)
        table.columns = [c.strip().lower() for c in table.columns]
        date_values, rain_values = table["date"], table["rain"]
    else:
        first_data = next((i for i, l in enumerate(lines) if DATE_LINE.match(l)), None)
        legend = [m.group(1).lower() for l in lines[:first_data or 0] if (m := LEGEND_LINE.match(l))]
        if first_data is None or "date" not in legend or "rain" not in legend:
            raise ValueError(f"Couldn't find daily data with a rain column in {path}")
        rain_col = legend.index("rain") - legend.index("date")
        table = pd.read_csv(path, skiprows=first_data, header=None, low_memory=False)
        date_values, rain_values = table[0], table[rain_col]
        print(f"  ({station}: no header row, so rain was read from column {rain_col + 1} using the legend)")

    dates = pd.to_datetime(date_values, format="%d-%b-%Y", errors="coerce")
    if dates.isna().all():  # some files use a different date style
        dates = pd.to_datetime(date_values, dayfirst=True, errors="coerce")
    rain = pd.Series(pd.to_numeric(rain_values, errors="coerce").values, index=dates).dropna()
    rain = rain[~rain.index.isna()]
    rain = rain[~rain.index.duplicated()].sort_index()
    return station, rain


def combine(primary, fill):
    """Main station where it has data, otherwise the fill station, otherwise missing."""
    series = [primary] + ([fill] if fill is not None else [])
    days = pd.date_range(min(s.index.min() for s in series), max(s.index.max() for s in series), freq="D")
    p = primary.reindex(days)
    if fill is None:
        return p, pd.Series(np.where(p.notna(), "local", "missing"), index=days)
    f = fill.reindex(days)
    source = np.where(p.notna(), "local", np.where(f.notna(), "filled", "missing"))
    return p.fillna(f), pd.Series(source, index=days)


def rain_between(rain, source, start, end):
    """Total rain after 'start' up to and including 'end', with counts of local / filled / missing days."""
    mask = (rain.index > start) & (rain.index <= end)
    total = float(rain[mask].sum())
    local = int((source[mask] == "local").sum())
    filled = int((source[mask] == "filled").sum())
    missing = (end - start).days - local - filled
    return total, local, filled, missing


def load_track(path):
    """A track's time series and per-pair movements."""
    data = np.load(path)
    dates = [datetime.strptime(str(d), "%Y%m%d") for d in data["dates"]]
    cube = data["vertical_mm"]
    return {"name": path.stem.replace("_timeseries", ""), "dates": dates, "cube": cube,
            "steps": np.diff(cube, axis=0), "x": data["x"], "y": data["y"], "epsg": int(data["epsg"])}


def near_far_masks(track, marker_key):
    to_map = Transformer.from_crs("EPSG:4326", f"EPSG:{track['epsg']}", always_xy=True)
    mx, my = to_map.transform(*MARKERS[marker_key])
    gx, gy = np.meshgrid(track["x"], track["y"])
    dist = np.hypot(gx - mx, gy - my) / 1000
    valid = np.all(np.isfinite(track["cube"]), axis=0)
    return valid & (dist <= NEAR_KM), valid & (dist >= FAR_KM), (mx, my)


def slope_and_r(x, y):
    """Per-pixel straight-line fit of movement (y: pairs x rows x cols) against rain (x: pairs)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xc = x - x.mean()
        yc = y - np.nanmean(y, axis=0)
        cov = np.nansum(xc[:, None, None] * yc, axis=0) / (len(x) - 1)
        slope = cov / np.var(x, ddof=1)
        r = cov / (np.std(x, ddof=1) * np.nanstd(y, axis=0, ddof=1))
    return slope, r


def rain_bars(ax, rain, source):
    """Bar chart of daily rain, coloured by where each day's value came from."""
    for kind, colour in COLOURS.items():
        part = rain[source == kind]
        if len(part):
            ax.bar(part.index, part.values, color=colour, width=1.0,
                   label="local station" if kind == "local" else "filled from second station")
    ax.set_ylabel("Daily rain (mm)")
    ax.legend(fontsize=8, loc="upper right")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\rainfall_check.py <site> <main station CSV> [gap-filling station CSV]")
        return
    site_id = sys.argv[1]
    main_name, main_rain = load_rain(sys.argv[2])
    fill_name, fill_rain = load_rain(sys.argv[3]) if len(sys.argv) > 3 else (None, None)
    rain, source = combine(main_rain, fill_rain)
    label = main_name + (f" (gaps filled from {fill_name})" if fill_name else "")
    print(f"Main station: {main_name} | {main_rain.index.min():%d %b %Y} to {main_rain.index.max():%d %b %Y}")
    if fill_name:
        print(f"Gap filling:  {fill_name} | {fill_rain.index.min():%d %b %Y} to {fill_rain.index.max():%d %b %Y}")

    tracks = [load_track(p) for p in sorted(WORK_DIR.glob(f"{site_id}_*_timeseries.npz"))]
    if not tracks:
        print("No time series found. Run build_timeseries.py first.")
        return

    # Coverage over the NISAR period.
    first = min(t["dates"][0] for t in tracks)
    last = max(t["dates"][-1] for t in tracks)
    period_mask = (source.index > first - pd.Timedelta(days=LAG_DAYS)) & (source.index <= last)
    counts = source[period_mask].value_counts()
    total_days = int(period_mask.sum())
    print(f"\nNISAR period coverage ({total_days} days): "
          f"{counts.get('local', 0)} local, {counts.get('filled', 0)} filled, {counts.get('missing', 0)} missing\n")

    fig_ts, (ax_rain, ax_move) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig_map, map_axes = plt.subplots(1, len(tracks), figsize=(8 * len(tracks), 7), squeeze=False)
    styles = ["-", "--", ":"]

    for k, track in enumerate(tracks):
        name = track["name"]
        near, far, (mx, my) = near_far_masks(track, name if name in MARKERS else site_id)

        # Rain in each pair, with where it came from.
        pair_rain, pair_rain_lag, usable = [], [], []
        print(f"{name}:")
        print("  Pair                rain (mm)  incl. week before  near - far (mm)   days: local/filled/missing")
        steps = track["steps"]
        nmf = np.array([np.nanmedian(s[near]) - np.nanmedian(s[far]) for s in steps])
        for (a, b), d in zip(zip(track["dates"][:-1], track["dates"][1:]), nmf):
            total, loc, fil, miss = rain_between(rain, source, a, b)
            lagged, _, _, miss_lag = rain_between(rain, source, a - pd.Timedelta(days=LAG_DAYS), b)
            ok = miss == 0 and miss_lag == 0
            pair_rain.append(total)
            pair_rain_lag.append(lagged)
            usable.append(ok)
            flag = "" if ok else "   <- left out (missing days)"
            print(f"  {a:%d %b} -> {b:%d %b}   {total:8.1f}   {lagged:12.1f}      {d:+8.1f}          "
                  f"{loc:3d}/{fil:3d}/{miss:3d}{flag}")
        pair_rain, pair_rain_lag, usable = np.array(pair_rain), np.array(pair_rain_lag), np.array(usable)

        if usable.sum() >= 3:
            r_now = np.corrcoef(pair_rain[usable], nmf[usable])[0, 1]
            r_lag = np.corrcoef(pair_rain_lag[usable], nmf[usable])[0, 1]
            print(f"  Correlation of near - far with rain ({usable.sum()} usable pairs): "
                  f"{r_now:+.2f} (during pair), {r_lag:+.2f} (incl. week before)")
            print("  (few pairs, so treat these as hints, not proof)\n")
        else:
            print(f"  Only {usable.sum()} usable pairs, too few for a correlation.\n")

        # Cumulative near and far movement.
        near_series = [float(np.nanmedian(c[near])) for c in track["cube"]]
        far_series = [float(np.nanmedian(c[far])) for c in track["cube"]]
        ls = styles[k % len(styles)]
        ax_move.plot(track["dates"], near_series, color="red", linestyle=ls, marker="o", label=f"{name}: near slide")
        ax_move.plot(track["dates"], far_series, color="blue", linestyle=ls, marker="o", label=f"{name}: far")

        # Rain-response map from usable pairs only.
        ax = map_axes[0][k]
        if usable.sum() >= 3:
            slope, r = slope_and_r(pair_rain[usable], steps[usable])
            slope10 = slope * 10
            slope10[~np.all(np.isfinite(track["cube"]), axis=0)] = np.nan
            limit = float(np.nanpercentile(np.abs(slope10), 98))
            flip = track["y"][0] < track["y"][-1]
            extent = (track["x"].min(), track["x"].max(), track["y"].min(), track["y"].max())
            img = ax.imshow(np.flipud(slope10) if flip else slope10, extent=extent, cmap="RdBu",
                            vmin=-limit, vmax=limit, interpolation="nearest")
            ax.plot(mx, my, marker="x", color="black", markersize=10, mew=2)
            ax.set_title(f"{name}: rain response (mm per 10 mm rain), {usable.sum()} pairs\n"
                         "blue = rises more than average when it rains")
            ax.tick_params(labelsize=7)
            fig_map.colorbar(img, ax=ax, shrink=0.8)
        else:
            ax.axis("off")
            ax.set_title(f"{name}: too few usable pairs for a map")

    # Rainfall over the NISAR period.
    rain_bars(ax_rain, rain[period_mask], source[period_mask])
    ax_rain.set_title(f"Rainfall at {label} and NISAR movement (relative to site median)")
    ax_move.axhline(0, color="black", linewidth=0.8)
    ax_move.set_ylabel("Movement since first date (mm)")
    ax_move.legend(fontsize=8)
    fig_ts.autofmt_xdate()
    fig_ts.tight_layout()
    fig_ts.savefig(WORK_DIR / f"{site_id}_rain_vs_motion.png", dpi=110)
    plt.close(fig_ts)

    fig_map.tight_layout()
    fig_map.savefig(WORK_DIR / f"{site_id}_rain_response_maps.png", dpi=110)
    plt.close(fig_map)

    # Rainfall before the 2020 slide.
    window = (rain.index >= SLIDE_DATE - pd.Timedelta(days=60)) & (rain.index <= SLIDE_DATE + pd.Timedelta(days=7))
    if (source[window] != "missing").sum() > 30:
        fig, ax = plt.subplots(figsize=(11, 4))
        rain_bars(ax, rain[window], source[window])
        ax.axvline(SLIDE_DATE, color="red", linewidth=2)
        ax.set_title(f"Rainfall at {label} before the 2020 slide (red line = 12 Nov 2020)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(WORK_DIR / f"{site_id}_rain_before_slide.png", dpi=110)
        plt.close(fig)
        for days in (7, 30):
            total, loc, fil, miss = rain_between(rain, source, SLIDE_DATE - pd.Timedelta(days=days), SLIDE_DATE)
            print(f"Rain in the {days} days before the 2020 slide: {total:.0f} mm "
                  f"({loc} local, {fil} filled, {miss} missing days)")
        print(f"Saved picture: {site_id}_rain_before_slide.png")
    else:
        print("Not enough rainfall data around the 2020 slide for a chart.")

    print(f"Saved pictures: {site_id}_rain_vs_motion.png and {site_id}_rain_response_maps.png")


if __name__ == "__main__":
    main()