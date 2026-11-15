# pipeline/config.py
# Settings for every study site, kept in one place.
# Each site has: a box (as WKT, longitude first), dates to search,
# and the direction / track / frame to use (filled in after running search.py).

from datetime import datetime
from pathlib import Path

# The project root is the folder above this file's folder (the "bogwatch" folder).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
WORK_DIR = PROJECT_ROOT / "data" / "work"
# Meenbog is covered by four NISAR tracks, so it has one entry per track,
# all sharing the same box. Scripts treat each entry as its own "site".
MEENBOG_WKT = "POLYGON((-7.98 54.67, -7.78 54.67, -7.78 54.77, -7.98 54.77, -7.98 54.67))"
MEENBOG_START = datetime(2026, 6, 1)
MEENBOG_END = datetime(2026, 11, 9)

SITES = {
    "viking": {
        "name": "Viking Wind Farm, Shetland",
        # Box around the wind farm plus surrounding bog. Longitude first, then latitude.
        "wkt": "POLYGON((-1.40 60.20, -1.05 60.20, -1.05 60.42, -1.40 60.42, -1.40 60.20))",
        "start": datetime(2026, 6, 1),
        "end": datetime(2026, 11, 8),
        # Fill these in after running search.py (it suggests the best combination).
        "direction": "DESCENDING",
        "track": 81,
        "frame": 58,
                # Pairs left out of the time series, with the reason.
        "exclude_pairs": {
        "20261005_20261017": "steep band across the north, likely ionosphere; not ground motion",
        },
    },
    "galway": {
        "name": "Galway Wind Park, Connemara",
        "wkt": "POLYGON((-9.60 53.27, -9.15 53.27, -9.15 53.42, -9.60 53.42, -9.60 53.27))",
        "start": datetime(2026, 6, 1),
        "end": datetime(2026, 11, 8),
        "direction": "ASCENDING", 
        "track": 159, 
        "frame": 28,
        "exclude_pairs": {},
    },
        # Kept for searching only (no track chosen).
    "meenbog": {
        "name": "Meenbog Wind Farm, Co. Donegal",
        "wkt": MEENBOG_WKT,
        "start": MEENBOG_START,
        "end": MEENBOG_END,
        "direction": None,
        "track": None,
        "frame": None,
        "exclude_pairs": {},
    },
    "meenbog_d9": {
        "name": "Meenbog (descending, track 9)",
        "wkt": MEENBOG_WKT,
        "start": MEENBOG_START,
        "end": MEENBOG_END,
        "direction": "DESCENDING",
        "track": 9,
        "frame": 61,
        "exclude_pairs": {},
    },
    "meenbog_d110": {
        "name": "Meenbog (descending, track 110)",
        "wkt": MEENBOG_WKT,
        "start": MEENBOG_START,
        "end": MEENBOG_END,
        "direction": "DESCENDING",
        "track": 110,
        "frame": 61,
        "exclude_pairs": {},
    },
    "meenbog_a58": {
        "name": "Meenbog (ascending, track 58)",
        "wkt": MEENBOG_WKT,
        "start": MEENBOG_START,
        "end": MEENBOG_END,
        "direction": "ASCENDING",
        "track": 58,
        "frame": 29,
                # Pairs left out of the time series, with the reason.
        "exclude_pairs": {
            "20261004_20261016": "sharp east-west boundary across the box, likely ionosphere/atmosphere; disagrees with descending",
        },
    },
    "meenbog_a130": {
        "name": "Meenbog (ascending, track 130)",
        "wkt": MEENBOG_WKT,
        "start": MEENBOG_START,
        "end": MEENBOG_END,
        "direction": "ASCENDING",
        "track": 130,
        "frame": 29,
        "exclude_pairs": {},
    },
}


def raw_dir_for(site_id: str) -> Path:
    """Folder where a site's untouched downloads live. Created if missing."""
    folder = RAW_DIR / site_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder