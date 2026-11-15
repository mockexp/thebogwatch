# pipeline/search.py
# Lists NISAR GUNW products over a site without downloading anything.
# Usage (Command Prompt, inside the bogwatch folder, with (.venv) showing):
#     python pipeline\search.py viking

import sys
from collections import Counter
from getpass import getpass

import asf_search as asf
import pandas as pd

from config1 import SITES, WORK_DIR


def get_prop(product, *keys):
    """Return the first metadata value found under any of the given names.
    NISAR's field names may differ from older missions, so we try a few."""
    for key in keys:
        value = product.properties.get(key)
        if value not in (None, ""):
            return value
    return None
def total_bytes(value):
    """NISAR stores file sizes as a dictionary (one entry per file in the product)
    instead of a single number. Add them all up; return 0 if the format is unknown."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        total = 0
        for entry in value.values():
            if isinstance(entry, dict):
                total += entry.get("bytes", 0) or 0
            elif isinstance(entry, (int, float)):
                total += entry
        return total
    return 0

def main():
    # 1. Which site? Read it from the command line, e.g. "viking".
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print(f"Usage: python pipeline\\search.py <site>   (sites: {', '.join(SITES)})")
        return

    site_id = sys.argv[1]
    site = SITES[site_id]
    print(f"Searching NISAR GUNW over: {site['name']}")

    # 2. Log in to Earthdata. The password is hidden as you type.
    session = asf.ASFSession()
    session.auth_with_creds(input("Earthdata username: "), getpass("Earthdata password: "))

    # 3. Search with the same filters you would set in Vertex.
    opts = asf.ASFSearchOptions(
        maxResults=500,
        intersectsWith=site["wkt"],
        start=site["start"],
        end=site["end"],
        dataset=["NISAR"],
        processingLevel=["GUNW"],
        productionConfiguration=["PR"],  # PR = provisional (calibrated). Avoids beta products.
        session=session,
    )
    results = asf.search(opts=opts)
    print(f"\nFound {len(results)} GUNW products.\n")

    if len(results) == 0:
        print("Nothing found. Check the dates or the box, or try again closer to the hackathon.")
        return

    # 4. Show every metadata field name once, so we know what NISAR calls things.
    print("Metadata fields available on each product:")
    print(", ".join(sorted(results[0].properties.keys())))
    print()

    # 5. One row per product.
    rows = []
    for product in results:
        rows.append({
            "name": get_prop(product, "sceneName", "fileID"),
            "start": get_prop(product, "startTime"),
            "stop": get_prop(product, "stopTime"),
            "direction": get_prop(product, "flightDirection"),
            "track": get_prop(product, "pathNumber", "relativeOrbit"),
            "frame": get_prop(product, "frameNumber", "frame"),
            "polarization": get_prop(product, "mainBandPolarization", "polarization"),
            "bytes": get_prop(product, "bytes"),
        })

    table = pd.DataFrame(rows).sort_values("start")
    # Convert file size to megabytes where possible (left blank if NISAR reports it differently).
    table["size_mb"] = table["bytes"].apply(total_bytes) / 1e6
    table = table.drop(columns=["bytes"])

    # 6. Save the list so the team can open it in Excel.
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = WORK_DIR / f"{site_id}_gunw_search.csv"
    table.to_csv(out_csv, index=False)

    print(table.to_string(index=False, max_colwidth=60))
    print(f"\nSaved the list to: {out_csv}")

    # 7. Count products per (direction, track, frame). The biggest group is your time series.
    combos = Counter(zip(table["direction"], table["track"], table["frame"]))
    print("\nProducts per direction / track / frame (most common first):")
    for (direction, track, frame), count in combos.most_common():
        print(f"  {direction}, track {track}, frame {frame}: {count} products")

    best = combos.most_common(1)[0][0]
    print("\nSuggested settings for config.py:")
    print(f'  "direction": "{best[0]}", "track": {best[1]}, "frame": {best[2]},')


if __name__ == "__main__":
    main()