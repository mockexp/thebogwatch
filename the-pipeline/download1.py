# pipeline/download.py
# Downloads NISAR GUNW files for one site, on the direction/track/frame chosen in config.py.
# Usage (Command Prompt, inside bogwatch, with (.venv) showing):
#     python pipeline\download.py viking 3      <- the 3 earliest products
#     python pipeline\download.py viking all    <- everything (large!)
import pandas as pd
import sys
from getpass import getpass
from pathlib import Path

import asf_search as asf

from config1 import SITES, WORK_DIR, raw_dir_for


def get_prop(product, *keys):
    """Return the first metadata value found under any of the given names."""
    for key in keys:
        value = product.properties.get(key)
        if value not in (None, ""):
            return value
    return None


def same_number(a, b):
    """Compare track/frame values that may arrive as text or as numbers."""
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return False


def main():
    # 1. Which site, and how many files?
    if len(sys.argv) < 3 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\download.py <site> <how many, or 'all'>")
        return

    site_id, how_many = sys.argv[1], sys.argv[2]
    site = SITES[site_id]

    # 2. Make sure the path in the sky has been chosen.
    if site["direction"] is None or site["track"] is None or site["frame"] is None:
        print("Fill in direction, track and frame for this site in config.py first (run search.py).")
        return

    # 3. Log in and search again.
    session = asf.ASFSession()
    session.auth_with_creds(input("Earthdata username: "), getpass("Earthdata password: "))

    opts = asf.ASFSearchOptions(
        maxResults=500,
        intersectsWith=site["wkt"],
        start=site["start"],
        end=site["end"],
        dataset=["NISAR"],
        processingLevel=["GUNW"],
        productionConfiguration=["PR"],
        session=session,
    )
    results = asf.search(opts=opts)

    # 4. Keep only products on the chosen direction, track and frame.
    chosen = [
        p for p in results
        if get_prop(p, "flightDirection") == site["direction"]
        and same_number(get_prop(p, "pathNumber"), site["track"])
        and same_number(get_prop(p, "frameNumber"), site["frame"])
    ]

    # 5. Oldest first, then take the number requested.
    chosen.sort(key=lambda p: get_prop(p, "startTime") or "")
    print(f"{len(chosen)} products match {site['direction']}, track {site['track']}, frame {site['frame']}.")
    if how_many != "all":
        chosen = chosen[: int(how_many)]

        # Record exactly which products and versions were selected, for reproducibility.
    # NISAR products can be reprocessed into newer versions (different CRID), so the
    # manifest pins down precisely what this analysis used. It is small: commit it to GitHub.
    manifest_rows = [{
        "name": get_prop(p, "sceneName", "fileID"),
        "file": get_prop(p, "fileName"),
        "start": get_prop(p, "startTime"),
        "stop": get_prop(p, "stopTime"),
        "direction": get_prop(p, "flightDirection"),
        "track": get_prop(p, "pathNumber"),
        "frame": get_prop(p, "frameNumber"),
        "crid": get_prop(p, "crid"),
        "processing_date": get_prop(p, "processingDate"),
        "url": get_prop(p, "url"),
    } for p in chosen]
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = WORK_DIR / f"{site_id}_manifest.csv"
    manifest = pd.DataFrame(manifest_rows)
    if manifest_path.exists():
        # Merge with earlier runs; if a product was re-selected, keep the newest record.
        manifest = pd.concat([pd.read_csv(manifest_path), manifest]).drop_duplicates(subset="name", keep="last")
    manifest.sort_values("start").to_csv(manifest_path, index=False)
    print(f"Manifest updated: {manifest_path.name}")

    # 6. From each product, take the main .h5 file and skip the QA statistics file.
    urls = asf.ASFSearchResults(chosen).find_urls(
        extension=".h5",
        pattern=r"^(?!.*QA_STATS).*",
        directAccess=False,
    )

    # 7. Skip anything already downloaded.
    out_dir = raw_dir_for(site_id)
    to_get = [u for u in urls if not (out_dir / Path(u).name).exists()]
    print(f"{len(urls)} files selected, {len(urls) - len(to_get)} already downloaded, {len(to_get)} to fetch.")
    print(f"Saving to: {out_dir}")

    # 8. Download, two at a time.
    if to_get:
        asf.download_urls(to_get, str(out_dir), session=session, processes=2)
    print("Done.")


if __name__ == "__main__":
    main()