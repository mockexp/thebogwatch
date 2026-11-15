# pipeline/inspect_file.py
# Lists every layer inside a NISAR .h5 file without loading the data.
# Usage (Command Prompt, inside bogwatch, with (.venv) showing):
#     python pipeline\inspect_file.py viking                 <- first GUNW file for the site
#     python pipeline\inspect_file.py viking <file name>     <- a specific file

import sys

import h5py

from config1 import SITES, WORK_DIR, raw_dir_for


def to_text(value):
    """Attributes are often stored as bytes; turn them into normal text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tobytes") and getattr(value, "dtype", None) is not None and value.dtype.kind == "S":
        return value.tobytes().decode("utf-8", errors="replace")
    return str(value)


def main():
    # 1. Which site and which file?
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\inspect_file.py <site> [file name]")
        return

    site_id = sys.argv[1]
    folder = raw_dir_for(site_id)

    if len(sys.argv) >= 3:
        path = folder / sys.argv[2]
    else:
        files = sorted(folder.glob("*GUNW*.h5"))
        if not files:
            print(f"No GUNW .h5 files found in {folder}")
            return
        path = files[0]

    print(f"Inspecting: {path.name}\n")

    all_lines = []      # everything, for the text file
    image_lines = []    # only image-sized layers, for the screen

    # 2. Visit every item inside the file.
    def visit(name, item):
        if isinstance(item, h5py.Dataset):
            # 3. Record path, shape, type, units and description.
            units = to_text(item.attrs.get("units", ""))
            description = to_text(item.attrs.get("description", ""))
            line = f"/{name} | shape {item.shape} | {item.dtype} | units: {units}"
            if description:
                line += f" | {description[:120]}"
            all_lines.append(line)

            # 5. Image-sized layers: at least 2 dimensions, each at least 50 pixels.
            if item.ndim >= 2 and min(item.shape[-2:]) >= 50:
                image_lines.append(f"/{name}\n    shape {item.shape} | {item.dtype} | units: {units}")
        else:
            all_lines.append(f"/{name}/   (folder)")

    with h5py.File(path, "r") as f:
        f.visititems(visit)

    # 4. Save the full list to a text file.
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = WORK_DIR / f"{site_id}_gunw_structure.txt"
    out_txt.write_text("\n".join(all_lines), encoding="utf-8")

    print(f"Image-sized layers ({len(image_lines)} found):\n")
    print("\n".join(image_lines))
    print(f"\nFull list of {len(all_lines)} items saved to: {out_txt}")


if __name__ == "__main__":
    main()