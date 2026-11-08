"""Inspect one flight folder: file counts, types, and a peek at the poses.

Usage (from the project root):
    python scripts/00_inspect_data.py data/raw/Raw_lowresRGB/<flight_folder>
"""
import sys
from collections import Counter
from pathlib import Path


def peek(path: Path, n_lines: int = 8, max_chars: int = 200) -> None:
    print(f"\n--- {path.name} ({path.stat().st_size} bytes) ---")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_lines:
                    print("...")
                    break
                print(line.rstrip()[:max_chars])
    except UnicodeDecodeError:
        print("(binary file, cannot show as text)")


def main(root: Path) -> None:
    if not root.exists():
        sys.exit(f"Folder not found: {root}")

    print(f"Inspecting: {root.resolve()}\n")
    print("Subfolders:")
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [p for p in sub.rglob("*") if p.is_file()]
        exts = Counter(p.suffix.lower() for p in files)
        print(f"  {sub.name:28s} {len(files):5d} files  {dict(exts)}")
        for p in sorted(files)[:3]:
            print(f"      e.g. {p.relative_to(sub)}")

    loose = [p for p in root.iterdir() if p.is_file()]
    if loose:
        print("\nFiles directly in this folder:")
        for p in loose:
            print(f"  {p.name}")

    poses = root / "poses"
    if poses.exists():
        text_files = sorted(p for p in poses.rglob("*") if p.is_file())[:3]
        for p in text_files:
            peek(p)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))