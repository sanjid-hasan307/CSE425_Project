"""
download_dataset.py
====================
Downloads and organises MIDI datasets for the Music Generation project.

Datasets:
  1. MAESTRO v3.0.0  — Classical Piano MIDI (Google Magenta)
  2. Lakh MIDI Dataset (LMD-matched subset, ~45k files, ~1.5 GB)

Usage:
    python download_dataset.py [--maestro] [--lakh] [--all]

After running, the folder layout will be:

    data/
    └── raw/
        ├── maestro/
        │   ├── maestro-v3.0.0.json        # metadata
        │   └── 2004/                       # year-based subfolders
        │       └── *.midi
        └── lakh/
            └── lmd_matched/
                └── <artist>/<track>.mid   # genre-tagged MIDI files
"""

import os
import json
import zipfile
import tarfile
import hashlib
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent
DATA_RAW     = ROOT / "data" / "raw"
MAESTRO_DIR  = DATA_RAW / "maestro"
LAKH_DIR     = DATA_RAW / "lakh"

MAESTRO_URL  = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
LAKH_URL     = "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz"

# ── Helpers ────────────────────────────────────────────────────────────────

def _sizeof_fmt(num: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _download(url: str, dest: Path, desc: str) -> None:
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already downloaded.")
        return

    print(f"  Downloading {desc} ...")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [ERROR] Could not reach {url}: {exc}")
        print("  Please download manually (see README for instructions).")
        return

    total = int(resp.headers.get("Content-Length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=f"  {dest.name}", ncols=80
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved -> {dest}  ({_sizeof_fmt(dest.stat().st_size)})")


def _extract_zip(archive: Path, dest: Path) -> None:
    """Extract a .zip archive."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive.name} ...")
    with zipfile.ZipFile(archive, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="  Unzipping", ncols=80):
            zf.extract(member, dest)
    print(f"  Extracted -> {dest}")


def _extract_tar(archive: Path, dest: Path) -> None:
    """Extract a .tar.gz archive."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive.name} …")
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for member in tqdm(members, desc="  Untarring", ncols=80):
            tf.extract(member, dest, filter="data")
    print(f"  Extracted -> {dest}")


def _count_midi(folder: Path) -> int:
    """Count all .mid / .midi files recursively."""
    return sum(1 for _ in folder.rglob("*") if _.suffix.lower() in {".mid", ".midi"})


# ── Dataset downloaders ────────────────────────────────────────────────────

def download_maestro() -> None:
    """Download and extract MAESTRO v3.0.0."""
    print("\n" + "=" * 55)
    print("  [1/2] MAESTRO v3.0.0 - Classical Piano")
    print("=" * 55)

    archive = DATA_RAW / "maestro-v3.0.0-midi.zip"
    _download(MAESTRO_URL, archive, "MAESTRO v3.0.0")

    if archive.exists():
        _extract_zip(archive, MAESTRO_DIR)
        n = _count_midi(MAESTRO_DIR)
        print(f"\n  [OK] MAESTRO ready - {n} MIDI files in {MAESTRO_DIR}")
    else:
        _print_manual_maestro()


def download_lakh() -> None:
    """Download and extract the Lakh MIDI Dataset (matched subset)."""
    print("\n" + "=" * 55)
    print("  [2/2] Lakh MIDI Dataset (LMD-matched)")
    print("=" * 55)

    archive = DATA_RAW / "lmd_matched.tar.gz"
    _download(LAKH_URL, archive, "LMD-matched")

    if archive.exists():
        _extract_tar(archive, LAKH_DIR)
        n = _count_midi(LAKH_DIR)
        print(f"\n  [OK] Lakh MIDI ready - {n} MIDI files in {LAKH_DIR}")
    else:
        _print_manual_lakh()


# ── Manual instructions (fallback) ─────────────────────────────────────────

def _print_manual_maestro() -> None:
    print("""
  MANUAL DOWNLOAD — MAESTRO v3.0.0
  ----------------------------------
  1. Visit: https://magenta.tensorflow.org/datasets/maestro
  2. Download: maestro-v3.0.0-midi.zip
  3. Extract into: data/raw/maestro/

  Expected layout after extraction:
    data/raw/maestro/
    ├── maestro-v3.0.0.json
    ├── 2004/
    │   └── *.midi
    ├── 2006/ … 2018/
""")


def _print_manual_lakh() -> None:
    print("""
  MANUAL DOWNLOAD — Lakh MIDI Dataset
  -------------------------------------
  1. Visit: https://colinraffel.com/projects/lmd/
  2. Download: lmd_matched.tar.gz  (~1.6 GB)
  3. Extract into: data/raw/lakh/

  Expected layout after extraction:
    data/raw/lakh/lmd_matched/
    └── <MSD_track_id>/
        └── *.mid
""")


# ── Print folder summary ───────────────────────────────────────────────────

def summarise() -> None:
    """Print a summary of what was downloaded."""
    print("\n" + "=" * 55)
    print("  Dataset Summary")
    print("=" * 55)

    for name, folder in [("MAESTRO", MAESTRO_DIR), ("Lakh MIDI", LAKH_DIR)]:
        if folder.exists():
            n = _count_midi(folder)
            size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
            print(f"  {name:<12}: {n:>6} MIDI files  |  {_sizeof_fmt(size)}")
        else:
            print(f"  {name:<12}: not downloaded")

    print("=" * 55)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MIDI datasets for the Music Generation project."
    )
    parser.add_argument("--maestro", action="store_true", help="Download MAESTRO only")
    parser.add_argument("--lakh",    action="store_true", help="Download Lakh MIDI only")
    parser.add_argument("--all",     action="store_true", help="Download all datasets (default)")
    args = parser.parse_args()

    if not any([args.maestro, args.lakh, args.all]):
        args.all = True

    if args.maestro or args.all:
        download_maestro()

    if args.lakh or args.all:
        download_lakh()

    summarise()


if __name__ == "__main__":
    main()
