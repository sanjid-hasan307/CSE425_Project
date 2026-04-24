"""
verify_setup.py
================
Verifies that all required packages are installed and functional.
Run this script before starting any training or preprocessing.

Usage:
    python verify_setup.py
"""

import sys
import importlib

# ── Colour helpers ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

OK_MARK   = "[OK]"
FAIL_MARK = "[FAIL]"

REQUIRED_PACKAGES = [
    ("torch",        "PyTorch"),
    ("numpy",        "NumPy"),
    ("pandas",       "Pandas"),
    ("matplotlib",   "Matplotlib"),
    ("tqdm",         "TQDM"),
    ("pretty_midi",  "PrettyMIDI"),
    ("miditoolkit",  "MIDIToolkit"),
    ("music21",      "Music21"),
    ("scipy",        "SciPy"),
    ("sklearn",      "Scikit-Learn"),
    ("seaborn",      "Seaborn"),
]


def check_package(import_name: str, friendly_name: str) -> bool:
    """Try to import a package and return True on success."""
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {GREEN}{OK_MARK}{RESET} {friendly_name:<20} (version: {version})")
        return True
    except ImportError as exc:
        print(f"  {RED}{FAIL_MARK}{RESET} {friendly_name:<20} - MISSING  [{exc}]")
        return False


def check_torch_extras() -> None:
    """Report CUDA availability and device information."""
    import torch
    print(f"\n  {'PyTorch device info':}")
    print(f"    CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA version   : {torch.version.cuda}")
        print(f"    GPU name       : {torch.cuda.get_device_name(0)}")
    else:
        print(f"    {YELLOW}Running on CPU - training will be slower.{RESET}")


def check_pretty_midi_functional() -> None:
    """Quick functional test of pretty_midi."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(program=0)
    note  = pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5)
    instr.notes.append(note)
    pm.instruments.append(instr)
    print(f"\n  {GREEN}{OK_MARK}{RESET} pretty_midi functional test PASSED")


def main() -> None:
    print("=" * 55)
    print("   Music Generation Project — Environment Verification")
    print("=" * 55)
    print(f"\n  Python  : {sys.version}")
    print()

    failures: list[str] = []
    for import_name, friendly in REQUIRED_PACKAGES:
        ok = check_package(import_name, friendly)
        if not ok:
            failures.append(friendly)

    # Extended checks only if core packages are present
    if "PyTorch" not in failures:
        check_torch_extras()

    if "PrettyMIDI" not in failures:
        check_pretty_midi_functional()

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    if failures:
        print(f"  {RED}FAILED - Missing packages: {', '.join(failures)}{RESET}")
        print(f"\n  Fix with:")
        print(f"    pip install -r requirements.txt")
        sys.exit(1)
    else:
        print(f"  {GREEN}ALL CHECKS PASSED - Environment is ready!{RESET}")
    print("=" * 55)


if __name__ == "__main__":
    main()
