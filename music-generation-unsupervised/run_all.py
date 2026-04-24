"""
run_all.py
===========
Master pipeline script — runs the entire project end-to-end:

  Step 1  : Verify dependencies
  Step 2  : Download datasets (skipped if already downloaded)
  Step 3  : Preprocess MAESTRO 
  Step 4  : Run baselines
  Step 5  : Train LSTM Autoencoder
  Step 6  : Train Music VAE
  Step 7  : Train Transformer
  Step 8  : Evaluate all models
  Step 9  : Generate final sample MIDI files

Each step is skipped if its output already exists (idempotent pipeline).

Usage
-----
    python run_all.py [--skip_download] [--epochs_ae 50] [--epochs_vae 60]
                      [--epochs_tf 60] [--max_files 500]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

os.environ["PYTHONUTF8"] = "1"

def header(msg: str) -> None:
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}{RESET}")


def run(cmd: list[str], check: bool = True) -> int:
    """Run a subprocess command, stream output, return exit code."""
    print(f"  {YELLOW}$ {' '.join(cmd)}{RESET}")
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=None, stderr=None,   # inherit terminal
    )
    if check and result.returncode != 0:
        print(f"  {RED}[ERROR] Command failed with exit code {result.returncode}{RESET}")
        sys.exit(result.returncode)
    return result.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full Music Generation pipeline")
    p.add_argument("--skip_download",  action="store_true",
                   help="Skip dataset download (data already present)")
    p.add_argument("--max_files",      type=int, default=None,
                   help="Limit MIDI files for quick testing (e.g. --max_files 200)")
    p.add_argument("--window",         type=int, default=64,
                   help="Segment window size (timesteps)")
    p.add_argument("--epochs_ae",      type=int, default=50)
    p.add_argument("--epochs_vae",     type=int, default=60)
    p.add_argument("--epochs_tf",      type=int, default=60)
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--beta",           type=float, default=1.0,
                   help="KL weight β for the VAE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py   = [sys.executable]

    # ── Step 1: Verify setup ────────────────────────────────────────────────
    header("Step 1 — Verify Environment")
    run(py + ["verify_setup.py"])

    # ── Step 2: Download datasets ───────────────────────────────────────────
    if not args.skip_download:
        header("Step 2 — Download Datasets")
        run(py + ["download_dataset.py", "--all"])
    else:
        header("Step 2 — Dataset Download (SKIPPED)")

    # ── Step 3: Preprocess ─────────────────────────────────────────────────
    header("Step 3 — Preprocess MAESTRO")
    maestro_out = "data/processed/maestro"
    mf_arg = (["--max_files", str(args.max_files)] if args.max_files else [])
    if not (PROJECT_ROOT / maestro_out / "roll_train.pt").exists():
        run(py + ["-m", "src.preprocessing.preprocess",
                  "--midi_dir", "data/raw/maestro",
                  "--out_dir",  maestro_out,
                  "--window",   str(args.window),
                  ] + mf_arg)
    else:
        print(f"  {GREEN}[skip] Preprocessed MAESTRO data already exists.{RESET}")

    # ── Step 4: Baselines ──────────────────────────────────────────────────
    header("Step 4 — Run Baselines")
    run(py + ["-m", "src.models.baselines",
              "--data_dir", maestro_out,
              "--out_dir",  "outputs/baselines",
              "--num_samples", "5"])

    # ── Step 5: Train AE ───────────────────────────────────────────────────
    header("Step 5 — Train LSTM Autoencoder")
    ae_ckpt = PROJECT_ROOT / "outputs" / "models" / "ae_checkpoint.pt"
    if not ae_ckpt.exists():
        run(py + ["-m", "src.training.train_ae",
                  "--data_dir",   maestro_out,
                  "--epochs",     str(args.epochs_ae),
                  "--batch_size", str(args.batch_size)])
    else:
        print(f"  {GREEN}[skip] AE checkpoint already exists.{RESET}")

    # ── Step 6: Train VAE ──────────────────────────────────────────────────
    header("Step 6 — Train Music VAE")
    vae_ckpt = PROJECT_ROOT / "outputs" / "models" / "vae_checkpoint.pt"
    if not vae_ckpt.exists():
        run(py + ["-m", "src.training.train_vae",
                  "--data_dir",   maestro_out,
                  "--epochs",     str(args.epochs_vae),
                  "--batch_size", str(args.batch_size),
                  "--beta",       str(args.beta)])
    else:
        print(f"  {GREEN}[skip] VAE checkpoint already exists.{RESET}")

    # ── Step 7: Train Transformer ──────────────────────────────────────────
    header("Step 7 — Train Autoregressive Transformer")
    tf_ckpt = PROJECT_ROOT / "outputs" / "models" / "transformer_ckpt.pt"
    if not tf_ckpt.exists():
        run(py + ["-m", "src.training.train_transformer",
                  "--data_dir",   maestro_out,
                  "--epochs",     str(args.epochs_tf),
                  "--batch_size", str(args.batch_size)])
    else:
        print(f"  {GREEN}[skip] Transformer checkpoint already exists.{RESET}")

    # ── Step 8: Evaluate all ────────────────────────────────────────────────
    header("Step 8 — Evaluate All Models")
    run(py + ["-m", "src.evaluation.evaluate_all",
              "--data_dir", maestro_out,
              "--seq_len",  str(args.window)])

    # ── Step 9: Generate extras ────────────────────────────────────────────
    header("Step 9 — Generate Extra Samples (VAE prior)")
    run(py + ["-m", "src.generation.generate",
              "--model",   "vae",
              "--num",     "5",
              "--data_dir", maestro_out,
              "--out_dir",  "outputs/midi/generated_extra"])

    print(f"\n{BOLD}{GREEN}{'=' * 60}")
    print("  ✓  PIPELINE COMPLETE!")
    print(f"{'=' * 60}{RESET}\n")
    print("  Outputs:")
    print("    outputs/plots/        — loss curves, pitch histograms")
    print("    outputs/midi/         — MIDI files from all models")
    print("    outputs/models/       — trained model checkpoints")
    print("    outputs/evaluation/   — metric comparison data")
    print()


if __name__ == "__main__":
    main()
