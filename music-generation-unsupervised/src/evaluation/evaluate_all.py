"""
src/evaluation/evaluate_all.py
================================
Comprehensive evaluation script that:
1. Loads all trained models.
2. Generates samples from each.
3. Computes quantitative metrics (note density, pitch entropy, etc.).
4. Generates comparison plots (pitch histograms, metric bar charts).
5. Prints a final ranked comparison table.

Usage
-----
    python -m src.evaluation.evaluate_all \\
        --data_dir   data/processed/maestro \\
        --out_dir    outputs \\
        --seq_len    64
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder import LSTMAutoencoder
from src.models.vae import MusicVAE
from src.models.baselines import RandomNoteGenerator, MarkovChainModel
from src.evaluation.metrics import evaluate_batch, compare_models, pitch_class_histogram
from src.preprocessing.midi_utils import save_piano_roll_as_midi


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_roll_data(data_dir: Path) -> torch.Tensor:
    path = data_dir / "roll_test.pt"
    if path.exists():
        return torch.load(path, weights_only=True).float()
    path = data_dir / "roll_train.pt"
    return torch.load(path, weights_only=True).float()


def _load_model_if_exists(ckpt_path: Path, model, device):
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return True
    return False


# ── Sample generators ──────────────────────────────────────────────────────

@torch.no_grad()
def _ae_samples(ckpt_dir: Path, test_tensor: torch.Tensor,
                seq_len: int, device, n: int = 32) -> np.ndarray:
    model = LSTMAutoencoder(seq_len=seq_len).to(device)
    ok = _load_model_if_exists(ckpt_dir / "ae_checkpoint.pt", model, device)
    if not ok:
        return None
    idx = torch.randperm(len(test_tensor))[:n]
    x   = test_tensor[idx].to(device)
    x_hat, _ = model(x)
    return x_hat.cpu().numpy()


@torch.no_grad()
def _vae_samples(ckpt_dir: Path, seq_len: int, device, n: int = 32) -> np.ndarray:
    model = MusicVAE(seq_len=seq_len).to(device)
    ok = _load_model_if_exists(ckpt_dir / "vae_checkpoint.pt", model, device)
    if not ok:
        return None
    rolls = model.sample(n, device=str(device))
    return rolls.cpu().numpy()


def _random_samples(seq_len: int, n: int = 32) -> np.ndarray:
    gen = RandomNoteGenerator(seed=0)
    return np.stack([gen.generate(seq_len) for _ in range(n)])


def _markov_samples(train_tensor: torch.Tensor, seq_len: int, n: int = 32) -> np.ndarray:
    m = MarkovChainModel(order=1, seed=0)
    m.fit(train_tensor[:500])   # limit for speed
    return np.stack([m.generate(seq_len) for _ in range(n)])


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_pitch_histograms(
    all_rolls: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """One pitch-class histogram per model, saved as a grid."""
    n = len(all_rolls)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    pitch_names = ["C", "C#", "D", "D#", "E", "F",
                   "F#", "G", "G#", "A", "A#", "B"]
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    for ax, (name, rolls), color in zip(axes, all_rolls.items(), colors):
        hist = np.zeros(12)
        for r in rolls:
            hist += pitch_class_histogram(r)
        hist /= len(rolls)
        ax.bar(pitch_names, hist, color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Pitch Class")
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("Relative Frequency")
    fig.suptitle("Pitch Class Distribution by Model", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "pitch_histograms.png", dpi=150)
    plt.close(fig)
    print(f"  Pitch histograms → {out_dir / 'pitch_histograms.png'}")


def plot_metric_bars(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    """Bar chart comparing each metric across models."""
    metrics = list(next(iter(results.values())).keys())
    models  = list(results.keys())
    n_m     = len(metrics)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax     = axes[i]
        vals   = [results[m].get(metric, 0) for m in models]
        colors = plt.cm.Paired(np.linspace(0, 1, len(models)))
        ax.bar(models, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for j in range(n_m, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Quantitative Evaluation — All Models", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "metric_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Metric bars → {out_dir / 'metric_comparison.png'}")


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/processed/maestro")
    p.add_argument("--out_dir",  type=str, default="outputs")
    p.add_argument("--seq_len",  type=int, default=64)
    p.add_argument("--n_eval",   type=int, default=32,
                   help="Number of samples to evaluate per model.")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = PROJECT_ROOT / args.data_dir
    out_root = PROJECT_ROOT / args.out_dir
    ckpt_dir = out_root / "models"
    eval_dir = out_root / "evaluation"
    plots_dir = out_root / "plots"

    test_tensor  = _load_roll_data(data_dir)
    train_tensor = torch.load(data_dir / "roll_train.pt", weights_only=True).float()
    seq_len      = test_tensor.shape[1]

    print(f"\n  Evaluating on {args.n_eval} samples per model …")

    # ── Collect samples from each model ────────────────────────────────────
    all_rolls: dict[str, np.ndarray] = {}
    results:   dict[str, dict]       = {}

    # Random baseline
    r = _random_samples(seq_len, args.n_eval)
    all_rolls["Random"] = r
    results["Random"]   = evaluate_batch(r)
    print("  [✓] Random baseline evaluated")

    # Markov baseline
    m = _markov_samples(train_tensor, seq_len, args.n_eval)
    all_rolls["Markov"] = m
    results["Markov"]   = evaluate_batch(m)
    print("  [✓] Markov baseline evaluated")

    # AE
    ae_r = _ae_samples(ckpt_dir, test_tensor, seq_len, device, args.n_eval)
    if ae_r is not None:
        all_rolls["LSTM-AE"] = ae_r
        results["LSTM-AE"]   = evaluate_batch(ae_r)
        print("  [✓] LSTM Autoencoder evaluated")
    else:
        print("  [!] LSTM-AE checkpoint not found — skipping")

    # VAE
    vae_r = _vae_samples(ckpt_dir, seq_len, device, args.n_eval)
    if vae_r is not None:
        all_rolls["VAE"] = vae_r
        results["VAE"]   = evaluate_batch(vae_r)
        print("  [✓] Music VAE evaluated")
    else:
        print("  [!] VAE checkpoint not found — skipping")

    # Ground truth (real data)
    idx = torch.randperm(len(test_tensor))[:args.n_eval]
    gt  = test_tensor[idx].numpy()
    all_rolls["Ground Truth"] = gt
    results["Ground Truth"]   = evaluate_batch(gt)
    print("  [✓] Ground truth evaluated")

    # ── Print table ────────────────────────────────────────────────────────
    compare_models(results)

    # ── Plots ──────────────────────────────────────────────────────────────
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_pitch_histograms(all_rolls, plots_dir)
    plot_metric_bars(results, plots_dir)

    print(f"\n  ✓ Evaluation complete. Plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
