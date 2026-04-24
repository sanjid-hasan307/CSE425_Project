"""
src/training/train_vae.py
==========================
Training script for the β-VAE on Lakh MIDI multi-genre piano-roll data.

Outputs
-------
  outputs/plots/vae_recon_loss.png     — reconstruction loss curve
  outputs/plots/vae_kl_loss.png        — KL divergence curve
  outputs/plots/vae_total_loss.png     — total ELBO loss curve
  outputs/midi/vae/vae_sample_*.mid    — 8 sampled MIDI files
  outputs/midi/vae/interpolation_*.mid — latent interpolation sequence
  outputs/models/vae_checkpoint.pt     — best model weights

Usage
-----
    python -m src.training.train_vae \\
        --data_dir data/processed/lakh \\
        --epochs   60 \\
        --beta     1.0
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vae import MusicVAE
from src.preprocessing.midi_utils import save_piano_roll_as_midi


# ── Dataset helper ─────────────────────────────────────────────────────────

def load_dataset(data_dir: Path) -> tuple[TensorDataset, TensorDataset]:
    roll_train = torch.load(data_dir / "roll_train.pt", weights_only=True).float()
    roll_test  = torch.load(data_dir / "roll_test.pt",  weights_only=True).float()
    print(f"  Train: {roll_train.shape}   Test: {roll_test.shape}")
    return TensorDataset(roll_train), TensorDataset(roll_test)


# ── One epoch ─────────────────────────────────────────────────────────────

def run_epoch(
    model:     MusicVAE,
    loader:    DataLoader,
    optimiser: torch.optim.Optimizer | None,
    device:    torch.device,
    train:     bool = True,
) -> tuple[float, float, float]:
    """Run one epoch. Returns (avg_total, avg_recon, avg_kl)."""
    model.train(train)
    tot_total = tot_recon = tot_kl = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, mu, logvar = model(batch)
            total, recon, kl  = model.loss(batch, x_hat, mu, logvar)

            if train and optimiser is not None:
                optimiser.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            tot_total += total.item()
            tot_recon += recon.item()
            tot_kl    += kl.item()

    n = len(loader)
    return tot_total / n, tot_recon / n, tot_kl / n


# ── Plotting ───────────────────────────────────────────────────────────────

def _plot(values: list, label: str, filename: str, out_dir: Path, color: str = "steelblue") -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(values) + 1), values, color=color, lw=1.8, label=f"Train {label}")
    ax.set_xlabel("Epoch"); ax.set_ylabel(label)
    ax.set_title(f"Music VAE — {label}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {out_dir / filename}")


def plot_combined(
    train_recon: list, train_kl: list, train_total: list,
    val_recon:   list, val_kl:   list, val_total:   list,
    out_dir: Path,
) -> None:
    """Three-panel figure: total, recon, KL."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    epochs = range(1, len(train_total) + 1)

    for ax, (tr, va, title, col) in zip(
        axes,
        [
            (train_total, val_total, "Total Loss (ELBO)",   "royalblue"),
            (train_recon, val_recon, "Reconstruction (BCE)", "darkorange"),
            (train_kl,   val_kl,   "KL Divergence",         "seagreen"),
        ],
    ):
        ax.plot(epochs, tr, "-",  label="Train", color=col,   lw=1.8)
        ax.plot(epochs, va, "--", label="Val",   color=col,   lw=1.8, alpha=0.7)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Music VAE Training", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "vae_combined_loss.png", dpi=150)
    plt.close(fig)
    print(f"  Combined loss plot → {out_dir / 'vae_combined_loss.png'}")


# ── Generation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_and_save(
    model:       MusicVAE,
    num_samples: int,
    out_dir:     Path,
    device:      torch.device,
) -> None:
    """Sample from the prior and save as MIDI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rolls = model.sample(num_samples, device=str(device))
    for i, roll in enumerate(rolls):
        path = str(out_dir / f"vae_sample_{i+1:02d}.mid")
        save_piano_roll_as_midi(roll.cpu().numpy(), path)
        print(f"  Generated → {path}")


@torch.no_grad()
def interpolate_and_save(
    model:    MusicVAE,
    test_ds:  TensorDataset,
    out_dir:  Path,
    device:   torch.device,
    steps:    int = 8,
) -> None:
    """Interpolate between two random test examples and save MIDI."""
    out_dir.mkdir(parents=True, exist_ok=True)

    x1 = test_ds[0][0].unsqueeze(0).to(device)
    x2 = test_ds[1][0].unsqueeze(0).to(device)

    interp_rolls = model.interpolate(x1, x2, steps=steps)  # (steps, T, 128)
    for i, roll in enumerate(interp_rolls):
        path = str(out_dir / f"interpolation_{i+1:02d}.mid")
        save_piano_roll_as_midi(roll.cpu().numpy(), path)
    print(f"  Interpolation saved ({steps} steps) → {out_dir}/")


# ── Quantitative comparison ────────────────────────────────────────────────

def print_comparison_table(
    ae_final_loss:  float,
    vae_final_recon: float,
    vae_final_kl:   float,
    vae_final_total: float,
) -> None:
    """Print a comparison table of AE vs VAE metrics."""
    print("\n" + "=" * 60)
    print(f"  {'Metric':<35} {'AE':>8}  {'VAE':>8}")
    print("-" * 60)

    ae_label  = f"{ae_final_loss:.6f}"
    vae_label = f"{vae_final_recon:.6f}"

    print(f"  {'Reconstruction Loss (final)':<35} {ae_label:>8}  {vae_label:>8}")
    print(f"  {'KL Divergence (final)':<35} {'N/A':>8}  {vae_final_kl:>8.4f}")
    print(f"  {'Total Loss (final)':<35} {ae_label:>8}  {vae_final_total:>8.4f}")
    print(f"  {'Model type':<35} {'Det.':>8}  {'Prob.':>8}")
    print(f"  {'Latent space':<35} {'Fixed':>8}  {'Reg.':>8}")
    print(f"  {'Can sample novel sequences':<35} {'No':>8}  {'Yes':>8}")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Music VAE")
    p.add_argument("--data_dir",    type=str, default="data/processed/lakh")
    # PROBLEM 5 FIX: Minimum recommended training config applied
    p.add_argument("--epochs",      type=int, default=60)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden_dim",  type=int, default=256)
    p.add_argument("--latent_dim",  type=int, default=64)
    p.add_argument("--num_layers",  type=int, default=2)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--beta",        type=float, default=1.0,
                   help="KL weight (β=1 → standard VAE, β>1 → β-VAE)")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--out_dir",     type=str, default="outputs")
    p.add_argument("--ae_loss",     type=float, default=None,
                   help="Final AE loss for comparison table (optional).")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}  |  β = {args.beta}")

    data_dir = PROJECT_ROOT / args.data_dir
    out_root = PROJECT_ROOT / args.out_dir

    # ── Data ───────────────────────────────────────────────────────────────
    train_ds, test_ds = load_dataset(data_dir)
    seq_len = train_ds.tensors[0].shape[1]
    print(f"  Sequence length : {seq_len}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────────
    model = MusicVAE(
        input_dim  = 128,
        hidden_dim = args.hidden_dim,
        latent_dim = args.latent_dim,
        num_layers = args.num_layers,
        dropout    = args.dropout,
        seq_len    = seq_len,
        beta       = args.beta,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5, min_lr=1e-5
    )

    # ── Training ───────────────────────────────────────────────────────────
    tr_tot, tr_rec, tr_kl = [], [], []
    va_tot, va_rec, va_kl = [], [], []
    best_val = float("inf")
    ckpt_dir = out_root / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Training for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        t_tot, t_rec, t_kl = run_epoch(model, train_loader, optimiser, device, train=True)
        v_tot, v_rec, v_kl = run_epoch(model, val_loader,   None,       device, train=False)

        tr_tot.append(t_tot); tr_rec.append(t_rec); tr_kl.append(t_kl)
        va_tot.append(v_tot); va_rec.append(v_rec); va_kl.append(v_kl)

        scheduler.step(v_tot)
        if v_tot < best_val:
            best_val = v_tot
            torch.save(model.state_dict(), ckpt_dir / "vae_checkpoint.pt")

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Total: {t_tot:.4f}  Recon: {t_rec:.4f}  KL: {t_kl:.4f}  "
                  f"VTotal: {v_tot:.4f}  LR: {lr_now:.2e}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plots_dir = out_root / "plots"
    plot_combined(tr_rec, tr_kl, tr_tot, va_rec, va_kl, va_tot, plots_dir)

    # ── Load best and generate ─────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_dir / "vae_checkpoint.pt",
                                     map_location=device, weights_only=True))
    sample_and_save(model, args.num_samples, out_root / "midi" / "vae", device)
    interpolate_and_save(model, test_ds, out_root / "midi" / "vae" / "interpolation",
                         device, steps=8)

    # ── Summary ────────────────────────────────────────────────────────────
    ae_loss = args.ae_loss if args.ae_loss is not None else float("nan")
    print_comparison_table(ae_loss, tr_rec[-1], tr_kl[-1], tr_tot[-1])

    print("\n" + "=" * 55)
    print("  MUSIC VAE — RESULTS SUMMARY")
    print("=" * 55)
    print(f"  Final Train Total Loss : {tr_tot[-1]:.6f}")
    print(f"  Final Train Recon Loss : {tr_rec[-1]:.6f}")
    print(f"  Final Train KL         : {tr_kl[-1]:.6f}")
    print(f"  Best Val Total Loss    : {best_val:.6f}")
    print(f"  β (KL weight)          : {args.beta}")
    print(f"  Parameters             : {n_params:,}")
    print(f"  Checkpoint             : outputs/models/vae_checkpoint.pt")
    print(f"  MIDI samples           : outputs/midi/vae/")
    print("=" * 55)


if __name__ == "__main__":
    main()
