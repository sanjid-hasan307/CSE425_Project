"""
src/training/train_ae.py
=========================
Training script for the LSTM Autoencoder on MAESTRO piano-roll data.

Outputs
-------
  outputs/plots/ae_loss_curve.png  — training & validation loss curve
  outputs/midi/ae/ae_sample_*.mid  — 5 generated MIDI files
  outputs/models/ae_checkpoint.pt  — best model weights

Usage
-----
    python -m src.training.train_ae \\
        --data_dir   data/processed/maestro \\
        --epochs     50 \
        --batch_size 32 \
        --lr         1e-3 \\
        --latent_dim 64
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── Resolve project root ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder import LSTMAutoencoder
from src.preprocessing.midi_utils import save_piano_roll_as_midi


# ── Dataset helpers ────────────────────────────────────────────────────────

def load_piano_roll_dataset(data_dir: Path) -> tuple[TensorDataset, TensorDataset]:
    """Load train and test piano-roll tensors."""
    roll_train = torch.load(data_dir / "roll_train.pt", weights_only=True).float()
    roll_test  = torch.load(data_dir / "roll_test.pt",  weights_only=True).float()
    print(f"  Train: {roll_train.shape}   Test: {roll_test.shape}")
    return TensorDataset(roll_train), TensorDataset(roll_test)


# ── Training Loop ──────────────────────────────────────────────────────────

def train_epoch(
    model:      LSTMAutoencoder,
    loader:     DataLoader,
    optimiser:  torch.optim.Optimizer,
    device:     torch.device,
) -> float:
    """Run one full training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        optimiser.zero_grad()
        x_hat, _ = model(batch)
        loss = LSTMAutoencoder.reconstruction_loss(batch, x_hat)
        loss.backward()
        # Gradient clipping stabilises LSTM training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model:   LSTMAutoencoder,
    loader:  DataLoader,
    device:  torch.device,
) -> float:
    """Evaluate the model and return average validation loss."""
    model.eval()
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        x_hat, _ = model(batch)
        loss = LSTMAutoencoder.reconstruction_loss(batch, x_hat)
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Plot ───────────────────────────────────────────────────────────────────

def plot_loss_curve(
    train_losses: list[float],
    val_losses:   list[float],
    save_path:    str,
) -> None:
    """Save a training/validation loss curve to disk."""
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss (MSE)")
    ax.plot(epochs, val_losses,   "r-s", markersize=3, label="Val Loss (MSE)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Reconstruction Loss (MSE)", fontsize=12)
    ax.set_title("LSTM Autoencoder — Reconstruction Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve saved → {save_path}")


# ── Generation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    model:      LSTMAutoencoder,
    test_data:  TensorDataset,
    num_samples: int,
    out_dir:    Path,
    device:     torch.device,
    seq_len:    int,
) -> None:
    """Reconstruct test examples and save as MIDI."""
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick random test examples
    indices = torch.randint(0, len(test_data), (num_samples,))
    for i, idx in enumerate(indices):
        x = test_data[idx.item()][0].unsqueeze(0).to(device)   # (1, T, 128)
        x_hat, z = model(x)

        roll = x_hat.squeeze(0).cpu().numpy()   # (T, 128)
        filepath = str(out_dir / f"ae_sample_{i+1:02d}.mid")
        save_piano_roll_as_midi(roll, filepath)
        print(f"  Generated → {filepath}")


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM Autoencoder on MAESTRO")
    p.add_argument("--data_dir",   type=str, default="data/processed/maestro")
    # PROBLEM 5 FIX: Minimum recommended training config applied
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--num_samples",type=int, default=5)
    p.add_argument("--out_dir",    type=str, default="outputs")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    data_dir  = PROJECT_ROOT / args.data_dir
    out_root  = PROJECT_ROOT / args.out_dir

    # ── Data ───────────────────────────────────────────────────────────────
    train_ds, test_ds = load_piano_roll_dataset(data_dir)
    seq_len = train_ds.tensors[0].shape[1]
    print(f"  Sequence length : {seq_len}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────────
    model = LSTMAutoencoder(
        input_dim  = 128,
        hidden_dim = args.hidden_dim,
        latent_dim = args.latent_dim,
        num_layers = args.num_layers,
        dropout    = args.dropout,
        seq_len    = seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    optimiser  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5, min_lr=1e-5
    )

    # ── Training ───────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    ckpt_dir = out_root / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Training for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, device)
        val_loss   = eval_epoch( model, val_loader,              device)

        train_losses.append(train_loss)
        val_losses.append(  val_loss)
        scheduler.step(val_loss)

        # Checkpoint on improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "ae_checkpoint.pt")

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  LR: {lr_now:.2e}")

    print(f"\n  ✓ Training complete.")
    print(f"  Best validation loss : {best_val_loss:.6f}")

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_loss_curve(
        train_losses, val_losses,
        save_path=str(out_root / "plots" / "ae_loss_curve.png"),
    )

    # ── Generate samples ───────────────────────────────────────────────────
    # Load best checkpoint
    model.load_state_dict(
        torch.load(ckpt_dir / "ae_checkpoint.pt", map_location=device, weights_only=True)
    )
    generate_samples(
        model, test_ds,
        num_samples=args.num_samples,
        out_dir=out_root / "midi" / "ae",
        device=device,
        seq_len=seq_len,
    )

    # ── Final printout ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  LSTM AUTOENCODER — RESULTS SUMMARY")
    print("=" * 55)
    print(f"  Final Train Loss  : {train_losses[-1]:.6f}")
    print(f"  Final Val Loss    : {val_losses[-1]:.6f}")
    print(f"  Best Val Loss     : {best_val_loss:.6f}")
    print(f"  Latent Dim        : {args.latent_dim}")
    print(f"  Parameters        : {n_params:,}")
    print(f"  Checkpoint        : outputs/models/ae_checkpoint.pt")
    print(f"  MIDI samples      : outputs/midi/ae/")
    print("=" * 55)


if __name__ == "__main__":
    main()
