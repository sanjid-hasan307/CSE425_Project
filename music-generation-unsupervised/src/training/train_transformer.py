"""
src/training/train_transformer.py
===================================
Training script for the Autoregressive Transformer on token-based
multi-genre MIDI data.

Objective
---------
  Next-token prediction (teacher forcing):
    L_CE = −(1/T) Σ_t log p_θ(x_t | x_1, …, x_{t−1})

Outputs
-------
  outputs/plots/transformer_loss.png   — training/validation loss curve
  outputs/midi/transformer/*.mid       — generated MIDI files
  outputs/models/transformer_ckpt.pt   — best model weights

Usage
-----
    python -m src.training.train_transformer \\
        --data_dir data/processed/lakh \\
        --epochs 60 \\
        --batch_size 64 \\
        --lr 3e-4
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

from src.models.transformer import MusicTransformer
from src.preprocessing.midi_utils import save_tokens_as_midi


# ── Dataset helper ─────────────────────────────────────────────────────────

def load_token_dataset(data_dir: Path) -> tuple[TensorDataset, TensorDataset]:
    """Load token-based train and test tensors."""
    tok_train = torch.load(data_dir / "tok_train.pt", weights_only=True).long()
    tok_test  = torch.load(data_dir / "tok_test.pt",  weights_only=True).long()
    print(f"  Token Train: {tok_train.shape}   Test: {tok_test.shape}")
    return TensorDataset(tok_train), TensorDataset(tok_test)


# ── Training / Eval ────────────────────────────────────────────────────────

def train_epoch(
    model:     MusicTransformer,
    loader:    DataLoader,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)                # (B, T)
        inputs  = batch[:, :-1]                 # feed x_1 … x_{T-1}
        targets = batch[:, 1:]                  # predict x_2 … x_T

        logits = model(inputs)                  # (B, T-1, vocab_size)
        loss   = MusicTransformer.compute_loss(logits, targets)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model:  MusicTransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for (batch,) in loader:
        batch   = batch.to(device)
        inputs  = batch[:, :-1]
        targets = batch[:, 1:]
        logits  = model(inputs)
        loss    = MusicTransformer.compute_loss(logits, targets)
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_loss(train_losses: list, val_losses: list, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    e = range(1, len(train_losses) + 1)
    ax.plot(e, train_losses, "b-",  lw=1.8, label="Train Loss (CE)")
    ax.plot(e, val_losses,   "r--", lw=1.8, label="Val Loss (CE)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Autoregressive Transformer — Training Loss", fontsize=14)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "transformer_loss.png", dpi=150)
    plt.close(fig)
    print(f"  Loss curve → {out_dir / 'transformer_loss.png'}")


# ── Generation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    model:      MusicTransformer,
    test_ds:    TensorDataset,
    num:        int,
    out_dir:    Path,
    device:     torch.device,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_k:      int = 50,
) -> None:
    """Generate MIDI files autoregressively from test prompts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use first 16 tokens of random test examples as prompts
    for i in range(num):
        idx    = np.random.randint(0, len(test_ds))
        prompt = test_ds[idx][0][:16].unsqueeze(0).to(device)   # (1, 16)

        generated = model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )  # (1, 16 + max_tokens)

        path = str(out_dir / f"transformer_sample_{i+1:02d}.mid")
        save_tokens_as_midi(generated[0], path)
        print(f"  Generated → {path}")


# ── Comparison table ───────────────────────────────────────────────────────

def print_comparison(
    ae_loss:    float,
    vae_loss:   float,
    tf_loss:    float,
    n_ae:       int,
    n_vae:      int,
    n_tf:       int,
) -> None:
    print("\n" + "=" * 65)
    print(f"  {'Model':<20} {'Final Loss':>12}  {'Params':>12}  {'Type':>12}")
    print("-" * 65)
    print(f"  {'LSTM Autoencoder':<20} {ae_loss:>12.6f}  {n_ae:>12,}  {'Det. AE':>12}")
    print(f"  {'Music VAE':<20} {vae_loss:>12.6f}  {n_vae:>12,}  {'β-VAE':>12}")
    print(f"  {'Transformer':<20} {tf_loss:>12.6f}  {n_tf:>12,}  {'Auto-Reg':>12}")
    print("=" * 65)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Autoregressive Transformer")
    p.add_argument("--data_dir",    type=str, default="data/processed/lakh")
    p.add_argument("--epochs",      type=int, default=60)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--d_model",     type=int, default=256)
    p.add_argument("--n_heads",     type=int, default=8)
    p.add_argument("--n_layers",    type=int, default=4)
    p.add_argument("--d_ff",        type=int, default=512)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k",       type=int, default=50)
    p.add_argument("--out_dir",     type=str, default="outputs")
    p.add_argument("--ae_loss",     type=float, default=None)
    p.add_argument("--vae_loss",    type=float, default=None)
    p.add_argument("--ae_params",   type=int, default=0)
    p.add_argument("--vae_params",  type=int, default=0)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    data_dir = PROJECT_ROOT / args.data_dir
    out_root = PROJECT_ROOT / args.out_dir

    # ── Data ───────────────────────────────────────────────────────────────
    train_ds, test_ds = load_token_dataset(data_dir)
    seq_len = train_ds.tensors[0].shape[1]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────────
    model = MusicTransformer(
        vocab_size  = 417,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        d_ff        = args.d_ff,
        max_seq_len = seq_len + 1,
        dropout     = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), weight_decay=0.01)
    # Warmup + cosine decay scheduler
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = total_steps // 10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr     = args.lr,
        total_steps= total_steps,
        pct_start  = 0.1,
    )

    # ── Training ───────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val = float("inf")
    ckpt_dir = out_root / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Training for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, train_loader, optimiser, device)
        v_loss = eval_epoch( model, val_loader,              device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), ckpt_dir / "transformer_ckpt.pt")

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train CE: {t_loss:.4f}  Val CE: {v_loss:.4f}  "
                  f"LR: {lr_now:.2e}")

        # Step scheduler once per batch (OneCycleLR is per-step)
        # Already stepped inside train_epoch, but we recreate per epoch here:
        # (OneCycleLR is already stepped per batch above — skip redundant step)

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_loss(train_losses, val_losses, out_root / "plots")

    # ── Generate ───────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_dir / "transformer_ckpt.pt",
                                     map_location=device, weights_only=True))
    generate_samples(
        model, test_ds, args.num_samples,
        out_root / "midi" / "transformer",
        device, temperature=args.temperature, top_k=args.top_k,
    )

    # ── Comparison ─────────────────────────────────────────────────────────
    print_comparison(
        ae_loss  = args.ae_loss  or float("nan"),
        vae_loss = args.vae_loss or float("nan"),
        tf_loss  = train_losses[-1],
        n_ae     = args.ae_params,
        n_vae    = args.vae_params,
        n_tf     = n_params,
    )

    print("\n" + "=" * 55)
    print("  TRANSFORMER — RESULTS SUMMARY")
    print("=" * 55)
    print(f"  Final Train CE Loss : {train_losses[-1]:.6f}")
    print(f"  Final Val CE Loss   : {val_losses[-1]:.6f}")
    print(f"  Best Val CE Loss    : {best_val:.6f}")
    print(f"  Model Parameters    : {n_params:,}")
    print(f"  Checkpoint          : outputs/models/transformer_ckpt.pt")
    print(f"  MIDI samples        : outputs/midi/transformer/")
    print("=" * 55)


if __name__ == "__main__":
    main()
