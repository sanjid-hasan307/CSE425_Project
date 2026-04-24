"""
src/generation/generate.py
============================
Unified generation script — load any trained model and produce MIDI files.

Supported models
----------------
  --model ae          : LSTM Autoencoder (reconstruct test samples)
  --model vae         : Music VAE       (sample from prior)
  --model transformer : Autoregressive Transformer (autoregressive decode)
  --model random      : Random baseline
  --model markov      : Markov Chain baseline

Usage
-----
    python -m src.generation.generate \\
        --model vae \\
        --num 10 \\
        --out_dir outputs/midi/custom \\
        --data_dir data/processed/lakh
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder import LSTMAutoencoder
from src.models.vae import MusicVAE
from src.models.transformer import MusicTransformer
from src.models.baselines import RandomNoteGenerator, MarkovChainModel
from src.preprocessing.midi_utils import save_piano_roll_as_midi, save_tokens_as_midi


# ── Loaders ────────────────────────────────────────────────────────────────

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_roll_data(data_dir: Path, split: str = "test") -> torch.Tensor:
    path = data_dir / f"roll_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Run preprocess.py first.")
    return torch.load(path, weights_only=True).float()


def _load_token_data(data_dir: Path, split: str = "test") -> torch.Tensor:
    path = data_dir / f"tok_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Run preprocess.py first.")
    return torch.load(path, weights_only=True).long()


# ── Generators ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_ae(
    num: int, data_dir: Path, ckpt_dir: Path, out_dir: Path, device
) -> None:
    roll_test = _load_roll_data(data_dir)
    seq_len   = roll_test.shape[1]

    model = LSTMAutoencoder(seq_len=seq_len).to(device)
    ckpt  = ckpt_dir / "ae_checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"AE checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = torch.randperm(len(roll_test))[:num]
    
    from src.preprocessing.midi_utils import piano_roll_to_midi
    
    for i, j in enumerate(idx):
        x     = roll_test[j].unsqueeze(0).to(device)
        x_hat, _ = model(x)
        x_hat = torch.sigmoid(x_hat)
        
        piano_roll = x_hat.squeeze().cpu().numpy()
        print(f"\n--- AE Sample {i+1} Output Debug ---")
        print("Piano roll shape:", piano_roll.shape)
        print("Max value:", piano_roll.max())
        print("Non-zero count:", (piano_roll > 0.1).sum())

        path  = str(out_dir / f"ae_{i+1:02d}.mid")
        midi = piano_roll_to_midi(piano_roll, fs=16, tempo=120)
        print("Notes generated:", sum(len(instr.notes) for instr in midi.instruments))
        midi.write(path)
        print(f"  [AE] → {path}")


@torch.no_grad()
def generate_vae(
    num: int, data_dir: Path, ckpt_dir: Path, out_dir: Path, device
) -> None:
    roll_test = _load_roll_data(data_dir)
    seq_len   = roll_test.shape[1]

    model = MusicVAE(seq_len=seq_len).to(device)
    ckpt  = ckpt_dir / "vae_checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    from src.preprocessing.midi_utils import piano_roll_to_midi
    
    for i in range(num):
        z = torch.randn(1, model.latent_dim).to(device)
        output = model.decode(z)
        output = torch.sigmoid(output)
        
        print(f"\n--- VAE Sample {i+1} Output Debug ---")
        print("Output shape:", output.shape)
        print("Max value:", output.max().item())
        print("Min value:", output.min().item())
        print("Non-zero (>0.1):", (output > 0.1).sum().item())
        
        if output.max().item() < 0.1:
            print("Model output max is below 0.1, forcing higher noise scale for test...")
            z = torch.randn(1, model.latent_dim).to(device) * 3.0
            output = model.decode(z)
            output = torch.sigmoid(output)
            print("New Max value:", output.max().item())
            print("New Non-zero (>0.1):", (output > 0.1).sum().item())

        piano_roll = output.squeeze().cpu().numpy()

        path = str(out_dir / f"vae_{i+1:02d}.mid")
        midi = piano_roll_to_midi(piano_roll, fs=16, tempo=120)
        
        import os
        notes_gen = sum(len(instr.notes) for instr in midi.instruments)
        print("Notes generated:", notes_gen)
        midi.write(path)
        print("MIDI file size (bytes):", os.path.getsize(path))
        print(f"  [VAE] -> {path}")


@torch.no_grad()
def generate_transformer(
    num: int, data_dir: Path, ckpt_dir: Path, out_dir: Path,
    device, temperature: float = 1.0, top_k: int = 50,
    max_new_tokens: int = 256,
) -> None:
    tok_test = _load_token_data(data_dir)

    model = MusicTransformer(vocab_size=417).to(device)
    ckpt  = ckpt_dir / "transformer_ckpt.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num):
        idx    = np.random.randint(0, len(tok_test))
        prompt = tok_test[idx, :16].unsqueeze(0).to(device)
        out    = model.generate(prompt, max_new_tokens, temperature, top_k)
        path   = str(out_dir / f"transformer_{i+1:02d}.mid")
        save_tokens_as_midi(out[0], path)
        print(f"  [Transformer] → {path}")


def generate_random(num: int, seq_len: int, out_dir: Path) -> None:
    gen = RandomNoteGenerator(seed=42)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num):
        path = str(out_dir / f"random_{i+1:02d}.mid")
        gen.generate_midi(path, num_steps=seq_len)


def generate_markov(
    num: int, data_dir: Path, out_dir: Path
) -> None:
    roll_train = torch.load(data_dir / "roll_train.pt", weights_only=True).float()
    seq_len    = roll_train.shape[1]
    m = MarkovChainModel(order=1, seed=42)
    m.fit(roll_train[:1000])
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num):
        path = str(out_dir / f"markov_{i+1:02d}.mid")
        m.generate_midi(path, num_steps=seq_len)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified music generation script")
    p.add_argument("--model",       type=str, required=True,
                   choices=["ae", "vae", "transformer", "random", "markov"])
    p.add_argument("--num",         type=int, default=5)
    p.add_argument("--data_dir",    type=str, default="data/processed/maestro")
    p.add_argument("--out_dir",     type=str, default="outputs/midi/generated")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k",       type=int, default=50)
    p.add_argument("--max_tokens",  type=int, default=256)
    p.add_argument("--seq_len",     type=int, default=64)
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    device   = _device()
    data_dir = PROJECT_ROOT / args.data_dir
    out_dir  = PROJECT_ROOT / args.out_dir
    ckpt_dir = PROJECT_ROOT / "outputs" / "models"

    print(f"\n  Model  : {args.model}")
    print(f"  Device : {device}")
    print(f"  Num    : {args.num}")
    print(f"  OutDir : {out_dir}\n")

    if args.model == "ae":
        generate_ae(args.num, data_dir, ckpt_dir, out_dir, device)
    elif args.model == "vae":
        generate_vae(args.num, data_dir, ckpt_dir, out_dir, device)
    elif args.model == "transformer":
        generate_transformer(
            args.num, data_dir, ckpt_dir, out_dir, device,
            temperature=args.temperature, top_k=args.top_k,
            max_new_tokens=args.max_tokens,
        )
    elif args.model == "random":
        generate_random(args.num, args.seq_len, out_dir)
    elif args.model == "markov":
        generate_markov(args.num, data_dir, out_dir)

    print(f"\n  OK Done - {args.num} MIDI files saved to {out_dir}/")


if __name__ == "__main__":
    main()
