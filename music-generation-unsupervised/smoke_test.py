"""
smoke_test.py
==============
Quick end-to-end smoke test that does NOT require any downloaded MIDI
datasets. It synthesises a small batch of random MIDI files, runs the
full pipeline (preprocess → baselines → AE forward pass → VAE forward
pass → Transformer forward pass), and verifies all outputs exist.

Run this first to confirm the entire codebase works before training.

    python smoke_test.py
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import torch

# ── Setup paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"
PASS  = f"{GREEN}[PASS]{RESET}"
FAIL  = f"{RED}[FAIL]{RESET}"

errors: list[str] = []


def check(condition: bool, msg: str) -> None:
    if condition:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  {msg}")
        errors.append(msg)


# ── Step 1: Synthesise fake MIDI files ────────────────────────────────────

def make_fake_midi(path: Path, n_notes: int = 20, duration: float = 4.0) -> None:
    """Create a random pretty_midi MIDI file for testing."""
    pm   = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instr = pretty_midi.Instrument(program=0)
    rng  = np.random.default_rng(42)
    for _ in range(n_notes):
        start = float(rng.uniform(0, duration - 0.5))
        end   = float(min(start + rng.uniform(0.1, 0.5), duration))
        pitch = int(rng.integers(48, 84))
        vel   = int(rng.integers(60, 100))
        instr.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch,
                                            start=start, end=end))
    pm.instruments.append(instr)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


print("\n" + "=" * 60)
print("  SMOKE TEST — Full Pipeline (Synthetic Data)")
print("=" * 60)

# Create temp directory for fake MIDI data
tmp_midi = PROJECT_ROOT / "data" / "raw" / "_smoke_test_midi"
tmp_midi.mkdir(parents=True, exist_ok=True)

print("\n  [1/6] Generating 20 synthetic MIDI files …")
for i in range(20):
    make_fake_midi(tmp_midi / f"fake_{i:02d}.mid", n_notes=30)
print(f"  Created 20 MIDI files in {tmp_midi}")

# ── Step 2: Preprocessing ─────────────────────────────────────────────────
print("\n  [2/6] Testing preprocessing pipeline …")
from src.preprocessing.preprocess import run_pipeline

tmp_proc = PROJECT_ROOT / "data" / "processed" / "_smoke_test"
run_pipeline(
    midi_dir    = str(tmp_midi),
    out_dir     = str(tmp_proc),
    window      = 32,
    steps_per_bar = 16,
    train_ratio = 0.8,
)
check((tmp_proc / "roll_train.pt").exists(), "roll_train.pt created")
check((tmp_proc / "roll_test.pt").exists(),  "roll_test.pt  created")
check((tmp_proc / "tok_train.pt").exists(),  "tok_train.pt  created")
check((tmp_proc / "tok_test.pt").exists(),   "tok_test.pt   created")
check((tmp_proc / "metadata.json").exists(), "metadata.json created")

roll_train = torch.load(tmp_proc / "roll_train.pt", weights_only=True)
check(roll_train.ndim == 3, f"roll_train shape OK: {roll_train.shape}")
check(roll_train.shape[1] == 32, f"window=32 OK: {roll_train.shape[1]}")
check(roll_train.shape[2] == 128, f"128 pitches OK")

# ── Step 3: MIDI utilities ────────────────────────────────────────────────
print("\n  [3/6] Testing MIDI utilities (piano-roll ↔ MIDI) …")
from src.preprocessing.midi_utils import save_piano_roll_as_midi, save_tokens_as_midi

test_roll = np.random.randint(0, 2, (32, 128)).astype(np.float32)
midi_out  = PROJECT_ROOT / "outputs" / "_smoke" / "test_roll.mid"
save_piano_roll_as_midi(test_roll, str(midi_out))
check(midi_out.exists(), "piano_roll_to_midi saved OK")

tok_out = PROJECT_ROOT / "outputs" / "_smoke" / "test_tokens.mid"
tok_seq = torch.load(tmp_proc / "tok_train.pt", weights_only=True)[0]
save_tokens_as_midi(tok_seq, str(tok_out))
check(tok_out.exists(), "tokens_to_midi saved OK")

# ── Step 4: Model forward passes ─────────────────────────────────────────
print("\n  [4/6] Testing LSTM Autoencoder forward pass …")
from src.models.autoencoder import LSTMAutoencoder

ae = LSTMAutoencoder(input_dim=128, hidden_dim=64, latent_dim=16,
                     num_layers=1, dropout=0.0, seq_len=32)
x_fake = torch.rand(4, 32, 128)
x_hat, z = ae(x_fake)
check(x_hat.shape == (4, 32, 128), f"AE output shape: {x_hat.shape}")
check(z.shape == (4, 16),          f"AE latent shape: {z.shape}")
loss = ae.reconstruction_loss(x_fake, x_hat)
check(loss.item() >= 0,            f"AE loss: {loss.item():.6f}")

print("\n  [4b/6] Testing VAE forward pass …")
from src.models.vae import MusicVAE

vae = MusicVAE(input_dim=128, hidden_dim=64, latent_dim=16,
               num_layers=1, dropout=0.0, seq_len=32, beta=1.0)
x_hat_v, mu, logvar = vae(x_fake)
check(x_hat_v.shape == (4, 32, 128), f"VAE output shape: {x_hat_v.shape}")
check(mu.shape == (4, 16),           f"VAE mu shape: {mu.shape}")
total, recon, kl = vae.loss(x_fake, x_hat_v, mu, logvar)
check(total.item() >= 0, f"VAE total loss: {total.item():.4f}")
check(kl.item() >= 0,    f"VAE KL div: {kl.item():.4f}")

samples = vae.sample(3)
check(samples.shape == (3, 32, 128), f"VAE sample shape: {samples.shape}")

x1 = x_fake[:1]; x2 = x_fake[1:2]
interp = vae.interpolate(x1, x2, steps=4)
check(interp.shape[0] == 4, f"Interpolation steps: {interp.shape[0]}")

print("\n  [4c/6] Testing Transformer forward pass …")
from src.models.transformer import MusicTransformer

tf = MusicTransformer(vocab_size=417, d_model=64, n_heads=4,
                      n_layers=2, d_ff=128, max_seq_len=64)
tok_batch = torch.randint(1, 417, (4, 32))
logits    = tf(tok_batch)
check(logits.shape == (4, 32, 417), f"Transformer logits: {logits.shape}")

loss_tf = MusicTransformer.compute_loss(logits[:, :-1, :], tok_batch[:, 1:])
check(loss_tf.item() >= 0, f"Transformer CE loss: {loss_tf.item():.4f}")

prompt    = torch.randint(1, 417, (1, 8))
generated = tf.generate(prompt, max_new_tokens=16, top_k=10)
check(generated.shape[1] == 24, f"Transformer generation: {generated.shape[1]} tokens")

# ── Step 5: Evaluation metrics ────────────────────────────────────────────
print("\n  [5/6] Testing evaluation metrics …")
from src.evaluation.metrics import evaluate_piano_roll, evaluate_batch, compare_models

roll_np = np.random.randint(0, 2, (64, 128)).astype(np.float32)
stats   = evaluate_piano_roll(roll_np)
check(len(stats) == 6,                    f"6 metrics computed: {list(stats.keys())}")
check(0 <= stats["note_density"] <= 1,    f"note_density: {stats['note_density']:.4f}")
check(stats["pitch_entropy"] >= 0,        f"pitch_entropy: {stats['pitch_entropy']:.4f}")
check(stats["unique_pitches"] >= 0,       f"unique_pitches: {int(stats['unique_pitches'])}")

batch   = np.random.rand(8, 64, 128).astype(np.float32)
b_stats = evaluate_batch(batch)
check(len(b_stats) == 6, f"batch evaluation ok ({len(b_stats)} metrics)")

# ── Step 6: Baselines ─────────────────────────────────────────────────────
print("\n  [6/6] Testing baseline generators …")
from src.models.baselines import RandomNoteGenerator, MarkovChainModel

rng_gen = RandomNoteGenerator(seed=0)
r_roll  = rng_gen.generate(num_steps=32)
check(r_roll.shape == (32, 128), f"Random roll shape: {r_roll.shape}")

markov  = MarkovChainModel(order=1, seed=0)
markov.fit(roll_train[:50])
m_roll  = markov.generate(num_steps=32)
check(m_roll.shape == (32, 128), f"Markov roll shape: {m_roll.shape}")

rng_midi = PROJECT_ROOT / "outputs" / "_smoke" / "random.mid"
rng_gen.generate_midi(str(rng_midi), num_steps=32)
check(rng_midi.exists(), "Random MIDI exported OK")

# ── Cleanup ────────────────────────────────────────────────────────────────
shutil.rmtree(tmp_midi, ignore_errors=True)
shutil.rmtree(tmp_proc, ignore_errors=True)

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"  {RED}SMOKE TEST FAILED — {len(errors)} error(s):{RESET}")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print(f"  {GREEN}ALL SMOKE TESTS PASSED ({len(errors)} failures){RESET}")
    print("  The codebase is fully functional. You are ready to train!")
print("=" * 60 + "\n")
