"""
src/preprocessing/preprocess.py
=================================
End-to-end MIDI preprocessing pipeline.

Responsibilities
----------------
1. Walk a root directory and collect all .mid / .midi files.
2. Parse each file with pretty_midi.
3. Extract TWO representations:
   a) Piano-roll   : (T, 128) binary matrix at `steps_per_bar` resolution.
   b) Token stream : list of (event_type, value, time_step) tokens.
4. Segment each file into fixed-length windows (64 and 128 steps).
5. Apply a reproducible 80/20 train/test split.
6. Save processed arrays to data/processed/ as .pt (PyTorch tensors).
7. Print a full dataset statistics summary.

Usage
-----
    python -m src.preprocessing.preprocess \\
        --midi_dir  data/raw/maestro \\
        --out_dir   data/processed/maestro \\
        --window    64 \\
        --steps_per_bar 16

    python -m src.preprocessing.preprocess \\
        --midi_dir  data/raw/lakh \\
        --out_dir   data/processed/lakh \\
        --window    128
"""

import os
import sys
import json
import math
import random
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi
import torch
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────

# MIDI pitch range (standard 88-key piano: 21–108, but we keep full 0–127)
PITCH_MIN    = 0
PITCH_MAX    = 127
NUM_PITCHES  = 128

# Token vocabulary indices
TOKEN_PAD      = 0
TOKEN_NOTE_ON  = 1   # value = MIDI pitch  (0–127)
TOKEN_NOTE_OFF = 2   # value = MIDI pitch  (0–127)
TOKEN_VELOCITY = 3   # value = velocity bin (0–31 → 32 bins)
TOKEN_TIME     = 4   # value = number of time steps to advance
VOCAB_SIZE     = 5 + 128 + 128 + 32 + 128   # total token vocab

VELOCITY_BINS  = 32
MAX_TIME_SHIFT = 128  # maximum time-shift in one token

RANDOM_SEED    = 42


# ── MIDI Parsing ───────────────────────────────────────────────────────────

def load_midi(filepath: Path) -> Optional[pretty_midi.PrettyMIDI]:
    """Load a MIDI file; returns None on parse error."""
    try:
        pm = pretty_midi.PrettyMIDI(str(filepath))
        return pm
    except Exception:
        return None


def midi_to_piano_roll(
    pm: pretty_midi.PrettyMIDI,
    steps_per_bar: int = 16,
    beats_per_bar: int = 4,
) -> np.ndarray:
    """
    Convert a PrettyMIDI object to a binary piano-roll.

    Parameters
    ----------
    pm             : loaded PrettyMIDI object
    steps_per_bar  : number of quantised steps per bar (default = 16)
    beats_per_bar  : assumed time signature numerator

    Returns
    -------
    roll : np.ndarray, shape (T, 128), dtype float32
           1.0 where a note is active, 0.0 otherwise.
    """
    # Compute fs as steps_per_beat * (beats / second)
    # tempo_change_times gives us BPM; we use a weighted mean tempo
    tempo_change_times, tempos = pm.get_tempo_changes()
    if len(tempos) == 0:
        avg_tempo = 120.0
    else:
        avg_tempo = float(np.mean(tempos))

    beats_per_second = avg_tempo / 60.0
    steps_per_second = beats_per_second * (steps_per_bar / beats_per_bar)

    # Get piano-roll from pretty_midi (returns shape: 128 × T_frames)
    roll_raw = pm.get_piano_roll(fs=steps_per_second)  # (128, T)

    # Binarise and transpose → (T, 128)
    roll = (roll_raw > 0).astype(np.float32).T
    return roll


def midi_to_tokens(
    pm: pretty_midi.PrettyMIDI,
    steps_per_bar: int = 16,
    beats_per_bar: int = 4,
) -> List[int]:
    """
    Convert a PrettyMIDI object to a flat token sequence.

    Token encoding (all packed into a single integer vocabulary):
      0                           → PAD
      1 … 128                     → NOTE_ON  (pitch 0–127)
      129 … 256                   → NOTE_OFF (pitch 0–127)
      257 … 288                   → VELOCITY bin (0–31)
      289 … 416                   → TIME_SHIFT (1–128 steps)

    Returns
    -------
    tokens : List[int]
    """
    # Pre-compute timing
    tempo_change_times, tempos = pm.get_tempo_changes()
    avg_tempo = float(np.mean(tempos)) if len(tempos) > 0 else 120.0
    beats_per_second  = avg_tempo / 60.0
    steps_per_second  = beats_per_second * (steps_per_bar / beats_per_bar)
    seconds_per_step  = 1.0 / steps_per_second

    def _vel_bin(v: int) -> int:
        return min(int(v / 128 * VELOCITY_BINS), VELOCITY_BINS - 1)

    # Collect all events: (time_step, type, value)
    # type: 'on' | 'off'
    events: List[Tuple[int, str, int, int]] = []  # (step, type, pitch, velocity)
    for instr in pm.instruments:
        if instr.is_drum:
            continue
        for note in instr.notes:
            on_step  = int(round(note.start / seconds_per_step))
            off_step = int(round(note.end   / seconds_per_step))
            off_step = max(off_step, on_step + 1)
            events.append((on_step,  "on",  note.pitch, note.velocity))
            events.append((off_step, "off", note.pitch, 0))

    # Sort by time, then note_off before note_on (to avoid overlap glitches)
    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

    tokens: List[int] = []
    current_step = 0

    for step, etype, pitch, vel in events:
        # Emit time-shift tokens
        delta = step - current_step
        while delta > 0:
            shift = min(delta, MAX_TIME_SHIFT)
            tokens.append(288 + shift)   # TIME_SHIFT token (1-indexed → 289…416)
            delta -= shift
        current_step = step

        if etype == "on":
            tokens.append(257 + _vel_bin(vel))   # VELOCITY token
            tokens.append(1   + pitch)            # NOTE_ON  token
        else:
            tokens.append(129 + pitch)            # NOTE_OFF token

    return tokens


# ── Segmentation ───────────────────────────────────────────────────────────

def segment_piano_roll(
    roll: np.ndarray,
    window: int = 64,
    stride: Optional[int] = None,
) -> np.ndarray:
    """
    Slice a (T, 128) piano-roll into overlapping windows.

    Returns
    -------
    segments : np.ndarray, shape (N, window, 128)
    """
    if stride is None:
        stride = window // 2   # 50% overlap

    T = roll.shape[0]
    if T < window:
        # Pad with zeros if too short
        pad = np.zeros((window - T, NUM_PITCHES), dtype=np.float32)
        roll = np.concatenate([roll, pad], axis=0)
        T = window

    segments = []
    for start in range(0, T - window + 1, stride):
        seg = roll[start : start + window]
        segments.append(seg)

    return np.stack(segments, axis=0)  # (N, window, 128)


def segment_tokens(
    tokens: List[int],
    window: int = 128,
    stride: Optional[int] = None,
) -> np.ndarray:
    """
    Slice a flat token list into fixed-length windows.

    Returns
    -------
    segments : np.ndarray, shape (N, window), dtype int64
               Shorter sequences are right-padded with TOKEN_PAD (0).
    """
    if stride is None:
        stride = window // 2

    T = len(tokens)
    if T < window:
        tokens = tokens + [TOKEN_PAD] * (window - T)
        T = window

    arr = np.array(tokens, dtype=np.int64)
    segments = []
    for start in range(0, T - window + 1, stride):
        segments.append(arr[start : start + window])

    return np.stack(segments, axis=0)  # (N, window)


# ── Train/Test Split ───────────────────────────────────────────────────────

def train_test_split(
    segments: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly shuffle and split a segment array 80/20.

    Returns (train_segments, test_segments).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(segments))
    cut = int(len(segments) * train_ratio)
    return segments[idx[:cut]], segments[idx[cut:]]


# ── Statistics ─────────────────────────────────────────────────────────────

def compute_statistics(
    rolls: List[np.ndarray],
    token_seqs: List[List[int]],
) -> Dict:
    """Compute dataset-level statistics for the summary printout."""
    lengths = [r.shape[0] for r in rolls]
    token_lengths = [len(t) for t in token_seqs]

    all_rolls  = np.concatenate(rolls,  axis=0) if rolls  else np.zeros((1, 128))
    note_density = all_rolls.mean()

    # Active pitch distribution
    pitch_usage = all_rolls.mean(axis=0)  # (128,)
    active_pitches = int((pitch_usage > 0).sum())

    return {
        "num_files_parsed":   len(rolls),
        "min_roll_length":    int(min(lengths)) if lengths else 0,
        "max_roll_length":    int(max(lengths)) if lengths else 0,
        "avg_roll_length":    float(np.mean(lengths)) if lengths else 0,
        "note_density":       float(note_density),
        "active_pitches":     active_pitches,
        "avg_token_length":   float(np.mean(token_lengths)) if token_lengths else 0,
        "max_token_length":   int(max(token_lengths)) if token_lengths else 0,
    }


def print_summary(stats: Dict, split_info: Dict) -> None:
    """Pretty-print dataset statistics."""
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            print(f"  {label:<30}: {v:.4f}")
        else:
            print(f"  {label:<30}: {v}")
    print("-" * 60)
    print("  Split Information")
    print("-" * 60)
    for k, v in split_info.items():
        print(f"  {k:<30}: {v}")
    print("=" * 60 + "\n")


# ── Main Pipeline ──────────────────────────────────────────────────────────

def run_pipeline(
    midi_dir: str,
    out_dir: str,
    window: int = 64,
    steps_per_bar: int = 16,
    train_ratio: float = 0.8,
    max_files: Optional[int] = None,
) -> None:
    """
    Full preprocessing pipeline:
      1. Collect MIDI files.
      2. Parse → piano-roll + tokens.
      3. Segment into windows.
      4. Train/test split.
      5. Save as .pt tensors.
    """
    midi_root = Path(midi_dir)
    out_path  = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Discover MIDI files ────────────────────────────────────────
    exts = {".mid", ".midi"}
    all_files = sorted([
        p for p in midi_root.rglob("*") if p.suffix.lower() in exts
    ])

    if not all_files:
        print(f"[ERROR] No MIDI files found under {midi_root}")
        sys.exit(1)

    if max_files is not None:
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(all_files)
        all_files = all_files[:max_files]

    print(f"\n  Found {len(all_files)} MIDI files in {midi_root}")

    # ── Step 2: Parse MIDI files ───────────────────────────────────────────
    piano_rolls: List[np.ndarray]  = []
    token_seqs:  List[List[int]]   = []
    skipped = 0

    for filepath in tqdm(all_files, desc="  Parsing MIDI", unit="file", ncols=80):
        pm = load_midi(filepath)
        if pm is None or len(pm.instruments) == 0:
            skipped += 1
            continue
        try:
            roll   = midi_to_piano_roll(pm, steps_per_bar=steps_per_bar)
            tokens = midi_to_tokens(pm,     steps_per_bar=steps_per_bar)
            if roll.shape[0] < 4 or len(tokens) < 4:
                skipped += 1
                continue
            piano_rolls.append(roll)
            token_seqs.append(tokens)
        except Exception:
            skipped += 1
            continue

    print(f"  Parsed {len(piano_rolls)} files  |  Skipped {skipped} (corrupt/empty)")

    if not piano_rolls:
        print("[ERROR] No usable MIDI files after parsing. Check your data directory.")
        sys.exit(1)

    # ── Step 3: Compute statistics ─────────────────────────────────────────
    stats = compute_statistics(piano_rolls, token_seqs)

    # ── Step 4: Segment ────────────────────────────────────────────────────
    print(f"\n  Segmenting piano-rolls  (window={window}) …")
    roll_segs_list:  List[np.ndarray] = []
    for roll in tqdm(piano_rolls, desc="  Segmenting rolls", ncols=80):
        segs = segment_piano_roll(roll, window=window)
        roll_segs_list.append(segs)
    roll_segments = np.concatenate(roll_segs_list, axis=0)  # (N, window, 128)

    print(f"  Segmenting token streams (window={window}) …")
    tok_segs_list: List[np.ndarray] = []
    for toks in tqdm(token_seqs, desc="  Segmenting tokens", ncols=80):
        segs = segment_tokens(toks, window=window)
        tok_segs_list.append(segs)
    tok_segments = np.concatenate(tok_segs_list, axis=0)   # (N, window)

    # ── Step 5: Train/test split ───────────────────────────────────────────
    roll_train, roll_test = train_test_split(roll_segments, train_ratio=train_ratio)
    tok_train,  tok_test  = train_test_split(tok_segments,  train_ratio=train_ratio)

    split_info = {
        "Total roll segments":    len(roll_segments),
        "Train roll segments":    len(roll_train),
        "Test roll segments":     len(roll_test),
        "Total token segments":   len(tok_segments),
        "Train token segments":   len(tok_train),
        "Test token segments":    len(tok_test),
        "Window (steps)":         window,
        "Steps per bar":          steps_per_bar,
    }

    # ── Step 6: Save ───────────────────────────────────────────────────────
    torch.save(torch.from_numpy(roll_train), out_path / "roll_train.pt")
    torch.save(torch.from_numpy(roll_test),  out_path / "roll_test.pt")
    torch.save(torch.from_numpy(tok_train),  out_path / "tok_train.pt")
    torch.save(torch.from_numpy(tok_test),   out_path / "tok_test.pt")

    # Save metadata JSON for reproducibility
    meta = {
        **stats,
        **split_info,
        "midi_dir":    str(midi_root),
        "out_dir":     str(out_path),
        "train_ratio": train_ratio,
        "seed":        RANDOM_SEED,
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✓ Saved tensors to {out_path}/")
    print(f"      roll_train.pt  → {roll_train.shape}")
    print(f"      roll_test.pt   → {roll_test.shape}")
    print(f"      tok_train.pt   → {tok_train.shape}")
    print(f"      tok_test.pt    → {tok_test.shape}")

    print_summary(stats, split_info)


# ── CLI Entry Point ────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MIDI preprocessing pipeline for the Music Generation project."
    )
    p.add_argument("--midi_dir",      type=str, required=True,
                   help="Root directory containing MIDI files.")
    p.add_argument("--out_dir",       type=str, required=True,
                   help="Output directory for processed tensors.")
    p.add_argument("--window",        type=int, default=64,
                   help="Segment window length in time steps (default: 64).")
    p.add_argument("--steps_per_bar", type=int, default=16,
                   help="Quantisation resolution per bar (default: 16).")
    p.add_argument("--train_ratio",   type=float, default=0.8,
                   help="Fraction of data used for training (default: 0.8).")
    p.add_argument("--max_files",     type=int, default=None,
                   help="Limit number of files processed (useful for testing).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        midi_dir    = args.midi_dir,
        out_dir     = args.out_dir,
        window      = args.window,
        steps_per_bar = args.steps_per_bar,
        train_ratio = args.train_ratio,
        max_files   = args.max_files,
    )
