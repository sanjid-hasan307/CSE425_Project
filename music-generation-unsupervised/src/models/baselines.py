"""
src/models/baselines.py
========================
Baseline music generation models.

Models
------
1. RandomNoteGenerator  — samples random pitches and durations.
2. MarkovChainModel     — learns n-gram note transitions from training data
                          and samples new sequences via weighted random walk.

Both models export playable MIDI files to outputs/baselines/.

Usage
-----
    python -m src.models.baselines \\
        --data_dir  data/processed/maestro \\
        --out_dir   outputs/baselines \\
        --num_samples 5
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi
import torch
from tqdm import tqdm

from src.preprocessing.midi_utils import save_piano_roll_as_midi


# ═══════════════════════════════════════════════════════════════════════════
# 1. Random Note Generator
# ═══════════════════════════════════════════════════════════════════════════

class RandomNoteGenerator:
    """
    Baseline 1: Random Note Generator.

    At each time step the generator independently samples each of the 128
    MIDI pitches with probability `note_prob`.  The resulting binary piano-
    roll is then written to MIDI.

    Parameters
    ----------
    note_prob      : probability that any given pitch is active per step
    pitch_range    : (low, high) MIDI pitch range to draw from
    seed           : random seed for reproducibility
    """

    def __init__(
        self,
        note_prob:   float = 0.04,
        pitch_range: Tuple[int, int] = (36, 84),
        seed:        int = 42,
    ) -> None:
        self.note_prob   = note_prob
        self.pitch_lo, self.pitch_hi = pitch_range
        self.rng = np.random.default_rng(seed)

    def generate(self, num_steps: int = 64) -> np.ndarray:
        """
        Generate a random piano-roll.

        Returns
        -------
        roll : np.ndarray, shape (num_steps, 128), dtype float32
        """
        roll = np.zeros((num_steps, 128), dtype=np.float32)
        for t in range(num_steps):
            # Pick 1–4 random pitches within the allowed range
            n_notes = self.rng.integers(1, 5)
            pitches = self.rng.integers(self.pitch_lo, self.pitch_hi, size=n_notes)
            roll[t, pitches] = 1.0
        return roll

    def generate_midi(
        self,
        filepath:     str,
        num_steps:    int = 64,
        steps_per_bar: int = 16,
        tempo:        float = 120.0,
    ) -> None:
        """Generate and save a MIDI file."""
        roll = self.generate(num_steps=num_steps)
        save_piano_roll_as_midi(roll, filepath, steps_per_bar=steps_per_bar, tempo=tempo)
        print(f"  [Random]  Saved → {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Markov Chain Model
# ═══════════════════════════════════════════════════════════════════════════

class MarkovChainModel:
    """
    Baseline 2: n-gram Markov Chain over active pitch sets.

    Training: walks through piano-roll windows and records how often each
    pitch set (represented as a frozenset of active MIDI pitches) transitions
    to the next pitch set.

    Sampling: starts from a random observed state and follows the
    transition distribution greedily.

    Parameters
    ----------
    order   : Markov order (n previous states used to predict next)
    seed    : random seed
    """

    def __init__(self, order: int = 1, seed: int = 42) -> None:
        self.order = order
        self.seed  = seed
        self.rng   = random.Random(seed)

        # transition_counts[history_tuple] → {next_state: count}
        self.transition_counts: Dict = defaultdict(lambda: defaultdict(int))
        self.transition_probs:  Dict = {}
        self.states: List[frozenset] = []
        self._fitted = False

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(self, roll_data: torch.Tensor) -> None:
        """
        Learn transition probabilities from a (N, T, 128) piano-roll tensor.

        Parameters
        ----------
        roll_data : Tensor of shape (N, T, 128) with binary values.
        """
        print(f"  [Markov] Fitting order-{self.order} chain on "
              f"{roll_data.shape[0]} segments …")

        all_states: set = set()

        for seg_idx in tqdm(range(roll_data.shape[0]),
                            desc="  Fitting", ncols=80, unit="seg"):
            seg = roll_data[seg_idx].numpy()   # (T, 128)

            # Convert each time step to a frozenset of active pitches
            states = []
            for t in range(seg.shape[0]):
                active = frozenset(int(p) for p in np.where(seg[t] > 0.5)[0])
                states.append(active)
                all_states.add(active)

            # Record n-gram transitions
            for i in range(len(states) - self.order):
                history = tuple(states[i : i + self.order])
                nxt     = states[i + self.order]
                self.transition_counts[history][nxt] += 1

        # Convert counts to probabilities
        for history, nexts in self.transition_counts.items():
            total = sum(nexts.values())
            self.transition_probs[history] = {
                s: c / total for s, c in nexts.items()
            }

        self.states    = list(all_states)
        self._fitted   = True
        print(f"  [Markov] Learned {len(self.transition_probs)} unique "
              f"transition entries from {len(all_states)} distinct states.")

    def _sample_next(self, history: tuple) -> frozenset:
        """Sample the next state given a history tuple."""
        if history in self.transition_probs:
            nxt_states  = list(self.transition_probs[history].keys())
            probs       = list(self.transition_probs[history].values())
            idx         = self.rng.choices(range(len(nxt_states)), weights=probs)[0]
            return nxt_states[idx]
        else:
            # Backoff: pick a random observed state
            return self.rng.choice(self.states) if self.states else frozenset()

    def generate(self, num_steps: int = 64) -> np.ndarray:
        """
        Sample a piano-roll sequence from the Markov chain.

        Returns
        -------
        roll : np.ndarray, shape (num_steps, 128), dtype float32
        """
        assert self._fitted, "Call .fit() before .generate()."

        # Start with a random observed state as the initial history
        init_state  = self.rng.choice(self.states) if self.states else frozenset()
        history     = [init_state] * self.order    # type: List[frozenset]
        roll        = np.zeros((num_steps, 128), dtype=np.float32)

        for t in range(num_steps):
            current_state = history[-1]
            for pitch in current_state:
                if 0 <= pitch < 128:
                    roll[t, pitch] = 1.0

            hist_tuple = tuple(history[-self.order:])
            nxt_state  = self._sample_next(hist_tuple)
            history.append(nxt_state)

        return roll

    def generate_midi(
        self,
        filepath:      str,
        num_steps:     int   = 64,
        steps_per_bar: int   = 16,
        tempo:         float = 120.0,
    ) -> None:
        """Generate and save a MIDI file."""
        roll = self.generate(num_steps=num_steps)
        save_piano_roll_as_midi(roll, filepath, steps_per_bar=steps_per_bar, tempo=tempo)
        print(f"  [Markov]  Saved → {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run baseline music generators and export MIDI files."
    )
    p.add_argument("--data_dir",    type=str, default="data/processed/maestro",
                   help="Directory with preprocessed .pt tensors.")
    p.add_argument("--out_dir",     type=str, default="outputs/baselines",
                   help="Output directory for generated MIDI files.")
    p.add_argument("--num_samples", type=int, default=5,
                   help="Number of MIDI files to generate per model.")
    p.add_argument("--num_steps",   type=int, default=64,
                   help="Length of generated sequence (time steps).")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    # ── Load training data ─────────────────────────────────────────────────
    roll_path = data_dir / "roll_train.pt"
    if not roll_path.exists():
        print(f"[ERROR] {roll_path} not found. Run preprocess.py first.")
        return

    roll_train = torch.load(roll_path, weights_only=True)
    print(f"\n  Loaded roll_train: {roll_train.shape}")

    # ── Baseline 1: Random Note Generator ─────────────────────────────────
    print("\n" + "═" * 55)
    print("  Baseline 1: Random Note Generator")
    print("═" * 55)
    rng_model  = RandomNoteGenerator(seed=42)
    random_dir = out_dir / "random"
    random_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_samples):
        rng_model.generate_midi(
            filepath=str(random_dir / f"random_sample_{i+1:02d}.mid"),
            num_steps=args.num_steps,
        )

    # ── Baseline 2: Markov Chain ───────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  Baseline 2: Markov Chain (order=1)")
    print("═" * 55)
    markov_model = MarkovChainModel(order=1, seed=42)
    markov_model.fit(roll_train)

    markov_dir = out_dir / "markov"
    markov_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_samples):
        markov_model.generate_midi(
            filepath=str(markov_dir / f"markov_sample_{i+1:02d}.mid"),
            num_steps=args.num_steps,
        )

    print(f"\n  ✓ All baseline samples exported to {out_dir}/")


if __name__ == "__main__":
    args = _parse_args()
    main(args)
