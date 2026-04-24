"""
src/preprocessing/midi_utils.py
=================================
Shared helpers for converting model outputs (piano-roll tensors / token
sequences) back to playable MIDI files using pretty_midi.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pretty_midi
import torch


# ── Piano-roll → MIDI ─────────────────────────────────────────────────────

def piano_roll_to_midi(piano_roll, fs=16, tempo=120):
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    instrument = pretty_midi.Instrument(program=0)
    if piano_roll.shape[0] != 128:
        piano_roll = piano_roll.T
    piano_roll = (piano_roll > 0.1).astype(int)
    for pitch in range(128):
        active = False
        start_t = 0
        for t in range(piano_roll.shape[1]):
            if piano_roll[pitch, t] == 1 and not active:
                active = True
                start_t = t / fs
            elif piano_roll[pitch, t] == 0 and active:
                active = False
                end_t = t / fs
                if end_t - start_t > 0.05:
                    instrument.notes.append(pretty_midi.Note(
                        velocity=80, pitch=pitch,
                        start=start_t, end=end_t))
        if active:
            instrument.notes.append(pretty_midi.Note(
                velocity=80, pitch=pitch,
                start=start_t, end=piano_roll.shape[1]/fs))
    midi.instruments.append(instrument)
    return midi


def save_piano_roll_as_midi(
    roll: np.ndarray,
    filepath: str,
    steps_per_bar: int = 16,
    tempo: float = 120.0,
    program: int = 0,
) -> None:
    """Save a piano-roll tensor/array directly to a .mid file."""
    if isinstance(roll, torch.Tensor):
        roll = roll.detach().cpu().numpy()

    # Squeeze batch dimension if present
    if roll.ndim == 3 and roll.shape[0] == 1:
        roll = roll[0]

    pm = piano_roll_to_midi(roll, fs=steps_per_bar, tempo=tempo)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(filepath))


# ── Tokens → MIDI ─────────────────────────────────────────────────────────

def tokens_to_midi(
    tokens: List[int],
    steps_per_bar: int = 16,
    beats_per_bar: int = 4,
    tempo: float = 120.0,
    program: int = 0,
) -> pretty_midi.PrettyMIDI:
    """
    Convert a flat token sequence (as produced by midi_to_tokens) back
    to a pretty_midi object.

    Token encoding (mirrors preprocess.py):
      0          → PAD  (ignored)
      1…128      → NOTE_ON  (pitch = token − 1)
      129…256    → NOTE_OFF (pitch = token − 129)
      257…288    → VELOCITY bin (bin = token − 257)
      289…416    → TIME_SHIFT (steps = token − 288)
    """
    beats_per_second = tempo / 60.0
    steps_per_second = beats_per_second * (steps_per_bar / beats_per_bar)
    seconds_per_step = 1.0 / steps_per_second
    velocity_step    = 128 // 32   # 32 velocity bins → 4 MIDI units per bin

    pm    = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instr = pretty_midi.Instrument(program=program)

    current_step = 0
    current_velocity = 80           # default MIDI velocity
    active_notes: dict[int, float] = {}   # pitch → start_time_in_seconds

    for tok in tokens:
        if tok == 0:
            continue
        elif 1 <= tok <= 128:          # NOTE_ON
            pitch = tok - 1
            start_time = current_step * seconds_per_step
            active_notes[pitch] = start_time
        elif 129 <= tok <= 256:        # NOTE_OFF
            pitch = tok - 129
            if pitch in active_notes:
                start_time = active_notes.pop(pitch)
                end_time   = current_step * seconds_per_step
                if end_time > start_time:
                    instr.notes.append(
                        pretty_midi.Note(
                            velocity=current_velocity,
                            pitch=pitch,
                            start=start_time,
                            end=end_time,
                        )
                    )
        elif 257 <= tok <= 288:        # VELOCITY
            vel_bin = tok - 257
            current_velocity = min(127, vel_bin * velocity_step + velocity_step // 2)
        elif 289 <= tok <= 416:        # TIME_SHIFT
            current_step += tok - 288

    # Close any notes still active at end of sequence
    end_time = current_step * seconds_per_step
    for pitch, start_time in active_notes.items():
        if end_time > start_time:
            instr.notes.append(
                pretty_midi.Note(
                    velocity=current_velocity,
                    pitch=pitch,
                    start=start_time,
                    end=end_time,
                )
            )

    pm.instruments.append(instr)
    return pm


def save_tokens_as_midi(
    tokens,
    filepath: str,
    steps_per_bar: int = 16,
    tempo: float = 120.0,
) -> None:
    """Save a token list/tensor directly to a .mid file."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().numpy().tolist()
    elif isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()

    pm = tokens_to_midi(tokens, steps_per_bar=steps_per_bar, tempo=tempo)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(filepath))
