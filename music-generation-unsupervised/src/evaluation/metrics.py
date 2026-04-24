import os
import numpy as np
import pretty_midi

def load_midi_as_piano_roll(path, fs=16):
    try:
        midi = pretty_midi.PrettyMIDI(path)
        roll = midi.get_piano_roll(fs=fs)
        roll = (roll > 0).astype(float)
        return roll
    except Exception:
        return np.zeros((128, 64))

def pitch_histogram_similarity(midi_folder):
    histograms = []
    if not os.path.exists(midi_folder): return 0.0
    for f in os.listdir(midi_folder):
        if not f.endswith('.mid'): continue
        piano_roll = load_midi_as_piano_roll(
            os.path.join(midi_folder, f))
        hist = piano_roll.sum(axis=1)
        hist = hist / (hist.sum() + 1e-8)
        histograms.append(hist)
    if len(histograms) < 2: return 1.0
    scores = []
    for i in range(len(histograms)):
        for j in range(i+1, len(histograms)):
            dot = np.dot(histograms[i], histograms[j])
            norm = (np.linalg.norm(histograms[i]) * 
                    np.linalg.norm(histograms[j]))
            scores.append(dot / (norm + 1e-8))
    return float(np.mean(scores))

def rhythm_diversity_score(midi_folder):
    scores = []
    if not os.path.exists(midi_folder): return 0.0
    for f in os.listdir(midi_folder):
        if not f.endswith('.mid'): continue
        piano_roll = load_midi_as_piano_roll(
            os.path.join(midi_folder, f))
        active = (piano_roll.sum(axis=0) > 0).astype(float)
        if active.sum() < 2: 
            scores.append(0.0)
            continue
        onsets = np.where(np.diff(active) == 1)[0]
        if len(onsets) < 2:
            scores.append(0.0)
            continue
        ioi = np.diff(onsets).astype(float)
        ioi_norm = ioi / (ioi.sum() + 1e-8)
        entropy = -np.sum(
            ioi_norm * np.log(ioi_norm + 1e-8))
        scores.append(entropy)
    return float(np.mean(scores)) if scores else 0.0

def repetition_ratio(midi_folder, n=4):
    ratios = []
    if not os.path.exists(midi_folder): return 0.0
    for f in os.listdir(midi_folder):
        if not f.endswith('.mid'): continue
        piano_roll = load_midi_as_piano_roll(
            os.path.join(midi_folder, f))
        cols = [tuple(piano_roll[:, t].astype(int)) 
                for t in range(piano_roll.shape[1])]
        ngrams = [tuple(cols[i:i+n]) 
                  for i in range(len(cols)-n+1)]
        if not ngrams:
            ratios.append(0.0)
            continue
        unique = len(set(ngrams))
        ratio = 1.0 - (unique / len(ngrams))
        ratios.append(ratio)
    return float(np.mean(ratios)) if ratios else 0.0
