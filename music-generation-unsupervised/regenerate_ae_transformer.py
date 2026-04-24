import os, sys, torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
PROJECT_ROOT = Path("C:/Users/Hp/Desktop/cse425/music-generation-unsupervised")
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder import LSTMAutoencoder
from src.models.transformer import MusicTransformer
from src.preprocessing.midi_utils import piano_roll_to_midi, tokens_to_midi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = PROJECT_ROOT / "data/processed/maestro"
ckpt_dir = PROJECT_ROOT / "outputs/models"

# ── Load AE ────────────────────────────────────────────────────────────────
roll_test = torch.load(data_dir / "roll_test.pt", weights_only=True).float()
seq_len = roll_test.shape[1]
model_ae = LSTMAutoencoder(seq_len=seq_len).to(device)
model_ae.load_state_dict(torch.load(ckpt_dir / "ae_checkpoint.pt", map_location=device, weights_only=True))
model_ae.eval()

# ── Load Transformer ───────────────────────────────────────────────────────
tok_test_path = data_dir / "tok_test.pt"
transformer_ckpt_path = ckpt_dir / "transformer_ckpt.pt"
if tok_test_path.exists() and transformer_ckpt_path.exists():
    tok_test = torch.load(tok_test_path, weights_only=True).long()
    model_transformer = MusicTransformer(vocab_size=417).to(device)
    model_transformer.load_state_dict(torch.load(transformer_ckpt_path, map_location=device, weights_only=True))
    model_transformer.eval()
    has_transformer = True
else:
    print(f"WARNING: Transformer checkpoint or data missing. Skipping Task 2.")
    has_transformer = False

# ── TASK 1: AE Regeneration ────────────────────────────────────────────────
print("--- TASK 1: AE Regeneration (5 samples) ---")
ae_out_dir = PROJECT_ROOT / "outputs/midi/ae"
os.makedirs(ae_out_dir, exist_ok=True)

# Use roll_test as the "loader" source
with torch.no_grad():
    for i in range(1, 6):
        torch.manual_seed(i * 7)
        # AE: encode a real sample then decode it
        # idx = i % len(roll_test)
        idx = torch.randint(0, len(roll_test), (1,)).item()
        sample = roll_test[idx:idx+1].to(device)
        z = model_ae.encode(sample)
        output = model_ae.decode(z)
        # Note: model_ae.decode already applies sigmoid. Applying it again as requested.
        output = torch.sigmoid(output)
        piano_roll = output.squeeze().cpu().numpy()
        midi = piano_roll_to_midi(piano_roll, fs=16, tempo=120)
        save_path = ae_out_dir / f"ae_sample_{i:02d}.mid"
        midi.write(str(save_path))
        size = os.path.getsize(save_path)
        notes = sum(len(inst.notes) for inst in midi.instruments)
        print(f"ae_sample_{i:02d}.mid -> {size} bytes, {notes} notes")

# ── TASK 2: Transformer Regeneration ───────────────────────────────────────
print("\n--- TASK 2: Transformer Generation (10 samples) ---")
trans_out_dir = PROJECT_ROOT / "outputs/midi/transformer"
os.makedirs(trans_out_dir, exist_ok=True)

if has_transformer:
    with torch.no_grad():
        for i in range(1, 11):
            torch.manual_seed(i * 13)
            # Autoregressive generation — start from a seed token
            idx = torch.randint(0, len(tok_test), (1,)).item()
            prompt = tok_test[idx, :16].unsqueeze(0).to(device)
            
            # Using model's actual generate method parameters
            generated = model_transformer.generate(
                prompt=prompt,
                max_new_tokens=128,
                temperature=1.0
            )
            
            # Transformer outputs tokens, so we use tokens_to_midi
            midi = tokens_to_midi(generated[0].cpu().numpy().tolist())
            
            save_path = trans_out_dir / f"transformer_sample_{i:02d}.mid"
            midi.write(str(save_path))
            size = os.path.getsize(save_path)
            notes = sum(len(inst.notes) for inst in midi.instruments)
            print(f"transformer_sample_{i:02d}.mid -> {size} bytes, {notes} notes")
else:
    print("Task 2 skipped due to missing checkpoint.")

print("\nDone.")
