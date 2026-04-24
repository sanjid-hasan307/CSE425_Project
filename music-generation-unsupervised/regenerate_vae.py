import os, sys, torch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("C:/Users/Hp/Desktop/cse425/music-generation-unsupervised")
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vae import MusicVAE
from src.preprocessing.midi_utils import piano_roll_to_midi, save_piano_roll_as_midi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = PROJECT_ROOT / "data/processed/maestro"

# Load test data to get seq_len
roll_test = torch.load(data_dir / "roll_test.pt", weights_only=True).float()
seq_len = roll_test.shape[1]
latent_dim = 64

model = MusicVAE(seq_len=seq_len).to(device)
ckpt = PROJECT_ROOT / "outputs" / "models" / "vae_checkpoint.pt"
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
model.eval()

# Delete old files and create directories
vae_dir = PROJECT_ROOT / "outputs" / "midi" / "vae"
interp_dir = vae_dir / "interpolation"

for d in [vae_dir, interp_dir]:
    if d.exists():
        for f in d.glob("*.mid"):
            try:
                os.remove(f)
            except:
                pass
    os.makedirs(d, exist_ok=True)

print("--- Regenerating 8 VAE Samples ---")
with torch.no_grad():
    for i in range(1, 9):
        torch.manual_seed(i * 42)
        z = torch.randn(1, latent_dim).to(device) * 2.0
        output = model.decode(z)
        output = torch.sigmoid(output)
        piano_roll = output.squeeze().cpu().numpy()
        midi = piano_roll_to_midi(piano_roll, fs=16, tempo=120)
        save_path = vae_dir / f"vae_sample_{i:02d}.mid"
        midi.write(str(save_path))
        size = os.path.getsize(save_path)
        notes = sum(len(inst.notes) for inst in midi.instruments)
        print(f"vae_sample_{i:02d}.mid -> {size} bytes, {notes} notes")

print("\n--- Regenerating Interpolation Samples ---")
with torch.no_grad():
    x1 = roll_test[0].unsqueeze(0).to(device)
    x2 = roll_test[1].unsqueeze(0).to(device)
    interp_rolls = model.interpolate(x1, x2, steps=8)
    for i, roll in enumerate(interp_rolls):
        path = interp_dir / f"interpolation_{i+1:02d}.mid"
        save_piano_roll_as_midi(roll.cpu().numpy(), str(path))
        print(f"interpolation_{i+1:02d}.mid -> {os.path.getsize(path)} bytes, {len(piano_roll_to_midi(roll.cpu().numpy()).instruments[0].notes)} notes")

print("\nDone.")
