import os, sys, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time

# Add project root to sys.path
PROJECT_ROOT = Path("C:/Users/Hp/Desktop/cse425/music-generation-unsupervised")
sys.path.insert(0, str(PROJECT_ROOT))

print("Script started...", flush=True)

from src.preprocessing.midi_utils import piano_roll_to_midi

# ── 1. Model Definition ───────────────────────────────────────────────────
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size=128, d_model=256, nhead=8, 
                 num_layers=4, max_seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        memory = torch.zeros(B, 1, 256).to(x.device)
        out = self.transformer(x, memory, tgt_mask=mask)
        return self.fc_out(out)

    def generate(self, max_length=128, temperature=1.0, device='cpu'):
        self.eval()
        with torch.no_grad():
            # Start with a random token
            x = torch.randint(0, 128, (1, 1)).to(device)
            for _ in range(max_length - 1):
                out = self.forward(x)
                logits = out[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                x = torch.cat([x, next_token], dim=1)
        return x.squeeze(0)

# ── 2. Data Loading ───────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = PROJECT_ROOT / "data/processed/maestro"
roll_train = torch.load(data_dir / "roll_train.pt", weights_only=True).float()[:500]
# Trim or pad to max_seq_len if necessary
# roll_train is likely (N, 64, 128). We'll use seq_len=64.
seq_len = roll_train.shape[1]
train_ds = TensorDataset(roll_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)

# ── 3. Training ───────────────────────────────────────────────────────────
model_transformer = MusicTransformer(max_seq_len=seq_len).to(device)
optimizer = torch.optim.Adam(model_transformer.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print(f"Device: {device}", flush=True)
print(f"Loaded {len(roll_train)} segments (subset for speed). Sequence length: {seq_len}", flush=True)
print("Training Transformer...", flush=True)

model_transformer.train()
for epoch in range(30):
    total_loss = 0
    for (batch,) in train_loader:
        # Convert piano roll (B, T, 128) to token sequence (B, T)
        # Using argmax to pick the dominant pitch per timestep
        tokens = batch.argmax(dim=-1).long().to(device)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        
        out = model_transformer(x)
        # out: (B, T-1, 128)
        loss = criterion(out.reshape(-1, 128), y.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1:2d}/30 | Loss: {total_loss/len(train_loader):.4f}", flush=True)

# Save checkpoint
ckpt_path = PROJECT_ROOT / 'outputs' / 'transformer_checkpoint.pth'
os.makedirs(ckpt_path.parent, exist_ok=True)
torch.save(model_transformer.state_dict(), ckpt_path)
print(f"Checkpoint saved to {ckpt_path}!")

# ── 4. Generation ─────────────────────────────────────────────────────────
print("\n--- Generating 10 Samples ---")
model_transformer.eval()
out_dir = PROJECT_ROOT / "outputs/midi/transformer"
os.makedirs(out_dir, exist_ok=True)

with torch.no_grad():
    for i in range(1, 11):
        torch.manual_seed(i * 13)
        # Generate tokens
        tokens = model_transformer.generate(
            max_length=seq_len, temperature=1.0, device=device)
        
        # Convert tokens to piano roll (128 pitches, T timesteps)
        piano_roll = torch.zeros(128, seq_len)
        for t, tok in enumerate(tokens[:seq_len]):
            if 0 <= tok.item() < 128:
                piano_roll[tok.item(), t] = 1.0
        
        piano_roll_np = piano_roll.numpy()
        midi = piano_roll_to_midi(piano_roll_np, fs=16, tempo=120)
        save_path = out_dir / f"transformer_sample_{i:02d}.mid"
        midi.write(str(save_path))
        
        size = os.path.getsize(save_path)
        notes = sum(len(inst.notes) for inst in midi.instruments)
        print(f"transformer_sample_{i:02d}.mid -> {size} bytes, {notes} notes")

print("\nDone.")
