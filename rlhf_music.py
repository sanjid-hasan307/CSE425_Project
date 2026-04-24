import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("C:/Users/Hp/Desktop/cse425/music-generation-unsupervised")
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.midi_utils import piano_roll_to_midi

# --- Model Definition ---
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
            x = torch.randint(0, 128, (1, 1)).to(device)
            for _ in range(max_length - 1):
                out = self.forward(x)
                next_token = torch.multinomial(
                    torch.softmax(out[:, -1, :] / temperature, dim=-1), 1)
                x = torch.cat([x, next_token], dim=1)
        return x.squeeze(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_transformer = MusicTransformer(max_seq_len=64).to(device)
ckpt_path = PROJECT_ROOT / 'outputs/transformer_checkpoint.pth'
model_transformer.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

# STEP 1 — Generate 10 samples and save them
os.makedirs("outputs/midi/rlhf", exist_ok=True)
model_transformer.eval()
samples = []

print("Generating 10 samples for rating...", flush=True)
with torch.no_grad():
    for i in range(10):
        torch.manual_seed(i * 13)
        # Using seq_len=64 from previous training
        tokens = model_transformer.generate(
            max_length=64, temperature=1.0, device=device)
        piano_roll = torch.zeros(128, 64)
        for t, tok in enumerate(tokens[:64]):
            piano_roll[tok.item(), t] = 1.0
        piano_roll_np = piano_roll.numpy()
        midi = piano_roll_to_midi(piano_roll_np, fs=16, tempo=120)
        path = f"outputs/midi/rlhf/rlhf_before_{i+1:02d}.mid"
        midi.write(path)
        samples.append((tokens, piano_roll_np))
        print(f"Generated: rlhf_before_{i+1:02d}.mid", flush=True)

print("\nOpen outputs/midi/rlhf/ folder and listen to each file.", flush=True)

# STEP 2 — Collect human ratings manually
print("\n=== HUMAN RATING INPUT ===")
print("Listen to each MIDI file and rate 1-5:")
print("1=Very Bad, 2=Bad, 3=OK, 4=Good, 5=Very Good\n", flush=True)

human_ratings = []
for i in range(10):
    while True:
        try:
            # Using print + input with flush to ensure user sees it
            print(f"Rate rlhf_before_{i+1:02d}.mid (1-5): ", end='', flush=True)
            rating_str = input()
            rating = int(rating_str)
            if 1 <= rating <= 5:
                human_ratings.append(rating / 5.0)
                break
            else:
                print("Please enter 1-5 only!", flush=True)
        except ValueError:
            print("Please enter a number!", flush=True)

print("\nYour ratings:", 
      [f"{r*5:.0f}" for r in human_ratings], flush=True)
print(f"Average human score: {sum(human_ratings)/len(human_ratings)*5:.2f}/5", flush=True)

# STEP 3 — Train reward model using human ratings
class RewardModel(nn.Module):
    def __init__(self, input_dim=128*64): # Adjusted for 64 timesteps
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

reward_model = RewardModel().to(device)
optimizer_rm = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
criterion_rm = nn.MSELoss()

# Prepare training data
X = torch.tensor(
    np.array([s[1] for s in samples]), dtype=torch.float32).to(device)
y = torch.tensor(
    human_ratings, dtype=torch.float32).unsqueeze(1).to(device)

print("\n=== Training Reward Model ===", flush=True)
for epoch in range(50):
    pred = reward_model(X)
    loss = criterion_rm(pred, y)
    optimizer_rm.zero_grad()
    loss.backward()
    optimizer_rm.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 | Loss: {loss.item():.4f}", flush=True)

# STEP 4 — REINFORCE fine-tuning using reward model
print("\n=== RLHF Fine-tuning ===", flush=True)
optimizer_rl = torch.optim.Adam(
    model_transformer.parameters(), lr=1e-4)

for episode in range(20):
    torch.manual_seed(episode * 99)
    model_transformer.train()
    tokens = model_transformer.generate(
        max_length=64, temperature=1.0, device=device)
    piano_roll = torch.zeros(1, 128, 64).to(device)
    for t, tok in enumerate(tokens[:64]):
        piano_roll[0, tok.item(), t] = 1.0
    
    with torch.no_grad():
        reward = reward_model(piano_roll).item()
    
    token_tensor = tokens.unsqueeze(0).to(device)
    x = token_tensor[:, :-1]
    y_tok = token_tensor[:, 1:]
    logits = model_transformer(x)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected = log_probs.gather(
        2, y_tok.unsqueeze(-1)).squeeze(-1)
    baseline = 0.5
    loss = -((reward - baseline) * selected.mean())
    optimizer_rl.zero_grad()
    loss.backward()
    optimizer_rl.step()
    
    if (episode+1) % 5 == 0:
        print(f"Episode {episode+1}/20 | "
              f"Reward: {reward:.4f} | Loss: {loss.item():.4f}", flush=True)

# STEP 5 — Generate 10 AFTER samples and compare
print("\n=== AFTER RLHF — Auto Scoring ===", flush=True)
model_transformer.eval()
after_rewards = []

with torch.no_grad():
    for i in range(10):
        torch.manual_seed(i * 13)
        tokens = model_transformer.generate(
            max_length=64, temperature=1.0, device=device)
        piano_roll = torch.zeros(1, 128, 64).to(device)
        for t, tok in enumerate(tokens[:64]):
            piano_roll[0, tok.item(), t] = 1.0
        reward = reward_model(piano_roll).item()
        after_rewards.append(reward)
        piano_roll_np = piano_roll.squeeze().cpu().numpy()
        midi = piano_roll_to_midi(piano_roll_np, fs=16, tempo=120)
        midi.write(
            f"outputs/midi/rlhf/rlhf_after_{i+1:02d}.mid")
        print(f"After Sample {i+1:02d} | "
              f"Reward Score: {reward:.4f}", flush=True)

avg_before = sum(human_ratings) / len(human_ratings)
avg_after = sum(after_rewards) / len(after_rewards)
print(f"\n{'='*40}", flush=True)
print(f"Avg Human Score BEFORE: {avg_before:.4f}", flush=True)
print(f"Avg Model Score AFTER:  {avg_after:.4f}", flush=True)
print(f"Improvement:            {avg_after - avg_before:+.4f}", flush=True)

torch.save(model_transformer.state_dict(),
           'outputs/rlhf_checkpoint.pth')
torch.save(reward_model.state_dict(),
           'outputs/reward_model_checkpoint.pth')
print("Both checkpoints saved!", flush=True)
