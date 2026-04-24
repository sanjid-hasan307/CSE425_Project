"""
src/models/transformer.py
==========================
Autoregressive Transformer decoder for token-based music generation.

Architecture
------------
  - Learnable token + positional embeddings
  - Stack of N causal self-attention (decoder-only) layers
  - Linear output head → vocabulary logits

Autoregressive factorisation
-----------------------------
  p(X) = ∏_{t=1}^{T}  p(x_t | x_{<t} ; θ)

Training objective : Cross-Entropy (next-token prediction / teacher forcing)
  L_CE = −(1/T) Σ_t log p_θ(x_t | x_{<t})

Causal mask ensures each position only attends to itself and prior positions,
so the model cannot "look ahead" during training.

Reference: Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ── Positional Encoding ────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                      # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to token embeddings."""
        # x : (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ── Causal Self-Attention Layer ────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal (lower-triangular) mask.

    Ensures p(x_t | x_{<t}) through masking: position t cannot attend
    to positions t+1, t+2, … (no look-ahead).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.attn    = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        out : (batch, seq_len, d_model)
        """
        T = x.size(1)
        # Build causal mask: True means "ignore this position"
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        return self.dropout(out)


# ── Feed-Forward Block ─────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Position-wise fully connected feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Transformer Decoder Layer ──────────────────────────────────────────────

class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder-only layer with:
      - Pre-norm (more stable training)
      - Causal self-attention
      - Feed-forward sublayer
      - Residual connections
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        d_ff:     int,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.attn    = CausalSelfAttention(d_model, n_heads, dropout)
        self.ff      = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention with residual
        x = x + self.attn(self.norm1(x))
        # Pre-norm feed-forward with residual
        x = x + self.ff(self.norm2(x))
        return x


# ── Full Autoregressive Transformer ───────────────────────────────────────

class MusicTransformer(nn.Module):
    """
    Decoder-only Transformer for autoregressive music token generation.

    Autoregressive factorisation:
        p(X; θ) = ∏_{t=1}^{T}  p(x_t | x_1, …, x_{t-1} ; θ)

    Training: cross-entropy between predicted logits and next token.

    Parameters
    ----------
    vocab_size  : total token vocabulary size (see preprocess.py)
    d_model     : embedding / model dimension
    n_heads     : number of attention heads
    n_layers    : number of Transformer decoder layers
    d_ff        : feed-forward intermediate dimension
    max_seq_len : maximum sequence length the model can handle
    dropout     : dropout probability
    """

    def __init__(
        self,
        vocab_size:  int   = 417,   # 5 special + 128 note_on + 128 note_off + 32 vel + 128 time_shift + 1 padding
        d_model:     int   = 256,
        n_heads:     int   = 8,
        n_layers:    int   = 4,
        d_ff:        int   = 512,
        max_seq_len: int   = 512,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc     = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm      = nn.LayerNorm(d_model)
        self.output    = nn.Linear(d_model, vocab_size)

        # Weight tying: share embedding and output weights (common practice)
        self.output.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise parameters with small normal values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len)  — token indices

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        """
        emb    = self.token_embed(x)    # (batch, T, d_model)
        emb    = self.pos_enc(emb)      # + positional

        out    = emb
        for layer in self.layers:
            out = layer(out)

        out    = self.norm(out)
        logits = self.output(out)       # (batch, T, vocab_size)
        return logits

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = 0,
    ) -> torch.Tensor:
        """
        Cross-entropy loss for next-token prediction.

        L_CE = −(1/T) Σ_t log p_θ(x_t | x_{<t})

        Parameters
        ----------
        logits  : (batch, seq_len, vocab_size)
        targets : (batch, seq_len) — ground-truth next tokens
        """
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=ignore_index,
        )

    # ── Autoregressive Generation ─────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        """
        Autoregressive sampling via temperature-scaled top-k sampling.

        Parameters
        ----------
        prompt         : (1, prompt_len) — seed token sequence
        max_new_tokens : number of tokens to generate beyond prompt
        temperature    : softmax temperature (lower = more peaked)
        top_k          : restrict sampling to top-k tokens (None = no restriction)

        Returns
        -------
        generated : (1, prompt_len + max_new_tokens)
        """
        self.eval()
        generated = prompt.clone()  # (1, T_prompt)

        for _ in range(max_new_tokens):
            # Forward pass on the current context
            logits = self.forward(generated)        # (1, T, vocab_size)
            last_logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            # Top-k filtering
            if top_k is not None:
                top_k_clamped = min(top_k, last_logits.size(-1))
                values, _ = torch.topk(last_logits, top_k_clamped)
                threshold = values[:, -1].unsqueeze(-1)
                last_logits = last_logits.masked_fill(
                    last_logits < threshold, float("-inf")
                )

            probs    = F.softmax(last_logits, dim=-1)     # (1, vocab_size)
            next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated
