"""
src/models/autoencoder.py
==========================
LSTM Autoencoder for music piano-roll reconstruction.

Architecture
------------
  Encoder : Bidirectional LSTM  →  Linear projection  →  latent z
  Decoder : LSTM (conditioned on z repeated T times)  →  Linear  →  sigmoid

Loss
----
  L_AE = (1/T·P) Σ_t Σ_p (x_{t,p} − x̂_{t,p})²     (MSE)

where T = sequence length, P = 128 (MIDI pitches).

Mathematical formulation
------------------------
  z   = f_φ(X)   =  LSTM_enc(X) → last hidden state, then projected
  X̂  = g_θ(z)   =  LSTM_dec(z  repeated over T steps) → sigmoid activations
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder.

    Input  : x  — (batch, seq_len, 128)
    Output : z  — (batch, latent_dim)
    """

    def __init__(
        self,
        input_dim:  int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout:    float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # After bidirectional LSTM the last hidden state has size hidden_dim*2
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, 128)

        Returns
        -------
        z : (batch, latent_dim)
        """
        _, (h_n, _) = self.lstm(x)
        # h_n : (num_layers * 2, batch, hidden_dim)
        # Take the last layer's forward and backward hidden states
        h_fwd = h_n[-2]  # (batch, hidden_dim)
        h_bwd = h_n[-1]  # (batch, hidden_dim)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (batch, hidden_dim*2)
        z = self.proj(h_cat)                        # (batch, latent_dim)
        return z


class LSTMDecoder(nn.Module):
    """
    LSTM decoder that reconstructs the piano-roll from latent code z.

    Strategy: z is projected and then repeated `seq_len` times to form the
    LSTM input sequence (teacher-forcing not used during generation).

    Input  : z       — (batch, latent_dim)
    Output : x_hat   — (batch, seq_len, 128) in [0, 1]
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout:    float = 0.2,
        seq_len:    int = 64,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project latent to input sequence
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            # PROBLEM 4 FIX: Ensure final layer is Sigmoid for piano-roll reconstruction
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (batch, latent_dim)

        Returns
        -------
        x_hat : (batch, seq_len, 128) in [0, 1]
        """
        batch = z.size(0)
        # Project z to hidden dim, then repeat across time
        z_proj = self.latent_proj(z)                           # (batch, hidden_dim)
        lstm_input = z_proj.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, T, hidden_dim)

        out, _ = self.lstm(lstm_input)                         # (batch, T, hidden_dim)
        x_hat  = self.output_layer(out)                        # (batch, T, 128)
        return x_hat


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder combining Encoder and Decoder.

    Loss
    ----
    L_AE = MSELoss(x̂, x)

    The latent space is a deterministic bottleneck (no stochasticity).
    """

    def __init__(
        self,
        input_dim:  int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout:    float = 0.2,
        seq_len:    int = 64,
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, num_layers, dropout, seq_len)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent code z."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code z to reconstructed piano-roll."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Parameters
        ----------
        x : (batch, seq_len, 128)

        Returns
        -------
        x_hat : (batch, seq_len, 128)  — reconstructed piano-roll
        z     : (batch, latent_dim)    — latent code
        """
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @staticmethod
    def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Error reconstruction loss.

        L_AE = (1 / (T × P)) Σ ||x_t − x̂_t||²
        """
        return nn.functional.mse_loss(x_hat, x, reduction="mean")
