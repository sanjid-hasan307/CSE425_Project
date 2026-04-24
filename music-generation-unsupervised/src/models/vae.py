"""
src/models/vae.py
==================
Variational Autoencoder (VAE) for multi-genre music generation.

Architecture
------------
  Encoder : Bidirectional LSTM  →  two Linear heads  →  μ and log σ²
  Reparameterise : z = μ + σ ⊙ ε,  ε ~ N(0, I)
  Decoder : LSTM (conditioned on z)  →  sigmoid output

Loss (ELBO)
-----------
  L_VAE = L_recon + β · KL(q(z|X) ‖ p(z))

  where:
    L_recon = BCE(x̂, x)           (binary cross-entropy)
    KL      = −½ Σ (1 + log σ² − μ² − σ²)
    β       : KL weight (β-VAE formulation, default β=1)

Reference: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAEEncoder(nn.Module):
    """
    Bidirectional LSTM encoder that outputs (μ, log σ²).

    Input  : x  — (batch, seq_len, 128)
    Output : mu — (batch, latent_dim)
             logvar — (batch, latent_dim)
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
        feat_dim = hidden_dim * 2   # bidirectional

        # Two separate heads for μ and log σ²
        self.fc_mu     = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, 128)

        Returns
        -------
        mu     : (batch, latent_dim)
        logvar : (batch, latent_dim)
        """
        _, (h_n, _) = self.lstm(x)
        # Take last layer's bidirectional hidden states
        h_fwd  = h_n[-2]                        # (batch, hidden_dim)
        h_bwd  = h_n[-1]                        # (batch, hidden_dim)
        h_cat  = torch.cat([h_fwd, h_bwd], -1)  # (batch, hidden_dim*2)

        mu     = self.fc_mu(h_cat)              # (batch, latent_dim)
        logvar = self.fc_logvar(h_cat)          # (batch, latent_dim)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    LSTM decoder conditioned on a sample from q(z|X).

    Input  : z     — (batch, latent_dim)
    Output : x_hat — (batch, seq_len, 128) in [0, 1]
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
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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
        z_proj     = self.latent_proj(z)                              # (batch, H)
        lstm_input = z_proj.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, T, H)
        out, _     = self.lstm(lstm_input)                            # (batch, T, H)
        x_hat      = self.output_layer(out)                           # (batch, T, 128)
        return x_hat


class MusicVAE(nn.Module):
    """
    Full β-VAE with LSTM encoder and decoder for piano-roll music.

    Parameters
    ----------
    input_dim  : number of MIDI pitches (128)
    hidden_dim : LSTM hidden units
    latent_dim : dimensionality of latent space z
    num_layers : LSTM depth
    dropout    : dropout probability inside LSTM
    seq_len    : sequence length T (must match training windows)
    beta       : KL weight in the ELBO (β=1 → standard VAE)
    """

    def __init__(
        self,
        input_dim:  int   = 128,
        hidden_dim: int   = 256,
        latent_dim: int   = 64,
        num_layers: int   = 2,
        dropout:    float = 0.2,
        seq_len:    int   = 64,
        beta:       float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta

        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim, num_layers, dropout, seq_len)

    # ── Core operations ────────────────────────────────────────────────────

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick:
          z = μ + σ ⊙ ε,   ε ~ N(0, I)

        Uses σ = exp(0.5 · log σ²) for numerical stability.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) for input x."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent sample z to reconstructed piano-roll."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Parameters
        ----------
        x : (batch, seq_len, 128)

        Returns
        -------
        x_hat  : (batch, seq_len, 128)   — reconstruction
        mu     : (batch, latent_dim)
        logvar : (batch, latent_dim)
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        x_hat      = self.decode(z)
        return x_hat, mu, logvar

    # ── Loss ───────────────────────────────────────────────────────────────

    def loss(
        self,
        x:      torch.Tensor,
        x_hat:  torch.Tensor,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute β-VAE loss components.

        L_VAE = L_recon + β · KL

        where:
          L_recon = BCE(x̂, x)   (averaged over all elements)
          KL      = −½ Σ (1 + log σ² − μ² − σ²)   (normalised by batch size)

        Returns
        -------
        total_loss : scalar Tensor
        recon_loss : scalar Tensor  (for logging)
        kl_loss    : scalar Tensor  (for logging)
        """
        # Binary cross-entropy reconstruction loss
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="mean")

        # KL divergence: −½ Σ (1 + log σ² − μ² − exp(log σ²))
        kl_loss = -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp()
        )

        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    # ── Generation ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, num_samples: int = 1, device: str = "cpu") -> torch.Tensor:
        """
        Sample from the prior p(z) = N(0, I) and decode.

        Returns
        -------
        x_hat : (num_samples, seq_len, 128)
        """
        z     = torch.randn(num_samples, self.latent_dim, device=device)
        x_hat = self.decode(z)
        return x_hat

    @torch.no_grad()
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        steps: int = 8,
    ) -> torch.Tensor:
        """
        Linearly interpolate between the latent codes of two inputs.

        Parameters
        ----------
        x1, x2 : (1, seq_len, 128)  — source piano-rolls
        steps   : number of interpolation steps (including endpoints)

        Returns
        -------
        sequence : (steps, seq_len, 128)
        """
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)

        alphas = torch.linspace(0.0, 1.0, steps, device=x1.device)
        results = []
        for alpha in alphas:
            z_interp = (1.0 - alpha) * mu1 + alpha * mu2
            results.append(self.decode(z_interp))

        return torch.cat(results, dim=0)   # (steps, seq_len, 128)
