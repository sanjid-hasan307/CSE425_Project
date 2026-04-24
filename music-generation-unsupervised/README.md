# Unsupervised Multi-Genre Music Generation

**CSE425 — Neural Networks Course Project**

> End-to-end deep learning pipeline for generating original music using MIDI data and PyTorch.  
> Three models (LSTM Autoencoder, VAE, Autoregressive Transformer) evaluated against two baselines.

---

## Project Structure

```
music-generation-unsupervised/
├── data/
│   ├── raw/
│   │   ├── maestro/          ← MAESTRO v3.0.0 MIDI files
│   │   └── lakh/             ← Lakh MIDI Dataset (matched subset)
│   └── processed/
│       ├── maestro/          ← Preprocessed tensors (.pt)
│       └── lakh/
├── notebooks/
│   └── music_generation_demo.ipynb
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py     ← MIDI → piano-roll + tokens, segmentation, split
│   │   └── midi_utils.py     ← Piano-roll / token → MIDI conversion helpers
│   ├── models/
│   │   ├── baselines.py      ← RandomNoteGenerator + MarkovChainModel
│   │   ├── autoencoder.py    ← LSTM Autoencoder (deterministic)
│   │   ├── vae.py            ← β-VAE with reparameterisation trick
│   │   └── transformer.py    ← Decoder-only autoregressive Transformer
│   ├── training/
│   │   ├── train_ae.py       ← AE training loop + MSE loss
│   │   ├── train_vae.py      ← VAE training loop + ELBO loss
│   │   └── train_transformer.py ← Transformer training + CE loss
│   ├── evaluation/
│   │   ├── metrics.py        ← Quantitative music metrics
│   │   └── evaluate_all.py   ← Full cross-model evaluation + plots
│   └── generation/
│       └── generate.py       ← Unified MIDI generation CLI
├── outputs/
│   ├── plots/                ← Loss curves, pitch histograms, metric bars
│   ├── midi/
│   │   ├── ae/               ← AE-generated MIDI
│   │   ├── vae/              ← VAE-generated MIDI + interpolations
│   │   └── transformer/      ← Transformer-generated MIDI
│   ├── baselines/            ← Random + Markov MIDI
│   └── models/               ← Saved model checkpoints (.pt)
├── report/                   ← Academic report location
├── verify_setup.py           ← Dependency verification
├── download_dataset.py       ← Dataset downloader
├── run_all.py                ← Master end-to-end pipeline
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python verify_setup.py
```

### 2. Download Datasets

```bash
# Download both datasets (~3 GB total)
python download_dataset.py --all

# Or individually:
python download_dataset.py --maestro
python download_dataset.py --lakh
```

**Manual download (if auto-download fails):**

| Dataset | URL | Target Directory |
|---------|-----|-----------------|
| MAESTRO v3.0.0 | https://magenta.tensorflow.org/datasets/maestro | `data/raw/maestro/` |
| Lakh MIDI (matched) | https://colinraffel.com/projects/lmd/ | `data/raw/lakh/lmd_matched/` |

### 3. Run the Full Pipeline

```bash
# Full pipeline (all steps, all models)
python run_all.py

# Quick test with limited data
python run_all.py --max_files 200 --epochs_ae 10 --epochs_vae 10 --epochs_tf 10
```

### 4. Run Individual Steps

```bash
# Step 3 — Preprocess MAESTRO
python -m src.preprocessing.preprocess \
    --midi_dir data/raw/maestro \
    --out_dir  data/processed/maestro \
    --window   64

# Step 4 — Baselines
python -m src.models.baselines --data_dir data/processed/maestro

# Step 5 — Train LSTM Autoencoder
python -m src.training.train_ae --data_dir data/processed/maestro --epochs 50

# Step 6 — Train VAE
python -m src.training.train_vae --data_dir data/processed/lakh --epochs 60

# Step 7 — Train Transformer
python -m src.training.train_transformer --data_dir data/processed/lakh --epochs 60

# Step 8 — Evaluate all
python -m src.evaluation.evaluate_all --data_dir data/processed/maestro

# Step 9 — Generate new samples
python -m src.generation.generate --model vae --num 10
```

---

## Models

### Baseline 1 — Random Note Generator
Samples random pitches at each time step (uniform probability).  
- No learning. Serves as lower-bound.

### Baseline 2 — Markov Chain (order-1)
Learns pitch-set transition probabilities from training data.  
- Captures local note patterns without long-range dependencies.

### Task 1 — LSTM Autoencoder

$$z = f_\phi(X), \quad \hat{X} = g_\theta(z)$$

$$\mathcal{L}_{AE} = \frac{1}{T \cdot P} \sum_{t,p} (x_{t,p} - \hat{x}_{t,p})^2$$

- Bidirectional LSTM encoder → latent code **z**
- Decoder LSTM conditioned on z
- Loss: MSE reconstruction

### Task 2 — Variational Autoencoder (β-VAE)

$$z = \mu + \sigma \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

$$\mathcal{L}_{VAE} = \mathcal{L}_{recon} + \beta \cdot KL(q(z|X) \| p(z))$$

- Encoder outputs **μ** and **log σ²**
- Reparameterisation trick for differentiable sampling
- Loss: BCE reconstruction + β-weighted KL divergence
- Supports latent interpolation between samples

### Task 3 — Autoregressive Transformer

$$p(X;\theta) = \prod_{t=1}^T p(x_t \mid x_1, \ldots, x_{t-1};\theta)$$

$$\mathcal{L}_{CE} = -\frac{1}{T} \sum_t \log p_\theta(x_t \mid x_{<t})$$

- Decoder-only architecture with causal self-attention
- Sinusoidal positional encodings
- Top-k temperature sampling for generation
- Weight-tied input/output embeddings

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Note Density** | Average fraction of active pitches per step |
| **Pitch Entropy** | Shannon entropy of pitch-class distribution (bits) |
| **Inter-Onset Interval** | Mean time between note onset events (steps) |
| **Empty Bar Ratio** | Fraction of bars containing no notes |
| **Polyphony Rate** | Fraction of steps with ≥2 simultaneous notes |
| **Unique Pitches** | Number of distinct MIDI pitches used |

---

## Outputs

After running the full pipeline:

| Output | Description |
|--------|-------------|
| `outputs/plots/ae_loss_curve.png` | AE training + val MSE loss |
| `outputs/plots/vae_combined_loss.png` | VAE total / recon / KL loss |
| `outputs/plots/transformer_loss.png` | Transformer cross-entropy loss |
| `outputs/plots/pitch_histograms.png` | Pitch class distributions by model |
| `outputs/plots/metric_comparison.png` | Bar chart of all metrics |
| `outputs/midi/ae/*.mid` | 5 AE-generated MIDI files |
| `outputs/midi/vae/*.mid` | 8 VAE-sampled MIDI files |
| `outputs/midi/vae/interpolation/*.mid` | 8-step latent interpolation |
| `outputs/midi/transformer/*.mid` | Transformer-generated MIDI files |
| `outputs/baselines/random/*.mid` | Random generator samples |
| `outputs/baselines/markov/*.mid` | Markov chain samples |
| `outputs/models/*.pt` | Trained model checkpoints |

---

## Requirements

```
torch>=2.0.0
pretty_midi>=0.2.10
miditoolkit>=0.1.16
music21>=9.1.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
requests>=2.28.0
```

---

## Academic References

1. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR 2014.
2. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
3. Hawthorne, C., et al. (2019). *Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset*. ICLR 2019.
4. Raffel, C. (2016). *Learning-Based Methods for Comparing Sequences*. PhD Thesis, Columbia University (Lakh MIDI Dataset).
5. Higgins, I., et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR 2017.
