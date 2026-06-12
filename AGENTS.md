# AGENTS.md

## Project Overview

PixelTileGAN: PyTorch research project generating seamless 32x32 pixel-art terrain tiles via VQ-VAE + GAN pipeline. Chinese-language comments and docs throughout.

Pipeline: raw tile PNGs → preprocessing → VQ-VAE training → latent-space GAN → texture generation → palette remapping → game engine import (Godot 4).

## Architecture

```
models/                     # All model definitions (PyTorch nn.Modules)
  vq_vae_v2.py              # Current VQ-VAE architecture (used by v5–v7)
  autoencoder*.py            # Earlier autoencoder variants
  conditional_gan.py         # Conditional GAN
  conditional_wgan_gp.py     # WGAN-GP variant

scripts/
  training/                  # One training script per experiment version
    train_vqvae_v7.py        # Latest VQ-VAE experiment (v7)
    train_vqvae_v5.py        # Stable reference VQ-VAE (v5)
  data/                      # Dataset preprocessing, quantization, extraction
  tools/                     # Visualization, analysis, web UIs
    texture_browser.py       # Flask web app (port 5002) for interactive texture generation
    review_server.py         # Flask web app (port 5000) for dataset review
    palette_remapper.py      # Remap quantized textures between palettes

palettes/                    # JSON color palettes (grass, stone, dirt, etc.)
datasets/                    # Training data (gitignored, must exist locally)
checkpoints/                 # Model checkpoints (gitignored)
generated_textures/          # Output textures (gitignored)
```

## Key Commands

Training scripts are run directly — no wrapper CLI, no Makefile:

```bash
python scripts/training/train_vqvae_v7.py
python scripts/training/train_vqvae_v5.py
```

Each training script has a `CONFIG` dict at the top that controls all hyperparameters, data paths, checkpoint paths, and device selection. Edit CONFIG in-script to change behavior.

Web tools use Flask:

```bash
python scripts/tools/texture_browser.py   # port 5002
python scripts/tools/review_server.py     # port 5000
```

## Environment

- Python + PyTorch + torchvision + PIL + numpy
- Optional: Flask (texture browser / review server), scikit-learn (PCA analysis), OpenCV
- Uses conda (`.claude/settings.local.json` permits `conda run *`)
- `requirements.txt` only lists scraping deps (requests, beautifulsoup4); torch/torchvision are not pinned there
- Windows platform; all scripts use `Path` correctly (forward slashes work)
- CUDA preferred; scripts fall back to CPU via `"cuda" if torch.cuda.is_available() else "cpu"`

## Data & Checkpoints

- `datasets/` and `checkpoints/` are gitignored — not in repo
- Training data format: 32x32 RGBA PNGs, quantized to palette colors
- Data path configured per-script in CONFIG dict (e.g. `datasets/classified/pixel_32_quantized`)
- Checkpoint path also in CONFIG (e.g. `checkpoints/vqvae_v7/`)
- Texture browser defaults to `checkpoints/vqvae_v5/vqvae_v5_best.pth` — update CONFIG if using a different version

## Model Versioning Convention

Multiple experiment versions coexist as separate files, not branches:
- `vq_vae.py` → v1, `vq_vae_v2.py` → v2+ architecture
- Training scripts: `train_vqvae.py` through `train_vqvae_v7.py`
- v5 is the stable reference; v7 is the latest experiment (same architecture as v5, different data scale)
- Each version has its own checkpoint subdirectory

## Common Patterns

- Every script adds `project_root` to `sys.path` at startup for cross-directory imports
- Model classes use GroupNorm (not BatchNorm), ResidualBlocks, circular padding for seamless tiles
- Input is 4-channel RGBA, not 3-channel RGB
- Palette JSON format: `{"name": "...", "colors": ["#RRGGBB", ...]}`

## Gotchas

- No test suite, no linting, no type checking — this is a research codebase
- No `__init__.py` files; imports rely on `sys.path` manipulation
- Generated texture filenames encode latent index and perturbation strength: `texture_{idx}_{strength}_{random}.png`
- The `scripts/tools/shader_graph/` subdirectory contains Godot shader integration code
