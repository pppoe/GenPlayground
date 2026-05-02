# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup** (uses `uv` as package manager):
```bash
uv sync
```

**Create dataset** (MNIST → 2D point clouds, saves `mnist_2d_pc.pt`):
```bash
python create_dataset.py
```

**Train** (requires `mnist_2d_pc.pt`, saves `model.pt`):
```bash
python train.py
```

**Generate samples** (requires `model.pt`, saves `data/samples.jpg`):
```bash
python gen.py
```

**Visualize dataset**:
```bash
python render_tensor.py
```

There are no tests.

## Architecture

This project implements **flow matching** to generate 2D point clouds of MNIST digits (128 points per cloud).

### Data pipeline
`create_dataset.py` downloads MNIST and converts each image into a fixed-size 2D point cloud by thresholding pixels and sampling coordinates. Output tensor shape: `(N, 128, 2)`, stored in `mnist_2d_pc.pt` with keys `data` and `labels`.

### Model (`gen.py`)
- `SinusoidalEmbedding` — encodes continuous timestep `t ∈ [0,1]` into a high-dimensional vector
- `TransformerBlock` — standard multi-head self-attention + FFN applied over the point set
- `PointFlowModel` — stacks multiple `TransformerBlock`s; input is `(B, N, 2)` points concatenated with time embeddings; outputs a velocity field of the same shape
- `sample()` — Euler integration from Gaussian noise to data distribution using the trained velocity field

### Training (`train.py`)
Flow matching objective: at each step, sample random `t ~ U[0,1]`, interpolate `x_t = (1-t)*x0 + t*x1` between real data `x0` and noise `x1`, and train `PointFlowModel` to predict the ground-truth velocity `x0 - x1`. Trained for 200 epochs; saves `model.pt`.

### Artifacts
| File | Description |
|---|---|
| `mnist_2d_pc.pt` | Preprocessed dataset |
| `model.pt` | Trained model weights |
| `data/samples.jpg` | 4×4 grid of generated point clouds |
| `data/tensor_vis.jpg` | Dataset visualization |
