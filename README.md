# RAAM-Net

**Role-Aware Asymmetric Network for Hyperspectral Image Change Detection**

RAAM-Net detects changes from full-scene bitemporal hyperspectral images.
It uses complementary reference and discrepancy representations, asymmetric
spatial modeling, and progressive cross-role fusion.

## Architecture

```text
T1, T2 -> shared spectral calibration -> complementary role cues
  Reference: concat(T1,T2) -> projection -> BDDM -> Lr
  Discrepancy: abs(T2-T1) -> projection -> dual-axis BiMamba -> Gc
  Enhanced-level fusion: ConvFFM(Gc,Lr)
  Original-level fusion: ConvFFM(Fc0,Fr0)
  Progressive cross-role fusion -> pixel-wise prediction
```

BDDM refines directional detail in the reference features. Role-conditioned
horizontal and vertical BiMamba paths aggregate discrepancy context. Two
independent convolutional fusion blocks form the enhanced- and original-level
bases. Residual corrections use same-role feature interactions, a learned
role gate, and channel/spatial selection.

## Installation

Install a CUDA-compatible PyTorch build and Mamba dependencies, then run:

```bash
pip install -r requirements.txt
pip install -e .
```

The experiments used Python 3.10, PyTorch 2.4.1+cu118, CUDA 11.8,
cuDNN 90100, and an NVIDIA RTX 3090. Build `mamba-ssm` and `causal-conv1d`
for the Python/PyTorch/CUDA combination in your environment.

## Data preparation

Place the MAT files under `data/` and set their locations in
`configs/Farmland.json`, `configs/USA.json`, and `configs/River.json`.
Each config specifies MAT variable names, label mapping, normalization,
and checksums for the data used in the experiments. For USA, raw label
**1 denotes changed pixels and 2 denotes unchanged pixels**.

Training, validation and test indices for seeds 42-51 are provided in
`reproducibility/splits/`. Pass the corresponding file with `--split`.

| Dataset | Spatial size | Bands | Train | Validation | Test |
|---|---|---:|---:|---:|---:|
| Farmland | 450 x 140 | 155 | 631 | 631 | 61,738 |
| USA | 307 x 241 | 154 | 741 | 741 | 72,505 |
| River | 463 x 241 | 198 | 1,116 | 1,116 | 109,351 |

Each class contributes approximately 1% training and 1% validation pixels.
The remaining pixels are used for testing. These are random pixel partitions
within each scene; normalization uses the full unlabeled image pair.
Dataset imagery and model weights are not included in this repository.

## Training

```bash
python train.py --scene configs/Farmland.json --seed 42 \
  --split reproducibility/splits/Farmland/seed42.npz \
  --output-dir runs/Farmland_seed42
```

The default settings are 300 epochs, full-scene FP32 training, feature width
64, Adam with learning rate `1e-4`, and StepLR with step size 30 and factor
0.9. Training minimizes unweighted cross-entropy. The checkpoint with the
lowest validation loss is selected for testing; predictions use `argmax`.
CUDA matmul TF32 is disabled and cuDNN TF32 is enabled.

For another dataset or seed, change the config, split file and output folder.
The trainer saves checkpoints, predictions, metrics, and data/split checksums.

## Evaluation

```bash
python evaluate.py --scene configs/Farmland.json \
  --checkpoint runs/Farmland_seed42/checkpoints/best_seed42.pt \
  --split reproducibility/splits/Farmland/seed42.npz
```

The loader checks tensor shapes and loads all model parameters strictly.
Aggregate results across seeds with:

```bash
python summarize.py --results runs/Farmland_seed*/results/seed*.json
```

## Results

Changed-class F1, mean and sample standard deviation over seeds 42-51:

| Dataset | F1 (%) | Parameters |
|---|---:|---:|
| Farmland | 96.7191 +/- 0.2249 | 213,000 |
| USA | 95.4455 +/- 0.3028 | 212,808 |
| River | 83.0049 +/- 1.4749 | 221,256 |

The corresponding confusion counts, data/split checksums and verification
records are available under [`reproducibility/`](reproducibility/).

## Checks and profiling

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tests/test_model.py
python profile_complexity.py
```

The analytical profiler counts convolutions and dense projections, including
Mamba projections and its depthwise sequence convolution. It excludes the
selective-scan recurrence and reports FLOPs as twice MACs.

## Repository structure

```text
raamnet/network.py       RAAM-Net architecture
raamnet/_layers.py       shared model layers and initialization
raamnet/data.py          data loading, normalization and partitions
raamnet/checkpoints.py   checkpoint loading
train.py                training entry point
evaluate.py             evaluation entry point
configs/                dataset configurations
reproducibility/        split indices, results and verification records
tests/                  model and data checks
```
