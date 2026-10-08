# RAAM-Net: Role-Aware Asymmetric Network for Hyperspectral Image Change Detection

This repository contains a clean, self-contained implementation of the final
RAAM-Net model for binary bitemporal hyperspectral image change detection.
It contains no exploratory `v*`, `trial`, or dataset-specific experiment code.

## Architecture

```text
Date 1, Date 2
      │
Temporal-Aware Spectral Gate (shared spectral calibration)
      │
Complementary reference/change role estimation
      │
 ┌────┴──────────────────────────┐
 │ Reference evidence             │ Change evidence
 │ BDDM: directional octave detail│ Role-conditioned dual-axis BiMamba
 └──────────────┬─────────────────┘
                │
Progressive Cross-Role Fusion (FFM + residual correction)
                │
Pixel-wise binary change logits
```

The reference branch models temporally consistent local directional detail.
The change branch uses the calibrated absolute difference and chooses between
continuous horizontal and vertical BiMamba contexts according to bitemporal
change-topology evidence. Complementary roles condition context injection but
are not interpreted as hard change labels.

## Installation

Create an environment with a PyTorch build compatible with your CUDA runtime,
then install the remaining packages:

```bash
pip install -r requirements.txt
pip install -e .
```

`mamba-ssm` must be installed for the same Python, PyTorch, CUDA, and compiler
combination that will run the experiment. Please consult the upstream
`mamba-ssm` installation instructions if a binary wheel is unavailable.

## Data configuration

Prepare each scene as two MAT cubes and one MAT label map. Copy
[`configs/scene_example.json`](configs/scene_example.json), set the file paths
and MAT variable names, then set `positive_values` explicitly. This last field
is required because some public HSI-CD datasets use a nonzero unchanged label.

Supported cube layouts are `HWC` and `CHW`. Per-band scene standardization is
enabled by default and can be disabled in the JSON configuration.

## Reproduction

The example below trains a single seed. It creates disjoint class-stratified
training (1%), validation (1%), and test splits. The validation loss selects
the best checkpoint; test labels are not used for epoch selection.

```bash
python train.py \
  --scene configs/scene_example.json \
  --output-dir runs/example_seed42 \
  --seed 42 \
  --epochs 300 \
  --save-interval 50
```

The command writes:

- `splits/seed*.npz`: the reproducible split indices;
- `checkpoints/best_seed*.pt`: validation-selected checkpoint;
- `checkpoints/epoch050...epoch300*.pt`: periodic checkpoints;
- `results/seed*.json`: OA, Kappa, F1, Precision, and Recall on the test set;
- `results/seed*_prediction.npy`: full-scene binary prediction map.

Evaluate an existing best checkpoint with:

```bash
python evaluate.py \
  --scene configs/scene_example.json \
  --checkpoint runs/example_seed42/checkpoints/best_seed42.pt \
  --split runs/example_seed42/splits/seed42.npz
```

Aggregate multiple independent seed results with:

```bash
python summarize.py --results runs/seed*/results/seed*.json
```

Measure trainable parameters and FLOPs for the four paper scenes:

```bash
python profile_complexity.py --device cuda
```

The script reports `FLOPs = 2 × MACs`. It includes Mamba's dense projections
and one-dimensional convolution but excludes the selective-scan recurrence;
use the same convention for every compared model.

Before training on a scene, verify the model environment with:

```bash
python tests/test_model.py
```

## Repository layout

```text
raamnet/
  model.py       # TASG, role estimator, BDDM, dual-axis BiMamba, PRCF
  data.py        # MAT loading, standardization, reproducible stratified split
  metrics.py     # OA, Kappa, F1, Precision, Recall
  engine.py      # training/evaluation helpers
train.py         # full-scene training entry point
evaluate.py      # checkpoint evaluation entry point
summarize.py      # mean ± sample standard deviation over seed results
profile_complexity.py  # Params, MACs, and FLOPs (selective scan excluded)
configs/         # scene configuration template
tests/           # model smoke test
```

## Notes

- The implementation performs full-scene inference. GPU memory scales with the
  image height, width, band count, and hidden dimension.
- Report mean and standard deviation over independent seeds, rather than a
  single selected test run.
- Retain the exact preprocessing, label mapping, split protocol, and model
  configuration used in any comparison for a fair reproduction.
