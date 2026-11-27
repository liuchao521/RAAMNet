# RAAM-Net: complete Stage39 P_full

Full-scene binary hyperspectral image change detection with the **validated
Stage39 P_full** architecture. This release has one public model and one
training/evaluation protocol. It does not expose experimental fusion or
operator-assignment switches.

## Architecture

```text
T1, T2 -> shared temporal-aware spectral gate -> complementary role cues
  reference: concat(T1,T2) -> projection -> feature-only BDDM -> Lr
  discrepancy: abs(T2-T1) -> projection -> role-conditioned dual-axis BiMamba -> Gc
  enhanced base Bc = ConvFFM(Gc,Lr)
  original base Br = ConvFFM(Fc0,Fr0)
  original gated/selected PRCF -> concatenate -> pixel classifier
```

BDDM computes directional magnitudes from the reference feature `Fr0`; it
has no separate temporal projection. Each independent ConvFFM uses a
balanced concatenation projection followed by depthwise-convolutional
residual refinement. PRCF constructs interactions from
`Ar=(Fr0+Lr)/2`, `Ac=(Gc+Fc0)/2`, `D=abs(Ac-Ar)`, and `A=Ac*Ar`.
Its learned role gate and channel/spatial selector scale the two residual
corrections. There are no additional interaction-projection layers.

`raamnet/stage39.py` contains the active full-model implementation.
`raamnet/_stage39_base.py` is the unchanged historical base snapshot. Its
initialization order and parameter names are retained so original Stage39
checkpoints load strictly and matched seeds start from the same tensors.
The reference BDDM and both fusion modules are replaced during construction;
the historical temporary modules do not remain in the final model.

## Installation

Use a CUDA-compatible PyTorch and mamba-ssm installation, then run:

```bash
pip install -r requirements.txt
pip install -e .
```

The archived experiment environment was Python 3.10, PyTorch 2.4.1+cu118,
CUDA 11.8, cuDNN 90100, and an NVIDIA RTX 3090. Build Mamba and causal-conv1d
for the same Python/PyTorch/CUDA combination. Full forward/backward execution
requires a supported CUDA device.

## Data and frozen splits

Place the original MAT files under `data/` and edit only the file locations
in `configs/Farmland.json`, `configs/USA.json`, or `configs/River.json`.
The configs specify the actual MAT variable names, raw positive labels,
per-date/per-band normalization and expected normalized-array hashes.
In USA, raw label **1 is changed and 2 is unchanged**. The loader rejects
hash mismatches rather than silently evaluating another data version.

`reproducibility/splits/` contains the 30 original full-model index files
(three scenes, seeds 42-51). Use `--split` to reproduce a reported run.
The default split generator also follows Stage39's Python `random.Random`
and per-class `ceil` policy; identical seed numbers alone do not make NumPy
random splits equivalent.

| Scene | H x W x bands | Train | Validation | Test |
|---|---|---:|---:|---:|
| Farmland | 450 x 140 x 155 | 631 | 631 | 61,738 |
| USA | 307 x 241 x 154 | 741 | 741 | 72,505 |
| River | 463 x 241 x 198 | 1,116 | 1,116 | 109,351 |

Random pixel partitions may share spatial context. Normalization uses the
full unlabeled scene. These are within-scene results, not geographically
independent generalization estimates. Dataset imagery and weights are not
bundled in this source repository.

## Train the complete model

```bash
python train.py --scene configs/Farmland.json --seed 42 \
  --split reproducibility/splits/Farmland/seed42.npz \
  --output-dir runs/Farmland_seed42
```

Defaults match the complete Stage39 protocol: **300 epochs**, full-scene
FP32, **Adam 1e-4**, no class weighting, StepLR(30,0.9), width 64,
minimum validation cross-entropy checkpoint selection, and argmax prediction.
No test threshold tuning, test-time augmentation, or postprocessing is used.
CUDA matmul TF32 is disabled and cuDNN TF32 is enabled, matching the archive.

Outputs include the best and periodic checkpoints, exact split arrays,
full-scene prediction, five test metrics, model-version metadata, and input
and index hashes. Existing best checkpoints are protected from overwrite.
Use the corresponding config/split/output path for USA or River and seeds
43-51. A different learning rate, input or split defines a different run.

## Evaluate release or original Stage39 weights

```bash
python evaluate.py --scene configs/Farmland.json \
  --checkpoint runs/Farmland_seed42/checkpoints/best_seed42.pt \
  --split reproducibility/splits/Farmland/seed42.npz
```

Original Stage39 dictionaries with `model_state_dict` and the release
checkpoint format are supported. Loading is strict; original pre-Stage39 or
projected-fusion weights are not silently adapted. Five metrics are printed
as fractions. Aggregate release training outputs using:

```bash
python summarize.py --results runs/Farmland_seed*/results/seed*.json
```

## Recorded complete-model results

Mean and sample standard deviation over seeds 42-51; F1 is in percent.
Raw counts for these 30 actual archived evaluations are provided in
`reproducibility/full_model_10seed_confusions.csv`.

| Scene | Changed-class F1 (%) | Parameters |
|---|---:|---:|
| Farmland | 96.7191 +/- 0.2249 | 213,000 |
| USA | 95.4455 +/- 0.3028 | 212,808 |
| River | 83.0049 +/- 1.4749 | 221,256 |

These are recorded results, not a claim that retraining on every software or
GPU version is bit-identical. Source SHA256 values and the frozen data/split
manifests are included under `reproducibility/`.

## Verification and profiling

Actual release checks passed on the RTX 3090: 30 original checkpoints loaded
strictly, all 30 generated/saved splits matched, and each scene's seed-42
full prediction map had zero differences from its archived map. Initialization
and RNG states matched exactly; training logits/loss were bit-identical.
Gradient differences were about 1e-9, comparable to original-model CUDA
repeat differences. Training/evaluation CLI smoke checks also passed.
See [the verification record](reproducibility/release_verification.json).

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tests/test_model.py
python profile_complexity.py
```

With the original immutable Stage39 source available, compare initialization,
RNG state, training logits, loss and gradients on an idle CUDA device:

```bash
python tests/verify_stage39_reference.py --reference-root /path/to/Stage39
```

`profile_complexity.py` counts convolution and dense-projection operations,
including Mamba projections/convolution, but excludes selective-scan
recurrence. FLOPs = 2 x MACs; normalization, nonlinearities and elementwise
operations are outside this analytical convention. `profile_legacy_thop.py`
is a separately labeled legacy counting utility, not an inference benchmark.

## Repository layout

```text
raamnet/stage39.py          complete Stage39 model
raamnet/_stage39_base.py    immutable historical initialization/base layers
raamnet/data.py             normalization, hashes and original split policy
raamnet/checkpoints.py      strict original/release checkpoint loader
train.py / evaluate.py      full-model training and evaluation
configs/                   three verified scene configurations
reproducibility/            full-model counts, source/data hashes and 30 splits
tests/                     partition, model and original-source parity checks
```
