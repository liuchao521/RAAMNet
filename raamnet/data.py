"""Dataset loading and reproducible full-scene splits for RAAM-Net."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io as sio


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_mat(path: str | Path, key: str) -> np.ndarray:
    payload = sio.loadmat(path, variable_names=[key])
    if key not in payload:
        available = sorted(name for name in payload if not name.startswith("__"))
        raise KeyError(f"{key!r} is not present in {path}; available keys: {available}")
    return np.asarray(payload[key])


def _as_hwc(cube: np.ndarray, layout: str) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"HSI cube must be 3D, got {cube.shape}")
    if layout.upper() == "HWC":
        return cube
    if layout.upper() == "CHW":
        return np.moveaxis(cube, 0, -1)
    raise ValueError("cube_layout must be either 'HWC' or 'CHW'")


def standardize_bands(cube: np.ndarray) -> np.ndarray:
    """Per-band scene normalization used by the paper experiments."""
    cube = np.asarray(cube, dtype=np.float32)
    flattened = cube.reshape(-1, cube.shape[-1])
    mean = flattened.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flattened.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((cube - mean) / std).astype(np.float32, copy=False)


def load_scene(config_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one HSI pair according to a JSON scene configuration.

    The label mapping is deliberately explicit. Set ``positive_values`` to the
    raw label values that denote changed pixels; this prevents accidental
    reversal of datasets whose unchanged label is nonzero.
    """
    config_path = Path(config_path)
    config = load_json(config_path)
    root = config_path.parent

    def resolve(name: str) -> Path:
        path = Path(config[name])
        return path if path.is_absolute() else root / path

    layout = config.get("cube_layout", "HWC")
    date1 = _as_hwc(_read_mat(resolve("date1_path"), config["date1_key"]), layout)
    date2 = _as_hwc(_read_mat(resolve("date2_path"), config["date2_key"]), layout)
    raw_labels = np.asarray(_read_mat(resolve("label_path"), config["label_key"])).squeeze()
    if date1.shape != date2.shape:
        raise ValueError(f"temporal cubes have different shapes: {date1.shape} and {date2.shape}")
    if raw_labels.shape != date1.shape[:2]:
        raise ValueError(f"label shape {raw_labels.shape} does not match spatial shape {date1.shape[:2]}")
    if "positive_values" not in config:
        raise KeyError("scene config must explicitly provide positive_values")
    labels = np.isin(raw_labels, np.asarray(config["positive_values"])).astype(np.int64)
    if np.unique(labels).size != 2:
        raise ValueError("the chosen positive_values do not produce two classes")
    if config.get("standardize", True):
        date1, date2 = standardize_bands(date1), standardize_bands(date2)
    return date1, date2, labels


def stratified_split(labels: np.ndarray, train_ratio: float, val_ratio: float,
                     seed: int) -> dict[str, np.ndarray]:
    """Create disjoint full-scene indices for train, validation, and test."""
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than one")
    flattened = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for value in (0, 1):
        index = np.flatnonzero(flattened == value)
        rng.shuffle(index)
        count_train = max(1, int(round(index.size * train_ratio)))
        count_val = max(1, int(round(index.size * val_ratio)))
        if count_train + count_val >= index.size:
            raise ValueError(f"class {value} is too small for the requested split")
        train.append(index[:count_train])
        validation.append(index[count_train:count_train + count_val])
        test.append(index[count_train + count_val:])
    return {
        "train": np.sort(np.concatenate(train)).astype(np.int64),
        "val": np.sort(np.concatenate(validation)).astype(np.int64),
        "test": np.sort(np.concatenate(test)).astype(np.int64),
    }
