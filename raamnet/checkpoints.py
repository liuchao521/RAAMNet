"""Strict loading of RAAM-Net checkpoint dictionaries."""
from __future__ import annotations

from pathlib import Path
import torch

from .model import RAAMNet


def load_checkpoint(path: str | Path, bands: int, device: str | torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    # Architecture compatibility is established by tensor shapes and strict
    # state-dict loading, including checkpoints with earlier metadata labels.
    state = payload.get("model_state_dict", payload.get("model", payload))
    dim, saved_bands = state["reference_reduction.0.weight"].shape[:2]
    if saved_bands != 2 * bands:
        raise ValueError("Checkpoint spectral band count differs from the input scene")
    d_state = state["change_context.sequence_model.forward_mamba.A_log"].shape[1]
    model = RAAMNet(bands=bands, dim=dim, d_state=d_state)
    model.load_state_dict(state, strict=True)
    return model.to(device), payload
