"""Training and evaluation utilities for full-scene binary HSI-CD."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import binary_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)

def flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    """[B, K, H, W] -> [B*H*W, K]."""
    return logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])


@torch.no_grad()
def evaluate(model: torch.nn.Module, date1: torch.Tensor, date2: torch.Tensor,
             labels: torch.Tensor, index: torch.Tensor) -> tuple[float, dict[str, float], np.ndarray]:
    model.eval()
    logits = flatten_logits(model(date1, date2))
    loss = F.cross_entropy(logits.index_select(0, index), labels.index_select(0, index)).item()
    prediction = logits.argmax(dim=1)
    metric = binary_metrics(
        prediction.index_select(0, index).cpu().numpy(),
        labels.index_select(0, index).cpu().numpy(),
    )
    return loss, metric, prediction.cpu().numpy()


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
