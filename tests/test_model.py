"""Minimal smoke test for a forward/backward pass of the open-source model."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raamnet import RAAMNet


def test_forward_backward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RAAMNet(bands=16, dim=16, d_state=8).to(device)
    first = torch.randn(1, 16, 8, 8, device=device)
    second = torch.randn(1, 16, 8, 8, device=device)
    target = torch.randint(0, 2, (1, 8, 8), device=device)
    logits = model(first, second)
    assert logits.shape == (1, 2, 8, 8)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()


if __name__ == "__main__":
    test_forward_backward()
    print("RAAM-Net model smoke test passed.")
