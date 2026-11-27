"""Evaluate a saved RAAM-Net checkpoint on its held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from raamnet import MODEL_VERSION
from raamnet.checkpoints import load_stage39_checkpoint
from raamnet.data import load_scene, load_split
from raamnet.engine import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a RAAM-Net checkpoint")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, help=".npz split file produced by train.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cube1, cube2, label_map = load_scene(args.scene)
    device = torch.device(args.device)
    model, checkpoint = load_stage39_checkpoint(args.checkpoint, cube1.shape[-1], device)
    date1 = torch.from_numpy(cube1).permute(2, 0, 1).unsqueeze(0).to(device)
    date2 = torch.from_numpy(cube2).permute(2, 0, 1).unsqueeze(0).to(device)
    labels = torch.from_numpy(label_map.reshape(-1)).long().to(device)
    split = load_split(args.split, label_map.size)
    test = torch.from_numpy(split["test"]).long().to(device)
    loss, metrics, _ = evaluate(model, date1, date2, labels, test)
    print(json.dumps({"model_version": MODEL_VERSION, "test_loss": loss, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
