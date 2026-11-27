"""Train RAAM-Net on one bitemporal HSI scene and one random seed.

The validation split selects the checkpoint. Test labels are evaluated exactly
once after model selection and are never used to choose an epoch.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from raamnet import MODEL_VERSION, RAAMNet
from raamnet.data import array_sha256, load_scene, load_split, stratified_split
from raamnet.engine import atomic_torch_save, evaluate, flatten_logits, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RAAM-Net on a binary HSI-CD scene")
    parser.add_argument("--scene", required=True, help="path to a scene JSON configuration")
    parser.add_argument("--output-dir", required=True, help="directory for splits, checkpoints, and results")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--split", help="archived Stage39 .npz indices (recommended for reproduction)")
    parser.add_argument("--train-ratio", default=0.01, type=float)
    parser.add_argument("--val-ratio", default=0.01, type=float)
    parser.add_argument("--save-interval", default=50, type=int)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.save_interval < 1:
        raise ValueError("epochs and save_interval must be positive")
    output_dir = Path(args.output_dir)
    seed_everything(args.seed)
    cube1, cube2, map_labels = load_scene(args.scene)
    height, width, bands = cube1.shape
    device = torch.device(args.device)
    date1 = torch.from_numpy(cube1).permute(2, 0, 1).unsqueeze(0).to(device)
    date2 = torch.from_numpy(cube2).permute(2, 0, 1).unsqueeze(0).to(device)
    labels = torch.from_numpy(map_labels.reshape(-1)).long().to(device)

    split_path = output_dir / "splits" / f"seed{args.seed}.npz"
    if (output_dir / "checkpoints" / f"best_seed{args.seed}.pt").exists():
        raise FileExistsError("An existing best checkpoint would be overwritten; use a new output directory")
    if args.split:
        split = load_split(args.split, map_labels.size)
        if split_path.exists():
            cached = load_split(split_path, map_labels.size)
            if not all(np.array_equal(cached[k], split[k]) for k in split):
                raise ValueError("existing output split differs from --split")
    elif split_path.exists():
        split = load_split(split_path, map_labels.size)
    else:
        split = stratified_split(map_labels, args.train_ratio, args.val_ratio, args.seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(split_path, **split)
    index = {name: torch.from_numpy(value).long().to(device) for name, value in split.items()}

    model = RAAMNet(bands=bands, dim=args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    best_path = output_dir / "checkpoints" / f"best_seed{args.seed}.pt"
    best_val_loss, best_epoch = float("inf"), -1
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = flatten_logits(model(date1, date2))
        loss = F.cross_entropy(logits.index_select(0, index["train"]), labels.index_select(0, index["train"]))
        loss.backward()
        optimizer.step()
        scheduler.step()

        val_loss, val_metrics, _ = evaluate(model, date1, date2, labels, index["val"])
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            atomic_torch_save({
                "epoch": epoch,
                "model_version": MODEL_VERSION,
                "bands": bands,
                "dim": args.dim,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "validation_loss": val_loss,
            }, best_path)
        if epoch % args.save_interval == 0:
            atomic_torch_save({
                "epoch": epoch,
                "model_version": MODEL_VERSION,
                "bands": bands,
                "dim": args.dim,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
            }, output_dir / "checkpoints" / f"epoch{epoch:03d}_seed{args.seed}.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} train_loss={loss.item():.5f} "
                f"val_loss={val_loss:.5f} val_f1={100 * val_metrics['f1']:.2f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics, prediction = evaluate(model, date1, date2, labels, index["test"])
    result = {
        "scene": str(args.scene), "seed": args.seed, "epochs": args.epochs,
        "model_version": MODEL_VERSION,
        "input_array_sha256": {"date1": array_sha256(cube1), "date2": array_sha256(cube2), "labels": array_sha256(map_labels)},
        "split_array_sha256": {k: array_sha256(v) for k, v in split.items()},
        "learning_rate": args.learning_rate, "bands": bands, "spatial_size": [height, width],
        "train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
        "checkpoint_selection": "minimum validation cross-entropy",
        "test_used_for_selection": False, "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss, "test_loss": test_loss,
        "metrics": test_metrics, "training_seconds": time.time() - started,
    }
    save_json(output_dir / "results" / f"seed{args.seed}.json", result)
    np.save(output_dir / "results" / f"seed{args.seed}_prediction.npy", prediction.reshape(height, width))
    formatted = ", ".join(f"{key}={100 * value:.3f}" for key, value in test_metrics.items())
    print(f"FINAL_RESULT seed={args.seed} best_epoch={best_epoch} {formatted}", flush=True)


if __name__ == "__main__":
    main()
