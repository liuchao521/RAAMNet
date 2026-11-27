"""Partition integrity and the released full-model split artifacts."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from raamnet.data import load_split, stratified_split, validate_split


class SplitTests(unittest.TestCase):
    def test_released_splits(self):
        records = json.loads((ROOT / "reproducibility/splits_manifest.json").read_text())
        self.assertEqual(len(records), 30)
        for row in records:
            path = ROOT / "reproducibility" / row["file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
            split = load_split(path, int(np.prod(row["label_shape"])))
            self.assertEqual({k: len(v) for k, v in split.items()}, row["counts"])

    def test_overlapping_indices_rejected(self):
        with self.assertRaises(ValueError):
            validate_split({"train": np.array([0]), "val": np.array([0]), "test": np.array([1, 2])}, 3)

    def test_training_budget_keeps_validation_fixed(self):
        labels = np.r_[np.zeros(5000, dtype=int), np.ones(5000, dtype=int)]
        small = stratified_split(labels, 0.01, 0.01, 42)
        larger = stratified_split(labels, 0.03, 0.01, 42)
        np.testing.assert_array_equal(small["val"], larger["val"])
        self.assertTrue(set(small["train"]).issubset(larger["train"]))
        self.assertEqual(len(larger["train"]), 300)


if __name__ == "__main__":
    unittest.main()
