"""Guard against silently evaluating truncated or reordered probability caches."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


class EvaluationCacheTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/06_evaluation.ipynb").read_text())
        self.namespace = {}
        exec("".join(notebook["cells"][0]["source"]), self.namespace)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("PROJECT_ROOT", "DATA_ROOT"):
            self.namespace[name] = self.root
        self.namespace["RESULTS_ROOT"] = self.root / "results"
        (self.root / "data/splits").mkdir(parents=True)
        self.folder = self.root / "results/comparison_probabilities"
        self.folder.mkdir(parents=True)
        self.images = np.array(["data/a.png", "data/b.png"])
        self.masks = np.array(["data/a_mask.png", "data/b_mask.png"])
        for name in (*self.images, *self.masks):
            (self.root / name).touch()
        pd.DataFrame({"image_path": self.images, "mask_path": self.masks}).to_csv(
            self.root / "data/splits/test.csv", index=False
        )

    def save_cache(self, images=None, probabilities=None):
        np.savez_compressed(
            self.folder / "test.npz",
            image_paths=self.images if images is None else images,
            mask_paths=self.masks,
            probabilities=np.zeros((2, 1, 256, 256), dtype=np.float32)
            if probabilities is None else probabilities,
        )

    def test_complete_cache_is_accepted(self):
        self.save_cache()
        data = self.namespace["load_probability_cache"]("comparison_probabilities", "test")
        np.testing.assert_array_equal(data["image_paths"], self.images)

    def test_reordered_cache_is_rejected(self):
        self.save_cache(images=self.images[::-1])
        with self.assertRaisesRegex(ValueError, "split order"):
            self.namespace["load_probability_cache"]("comparison_probabilities", "test")

    def test_truncated_cache_is_rejected(self):
        self.save_cache(probabilities=np.zeros((1, 1, 256, 256), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "all 2 patches"):
            self.namespace["load_probability_cache"]("comparison_probabilities", "test")

    def test_nonfinite_cache_is_rejected(self):
        self.save_cache(probabilities=np.full((2, 1, 256, 256), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.namespace["load_probability_cache"]("comparison_probabilities", "test")
