from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.demo_inference import main, patch_metrics


class DemoTests(unittest.TestCase):
    def test_known_foreground_overlap(self):
        prediction = np.array([[1, 1], [0, 0]], dtype=bool)
        target = np.array([[1, 0], [1, 0]], dtype=bool)
        metrics = patch_metrics(prediction, target)
        self.assertAlmostEqual(metrics["iou"], 1 / (3 + 1e-7))
        self.assertAlmostEqual(metrics["dice"], 2 / (4 + 1e-7))

    def test_empty_masks_follow_experiment_denominator_epsilon(self):
        empty = np.zeros((2, 2), dtype=bool)
        self.assertEqual(patch_metrics(empty, empty), {"iou": 0.0, "dice": 0.0})

    def test_different_mask_shapes_are_not_broadcast(self):
        with self.assertRaises(ValueError):
            patch_metrics(np.zeros((2, 1)), np.zeros((2, 2)))

    def test_existing_output_is_preserved_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            sentinel = Path(temporary) / "keep.txt"
            sentinel.write_text("existing output")
            with patch("scripts.demo_inference.torch.load") as load:
                with self.assertRaises(SystemExit) as error:
                    main(["--output-dir", temporary])
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()
            self.assertEqual(sentinel.read_text(), "existing output")
