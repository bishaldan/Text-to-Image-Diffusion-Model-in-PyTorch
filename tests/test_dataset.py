from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.generate_synthetic_dataset import generate_split, sample_scene


class DatasetGenerationTests(unittest.TestCase):
    def test_same_seed_produces_identical_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            first = base / "first"
            second = base / "second"

            first_metadata = generate_split(first, "train", count=4, image_size=64, seed=123)
            second_metadata = generate_split(second, "train", count=4, image_size=64, seed=123)

            self.assertEqual(first_metadata.read_text(), second_metadata.read_text())

            first_hashes = sorted(
                hashlib.sha256(path.read_bytes()).hexdigest() for path in (first / "train").glob("*.png")
            )
            second_hashes = sorted(
                hashlib.sha256(path.read_bytes()).hexdigest() for path in (second / "train").glob("*.png")
            )
            self.assertEqual(first_hashes, second_hashes)

    def test_caption_matches_supported_templates(self) -> None:
        for seed in range(20):
            _, metadata = sample_scene(seed=seed, image_size=64)
            caption = metadata["caption"]
            valid = (
                (" on a " in caption and " background" in caption)
                or (" above a " in caption)
                or (" below a " in caption)
                or (" left of a " in caption)
                or (" right of a " in caption)
            )
            self.assertTrue(valid, msg=caption)

    def test_renderer_outputs_valid_rgb_image(self) -> None:
        image, metadata = sample_scene(seed=7, image_size=64)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (64, 64))
        array = np.asarray(image)
        self.assertEqual(array.shape, (64, 64, 3))
        self.assertEqual(metadata["seed"], 7)


if __name__ == "__main__":
    unittest.main()

