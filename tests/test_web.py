from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if FASTAPI_AVAILABLE and TORCH_AVAILABLE:
    from fastapi.testclient import TestClient
    from PIL import Image

    from src.inference import CheckpointSummary
    from src.web.app import app


@unittest.skipUnless(FASTAPI_AVAILABLE and TORCH_AVAILABLE, "fastapi and torch are required for web tests")
class WebAppTests(unittest.TestCase):
    def test_index_page_loads(self) -> None:
        client = TestClient(app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Tiny Diffusion Playground", response.text)

    def test_checkpoints_endpoint_returns_payload(self) -> None:
        summary = CheckpointSummary(
            run_id="demo-run",
            checkpoint_path="checkpoints/demo-run/best.pt",
            config_path="outputs/runs/demo-run/config.snapshot.yaml",
            training_prompts=["a red circle on a blue background"],
            guidance_scale=2.75,
            image_size=64,
            timesteps=96,
        )
        client = TestClient(app)
        with patch("src.web.app.service.list_checkpoints", return_value=[summary]):
            response = client.get("/api/checkpoints")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["default_checkpoint"], summary.checkpoint_path)
        self.assertEqual(payload["checkpoints"][0]["run_id"], "demo-run")

    def test_generate_endpoint_returns_base64_image(self) -> None:
        client = TestClient(app)
        image = Image.new("RGB", (64, 64), color=(255, 0, 0))
        with patch.object(Path, "exists", return_value=True):
            with patch(
                "src.web.app.service.generate",
                return_value=(image, {"run_id": "demo-run", "guidance_scale": 2.0, "timesteps": 96, "image_size": 64}),
            ):
                response = client.post(
                    "/api/generate",
                    json={
                        "prompt": "a red circle on a blue background",
                        "checkpoint_path": "checkpoints/demo-run/best.pt",
                    },
                )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prompt"], "a red circle on a blue background")
        self.assertTrue(payload["image_base64"])
        self.assertEqual(payload["meta"]["run_id"], "demo-run")


if __name__ == "__main__":
    unittest.main()
