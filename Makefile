PYTHON ?= python
DOCKER_COMPOSE ?= docker compose

.PHONY: test smoke-data smoke-train showcase-data showcase-train full-data full-train sample docker-test docker-build web docker-web

test:
	$(PYTHON) -m unittest

docker-build:
	$(DOCKER_COMPOSE) build

docker-test:
	$(DOCKER_COMPOSE) run --rm app python -m unittest

web:
	$(PYTHON) -m uvicorn src.web.app:app --reload

docker-web:
	$(DOCKER_COMPOSE) up web

smoke-data:
	$(PYTHON) -m src.data.generate_synthetic_dataset --out data/synth_smoke --num-train 200 --num-val 40 --image-size 64 --seed 42

smoke-train:
	$(PYTHON) -m src.train --config configs/text2img_smoke.yaml

showcase-data:
	$(PYTHON) -m src.data.generate_synthetic_dataset --out data/synth_showcase --num-train 400 --num-val 80 --image-size 64 --seed 42

showcase-train:
	$(PYTHON) -m src.train --config configs/text2img_showcase.yaml

full-data:
	$(PYTHON) -m src.data.generate_synthetic_dataset --out data/synth_v1 --num-train 20000 --num-val 2000 --image-size 64 --seed 42

full-train:
	$(PYTHON) -m src.train --config configs/text2img_tiny.yaml

sample:
	$(PYTHON) -m src.sample --checkpoint checkpoints/<run-id>/best.pt --prompts "a red circle on a blue background"
