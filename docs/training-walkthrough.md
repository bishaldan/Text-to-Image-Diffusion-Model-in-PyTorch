# Training Walkthrough

This repo keeps training intentionally direct so the full loop is easy to follow.

## Training Flow

1. Load images and captions from the synthetic dataset.
2. Convert captions into frozen text embeddings.
3. Sample random diffusion timesteps.
4. Add noise to the clean images.
5. Ask the conditioned U-Net to predict that noise.
6. Compute MSE loss between true noise and predicted noise.
7. Save checkpoints and validation samples at regular intervals.

## What Gets Saved

Training writes three useful artifact types:

- checkpoints under `checkpoints/<run-id>/`
- sample grids under `outputs/samples/<run-id>/`
- config snapshots under `outputs/runs/<run-id>/`

That makes the runs reproducible and easy to inspect later.

## What to Look For During Training

Healthy signs:

- loss trends downward overall,
- validation sample grids become less chaotic,
- prompt-conditioned colors and shapes begin to match the request.

Things that still commonly fail in a tiny model:

- object counts,
- precise spatial relations,
- sharp geometric boundaries.

## Read the Code

The main entrypoint is:

- `src/train.py`

That file coordinates:

- config loading,
- dataloaders,
- text encoding,
- optimizer and scheduler,
- checkpointing,
- validation sampling.

