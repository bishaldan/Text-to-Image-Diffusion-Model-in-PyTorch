# Experiment Summary

## Showcase Goal

Produce a small but credible public result set that demonstrates:

- prompt-conditioned color control,
- count prompts,
- simple spatial relations,
- end-to-end checkpoint sampling.

## Public Prompt Suite

- `a red circle on a blue background`
- `two green squares on a white background`
- `a small white circle above a black square`
- `three yellow triangles on a black background`
- `a large blue square left of a red circle`

## Most Useful Current Runs

### Smoke Run

- Config: `configs/text2img_smoke.yaml`
- Dataset size: `200 train / 40 val`
- Run id: `20260403-184543`
- Purpose: verify setup, training loop, checkpointing, and sampling
- Outcome: complete end-to-end run with saved checkpoints and sample grids

### Showcase Run

- Config: `configs/text2img_showcase.yaml`
- Dataset size: `400 train / 80 val`
- Run id: `20260403-184651`
- Training length: `4 epochs`, roughly `4-5 minutes` in the Docker path on this machine
- Purpose: produce cleaner qualitative assets for the README
- Outcome: intended public-facing run for curated GitHub screenshots and progression assets

## What Improved

- Docker workflow now supports the required tokenizer dependencies.
- The default text encoder was changed to a smaller BERT variant that loads reliably in the container.
- The project now has a stable smoke path and a separate showcase path.
- Showcase loss fell substantially from the first epoch to the final epoch, indicating stable learning.

## What Still Fails

- Samples are still far from photorealistic and remain visibly noisy.
- Complex relations can remain inconsistent.
- Count and shape precision may drift because the model is deliberately small.
- The repo is strongest today as an educational diffusion build, not a high-fidelity image generator.
