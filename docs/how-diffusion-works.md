# How Diffusion Works

This repo uses a **DDPM-style diffusion model**. The easiest way to think about it is:

> training teaches the model how to remove noise from an image, one step at a time.

## Core Idea

During training:

1. Take a clean training image.
2. Choose a timestep.
3. Add the amount of noise associated with that timestep.
4. Ask the model to predict the noise that was added.

The model is not directly asked to "draw a red circle." It is asked to answer:

> "Given this noisy image, this timestep, and this prompt, what noise should be removed?"

## Why This Works

If the model gets good at predicting noise across many timesteps, then at inference time we can:

1. start from pure noise,
2. repeatedly predict the noise present,
3. subtract that predicted noise,
4. slowly move toward a clean image that matches the prompt.

That repeated denoising process is what generates the final image.

## What Happens in This Project

This implementation stays intentionally small:

- images are `64x64`,
- the dataset is synthetic,
- the model predicts epsilon (noise),
- sampling uses classifier-free guidance.

That makes the mechanics easier to inspect than a production image model.

## Read the Code

The most important file for the diffusion logic is:

- `src/model/diffusion.py`

That module contains:

- the noise schedule,
- forward noising (`q_sample`),
- training loss,
- reverse denoising and sampling.

