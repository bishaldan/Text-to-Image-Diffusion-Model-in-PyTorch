# How Text Conditioning Works

Text-to-image models need a way to turn words into numbers the image model can use.

In this repo, the text path is intentionally simple:

1. tokenize the caption,
2. run it through a frozen BERT encoder,
3. pool the token embeddings into one prompt embedding,
4. inject that embedding into the U-Net during denoising.

## Why Use a Frozen Encoder

The diffusion model, dataset generator, and training loop are already enough moving parts for one educational project.

Freezing the text encoder gives us:

- prompt embeddings that already carry useful language structure,
- a much simpler training problem,
- clearer code boundaries between text understanding and image generation.

## How the Conditioning Reaches the Image Model

The prompt embedding is projected into a conditioning vector and passed into residual blocks of the U-Net.

This project combines:

- a timestep embedding,
- a text-conditioning embedding.

Those signals modulate feature maps while the model predicts noise.

## Why This Matters

Without text conditioning, the model can only learn “what synthetic images look like.”

With text conditioning, it can learn:

- which color is requested,
- which shape is requested,
- how many objects to draw,
- simple spatial relations like `above` or `left of`.

## Read the Code

Start with:

- `src/model/text_encoder.py`

Then see how those embeddings are used in:

- `src/model/unet.py`

