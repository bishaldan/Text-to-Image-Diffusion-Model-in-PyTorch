# Model Architecture

This document explains the actual neural network used in this repo and how data flows through it.

## High-Level Architecture

The project combines five major components:

1. a synthetic image-and-caption generator,
2. a frozen text encoder,
3. a timestep embedding module,
4. a conditioned U-Net,
5. a DDPM-style diffusion process.

The core training task is:

> predict the noise added to an image, conditioned on both timestep and text prompt.

## 1. Synthetic Dataset Layer

The dataset generator creates controlled image-caption pairs such as:

- `a red circle on a blue background`
- `two green squares on a white background`
- `a small white circle above a black square`

That gives the model a compact concept space:

- color,
- shape,
- count,
- simple relation,
- size adjective.

Because the captions and images are generated together, supervision is clean and deterministic.

## 2. Frozen Text Encoder

The project uses:

- `google/bert_uncased_L-2_H-128_A-2`

This encoder is frozen and not fine-tuned during diffusion training.

### Why freeze it

Freezing the text encoder:

- simplifies optimization,
- reduces compute,
- keeps the project focused on image-generation mechanics,
- provides stable prompt embeddings from the start.

### Output

The tokenizer converts a prompt into token IDs. The BERT encoder produces contextual token embeddings. Those token embeddings are pooled into a single prompt embedding vector.

That pooled embedding becomes the conditioning signal for the image model.

## 3. Timestep Embeddings

Diffusion models need to know how noisy the current image is.

This repo uses sinusoidal timestep embeddings, then passes them through a small MLP.

The timestep embedding tells the model whether it is working on:

- a lightly corrupted image,
- a heavily corrupted image,
- or something close to pure noise.

Without this signal, the model would not know which denoising behavior to apply.

## 4. Conditioned U-Net

The image backbone is a compact U-Net with residual blocks.

### Why U-Net

U-Nets are widely used in diffusion models because they:

- preserve image structure through skip connections,
- work well for dense prediction tasks,
- combine local and global image information.

### What enters the U-Net

The U-Net receives:

- the noisy image,
- the timestep embedding,
- the text embedding.

### How conditioning is applied

Each residual block mixes:

- image features,
- timestep features,
- prompt features.

In this implementation, time and text projections are combined and used as modulation terms inside residual blocks. That gives the model a way to adapt denoising behavior to both:

- the noise level,
- the prompt meaning.

### Output

The final U-Net output is a tensor with the same shape as the input image.

It represents the model’s prediction of the noise added at that timestep.

## 5. Diffusion Process

The repo follows the DDPM pattern:

- define a beta schedule,
- construct alpha and cumulative alpha terms,
- corrupt images during training,
- reverse the process during sampling.

### Training target

The model learns epsilon prediction:

- sample a timestep `t`,
- sample Gaussian noise,
- produce a noisy image `x_t`,
- train the U-Net to predict the sampled noise.

Loss is mean squared error between:

- true noise,
- predicted noise.

### Sampling

At generation time:

1. start from random noise,
2. run the denoising step many times,
3. use classifier-free guidance to strengthen prompt alignment,
4. decode the final tensor into an image.

## Classifier-Free Guidance

This repo uses classifier-free guidance during sampling.

### During training

Some conditioning vectors are randomly dropped.

That teaches the model both:

- conditional denoising,
- unconditional denoising.

### During sampling

The model computes:

- a conditional prediction,
- an unconditional prediction.

Those are combined using a guidance scale:

- low guidance: more diversity, weaker prompt adherence,
- high guidance: stronger prompt pull, sometimes less stability.

## Why This Architecture Is Good for Learning

This is not meant to compete with large production text-to-image systems.

It is meant to expose the core mechanics clearly:

- prompt encoding,
- timestep awareness,
- conditioned residual learning,
- diffusion loss,
- iterative denoising.

That makes it a good teaching architecture and a strong educational portfolio project.

## Read the Code

Start in this order:

- `src/data/generate_synthetic_dataset.py`
- `src/model/text_encoder.py`
- `src/model/embeddings.py`
- `src/model/unet.py`
- `src/model/diffusion.py`
- `src/train.py`
- `src/sample.py`

