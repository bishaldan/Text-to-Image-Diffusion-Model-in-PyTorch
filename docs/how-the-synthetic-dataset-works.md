# How the Synthetic Dataset Works

The dataset in this repo is generated programmatically instead of downloaded from the internet.

That is a design choice, not a shortcut.

## Why a Synthetic Dataset

Using a synthetic dataset gives us:

- deterministic generation,
- a very small and understandable visual domain,
- captions that exactly match the scene,
- an easier path for learners to connect prompts to outputs.

## What the Generator Produces

Each sample includes:

- a `64x64` RGB image,
- a caption,
- structured metadata describing the scene.

Scene types include:

- single object on a colored background,
- count-based prompts such as `two green squares`,
- relation-based prompts such as `a small white circle above a black square`.

## Why This Is Useful for Learning

A realistic internet-scale dataset would make the project feel more impressive, but much harder to understand.

Here, the model can focus on a few clear concepts:

- color,
- shape,
- count,
- spatial relation,
- size adjective.

That makes it easier to see whether training is working.

## Read the Code

The best place to start is:

- `src/data/generate_synthetic_dataset.py`

That file contains:

- scene templates,
- caption grammar,
- shape rendering,
- deterministic split generation.

