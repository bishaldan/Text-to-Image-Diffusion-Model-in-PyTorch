# GitHub Launch Checklist

Use this checklist when you are ready to publish the repo.

## Repository Setup

- Choose the final repository title and description.
- Add the MIT license.
- Make sure large generated outputs are not accidentally committed.
- Keep only curated media under `assets/`.

## README Quality

- Hero section explains what the project is in one screenful.
- Screenshots or generated outputs render correctly.
- Architecture section is visible without digging through code.
- Quickstart commands are copy-paste friendly.
- Limitations are stated honestly.
- Future improvements are listed clearly.

## Technical Checks

- `python3 -m compileall src tests`
- `python3 -m unittest`
- container tests pass when Docker is healthy
- smoke run commands still work from the README

## Content Checks

- Docs are linked from the README.
- Sample prompts are useful and realistic for this project.
- Experiment summary matches the actual tracked assets.
- Frontend instructions are present but do not oversell current quality.

## Publishing Extras

- Add a short GitHub repository description.
- Add repository topics such as:
  - `diffusion-model`
  - `text-to-image`
  - `pytorch`
  - `machine-learning`
  - `deep-learning`
  - `generative-ai`
  - `educational-project`
- Optionally pin the repo on your GitHub profile.

