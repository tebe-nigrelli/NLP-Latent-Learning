# Project Overview

# Setup

We provide detailed instructions to guarantee reproducibility of our results.

## Dataset Download

We validate our Research with Google Research's `goemotions` dataset, a large-scale collection of Reddit comments annotated with 27 one-hot encoded emotions, plus neutrality. We use it extensively to train our model and finetune LLMs for baseline comparison.

We use google-research's dataset mirror on [Huggingface Website](https://huggingface.co/datasets/google-research-datasets/go_emotions). 

## Python Environment

The project uses `Python 3.14.2`. We `uv` as our preferred dependency manager, as it is simple to . After installing [uv](https://docs.astral.sh/uv/), it is sufficient to run the command `uv sync` in the folder of the repository, which will automatically download all the necessary packages. 

Installation of `uv` is recommended but not necessary. People wishing to use environment managers may find it helpful to read the `pyproject.toml` file for reference on packages used.

## LLMs for Finetuning 


## Commercial Models