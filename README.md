# Wrangl

[![Tests](https://github.com/r2llab/wrangl/actions/workflows/test.yml/badge.svg)](https://github.com/r2llab/wrangl/actions/workflows/test.yml)

Parallel data preprocessing and fast experiments for NLP and ML.
See [docs here](https://r2llab.github.io/wrangl/).

## Why?
I built this library to prototype ideas quickly.
In essence it combines [Hydra](https://hydra.cc), [Pytorch Lightning](https://www.pytorchlightning.ai), and [Ray](https://ray.io) for some fast data processing and supervised learning.
The following are supported with command line or config tweaks (e.g. no additional boilerplate code):

- checkpointing
- early stopping
- auto git diffs
- logging to S3 (along with auto-generated seaborn plot), wandb
- Slurm launcher


## Installation

```bash
pip install -e .  # add [dev] if you want to run tests and build docs.

# for latest
pip install git+https://github.com/r2llab/wrangl

# pypi release
pip install wrangl
```

## Usage

See [the documentation](https://wrangl.pages.dev) for how to use Wrangl.
Examples of projects using Wrangl are found in `wrangl.examples`.
In particular `wrangl.examples.learn.xor_clf` shows an example of using Wrangl to quickly set up a supervised classification task.
For parallel data preprocessing `wrangl.examples.preprocess.using_stanza` shows an example of using Stanford NLP Stanza to parse text in parallel across CPU cores.
