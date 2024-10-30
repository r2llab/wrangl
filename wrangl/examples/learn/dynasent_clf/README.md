# Dynasent CLF

This example trains a sentence sentiment classifier.

```bash
git clone https://github.com/r2llab/wrangl
cd wrangl/wrangl/examples/learn/dynasent_clf
```

## Training
To train locally (settings in `conf/default.yaml`):

```bash
python train.py
```

To train using Slurm via the Slurm launcher (settings in `conf/hydra/launcher/slurm.yaml`):

```bash
python train.py --multirun hydra/launcher=slurm hydra.launcher.partition=<name of your partition>
```

## Git
To track code changes on a run-to-run basis:

```bash
python train.py git.enable=true
```

This will save Git diffs to the run output directory in `saves/<project name>/<run name>/{git.head.json, git.path.diff}`.


## WanDB
To log results onto Wandb (assuming your `wandb` user is your current `$USER`, if not, you can specify `wandb.entity=<your wandb username>`):

```bash
python train.py wandb.enable=true
```

[Here is an example](https://wandb.ai/vzhong/wrangl-examples-dynasent_clf) of the Wandb run for this job.
