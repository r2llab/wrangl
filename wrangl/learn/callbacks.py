"""
Callbacks that can be used during PytorchLightning training.
"""

import json
import wandb
from hydra.utils import get_original_cwd
from lightning.pytorch.callbacks import Callback


class WandbTableCallback(Callback):
    """
    Uploads sample predictions to Wandb.
    """

    def on_validation_end(self, trainer, model):
        wandb_logger = [exp for exp in trainer.logger.experiment if isinstance(exp, wandb.sdk.wandb_run.Run)]
        assert wandb_logger
        run = wandb.Api().run(path=wandb_logger[0].path)
        for artifact in run.logged_artifacts():
            artifact.delete(delete_aliases=True)
        table = wandb.Table(columns=['context', 'pred', 'gold'])
        for context, gen, gold in model.pred_samples:
            table.add_data(repr(context), repr(gen), repr(gold))
        wandb.log(dict(gen=table))


class GitCallback(Callback):
    """
    Dumps git diffs to work directory.
    """

    def on_init_end(self, trainer=None):
        import git
        repo = git.Repo(get_original_cwd(), search_parent_directories=True)
        with open('git.patch.diff', 'wt') as f:
            f.write(repo.git.diff(repo.head.commit.tree))
        with open('git.head.json', 'wt') as f:
            c = repo.head.commit
            json.dump(dict(hexsha=c.hexsha, message=c.message, date=c.committed_date, committer=c.committer.name), f, indent=2)
