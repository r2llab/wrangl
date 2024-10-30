#!/usr/bin/env python
import hydra
import torch
import random
import logging
from wrangl.learn import SupervisedModel
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class MyDataset(list, Dataset):
    pass


def generate_dataset(n, seed=0):
    rng = random.Random(seed)
    dataset = MyDataset()
    for _ in range(n):
        raw = dict(x=rng.uniform(-1, 1), y=rng.uniform(-1, 1))
        raw.update(dict(
            feat=torch.tensor([raw['x'], raw['y']], dtype=torch.float32),
            label=1 if raw['x'] > 0 and raw['y'] > 0 else 0,
        ))
        dataset.append(raw)
    return dataset


@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    Model = SupervisedModel.load_model_class(cfg.model)
    train = generate_dataset(10000)
    val = generate_dataset(1000)
    Model.run_train_test(cfg, train, val)


if __name__ == '__main__':
    main()
