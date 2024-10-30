#!/usr/bin/env python
import json
import hydra
import random
import logging
from wrangl.learn import SupervisedModel
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class MyDataset(list, Dataset):
    pass


def generate_dataset(fname):
    dataset = MyDataset()
    with open(fname) as f:
        for line in f:
            row = json.loads(line)
            if row['gold_label'] not in {None, 'mixed'}:
                row.update(dict(
                    sent=row['sentence'],
                    label_text=row['gold_label'],
                    label_idx=['positive', 'neutral', 'negative'].index(row['gold_label'])
                ))
                dataset.append(row)
    return dataset


@hydra.main(config_path='conf', config_name='default', version_base='1.1')
def main(cfg):
    Model = SupervisedModel.load_model_class(cfg.model)
    train = generate_dataset(cfg.ftrain)
    val = generate_dataset(cfg.feval)
    Model.run_train_test(cfg, train, val)


if __name__ == '__main__':
    main()
