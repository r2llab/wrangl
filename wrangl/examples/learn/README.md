### Supervised learning from scratch

You can find examples of how to use Wrangl for supervised learning in `wrangl.examples.learn.xor_clf`.

Here, we'll build from scratch a sentiment classifier on the [DynaSent dataset](https://github.com/cgpotts/dynasent).
The finished project for this example is in `wrangl.examples.learn.dynasent_clf`.
First, let's bootstrap our model using the `xor_clf` example and download the dataset.

```bash
wrangl project --source xor_clf --name dynasent_clf
cd dynasent_clf
wget https://github.com/cgpotts/dynasent/raw/main/dynasent-v1.1.zip
unzip dynasent-v1.1.zip
rm -rf __MACOSX dynasent-v1.1.zip
mv dynasent-v1.1 data
```

Now we'll modify `train.py` to use these data files.

```python
# train.py
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


@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    Model = SupervisedModel.load_model_class(cfg.model)
    train = generate_dataset(cfg.ftrain)
    val = generate_dataset(cfg.fval)
    Model.run_train_test(cfg, train, val)
```

Next, we'll update the config file to use these data files, and specify what pretrained LM we'll use.
We'll also drop the learning rate a bit since we'll be finetuning a pretrained LM.

```yaml
# conf/default.yaml
...
    lr: 0.00001
...
ftrain: '${oc.env:PWD}/data/dynasent-v1.1-round01-yelp-train.jsonl'
feval: '${oc.env:PWD}/data/dynasent-v1.1-round01-yelp-dev.jsonl'
num_workers: 4
lm: bert-base-uncased
```

Next, we'll overload the existing model as follows:
- tell it out how extract out the context, predictions, and ground truth labels for this dataset.
- instantiate a BERT encoder and use it during the forward pass.

```python
# model/mymodel.py
...
from transformers import AutoModel, AutoTokenizer


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lm = AutoModel.from_pretrained(cfg.lm)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        self.labels = ['positive', 'neutral', 'negative']
        self.mlp = nn.Linear(self.lm.config.hidden_size, len(self.labels))
        self.acc = M.Accuracy()

...

    def featurize(self, batch: list):
        """
        Converts a batch of examples to features.
        By default this returns the batch as is.

        Alternatively you may want to set `collate_fn: "ignore"` in your config and use `featurize` to convert raw examples into features.
        """
        return dict(
            sent=self.tokenizer.batch_encode_plus(
                [x['sent'] for x in batch],
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=80,
                return_tensors='pt',
            ).to(self.device),
            label=torch.tensor([x['label_idx'] for x in batch], dtype=torch.long, device=self.device),
        )

...

    def extract_context(self, feat, batch):
        return [ex['sent'] for ex in batch]

    def extract_pred(self, out, feat, batch):
        return [self.labels[x] for x in out.max(1)[1].tolist()]

    def extract_gold(self, feat, batch):
        return [ex['label_text'] for ex in batch]

    def forward(self, feat, batch):
        out = self.lm(**feat['sent']).last_hidden_state
        return self.mlp(out[:, 0])  # use the [CLS] token for classification
```

Let's train this.

```bash
python train.py devices=1 git.enable=true
```

Here is the validation accuracy across steps, logged onto S3:

![validation curve](https://github.com/r2llab/wrangl/raw/main/wrangl/examples/learn/dynasent_clf/static/step_vs_val_acc.jpg)
