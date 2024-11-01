import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from wrangl.learn import SupervisedModel, metrics as M


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lm = AutoModel.from_pretrained(cfg.lm)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        self.labels = ['positive', 'neutral', 'negative']
        self.mlp = nn.Linear(self.lm.config.hidden_size, len(self.labels))
        self.acc = M.Accuracy()

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

    def compute_metrics(self, pred: list, gold: list, batch: list) -> dict:
        return self.acc(pred, gold)

    def compute_loss(self, out, feat, batch):
        return F.cross_entropy(out, feat['label'])

    def extract_context(self, feat, batch):
        return [ex['sent'] for ex in batch]

    def extract_pred(self, out, feat, batch):
        return [self.labels[x] for x in out.max(1)[1].tolist()]

    def extract_gold(self, feat, batch):
        return [ex['label_text'] for ex in batch]

    def forward(self, feat, batch):
        out = self.lm(**feat['sent']).last_hidden_state
        return self.mlp(out[:, 0])  # use the [CLS] token for classification
