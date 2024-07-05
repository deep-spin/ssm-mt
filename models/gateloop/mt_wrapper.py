from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

import evaluate

from gateloop_transformer import Transformer


class GateLoopMT(pl.LightningModule):
    def __init__(
        self,
        config=None,
        vocab_size=None,
        embedding_dim=None,
        ffn_dim=None,  # as multiplier of embed dim
        heads=None,  # heads is usually close to the dim, ie. 512ish
        layers=None,
        tokenizer=None,
        dropout=0.1,  # not supported as far as I can see
        **kwargs,
    ):
        super(GateLoopMT, self).__init__()

        self.config = config

        self.tokenizer: AutoTokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.bleu = evaluate.load("bleu")

        self.model = Transformer(
            num_tokens=vocab_size,
            dim=embedding_dim,
            heads=heads,
            depth=layers,
            ff_mult=ffn_dim,
            use_gate_looped_attn=True,
        )

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        ids = labels = batch["input_ids"]

        lm_logits = self.model(ids)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Find the SEP token indices in the shifted labels
        sep_indices = (ids == self.tokenizer.sep_token_id).nonzero(as_tuple=False)[:, 1]
        mask = torch.arange(shift_labels.size(1), device=ids.device).unsqueeze(
            0
        ) >= sep_indices.unsqueeze(1)

        # Apply the mask to shift_labels by setting tokens before SEP to pad_token_id
        shift_labels[~mask] = self.tokenizer.pad_token_id

        # Flatten the tokens for loss calculation
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_index):
        ids, labels = batch["input_ids"], batch["labels"]

        preds = []
        embs = []

        # FIXME note that this is parallel instead of RNN like
        # watch repo until this is pushed
        for idx in range(labels.size(1)):
            lm_logits = self.model(ids)
            next_token_logits = lm_logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            embs.append(next_token_logits)
            preds.append(next_token)
            ids = torch.cat((ids, next_token), dim=-1)

        preds = torch.cat(preds, dim=1)
        embs = torch.stack(embs, dim=1)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(
            embs.view(-1, embs.size(-1)),
            labels.view(-1),
        )
        self.log("val_loss", loss, sync_dist=True)

        tpreds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        tlabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["bleu"]
        self.log("val_bleu", bleu_score, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config["learning_rate"],
                total_steps=self.config["max_steps"],
                pct_start=self.config["warmup_steps"] / self.config["max_steps"],
                anneal_strategy="cos",
            ),
            "interval": "step",
            "name": "learning_rate",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
