import pytorch_lightning as pl

from models.retnet.base import RetNetDecoder
from torchscale.architecture.config import RetNetConfig
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import evaluate


class RetNetClassification(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        ffn_dim,
        heads,
        layers,
        num_classes,
        double_v_dim=True,
        no_output_layer=True,
        config=None,
        **kwargs
    ):
        super(RetNetClassification, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        rn_config = RetNetConfig(
            decoder_embed_dim=embedding_dim,
            decoder_ffn_embed_dim=ffn_dim,
            decoder_value_embed_dim=2 * embedding_dim
            if double_v_dim
            else embedding_dim,
            decoder_retention_heads=heads,
            decoder_layers=layers,
            no_output_layer=no_output_layer,
        )

        # Initialize the RetNet model.
        self.retnet = RetNetDecoder(rn_config)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, num_classes),
        )

        self.acc = evaluate.load("accuracy")
        self.config = config["run"]

    def forward(self, x):
        emb = self.embedding(x)
        retnet_out = self.retnet(
            prev_output_tokens=x, token_embeddings=emb, features_only=True
        )
        output = self.mlp(retnet_out[0].mean(1))
        return output

    def forward_recurrent(self, x):
        bsz, seq_len, emb_dim = x.shape
        emb = self.embedding(x)

        token_embeddings = []
        hidden_states = []
        hs = {"is_first_step": True}
        for idx in range(seq_len):
            out, hs = self.retnet.forward(
                prev_output_tokens=x[:, idx].view(bsz, 1),
                incremental_state=hs,
                token_embeddings=emb[:, idx, :].view(bsz, 1, emb_dim),
                features_only=True,
            )
            token_embeddings.append(out)
            hidden_states.append(hs)

        rn_out = torch.cat(token_embeddings, dim=1)
        mean_tok_emb = rn_out.mean(1)
        ff_out = self.mlp(mean_tok_emb)
        return ff_out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["input_ids"], batch["label"]
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["input_ids"], batch["label"]
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        yhat = torch.argmax(outputs, dim=1)
        acc = self.acc.compute(predictions=yhat, references=labels)

        self.log_dict(
            {"val_accuracy": acc["accuracy"], "val_loss": loss}, sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        inputs, labels = batch["input_ids"], batch["label"]
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        yhat = torch.argmax(outputs, dim=1)
        acc = self.acc.compute(predictions=yhat, references=labels)
        self.log_dict(
            {"test_accuracy": acc["accuracy"], "test_loss": loss}, sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])

        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.config["learning_rate"],
                total_steps=self.config["max_steps"],
                pct_start=self.config["warmup_steps"] / self.config["max_steps"],
                anneal_strategy="linear",
            ),
            "interval": "step",
            "name": "learning_rate",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class RetNetDualClassification(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        ffn_dim,
        heads,
        layers,
        num_classes,
        double_v_dim=True,
        no_output_layer=True,
        config=None,
        **kwargs
    ):
        super(RetNetDualClassification, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        rn_config = RetNetConfig(
            decoder_embed_dim=embedding_dim,
            decoder_ffn_embed_dim=ffn_dim,
            decoder_value_embed_dim=2 * embedding_dim
            if double_v_dim
            else embedding_dim,
            decoder_retention_heads=heads,
            decoder_layers=layers,
            no_output_layer=no_output_layer,
        )

        # Initialize the RetNet model.
        self.retnet = RetNetDecoder(rn_config)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, num_classes),
        )

        self.acc = evaluate.load("accuracy")
        self.config = config["run"]

    def forward(self, x0, x1):
        emb0 = self.embedding(x0)
        emb1 = self.embedding(x1)
        rn_0out = self.retnet(
            prev_output_tokens=x0, token_embeddings=emb0, features_only=True
        )[0].mean(1)
        rn_1out = self.retnet(
            prev_output_tokens=x1, token_embeddings=emb1, features_only=True
        )[0].mean(1)
        out = self.mlp(
            torch.cat([rn_0out, rn_1out, rn_0out * rn_1out, rn_0out - rn_1out], dim=-1)
        )
        return out

    def training_step(self, batch, batch_idx):
        inputs0, inputs1, labels = (
            batch["input_ids_0"],
            batch["input_ids_1"],
            batch["label"],
        )
        outputs = self(inputs0, inputs1)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs0, inputs1, labels = (
            batch["input_ids_0"],
            batch["input_ids_1"],
            batch["label"],
        )
        outputs = self(inputs0, inputs1)
        loss = F.cross_entropy(outputs, labels)
        yhat = torch.argmax(outputs, dim=1)
        acc = self.acc.compute(predictions=yhat, references=labels)
        self.log_dict(
            {"val_accuracy": acc["accuracy"], "val_loss": loss}, sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        inputs0, inputs1, labels = (
            batch["input_ids_0"],
            batch["input_ids_1"],
            batch["label"],
        )
        outputs = self(inputs0, inputs1)
        loss = F.cross_entropy(outputs, labels)
        yhat = torch.argmax(outputs, dim=1)
        acc = self.acc.compute(predictions=yhat, references=labels)
        self.log_dict(
            {"val_accuracy": acc["accuracy"], "val_loss": loss}, sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])

        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.config["learning_rate"],
                total_steps=self.config["max_steps"],
                pct_start=self.config["warmup_steps"] / self.config["max_steps"],
                anneal_strategy="linear",
            ),
            "interval": "step",
            "name": "learning_rate",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
