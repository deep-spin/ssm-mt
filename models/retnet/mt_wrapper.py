import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from torchscale.architecture.config import RetNetConfig
from transformers import AutoTokenizer

from transformers.optimization import get_inverse_sqrt_schedule
from models.retnet.base import RetNetDecoder
import evaluate

from utils.beam_search import BeamSearch
import json
from utils.mt.comet import load_comet


class RetNetMT(pl.LightningModule):
    is_encoder_decoder = False
    is_concat = False
    model_name = "retnet"

    configs = {
        "default": {
            "embedding_dim": 512,  # 77M
            "value_dim": 1024,
            "ffn_dim": 1024,
            "heads": 4,
            "layers": 12,
        }
    }

    def __init__(
        self,
        config=None,
        vocab_size=None,
        embedding_dim=None,
        value_dim=None,
        ffn_dim=None,
        heads=None,
        layers=None,
        tokenizer=None,
        activation_fn="swish",  # default is gelu
        subln=False,
        layernorm_eps=1e-5,
        no_scale_embedding=False,
        use_padding=True,
        dropout=0.1,
        test_per_sample=False,
        **kwargs,
    ):
        self.config = config
        # see https://github.com/microsoft/torchscale/blob/main/examples/fairseq/models/retnet.py for updated config
        config = RetNetConfig(
            vocab_size=vocab_size,
            decoder_embed_dim=embedding_dim,
            decoder_ffn_embed_dim=ffn_dim,
            decoder_value_embed_dim=value_dim,
            decoder_retention_heads=heads,
            decoder_layers=layers,
            dropout=dropout,
            activation_fn=activation_fn,
            subln=subln,
            layernorm_eps=layernorm_eps,
            no_scale_embedding=no_scale_embedding,
        )

        super(RetNetMT, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        # retnet decoder builds a output linear layer based on the vocab size
        self.retnet = RetNetDecoder(config, embed_tokens=self.embedding_layer)
        self.tokenizer: AutoTokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.bleu = evaluate.load("sacrebleu")
        self.comet = load_comet()

        self.use_padding = use_padding
        self.test_per_sample = test_per_sample
        self.test_res = []

    def forward(self, x):
        return self.retnet.forward(prev_output_tokens=x)

    def training_step(self, batch, batch_idx):
        (
            ids,
            labels,
        ) = (
            batch["input_ids"][:, :-1].contiguous(),
            batch["input_ids"][:, 1:].contiguous(),
        )

        padding_mask = (
            (ids == self.tokenizer.pad_token_id).to(ids.device)
            if self.use_padding
            else None
        )
        lm_logits, _ = self.retnet.forward(
            prev_output_tokens=ids, padding_mask=padding_mask
        )

        sep_mask = (ids == self.tokenizer.sep_token_id).cumsum(dim=1) > 0
        labels[~sep_mask] = self.tokenizer.pad_token_id

        loss = F.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_index):
        input_ids, labels = batch["input_ids"], batch["labels"]

        bsz, seq_len = input_ids.shape
        _, tgt_len = labels.shape
        preds = []
        hs = {}
        done = torch.tensor([False] * bsz).to(input_ids.device)
        for idx in range(seq_len + tgt_len - 1):
            padding_mask = (
                (input_ids == self.tokenizer.pad_token_id).to(input_ids.device)
                if self.use_padding
                else None
            )
            logits, _ = self.retnet.forward(
                prev_output_tokens=input_ids[:, : idx + 1],
                incremental_state=hs,
                padding_mask=padding_mask[:, idx] if self.use_padding else None,
            )

            if idx >= seq_len - 1:
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat((input_ids, next_token), dim=-1)
                preds.append(next_token)

                is_eos = next_token == self.tokenizer.eos_token_id
                done = done | is_eos.squeeze(-1)

                if done.all():
                    break

        preds = torch.cat(preds, dim=1)

        # Parallel masking of tokens after EOS
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # Create a cumulative sum mask where positions after EOS become True
        eos_mask = (preds == eos_token_id).cumsum(dim=1) > 0

        # Apply mask to predictions and embeddings
        preds[eos_mask] = pad_token_id

        tpreds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        tlabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["score"]
        self.log("val_bleu", bleu_score, sync_dist=True)

    def test_step(self, batch, batch_index):
        num_beams = 5
        max_seq_len = 256
        input_ids: torch.Tensor = batch["input_ids"]
        batch_size, decoder_prompt_len = input_ids.shape
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)

        search = BeamSearch(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_beams=num_beams,
            max_length=decoder_prompt_len + max_seq_len,
            device=input_ids.device,
            # decoder_prompt_len=decoder_prompt_len,
        )

        cache = {}
        for idx in range(decoder_prompt_len + max_seq_len - 1):
            padding_mask = (input_ids == self.tokenizer.pad_token_id).to(
                input_ids.device
            )
            outputs, _ = self.retnet.forward(
                prev_output_tokens=input_ids[:, : idx + 1],
                incremental_state=cache,
                padding_mask=padding_mask[:, idx],
            )

            if idx >= decoder_prompt_len - 1:
                next_token_logits = outputs[:, -1, :]
                input_ids, cache = search.step(
                    ids=input_ids,
                    logits=next_token_logits,
                    cache=cache,
                    reorder_cache_fn=self._reorder_cache,
                )

                if search.is_done:
                    break

        seqs = search.finalize(ids=input_ids)

        source_mask = (seqs == self.tokenizer.sep_token_id).cumsum(dim=1) == 0
        eos_mask = (seqs == self.tokenizer.eos_token_id).cumsum(dim=1) > 0

        src = seqs.clone()
        src[~source_mask] = self.tokenizer.pad_token_id
        tsrcs = self.tokenizer.batch_decode(src, skip_special_tokens=True)

        seqs[source_mask] = self.tokenizer.pad_token_id
        seqs[eos_mask] = self.tokenizer.pad_token_id
        tpreds = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        tlabels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["score"]
        self.log("test_bleu", bleu_score, sync_dist=True)

        res = self.comet.compute(
            sources=tsrcs,
            predictions=tpreds,
            references=tlabels,
            devices=self.config["devices"],
            progress_bar=False,
        )

        self.log("test_comet", res["mean_score"], sync_dist=True)

        if self.test_per_sample:
            bleu_scores = [
                self.bleu.compute(predictions=[tpreds[i]], references=[tlabels[i]])[
                    "score"
                ]
                for i in range(batch_size)
            ]
            self.test_res.append((tsrcs, tpreds, tlabels, bleu_scores, res["scores"]))

        return bleu_score, res["mean_score"]

    def on_test_epoch_end(self):
        if self.test_per_sample:
            source, target = self.config["language_pair"]

            with open(
                f"mt/res/{self.config['dataset']}/{self.config['dataset']}-{source}-{target}-{self.model_name}.json",
                "w",
            ) as f:
                json.dump(self.test_res, f)

    def _reorder_cache(self, cache, beam_idx):
        # cache has a is_first_step key besides the layers
        for layer_idx in range(len(cache)):
            device = cache[layer_idx]["prev_key_value"].device
            cache[layer_idx]["prev_key_value"] = cache[layer_idx][
                "prev_key_value"
            ].index_select(0, beam_idx.to(device))

        return cache

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            fused=True,
        )

        scheduler = {
            "scheduler": get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=self.config["warmup_steps"],
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
