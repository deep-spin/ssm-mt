from typing import Any
import pytorch_lightning as pl
import torch

import torch.nn as nn
import torch.nn.functional as F
from torchscale.architecture.config import DecoderConfig

from torchscale.architecture.decoder import Decoder

from transformers import AutoTokenizer
import evaluate
from transformers.optimization import get_inverse_sqrt_schedule
from fairseq.modules import PositionalEmbedding

from utils.beam_search import BeamSearch
from utils.mt.comet import load_comet


class TransformerMT(pl.LightningModule):
    is_encoder_decoder = False
    model_name = "transformer"

    configs = {
        "default": {
            "embedding_dim": 544,  # 77 M
            "ffn_dim": 4,  # multiplier
            "heads": 8,
            "layers": 12,
            "dropout": 0.1,
            "max_seq_len": 512,
        }
    }

    def __init__(
        self,
        config=None,
        vocab_size=None,
        tokenizer=None,
        embedding_dim=None,
        ffn_dim=None,  # multiplier
        heads=None,
        layers=None,
        max_seq_len=512,
        dropout=0.1,
        decoder_normalize_before=True,
        no_scale_embedding=False,
        use_padding=True,
        activation_fn="relu",
        flash_attention=True,
        test_per_sample=False,
        **kwargs,
    ) -> None:
        super().__init__()

        args = DecoderConfig(
            vocab_size=vocab_size,
            decoder_attention_heads=heads,
            decoder_embed_dim=embedding_dim,
            decoder_ffn_embed_dim=embedding_dim * ffn_dim,
            decoder_layers=layers,
            dropout=dropout,
            decoder_normalize_before=decoder_normalize_before,
            no_scale_embedding=no_scale_embedding,
            activation_fn=activation_fn,
            flash_attention=flash_attention,
        )

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos = PositionalEmbedding(max_seq_len, embedding_dim, 0)
        self.model = Decoder(
            args=args,
            embed_tokens=self.emb,
            embed_positions=self.pos,
        )

        self.config = config
        self.tokenizer: AutoTokenizer = tokenizer
        self.bleu = evaluate.load("sacrebleu")
        self.comet = load_comet()
        self.use_padding = use_padding
        self.test_per_sample = test_per_sample
        self.test_res = []

    def forward(self, ids, **kwargs) -> Any:
        return self.model(ids, **kwargs)

    def training_step(self, batch, batch_idx):
        ids, labels = (
            batch["input_ids"][:, :-1].contiguous(),
            batch["input_ids"][:, 1:].contiguous(),
        )

        self_attn_padding_mask = (
            ~(batch["attention_mask"][:, :-1].bool()).contiguous()
            if self.use_padding
            else None
        )

        lm_logits, _ = self.model.forward(
            prev_output_tokens=ids, self_attn_padding_mask=self_attn_padding_mask
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

    def validation_step(self, batch, batch_idx):
        ids, labels = batch["input_ids"], batch["labels"]

        preds = []

        # kv cache
        cache = {"is_first_step": True}
        done = torch.tensor([False] * ids.size(0)).to(ids.device)

        self_attn_padding_mask = (
            ~(batch["attention_mask"].bool()) if self.use_padding else None
        )

        zeros = torch.zeros((ids.size(0), 1), dtype=torch.bool).to(ids.device)

        for i in range(labels.size(1)):
            lm_logits, _ = self.model.forward(
                ids,
                self_attn_padding_mask=self_attn_padding_mask,
                incremental_state=cache,
            )
            next_token_logits = lm_logits[:, -1, :]

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_token), dim=-1)
            self_attn_padding_mask = (
                torch.cat((self_attn_padding_mask, zeros), dim=-1)
                if self.use_padding
                else None
            )

            preds.append(next_token)

            is_eos = next_token == self.tokenizer.eos_token_id
            done = done | is_eos.squeeze(-1)

            cache["is_first_step"] = False  # FIXME only necessary for first step

            if done.all():
                break

        preds = torch.cat(preds, dim=1)

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        eos_mask = (preds == eos_token_id).cumsum(dim=1) > 0

        preds[eos_mask] = pad_token_id
        tpreds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        tlabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["score"]
        self.log("val_bleu", bleu_score, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """beam search with parallel formulation"""
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
        )

        cache = {"is_first_step": True}

        for idx in range(max_seq_len):
            self_attn_padding_mask = (input_ids == self.tokenizer.pad_token_id).to(
                input_ids.device
            )
            outputs, _ = self.model.forward(
                prev_output_tokens=input_ids,
                self_attn_padding_mask=self_attn_padding_mask,
                incremental_state=cache,
            )
            next_token_logits = outputs[:, -1, :]
            input_ids, cache = search.step(
                ids=input_ids,
                logits=next_token_logits,
                cache=cache,
                reorder_cache_fn=self._reorder_cache,
            )

            cache["is_first_step"] = False

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

    def _reorder_cache(self, cache, beam_idx):
        # cache has a is_first_step key besides the layers

        for layer_idx in range(len(cache) - 1):
            device = cache[layer_idx]["prev_key"].device
            cache[layer_idx]["prev_key"] = cache[layer_idx]["prev_key"].index_select(
                0, beam_idx.to(device)
            )
            cache[layer_idx]["prev_value"] = cache[layer_idx][
                "prev_value"
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
