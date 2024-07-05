from typing import Any
import pytorch_lightning as pl
import torch

import torch.nn as nn
import torch.nn.functional as F
from torchscale.architecture.config import EncoderDecoderConfig
from torchscale.architecture.encoder_decoder import EncoderDecoder

import evaluate
from transformers.optimization import (
    get_inverse_sqrt_schedule,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from fairseq.modules import PositionalEmbedding

from utils.beam_search import BeamSearch
from utils.mt.comet import load_comet
import json


class TransformerEncDecMT(pl.LightningModule):
    is_encoder_decoder = True
    is_concat = False
    model_name = "transformer_encdec"

    configs = {
        "default": {
            "encoder_embed_dim": 512,  # 77 M
            "encoder_attention_heads": 8,
            "encoder_ffn_embed_dim": 2048,
            "encoder_layers": 6,
            "decoder_embed_dim": 512,
            "decoder_attention_heads": 8,
            "decoder_ffn_embed_dim": 2048,
            "decoder_layers": 6,
            "max_seq_len": 512,
        }
    }

    def __init__(
        self,
        config=None,
        tokenizer=None,
        vocab_size=None,
        encoder_embed_dim=512,
        encoder_attention_heads=8,
        encoder_ffn_embed_dim=2048,
        encoder_layers=6,
        decoder_embed_dim=512,
        decoder_attention_heads=8,
        decoder_ffn_embed_dim=2048,
        decoder_layers=6,
        dropout=0.1,
        max_seq_len=1024,
        flash_attention=True,
        test_per_sample=False,
        test=False,
        test_suffix="",
        **kwargs,
    ):
        super().__init__()

        args = EncoderDecoderConfig(
            vocab_size=vocab_size,
            encoder_embed_dim=encoder_embed_dim,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_embed_dim=encoder_ffn_embed_dim,
            encoder_layers=encoder_layers,
            decoder_embed_dim=decoder_embed_dim,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_embed_dim=decoder_ffn_embed_dim,
            decoder_layers=decoder_layers,
            flash_attention=flash_attention,
            dropout=dropout,
        )

        self.emb = nn.Embedding(vocab_size, encoder_embed_dim)  # same as dec_emb_dim
        self.pos = PositionalEmbedding(
            max_seq_len, encoder_embed_dim, 0
        )  # 0 for pad idx

        self.model = EncoderDecoder(
            args,
            encoder_embed_tokens=self.emb,
            encoder_embed_positions=self.pos,
            decoder_embed_tokens=self.emb,
            decoder_embed_positions=self.pos,
        )

        self.config = config
        self.tokenizer = tokenizer
        self.bleu = evaluate.load("sacrebleu")

        if test:
            self.comet = load_comet()
            self.test_per_sample = test_per_sample
            self.test_suffix = test_suffix
            self.test_res = []

    def training_step(self, batch, batch_idx):
        source, target = (
            batch["input_ids"],
            batch["labels"],
        )

        encoder_padding_mask = (source == self.tokenizer.pad_token_id).to(source.device)
        decoder_in = target[:, :-1].contiguous()
        decoder_padding_mask = (decoder_in == self.tokenizer.pad_token_id).to(
            source.device
        )

        logits, _ = self.model.forward(
            src_tokens=source,
            encoder_padding_mask=encoder_padding_mask,
            self_attn_padding_mask=decoder_padding_mask,
            prev_output_tokens=decoder_in,
        )

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target[:, 1:].contiguous().view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = (
            batch["input_ids"],
            batch["labels"],
        )

        encoder_padding_mask = input_ids.eq(self.tokenizer.pad_token_id).to(
            input_ids.device
        )
        preds = (
            torch.ones((input_ids.size(0), 1), dtype=torch.long).to(input_ids.device)
            * self.tokenizer.bos_token_id
        )

        done = torch.tensor([False] * input_ids.size(0)).to(input_ids.device)

        for i in range(labels.size(1)):
            logits, _ = self.model.forward(
                src_tokens=input_ids,
                prev_output_tokens=preds,
                encoder_padding_mask=encoder_padding_mask,
            )
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            is_eos = next_token == self.tokenizer.eos_token_id
            done = done | is_eos.squeeze(-1)
            preds = torch.cat((preds, next_token), dim=-1)

            if done.all():
                break

        # Parallel masking of tokens after EOS
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # Create a cumulative sum mask where positions after EOS become True
        eos_mask = (preds == eos_token_id).cumsum(dim=1) > 0
        preds[eos_mask] = pad_token_id

        tpreds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        tlabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["score"]
        self.log("val_bleu", bleu_score, sync_dist=True)

    def validation_step_beam(self, batch, batch_idx):
        num_beams = 5
        max_seq_len = 256
        batch_size, _ = batch["input_ids"].shape
        batch_beam_size = batch_size * num_beams

        input_ids = batch["input_ids"].repeat_interleave(num_beams, dim=0)

        search = BeamSearch(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_beams=num_beams,
            max_length=max_seq_len + 1,  # prompt is only <BOS>
            device=input_ids.device,
        )

        preds = (
            torch.ones((batch_beam_size, 1), dtype=torch.long).to(input_ids.device)
            * self.tokenizer.bos_token_id
        )
        encoder_padding_mask = input_ids.eq(self.tokenizer.pad_token_id).to(
            input_ids.device
        )

        for idx in range(max_seq_len):
            outputs, _ = self.model.forward(
                src_tokens=input_ids,
                encoder_padding_mask=encoder_padding_mask,
                prev_output_tokens=preds,
            )
            next_token_logits = outputs[:, -1, :]
            preds = search.step(ids=preds, logits=next_token_logits)

            if search.is_done:
                break

        seqs = search.finalize(ids=preds)
        eos_mask = (seqs == self.tokenizer.eos_token_id).cumsum(dim=1) > 0
        seqs[eos_mask] = self.tokenizer.pad_token_id

        tsrcs = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        tpreds = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        tlabels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["score"]
        self.log("val_bleu", bleu_score, sync_dist=True)

    def test_step(self, batch, batch_idx):
        num_beams = 5
        batch_size, seq_len = batch["input_ids"].shape
        max_seq_len = int(seq_len * 1.5)
        # max_seq_len = 1024
        batch_beam_size = batch_size * num_beams
        input_ids = batch["input_ids"].repeat_interleave(num_beams, dim=0)

        search = BeamSearch(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_beams=num_beams,
            max_length=max_seq_len + 1,  # prompt is only <BOS>
            device=input_ids.device,
        )

        preds = (
            torch.ones((batch_beam_size, 1), dtype=torch.long).to(input_ids.device)
            * self.tokenizer.bos_token_id
        )
        encoder_padding_mask = input_ids.eq(self.tokenizer.pad_token_id).to(
            input_ids.device
        )

        for idx in range(max_seq_len):
            outputs, _ = self.model.forward(
                src_tokens=input_ids,
                encoder_padding_mask=encoder_padding_mask,
                prev_output_tokens=preds,
            )
            next_token_logits = outputs[:, -1, :]
            preds = search.step(ids=preds, logits=next_token_logits)

            if search.is_done:
                break

        seqs = search.finalize(ids=preds)
        eos_mask = (seqs == self.tokenizer.eos_token_id).cumsum(dim=1) > 0
        seqs[eos_mask] = self.tokenizer.pad_token_id

        tsrcs = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        tpreds = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        tlabels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["score"]
        self.log("test_bleu", bleu_score, sync_dist=True)

        res = self.comet.compute(
            sources=tsrcs,
            predictions=tpreds,
            references=tlabels,
            devices=self.config['devices'],
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

        print(f"BLEU: {bleu_score}, COMET: {res['mean_score']}")
        return bleu_score, res["mean_score"]

    def on_test_epoch_end(self):
        if self.test_per_sample:
            source, target = self.config["language_pair"]

            with open(
                f"mt/res/{self.config['dataset']}/{source}-{target}-{self.model_name}-{self.test_suffix}.json",
                "w",
            ) as f:
                json.dump(self.test_res, f)

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
