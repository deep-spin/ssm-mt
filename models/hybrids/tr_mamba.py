from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams


from utils.beam_search import BeamSearch
import json
from utils.mt.comet import load_comet
from transformers.optimization import get_inverse_sqrt_schedule
from transformers import PreTrainedTokenizerFast
import evaluate

from x_transformers import TransformerWrapper, Encoder
from models.hybrids.helpers.flash_cross_attention import FlashCrossAttentionWrapper
from models.hybrids.helpers.ffn import FeedForwardWrapper
from models.hybrids.helpers.mamba import MambaDecoder, MixerModel


class HybridTrMamba(pl.LightningModule):
    is_encoder_decoder = True
    is_concat = False  # FIXME remove
    model_name = "tr_mamba"
    configs = {
        "default": {
            "enc_n_layer": 6,
            "enc_n_heads": 8,
            # mamba config
            "d_model": 512,
            "n_layer": 18,
            "rms_norm": True,
            "fused_add_norm": True,
            "use_fast_path": False,
        }
    }

    def __init__(
        self,
        config=None,
        tokenizer=PreTrainedTokenizerFast,
        vocab_size=32000,
        max_seq_len=1024,
        d_model=None,
        n_layer=None,
        enc_n_layer=None,
        enc_n_heads=None,
        rms_norm=None,
        fused_add_norm=None,
        use_fast_path=None,
        dropout=None,
        use_padding=None,
        precision="bf16-mixed",
        test_per_sample=True,
        test=False,
        **kwargs,
    ):
        super().__init__()

        self.config = MambaConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layer=n_layer,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            use_fast_path=use_fast_path,
            ssm_cfg={"dropout": dropout},
        )

        self.encoder = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            # scaled_sinu_pos_emb=True,
            use_abs_pos_emb=False,
            tie_embedding=True,
            attn_layers=Encoder(
                dim=d_model,
                depth=enc_n_layer,
                heads=enc_n_heads,
                layer_dropout=dropout,
                # ff_swish=True,
                attn_flash=True,
            ),
        )

        self.layers = (0, 3, 6, 9, 12, 15)
        x_attention_layers = [
            (i, FlashCrossAttentionWrapper) for i in (1, 4, 7, 10, 13, 16)
        ]
        ffn_layers = [(i, FeedForwardWrapper) for i in (2, 5, 8, 11, 14, 17)]

        layer_dict = dict(x_attention_layers + ffn_layers)

        self.decoder = MambaDecoder(
            config=self.config,
            layer_dict=layer_dict,
        )

        self.tokenizer = tokenizer
        self.bleu = evaluate.load("sacrebleu")
        self.config = config
        self.use_padding = use_padding
        dtype_map = {
            "bf16-mixed": torch.bfloat16,
            "16-true": torch.float16,
            "32-true": torch.float32,
        }
        self.precision = dtype_map[precision]

        if test:
            self.comet = load_comet()
            self.test_per_sample = test_per_sample
            self.test_res = []

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.decoder.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        context_tokens,
        input_ids,
        source_attention_mask=None,
        target_attention_mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
    ):

        source_attention_mask = source_attention_mask.to(torch.bool)
        target_attention_mask = target_attention_mask.to(torch.bool)

        source_vec = self.encoder.forward(
            context_tokens,
            mask=source_attention_mask,
            return_embeddings=True,
        )

        logits = self.decoder.forward(
            input_ids,
            context=source_vec,
            context_mask=source_attention_mask,
            attention_mask=target_attention_mask,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
        )
        return logits

    def training_step(self, batch, batch_idx):
        source, target, source_attention_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )

        target_attention_mask = (
            (target != self.tokenizer.pad_token_id).to(torch.bool).to(source.device)
        )

        lm_logits = self.forward(
            context_tokens=source,
            source_attention_mask=source_attention_mask,
            target_attention_mask=target_attention_mask,
            input_ids=target,
        )

        logits = lm_logits[:, :-1].contiguous()
        labels = target[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src_tokens, labels, source_attention_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        batch_size, seq_len = src_tokens.shape
        max_length = 256

        cache = self.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_length + seq_len + 1,  # source + BOS
            dtype=self.precision,
        )
        inference_params = InferenceParams(
            max_seqlen=max_length + seq_len + 1,
            max_batch_size=batch_size,
            key_value_memory_dict=cache,
        )

        done = torch.tensor([False] * batch_size).to(src_tokens.device)
        preds = (
            torch.ones((batch_size, 1), dtype=torch.long).to(src_tokens.device)
            * self.tokenizer.bos_token_id
        )

        source_vec = self.encoder.forward(
            src_tokens,
            mask=source_attention_mask.to(torch.bool),
            return_embeddings=True,
        )

        position_ids = None

        for idx in range(labels.size(1)):

            if idx > 0:
                last_tokens = preds[:, -1:]  # (B, 1)
                position_ids = torch.full(
                    (batch_size, 1),
                    inference_params.seqlen_offset,
                    dtype=torch.long,
                    device=src_tokens.device,
                )

            logits = self.decoder.forward(
                input_ids=preds if idx == 0 else last_tokens,
                context=source_vec,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            )

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            preds = torch.cat((preds, next_token), dim=-1)
            inference_params.seqlen_offset += 1

            is_eos = next_token == self.tokenizer.eos_token_id
            done = done | is_eos.squeeze(-1)

            if done.all():
                break

        # Create a cumulative sum mask where positions after EOS become True
        eos_token_id = self.tokenizer.eos_token_id
        eos_mask = (preds == eos_token_id).cumsum(dim=1) > 0
        preds[eos_mask] = self.tokenizer.pad_token_id

        tpreds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        tlabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["score"]

        self.log("val_bleu", bleu_score, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """beam search with parallel formulation"""
        num_beams = 5
        maxseq_len = 256

        source_tokens, labels, source_attention_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        batch_size, seq_len = source_tokens.shape
        beam_size = num_beams * batch_size
        input_ids = source_tokens.repeat_interleave(num_beams, dim=0)
        source_attention_mask = source_attention_mask.repeat_interleave(
            num_beams, dim=0
        )

        cache = self.allocate_inference_cache(
            batch_size=beam_size,
            max_seqlen=maxseq_len + seq_len,
            dtype=self.precision,
        )
        inference_params = InferenceParams(
            max_seqlen=maxseq_len + seq_len,
            max_batch_size=beam_size,
            key_value_memory_dict=cache,
        )

        search = BeamSearch(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_beams=num_beams,
            max_length=maxseq_len + seq_len,
            device=input_ids.device,
        )

        source_vec = self.encoder.forward(
            input_ids=input_ids,
            mask=source_attention_mask,
            return_embeddings=True,
        )

        position_ids = None
        preds = (
            torch.ones((beam_size, 1), dtype=torch.long).to(input_ids.device)
            * self.tokenizer.bos_token_id
        )

        for idx in range(maxseq_len):
            if idx > 0:
                last_tokens = preds[:, -1:]  # (B, 1)
                position_ids = torch.full(
                    (beam_size, 1),
                    inference_params.seqlen_offset,
                    dtype=torch.long,
                    device=input_ids.device,
                )

            outputs = self.decoder.forward(
                input_ids=preds if idx == 0 else last_tokens,
                context=source_vec,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            )

            next_token_logits = outputs[:, -1, :]
            preds, cache = search.step(
                ids=preds,
                logits=next_token_logits,
                cache=inference_params.key_value_memory_dict,
                reorder_cache_fn=self._reorder_cache,
            )
            inference_params.seqlen_offset += 1
            inference_params.key_value_memory_dict = cache

            # generated EOS for all beams
            if search.is_done:
                break

        seqs = search.finalize(ids=preds)

        eos_mask = (seqs == self.tokenizer.eos_token_id).cumsum(dim=1) > 0

        seqs[eos_mask] = self.tokenizer.pad_token_id

        tsrcs = self.tokenizer.batch_decode(source_tokens, skip_special_tokens=True)
        tpreds = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        tlabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

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

        print(f"bleu: {bleu_score}, comet: {res['mean_score']}")
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
        for layer_idx in self.layers:
            device = cache[layer_idx][0].device
            # {0:(conv_state, ssm_state)}
            cache[layer_idx] = (
                cache[layer_idx][0].index_select(0, beam_idx.to(device)),
                cache[layer_idx][1].index_select(0, beam_idx.to(device)),
            )
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
