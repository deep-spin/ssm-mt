from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams
from x_transformers.x_transformers import Intermediates

from utils.beam_search import BeamSearch
import json
from utils.mt.comet import load_comet
from transformers.optimization import get_inverse_sqrt_schedule
from transformers import PreTrainedTokenizerFast
import evaluate

from models.hybrids.helpers.mistral_attention import MistralAttentionWrapper

# from models.hybrids.helpers.self_attenttion import SelfAttentionWrapper
from models.hybrids.helpers.mamba import MambaDecoder


class HybridMistralMamba(pl.LightningModule):
    is_encoder_decoder = False
    model_name = "mamba_mistral"
    configs = {
        "default": {
            # mamba config
            "d_model": 624,
            "n_layer": 24,
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
        d_model=None,
        n_layer=None,
        rms_norm=None,
        fused_add_norm=None,
        use_fast_path=None,
        dropout=0,
        use_padding=None,
        precision="bf16-mixed",
        test_per_sample=True,
        layers_config_key="interleaved",
        window_size=64,
        test=False,
        **kwargs,
    ):
        super().__init__()

        self.dec_config = MambaConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layer=n_layer,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            use_fast_path=use_fast_path,
            ssm_cfg={"dropout": dropout},
        )

        assert layers_config_key in [
            "interleaved",
            "h3",
            "reverse_h3",
            "griffin",
        ], "unknown layer key"
        layers = {
            "interleaved": [i for i in range(n_layer) if i % 2 == 1],
            "h3": [1, 11],
            "reverse_h3": [11, 23],
            "griffin": [],
        }
        self.layers = layers[layers_config_key]

        self.decoder = MambaDecoder(
            config=self.dec_config,
            layer_dict=dict((i, MistralAttentionWrapper) for i in self.layers),
            layer_kwargs={"window_size": window_size},
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
        input_ids,
        attention_mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
    ):

        logits = self.decoder.forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
        )
        return logits

    def training_step(self, batch, batch_idx):
        ids, attention_mask, labels = (
            batch["input_ids"][:, :-1].contiguous(),
            batch["attention_mask"][:, :-1].to(torch.bool).contiguous(),
            batch["input_ids"][:, 1:].contiguous(),
        )

        lm_logits = self.forward(input_ids=ids, attention_mask=attention_mask)
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
        input_ids, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        batch_size, seq_len = input_ids.shape
        position_ids = None
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

        done = torch.tensor([False] * batch_size).to(input_ids.device)

        for idx in range(labels.size(1)):

            if idx > 0:
                last_tokens = input_ids[:, -1:]  # (B, 1)

            outputs = self.forward(
                input_ids=input_ids if idx == 0 else last_tokens,
                position_ids=position_ids,
                inference_params=inference_params,
                attention_mask=attention_mask if idx == 0 else None,
                num_last_tokens=1,
            )

            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            inference_params.seqlen_offset += 1

            is_eos = next_token == self.tokenizer.eos_token_id
            done = done | is_eos.squeeze(-1)
            if done.all():
                break

        # mask source sentence
        source_mask = ((input_ids == self.tokenizer.sep_token_id).cumsum(dim=1)) == 0
        input_ids[source_mask] = self.tokenizer.pad_token_id

        eos_mask = ((input_ids == self.tokenizer.eos_token_id).cumsum(dim=1)) > 0
        input_ids[eos_mask] = self.tokenizer.eos_token_id

        tpreds = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        tlabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["score"]

        self.log("val_bleu", bleu_score, sync_dist=True)
        return bleu_score

    def test_step(self, batch, batch_idx):
        """beam search with parallel formulation"""
        num_beams = 5
        maxseq_len = 1024
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        beam_size = num_beams * batch_size
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)

        search = BeamSearch(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_beams=num_beams,
            max_length=maxseq_len + seq_len,
            device=input_ids.device,
        )

        cache = self.allocate_inference_cache(
            batch_size=beam_size,
            max_seqlen=maxseq_len + seq_len + 1,  # source + BOS
            dtype=self.precision,
        )
        inference_params = InferenceParams(
            max_seqlen=maxseq_len + seq_len + 1,
            max_batch_size=beam_size,
            key_value_memory_dict=cache,
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        for idx in range(maxseq_len):
            if idx > 0:
                last_tokens = input_ids[:, -1:]  # (B, 1)

            outputs = self.forward(
                input_ids=input_ids if idx == 0 else last_tokens,
                inference_params=inference_params,
                attention_mask=attention_mask if idx == 0 else None,
                num_last_tokens=1,
            )

            next_token_logits = outputs[:, -1, :]
            input_ids, cache = search.step(
                ids=input_ids,
                logits=next_token_logits,
                cache=inference_params.key_value_memory_dict,
                reorder_cache_fn=self._reorder_cache,
            )
            inference_params.seqlen_offset += 1

            if search.is_done:
                break

        seqs = search.finalize(ids=input_ids)

        source_mask = (seqs == self.tokenizer.sep_token_id).cumsum(dim=1) == 0

        src = seqs.clone()
        src[~source_mask] = self.tokenizer.pad_token_id
        tsrcs = self.tokenizer.batch_decode(src, skip_special_tokens=True)
        seqs[source_mask] = self.tokenizer.pad_token_id

        eos_mask = ((seqs == self.tokenizer.eos_token_id).cumsum(dim=1)) > 0
        seqs[eos_mask] = self.tokenizer.eos_token_id

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

        print(f"bleu: {bleu_score}, comet: {res['mean_score']}")

    def on_test_epoch_end(self):
        if self.test_per_sample:
            source, target = self.config["language_pair"]

            with open(
                f"mt/res/{self.config['dataset']}/{self.config['dataset']}-{source}-{target}-{self.model_name}.json",
                "w",
            ) as f:
                json.dump(self.test_res, f)

    def _reorder_cache(self, cache, beam_idx):
        device = None
        for layer_idx in range(len(cache)):

            if layer_idx not in self.layers:
                device = cache[layer_idx][0].device
                beam_idx = beam_idx.to(device)
                # {0:(conv_state, ssm_state)}
                cache[layer_idx] = (
                    cache[layer_idx][0].index_select(0, beam_idx),
                    cache[layer_idx][1].index_select(0, beam_idx),
                )
            else:
                cache[layer_idx].reorder_cache(beam_idx)
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
