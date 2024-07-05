import time
from typing import Any
from transformers import LlamaForCausalLM, LlamaConfig, PreTrainedTokenizerFast
import pytorch_lightning as pl
import evaluate
import torch.nn.functional as F
import torch
from utils.beam_search import BeamSearch
from utils.mt import load_comet
from transformers import get_inverse_sqrt_schedule
import json

from torch.profiler import profile, record_function, ProfilerActivity
from functools import wraps


def torch_profiler(method):
    @wraps(method)
    def wrapper(self, batch, batch_idx, tokens, *args, **kwargs):
        with profile(
            activities=[ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=False,
        ) as prof:
            with record_function("model_inference"):
                res = method(self, batch, batch_idx, tokens, *args, **kwargs)

        exec_time = (
            sum(event.cuda_time_total for event in prof.key_averages()) * 1e-6
        )  # Convert to milliseconds

        memory_usage = sum(event.cuda_memory_usage for event in prof.key_averages())

        memory_usage_gb = memory_usage / (1024**3)  # Convert to gigabytes

        print(f"Execution Time (ms): {exec_time}")
        print(f"Memory Usage (GB): {memory_usage_gb}")

        return res, exec_time, memory_usage_gb

    return wrapper


class LlamaMT(pl.LightningModule):
    is_encoder_decoder = False
    is_concat = False
    model_name = "llama"

    configs = {
        "default": {
            "hidden_size": 496,  # 79M
            "intermediate_size": 4,  # multiplier
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
        },
        "xl": {
            "hidden_size": 1280,
            "intermediate_size": 4,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
        },
    }

    def __init__(
        self,
        config=None,
        tokenizer: PreTrainedTokenizerFast = None,
        vocab_size=None,
        hidden_size=None,
        num_hidden_layers=8,
        intermediate_size=None,
        num_attention_heads=8,
        use_padding=True,
        test=False,
        test_per_sample=False,
        test_suffix="",
        **kwargs,
    ):
        super().__init__()
        cfg = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * intermediate_size,
            _attn_implementation="flash_attention_2",
        )
        self.model = LlamaForCausalLM(
            config=cfg,
        )
        self.tokenizer = tokenizer
        self.bleu = evaluate.load("sacrebleu")
        self.config = config
        self.use_padding = use_padding

        if test:
            self.comet = load_comet()
            self.test_per_sample = test_per_sample
            self.test_res = []
            self.test_suffix = test_suffix

    def training_step(self, batch, batch_idx):
        ids, labels = (
            batch["input_ids"][:, :-1].contiguous(),
            batch["input_ids"][:, 1:].contiguous(),
        )

        attention_mask = batch["attention_mask"][:, :-1] if self.use_padding else None
        lm_logits = self.model.forward(
            input_ids=ids, attention_mask=attention_mask
        ).logits

        sep_mask = (ids == self.tokenizer.sep_token_id).cumsum(dim=1) > 0
        labels[~sep_mask] = self.tokenizer.pad_token_id

        loss = F.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=0.1,
        )
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, labels = batch["input_ids"], batch["labels"]

        preds = []
        done = torch.tensor([False] * ids.size(0)).to(ids.device)

        ones = torch.ones((ids.size(0), 1), dtype=torch.long).to(ids.device)
        cache = None

        attention_mask = (
            (ids != self.tokenizer.pad_token_id) if self.use_padding else None
        )

        for i in range(labels.size(1)):
            out = self.model.forward(
                input_ids=ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=cache,
            )

            lm_logits = out.logits
            cache = out.past_key_values

            next_token_logits = lm_logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            attention_mask = (
                torch.cat((attention_mask, ones), dim=-1) if self.use_padding else None
            )
            ids = next_token
            preds.append(next_token)

            is_eos = next_token == self.tokenizer.eos_token_id
            done = done | is_eos.squeeze(-1)

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

    @torch_profiler
    def test_step_profiled(self, batch, batch_idx, tokens):
        """beam search with parallel formulation"""
        num_beams = 5
        input_ids: torch.Tensor = batch["input_ids"]
        batch_size, decoder_prompt_len = input_ids.shape
        max_seq_len = tokens
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)

        search = BeamSearch(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_beams=num_beams,
            max_length=decoder_prompt_len + max_seq_len,
            device=input_ids.device,
            # decoder_prompt_len=decoder_prompt_len,
        )
        cache = None

        for idx in range(max_seq_len):
            attention_mask = (
                (input_ids != self.tokenizer.pad_token_id).to(input_ids.device)
                if self.use_padding
                else None
            )

            outputs = self.model.forward(
                input_ids=input_ids if idx == 0 else last_tokens,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=cache,
            )

            lm_logits = outputs.logits
            cache = outputs.past_key_values

            next_token_logits = lm_logits[:, -1, :]
            input_ids, cache = search.step(
                ids=input_ids,
                logits=next_token_logits,
                cache=cache,
                reorder_cache_fn=self._reorder_cache,
            )
            last_tokens = input_ids[:, -1:]

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


        print(f"BLEU: {bleu_score}")
        return (
            bleu_score,
     
        )  

    def test_step(self, batch, batch_idx):
        """beam search with parallel formulation"""
        num_beams = 5
        input_ids: torch.Tensor = batch["input_ids"]
        batch_size, decoder_prompt_len = input_ids.shape
        max_seq_len = int(decoder_prompt_len * 2.5)
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)

        search = BeamSearch(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_beams=num_beams,
            max_length=decoder_prompt_len + max_seq_len,
            device=input_ids.device,
            # decoder_prompt_len=decoder_prompt_len,
        )
        cache = None

        for idx in range(max_seq_len):
            attention_mask = (
                (input_ids != self.tokenizer.pad_token_id).to(input_ids.device)
                if self.use_padding
                else None
            )

            outputs = self.model.forward(
                input_ids=input_ids if idx == 0 else last_tokens,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=cache,
            )

            lm_logits = outputs.logits
            cache = outputs.past_key_values

            next_token_logits = lm_logits[:, -1, :]
            input_ids, cache = search.step(
                ids=input_ids,
                logits=next_token_logits,
                cache=cache,
                reorder_cache_fn=self._reorder_cache,
            )
            last_tokens = input_ids[:, -1:]

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

        print(f"BLEU: {bleu_score}, COMET: {res['mean_score']}")
        return bleu_score, res["mean_score"]

    def on_test_epoch_end(self):
        if self.test_per_sample:
            source, target = self.config["language_pair"]

            with open(
                f"mt/res/{self.config['dataset']}/{self.config['dataset']}-{source}-{target}-{self.model_name}-{self.test_suffix}.json",
                "w",
            ) as f:
                json.dump(self.test_res, f)

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

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
