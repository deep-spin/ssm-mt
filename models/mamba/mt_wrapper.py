import time
import pytorch_lightning as pl
import evaluate
import torch
import torch.nn.functional as F
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams
from transformers.optimization import get_inverse_sqrt_schedule
from utils.beam_search import BeamSearch
from utils.mt.comet import load_comet
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

        memory_usage_gb = memory_usage / (1024 ** 3)  # Convert to gigabytes

        print(f"Execution Time (ms): {exec_time}")
        print(f"Memory Usage (GB): {memory_usage_gb}")

        return res, exec_time, memory_usage_gb

    return wrapper


class MambaMT(pl.LightningModule):
    is_encoder_decoder = False
    is_concat = False
    model_name = "mamba"
    configs = {
        "default": {
            "d_model": 610,
            "n_layer": 24,
            "rms_norm": True,
            "fused_add_norm": True,
            "use_fast_path": False,
        },
        "xl": {
            "d_model": 1280,
            "n_layer": 32,
            "rms_norm": True,
            "fused_add_norm": True,
            "use_fast_path": True,
        },
    }

    def __init__(
        self,
        config=None,
        tokenizer=None,
        vocab_size=None,
        d_model=None,
        n_layer=None,
        rms_norm=None,
        fused_add_norm=False,
        use_padding=True,
        use_fast_path=False,
        precision=None,  # default is bf16-mixed
        dropout=None,
        device=None,
        test=False,
        test_per_sample=False,
        test_suffix="",
        **kwargs,
    ):
        super().__init__()

        cfg = MambaConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layer=n_layer,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            use_fast_path=use_fast_path,
            ssm_cfg={"dropout": dropout},
        )

        # FIXME dtype hack, should be set dynamically
        self.model = MambaLMHeadModel(
            device=device,
            config=cfg,
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
            self.test_suffix = test_suffix

    def training_step(self, batch, batch_idx):
        ids, labels = (
            batch["input_ids"][:, :-1].contiguous(),
            batch["input_ids"][:, 1:].contiguous(),
        )

        attention_mask = (
            (ids != self.tokenizer.pad_token_id) if self.use_padding else None
        )
        lm_logits = self.model.forward(
            input_ids=ids, attention_mask=attention_mask
        ).logits

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
        input_ids, labels = batch["input_ids"], batch["labels"]

        batch_size, seq_len = input_ids.shape

        max_length = 256

        attention_mask = (
            (input_ids != self.tokenizer.pad_token_id) if self.use_padding else None
        )

        cache = self.model.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_length + seq_len,
            dtype=self.precision,
        )
        inference_params = InferenceParams(
            max_seqlen=max_length + seq_len,
            max_batch_size=batch_size,
            key_value_memory_dict=cache,
        )

        preds = self.model.generate(
            input_ids,
            max_length=max_length + seq_len,
            eos_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask,
            inference_params=inference_params,
            # cg=True
        ).clone()

        # Create a cumulative sum mask where positions after EOS become True
        eos_token_id = self.tokenizer.eos_token_id
        eos_mask = (preds == eos_token_id).cumsum(dim=1) > 0
        preds[eos_mask] = self.tokenizer.pad_token_id

        # mask source sentence
        source_mask = (preds == self.tokenizer.sep_token_id).cumsum(dim=1) == 0
        preds[source_mask] = self.tokenizer.pad_token_id

        tpreds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        tlabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu_score = self.bleu.compute(predictions=tpreds, references=tlabels)["score"]

        self.log("val_bleu", bleu_score, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """beam search with parallel formulation"""
        num_beams = 5
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        maxseq_len = int(seq_len * 2.5)
        beam_size = num_beams * batch_size
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)

        cache = self.model.allocate_inference_cache(
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

        position_ids = None

        # attn mask is not used in .step(), no need to update
        attention_mask = (
            (input_ids != self.tokenizer.pad_token_id) if self.use_padding else None
        )
        for idx in range(maxseq_len):
            if idx > 0:
                last_tokens = input_ids[:, -1:]  # (B, 1)
                position_ids = torch.full(
                    (batch_size, 1),
                    inference_params.seqlen_offset,
                    dtype=torch.long,
                    device=input_ids.device,
                )

            outputs = self.model.forward(
                input_ids=input_ids if idx == 0 else last_tokens,
                position_ids=position_ids,
                inference_params=inference_params,
                attention_mask=attention_mask,
                num_last_tokens=1,
            ).logits

            next_token_logits = outputs[:, -1, :]
            input_ids, cache = search.step(
                ids=input_ids,
                logits=next_token_logits,
                cache=inference_params.key_value_memory_dict,
                reorder_cache_fn=self._reorder_cache,
            )
            inference_params.seqlen_offset += 1
            inference_params.key_value_memory_dict = cache

            # generated EOS for all beams
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
            gpus=len(self.config["devices"]),
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

    @torch_profiler
    def test_step_profiled(self, batch, batch_idx, tokens):
        """beam search with parallel formulation"""
        num_beams = 5
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        maxseq_len = tokens
        beam_size = num_beams * batch_size
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)


        cache = self.model.allocate_inference_cache(
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

        position_ids = None

        # attn mask is not used in .step(), no need to update
        attention_mask = (
            (input_ids != self.tokenizer.pad_token_id) if self.use_padding else None
        )

        for idx in range(maxseq_len):
            if idx > 0:
                last_tokens = input_ids[:, -1:]  # (B, 1)
                position_ids = torch.full(
                    (batch_size, 1),
                    inference_params.seqlen_offset,
                    dtype=torch.long,
                    device=input_ids.device,
                )

            outputs = self.model.forward(
                input_ids=input_ids if idx == 0 else last_tokens,
                position_ids=position_ids,
                inference_params=inference_params,
                attention_mask=attention_mask,
                num_last_tokens=1,
            ).logits

            next_token_logits = outputs[:, -1, :]
            input_ids, cache = search.step(
                ids=input_ids,
                logits=next_token_logits,
                cache=inference_params.key_value_memory_dict,
                reorder_cache_fn=self._reorder_cache,
            )
            inference_params.seqlen_offset += 1
            inference_params.key_value_memory_dict = cache

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


        return bleu_score

    def on_test_epoch_end(self):
        if self.test_per_sample:
            source, target = self.config["language_pair"]

            with open(
                f"mt/res/{self.config['dataset']}/{self.config['dataset']}-{source}-{target}-{self.model_name}-{self.test_suffix}.json",
                "w",
            ) as f:
                json.dump(self.test_res, f)

    def _reorder_cache(self, cache, beam_idx):
        for layer_idx in range(len(cache)):
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
