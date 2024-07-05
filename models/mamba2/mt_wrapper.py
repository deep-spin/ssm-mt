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

from einops import rearrange


class Mamba2MT(pl.LightningModule):
    is_encoder_decoder = False
    is_concat = False
    model_name = "mamba2"
    configs = {
        "default": {
            "d_model": 512,
            "n_layer": 36,
            "rms_norm": True,
            "fused_add_norm": True,
            "use_fast_path": True,
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
            ssm_cfg={"layer": "Mamba2", "dropout": dropout},
        )

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

    def pack_2d(self, tokens, cu_seqlens):
        """
        pack function: convert tokens to packed_tokens (batch_size=1)

        Args:
        tokens (torch.Tensor): Input tensor of shape (batch_size, max_seq_len)
        cu_seqlens (torch.Tensor): Cumulative sequence lengths tensor

        Returns:
        torch.Tensor: Packed tokens of shape (total_tokens,)
        """
        batch_size, max_seq_len = tokens.shape
        seq_len_list = cu_seqlens[1:] - cu_seqlens[:-1]

        # Create a mask for valid tokens
        indices_2d = (
            torch.arange(max_seq_len, device=tokens.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        mask_2d = indices_2d < seq_len_list.unsqueeze(1)

        # Apply the mask and flatten the result
        packed_tokens = tokens[mask_2d]

        return packed_tokens

    def unpack_3d(self, packed_hidden_states, cu_seqlens):
        batch_size = cu_seqlens.shape[0] - 1
        seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        packed_hidden_states = packed_hidden_states.squeeze(0)

        ori_indices = (
            torch.arange(seq_len, device=cu_seqlens.device)
            .unsqueeze(0)
            .expand((batch_size, seq_len))
        )

        ori_indices = (ori_indices + cu_seqlens[:-1].unsqueeze(1)) % (
            len(packed_hidden_states)
        )

        return packed_hidden_states[ori_indices]

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = (
            batch["input_ids"][:, :-1].contiguous(),
            batch["attention_mask"][:, :-1].to(torch.bool).contiguous(),
            batch["input_ids"][:, 1:].contiguous(),
        )
        batch_size = attention_mask.shape[0]

        seqlens = attention_mask.sum(dim=1, dtype=torch.int32)
        seq_idx = torch.cat(
            [
                torch.full((seqlen,), i, dtype=torch.int32, device=input_ids.device)
                for i, seqlen in enumerate(seqlens)
            ],
            dim=0,
        ).unsqueeze(0)
        cu_seqlens = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=attention_mask.device
        )
        cu_seqlens[1:] = seqlens.cumsum(0)

        packed_ids = input_ids[attention_mask].unsqueeze(0)
        lm_logits = self.model.forward(
            input_ids=packed_ids,
            cu_seqlens=cu_seqlens,
            seq_idx=seq_idx,
        ).logits

        unpacked = self.unpack_3d(lm_logits, cu_seqlens)
        sep_mask = (input_ids == self.tokenizer.sep_token_id).cumsum(dim=1) > 0
        labels[~sep_mask] = self.tokenizer.pad_token_id

        loss = F.cross_entropy(
            unpacked.view(-1, lm_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        batch_size, seq_len = input_ids.shape
        max_length = 512

        seqlens = attention_mask.sum(dim=1, dtype=torch.int32)
        seq_idx = torch.cat(
            [
                torch.full((seqlen,), i, dtype=torch.int32, device=input_ids.device)
                for i, seqlen in enumerate(seqlens)
            ],
            dim=0,
        ).unsqueeze(0)
        cu_seqlens = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=attention_mask.device
        )
        cu_seqlens[1:] = seqlens.cumsum(0)

        done = torch.tensor([False] * batch_size).to(input_ids.device)

        inference_params = InferenceParams(
            max_seqlen=max_length + seq_len,
            max_batch_size=batch_size,
        )

        packed_ids = input_ids[attention_mask.bool()].unsqueeze(0)
        lm_logits = self.model.forward(
            input_ids=packed_ids,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            inference_params=inference_params,
        ).logits

        # rearrange(lm_logits[0, cu_seqlens[1:] - 1], "b d -> b 1 d").shape

        next_token_logits = lm_logits[:, cu_seqlens[1:] - 1].view(batch_size, -1)
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)

        for i in range(1, max_length):
            out = self.model.forward(
                input_ids=next_tokens,
                inference_params=inference_params,
            )
            next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_tokens), dim=-1)
            is_eos = next_tokens == self.tokenizer.eos_token_id
            done = done | is_eos.squeeze(-1)
            if done.all():
                break

        eos_token_id = self.tokenizer.eos_token_id
        eos_mask = (input_ids == eos_token_id).cumsum(dim=1) > 0
        input_ids[eos_mask] = self.tokenizer.pad_token_id

        # mask source sentence
        source_mask = (input_ids == self.tokenizer.sep_token_id).cumsum(dim=1) == 0
        input_ids[source_mask] = self.tokenizer.pad_token_id

        tpreds = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
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
