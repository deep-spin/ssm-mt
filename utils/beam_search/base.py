from utils.beam_search.hf import BeamSearchScorer
from transformers import LogitsProcessorList, PreTrainedTokenizerFast
import torch.nn.functional as F
import torch


class BeamSearch:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int,
        num_beams: int,
        max_length: int,
        logits_processors=None,
        decoder_prompt_len=0,
        device="cpu",
    ):
        self.num_beams = num_beams
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.batch_beam_size = batch_size * num_beams
        self.decoder_prompt_len = decoder_prompt_len
        self.scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=device,
        )

        self.eos_token_id = [tokenizer.eos_token_id]
        self.logits_processor = (
            LogitsProcessorList(logits_processors)
            if logits_processors
            else LogitsProcessorList()
        )

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=device
        )
        beam_scores[:, 1:] = -1e9
        self.beam_scores = beam_scores.view((self.batch_beam_size,))
        self.beam_indices = None

        self.next_tokens = None
        self.next_indices = None

    @property
    def is_done(self):
        return self.scorer.is_done

    def step(self, ids, logits, cache=None, reorder_cache_fn=None):
        next_token_scores = F.log_softmax(
            logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)
        next_token_scores_processed = self.logits_processor(ids, next_token_scores)
        next_token_scores = next_token_scores_processed + self.beam_scores[
            :, None
        ].expand_as(next_token_scores_processed)

        vocab_size = next_token_scores.size(-1)
        next_token_scores = next_token_scores.view(
            self.batch_size, self.num_beams * vocab_size
        )

        n_eos_tokens = len(self.eos_token_id) if self.eos_token_id else 0
        next_token_scores, self.next_tokens = torch.topk(
            next_token_scores,
            max(2, 1 + n_eos_tokens) * self.num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )

        self.next_indices = torch.div(
            self.next_tokens, vocab_size, rounding_mode="floor"
        )
        self.next_tokens = self.next_tokens % vocab_size

        beam_outputs = self.scorer.process(
            ids,
            next_token_scores,
            self.next_tokens,
            self.next_indices,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            beam_indices=self.beam_indices,
            decoder_prompt_len=self.decoder_prompt_len,
        )

        self.beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        ids = torch.cat([ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        if cache is not None:
            assert reorder_cache_fn is not None
            cache = reorder_cache_fn(cache, beam_idx)
            return ids, cache

        return ids

    def finalize(self, ids):
        return self.scorer.finalize(
            ids,
            self.beam_scores,
            self.next_tokens,
            self.next_indices,
            max_length=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            beam_indices=self.beam_indices,
        )["sequences"]
