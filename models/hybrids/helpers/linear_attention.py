from typing import Optional
from functools import partial

import torch
import torch.nn as nn

from mamba_ssm.utils.generation import InferenceParams
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


from fla.layers.linear_attn import LinearAttention


class LinearAttentionWrapper(nn.Module):

    def __init__(
        self,
        layer_idx=0,
        d_model: int = 512,
        n_head: int = 8,
        rms_norm: bool = True,
        norm_epsilon: float = 1e-5,
        residual_in_fp32=True,
        fused_add_norm=True,
        feature_map="hedgehog",
        output_norm="identity",
        mode: str = "fused_chunk",
        **kwargs,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.mode: str = mode
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.attention = LinearAttention(
            d_model=d_model,
            num_heads=n_head,
            feature_map=feature_map,
            output_norm=output_norm,
            mode=mode,
        )

        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm,
            eps=norm_epsilon,  # **factory_kwargs
        )

        self.norm = norm_cls(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        inference_params: InferenceParams = None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )

        initial_state, output_final_state = None, False
        seqlen_offset, max_seqlen = 0, None

        if inference_params is not None:
            prev_mode = self.attention.mode
            seqlen_offset = inference_params.seqlen_offset
            max_seqlen = inference_params.max_seqlen
            self.attention.mode = "fused_recurrent"
            initial_state = inference_params.key_value_memory_dict[self.layer_idx]
            output_final_state = True

        hidden_states, last_state = self.attention.forward(
            hidden_states,
            initial_state=initial_state,
            output_final_state=output_final_state,
            seqlen_offset=seqlen_offset,
            max_seqlen=max_seqlen,
        )

        if output_final_state:
            inference_params.key_value_memory_dict[self.layer_idx] = last_state
            self.attention.mode = prev_mode

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return None
