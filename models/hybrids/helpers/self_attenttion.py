from typing import Optional
from functools import partial

import torch
import torch.nn as nn

from mamba_ssm.utils.generation import InferenceParams
from flash_attn.modules.mha import MHA
from torchscale.component.multihead_attention import MultiheadAttention
from torchscale.architecture.config import DecoderConfig

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class SelfAttentionWrapper(nn.Module):

    def __init__(
        self,
        layer_idx=0,
        d_model: int = 512,
        n_head: int = 8,
        rms_norm: bool = True,
        norm_epsilon: float = 1e-5,
        residual_in_fp32=True,
        fused_add_norm=True,
        dropout=0,
        **kwargs,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        config = DecoderConfig(xpos_rel_pos=True)

        self.attention = MultiheadAttention(
            args=config,
            embed_dim=d_model,
            num_heads=n_head,
            self_attention=True,
            dropout=dropout,
        )

        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm,
            eps=norm_epsilon,  # **factory_kwargs
        )

        self.norm = norm_cls(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
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

        incremental_state = None
        is_first_step = False

        if inference_params is not None and inference_params.seqlen_offset == 0:
            incremental_state = {}
            is_first_step = True

        elif inference_params is not None and inference_params.seqlen_offset > 0:
            incremental_state = inference_params.key_value_memory_dict[self.layer_idx]

        self_attn_mask = torch.triu(
            torch.zeros([hidden_states.size(1), hidden_states.size(1)])
            .float()
            .fill_(float("-inf"))
            .type_as(hidden_states),
            1,
        )
        hidden_states, _ = self.attention.forward(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=~mask,
            attn_mask=self_attn_mask,
            incremental_state=incremental_state,
            is_first_step=is_first_step,
            is_causal=True,
        )

        if incremental_state is not None:
            inference_params.key_value_memory_dict[self.layer_idx] = incremental_state

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return None
