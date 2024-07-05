from typing import Optional
from functools import partial

import torch
import torch.nn as nn

from mamba_ssm.utils.generation import InferenceParams
from flash_attn.modules.mha import MHA

from x_transformers.x_transformers import Attention, RotaryEmbedding

from local_attention.transformer import LocalMHA


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class LocalAttentionWrapper(nn.Module):

    def __init__(
        self,
        layer_idx=0,
        d_model: int = 512,
        n_head: int = 8,
        rms_norm: bool = True,
        norm_epsilon: float = 1e-5,
        residual_in_fp32=True,
        fused_add_norm=True,
        window_size=64,
        dropout=0,
        **kwargs,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.rotary_pos_emb = RotaryEmbedding(
            dim=16,
        )

        self.attention = LocalMHA(
            dim=d_model,
            dim_head=d_model // n_head,
            heads=n_head,
            window_size=1,
            causal=True,
            dropout=dropout,
            look_backward=window_size,  # measured in window_size
            use_rotary_pos_emb=False
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

        hidden_states = self.attention.forward(
            x=hidden_states,
            mask=mask,
        )

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        # return self.attention.allocate_inference_cache(
        #     batch_size=batch_size, max_seqlen=max_seqlen, dtype=dtype, **kwargs
        # )

        return None
