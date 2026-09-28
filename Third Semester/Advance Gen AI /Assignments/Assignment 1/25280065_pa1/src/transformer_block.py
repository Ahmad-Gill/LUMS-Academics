import torch
from torch import nn

from src.rmsnorm import RMSNorm
from src.attention import GroupedQuerySelfAttention
from src.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float,
        norm_eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.attention_norm = RMSNorm(
            d_model=d_model,
            norm_eps=norm_eps,
            device=device,
            dtype=dtype,
        )

        self.attention = GroupedQuerySelfAttention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            context_length=context_length,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )

        self.ffn_norm = RMSNorm(
            d_model=d_model,
            norm_eps=norm_eps,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Attention + residual
        x = x + self.attention(
            self.attention_norm(x),
            token_positions=token_positions,
        )

        # SwiGLU + residual
        x = x + self.ffn(
            self.ffn_norm(x)
        )

        return x