import math

import torch
from torch import nn
from src.linear import Linear
from src.rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    
        # Q and K must have the same feature dimension
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("queries and keys must have the same feature dimension")
    # K and V must have the same sequence length
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError("keys and values must have the same sequence length")
    # Check mask
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")
        if mask.shape[-2:] != (
            queries.shape[-2],
            keys.shape[-2],
        ):
            raise ValueError(
                "mask must have final dimensions matching query and key lengths"
            )
        if torch.any(mask.sum(dim=-1) == 0):
            raise ValueError(
                "Each query must have at least one unmasked key"
            )
    # Q @ K^T
    scores = queries @ keys.transpose(-2, -1)

    # Scale by sqrt(d_k)
    d_k = queries.shape[-1]
    scores = scores / math.sqrt(d_k)

    # Apply mask before softmax
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")

        # Make sure every query has at least one allowed key
        if torch.any(mask.sum(dim=-1) == 0):
            raise ValueError(
                "Each query must have at least one unmasked key"
            )

        scores = scores.masked_fill(~mask, float("-inf"))

    # Stable softmax
    max_score = torch.max(scores, dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - max_score)
    probabilities = exp_scores / torch.sum(
        exp_scores,
        dim=-1,
        keepdim=True,
    )

    # Attention weights @ V
    return probabilities @ values
class GroupedQuerySelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        context_length: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if d_model % n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")

        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")

        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.group_size = n_q_heads // n_kv_heads

        self.q_proj = Linear(
            d_model,
            n_q_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )

        self.k_proj = Linear(
            d_model,
            n_kv_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )

        self.v_proj = Linear(
            d_model,
            n_kv_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )

        self.output_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )

        self.rope = RotaryPositionalEmbedding(
            rope_theta=rope_theta,
            head_dim=self.head_dim,
            context_length=context_length,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads
        q = q.reshape(
            batch_size,
            sequence_length,
            self.n_q_heads,
            self.head_dim,
        )

        k = k.reshape(
            batch_size,
            sequence_length,
            self.n_kv_heads,
            self.head_dim,
        )

        v = v.reshape(
            batch_size,
            sequence_length,
            self.n_kv_heads,
            self.head_dim,
        )

        # Move heads before sequence
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Q and K get RoPE
        if token_positions is None:
            token_positions = torch.arange(
                sequence_length,
                device=x.device,
            )

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # Group query heads
        q = q.reshape(
            batch_size,
            self.n_kv_heads,
            self.group_size,
            sequence_length,
            self.head_dim,
        )

        # Attention scores
        scores = torch.einsum(
            "bhrid,bhjd->bhrij",
            q,
            k,
        )

        scores = scores / math.sqrt(self.head_dim)

        # Causal mask: j <= i
        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            )
        )

        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf"),
        )

        # Attention probabilities
        max_score = torch.max(
            scores,
            dim=-1,
            keepdim=True,
        ).values

        exp_scores = torch.exp(scores - max_score)

        probabilities = exp_scores / torch.sum(
            exp_scores,
            dim=-1,
            keepdim=True,
        )

        # Attention output
        output = torch.einsum(
            "bhrij,bhjd->bhrid",
            probabilities,
            v,
        )

        # [B, hkv, group, N, head_dim]
        # → [B, N, hq, head_dim]
        output = output.reshape(
            batch_size,
            self.n_q_heads,
            sequence_length,
            self.head_dim,
        )

        output = output.transpose(1, 2)

        # Merge heads
        output = output.reshape(
            batch_size,
            sequence_length,
            self.d_model,
        )

        return self.output_proj(output)