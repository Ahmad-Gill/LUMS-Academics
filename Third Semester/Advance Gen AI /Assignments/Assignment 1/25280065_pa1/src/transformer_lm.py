import torch
from torch import nn

from src.embedding import Embedding
from src.linear import Linear
from src.rmsnorm import RMSNorm
from src.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        rope_theta: float,
        norm_eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_q_heads=n_q_heads,
                    n_kv_heads=n_kv_heads,
                    d_ff=d_ff,
                    context_length=context_length,
                    rope_theta=rope_theta,
                    norm_eps=norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            d_model=d_model,
            norm_eps=norm_eps,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        batch_size, sequence_length = token_ids.shape

        if sequence_length < 1:
            raise ValueError("sequence length must be at least 1")

        if sequence_length > self.context_length:
            raise ValueError(
                "sequence length cannot exceed context_length"
            )

        x = self.token_embedding(token_ids)

        for block in self.blocks:
            x = block(
                x,
                token_positions=token_positions,
            )

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits