import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        rope_theta: float,
        head_dim: int,
        context_length: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        self.head_dim = head_dim
        self.context_length = context_length

        # Frequencies for each adjacent coordinate pair
        pair_indices = torch.arange(
            0,
            head_dim // 2,
            device=device,
            dtype=torch.float32,
        )

        frequencies = rope_theta ** (-2 * pair_indices / head_dim)

        positions = torch.arange(
            context_length,
            device=device,
            dtype=torch.float32,
        )

        angles = positions[:, None] * frequencies[None, :]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer(
            "cos",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "sin",
            sin,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        if x.shape[-1] != self.head_dim:
            raise ValueError("Last dimension of x must equal head_dim")

        if token_positions.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("token_positions must be an integer tensor")

        if torch.any(token_positions < 0):
            raise ValueError("token_positions are outside the valid context range")

        if torch.any(token_positions >= self.context_length):
            raise ValueError("token_positions are outside the valid context range")
        # Compute in at least float32, then cast back to the input dtype
        in_dtype = x.dtype
        compute_dtype = torch.promote_types(in_dtype, torch.float32)

        # Get cos/sin for the requested positions
        cos = self.cos[token_positions].to(compute_dtype)
        sin = self.sin[token_positions].to(compute_dtype)

        # Add dimensions so cos/sin broadcast over head dimensions
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # Split adjacent coordinates into pairs
        x_pairs = x.to(compute_dtype).reshape(*x.shape[:-1], self.head_dim // 2, 2)

        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        result = torch.stack(
            (rotated_x1, rotated_x2),
            dim=-1,
        )

        return result.reshape_as(x).to(in_dtype)
