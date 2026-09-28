import torch
from torch import nn

from src.linear import Linear
from src.silu import SiLU


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.gate = Linear(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )

        self.up = Linear(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )

        self.down = Linear(
            d_ff,
            d_model,
            device=device,
            dtype=dtype,
        )

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.silu(self.gate(x))
        up = self.up(x)

        hidden = gate * up

        return self.down(hidden)