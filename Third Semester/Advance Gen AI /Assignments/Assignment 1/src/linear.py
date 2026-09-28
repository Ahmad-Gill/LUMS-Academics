import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Weight shape: [out_features, in_features]
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        # std = sqrt(2 / (din + dout))
        std = (2 / (in_features + out_features)) ** 0.5

        # Truncated normal initialization
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # weight: (out_features, in_features)
        # output: (..., out_features)
        return x @ self.weight.T
