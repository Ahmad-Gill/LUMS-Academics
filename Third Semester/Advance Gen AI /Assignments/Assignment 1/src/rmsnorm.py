import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.norm_eps = norm_eps

        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)

        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True) + self.norm_eps
        )

        result = (x / rms) * self.weight

        return result.to(in_dtype)