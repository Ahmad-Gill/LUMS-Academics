import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_value = torch.max(x, dim=dim, keepdim=True).values

    exp_x = torch.exp(x - max_value)

    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)