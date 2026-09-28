import torch


def gradient_clipping(
    parameters,
    max_norm: float,
    eps: float = 1e-6,
) -> float:

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    total_squared_norm = 0.0
    gradients = []

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = parameter.grad

        if gradient.is_sparse:
            raise RuntimeError(
                "gradient clipping does not support sparse gradients"
            )

        gradients.append(gradient)

        # Square in at least float32 so fp16 gradients cannot overflow
        g = gradient.detach().to(
            torch.promote_types(gradient.dtype, torch.float32)
        )
        total_squared_norm += torch.sum(g * g).item()

    if len(gradients) == 0:
        return 0.0

    total_norm = total_squared_norm ** 0.5

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)

        with torch.no_grad():
            for gradient in gradients:
                gradient.mul_(scale)

    return float(total_norm)