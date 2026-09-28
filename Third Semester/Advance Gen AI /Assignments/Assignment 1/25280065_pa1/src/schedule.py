import math
import numbers


def get_lr_cosine_schedule(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    cosine_end: int,
) -> float:

    # bool is a subclass of int, so reject it explicitly
    for name, value in (
        ("step", step),
        ("warmup_steps", warmup_steps),
        ("cosine_end", cosine_end),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be an integer")

    if step < 0:
        raise ValueError("step must be non-negative")

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    if warmup_steps >= cosine_end:
        raise ValueError("warmup_steps must be less than cosine_end")

    if min_lr < 0 or max_lr < 0:
        raise ValueError("learning rates must be non-negative")

    if min_lr > max_lr:
        raise ValueError("min_lr must be less than or equal to max_lr")

    # Warmup
    if step < warmup_steps:
        return float((step / warmup_steps) * max_lr)

    # After cosine decay
    if step >= cosine_end:
        return float(min_lr)

    # Cosine decay
    progress = (
        (step - warmup_steps)
        / (cosine_end - warmup_steps)
    )

    cosine = 0.5 * (
        1 + math.cos(math.pi * progress)
    )

    return float(min_lr + cosine * (max_lr - min_lr))
