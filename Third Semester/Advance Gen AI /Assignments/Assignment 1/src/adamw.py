import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError("lr must be non-negative")

        if eps < 0:
            raise ValueError("eps must be non-negative")

        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        beta1, beta2 = betas

        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")

        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

        # Validate effective values in every parameter group
        for group in self.param_groups:
            self._validate_group(group)

    def _validate_group(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        if lr < 0:
            raise ValueError("lr must be non-negative")

        if eps < 0:
            raise ValueError("eps must be non-negative")

        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")

        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Reject invalid values before modifying any parameter
        for group in self.param_groups:
            self._validate_group(group)

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                grad = p.grad

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Increment only when this parameter is actually updated
                state["step"] += 1
                step = state["step"]

                # First moment
                exp_avg.mul_(beta1)
                exp_avg.add_(grad, alpha=1 - beta1)

                # Second moment
                exp_avg_sq.mul_(beta2)
                exp_avg_sq.addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                # Decoupled weight decay
                with torch.no_grad():
                    p.mul_(1 - lr * weight_decay)

                    # Adaptive Adam update
                    denominator = (
                        exp_avg_sq.sqrt() / bias_correction2 ** 0.5
                    ) + eps

                    p.addcdiv_(
                        exp_avg,
                        denominator,
                        value=-step_size,
                    )

        return loss