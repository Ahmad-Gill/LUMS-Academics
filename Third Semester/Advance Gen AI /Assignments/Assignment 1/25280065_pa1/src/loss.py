import torch


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    # Get the largest logit for numerical stability
    max_logits = torch.max(
        logits,
        dim=-1,
        keepdim=True,
    ).values

    # Shift logits
    shifted_logits = logits - max_logits

    # Compute log(sum(exp(logits))) using the max shift
    log_sum_exp = (
        torch.log(
            torch.sum(
                torch.exp(shifted_logits),
                dim=-1,
                keepdim=True,
            )
        )
        + max_logits
    ).squeeze(-1)

    # Get the logit of the correct target
    target_logits = logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # Cross-entropy
    loss = -target_logits + log_sum_exp

    # Average over all positions
    return loss.mean()