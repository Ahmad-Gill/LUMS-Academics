import torch


def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:

    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be one-dimensional")

    if prompt_ids.dtype != torch.long:
        raise TypeError("prompt_ids must have dtype torch.long")

    if prompt_ids.numel() == 0:
        raise ValueError("prompt_ids must not be empty")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    if context_length <= 0:
        raise ValueError("context_length must be positive")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if not hasattr(model, "context_length"):
        raise ValueError(
            "model must expose context_length"
        )

    if context_length != model.context_length:
        raise ValueError(
            "context_length must match model.context_length"
        )

    if max_new_tokens == 0:
        return prompt_ids

    was_training = model.training

    model.eval()

    sequence = prompt_ids.clone()

    with torch.inference_mode():

        for _ in range(max_new_tokens):

            # Keep only the most recent context_length tokens
            context = sequence[-context_length:]

            # Add batch dimension
            model_input = context.unsqueeze(0)

            # Get model logits
            logits = model(model_input)

            # Get logits for the final position
            next_token_logits = logits[0, -1]

            # Temperature scaling
            next_token_logits = (
                next_token_logits / temperature
            )

            # Numerically stable softmax
            max_logit = torch.max(
                next_token_logits
            )

            probabilities = torch.exp(
                next_token_logits - max_logit
            )

            probabilities = probabilities / probabilities.sum()

            # Sort probabilities from largest to smallest
            sorted_probabilities, sorted_indices = torch.sort(
                probabilities,
                descending=True,
            )

            # Cumulative probabilities
            cumulative_probabilities = torch.cumsum(
                sorted_probabilities,
                dim=0,
            )

            # Keep tokens until cumulative probability reaches top_p
            keep = cumulative_probabilities <= top_p

            # Always keep the first token that crosses top_p
            keep[0] = True

            first_above = torch.nonzero(
                cumulative_probabilities >= top_p,
                as_tuple=False,
            )

            if first_above.numel() > 0:
                cutoff = first_above[0].item()
                keep[:cutoff + 1] = True

            # Remove low-probability tokens
            filtered_probabilities = torch.where(
                keep,
                sorted_probabilities,
                torch.zeros_like(sorted_probabilities),
            )

            # Renormalize
            filtered_probabilities = (
                filtered_probabilities
                / filtered_probabilities.sum()
            )

            # Sample a rank in the sorted vocabulary
            sampled_rank = torch.multinomial(
                filtered_probabilities,
                num_samples=1,
                generator=generator,
            )

            # Convert rank back to actual vocabulary ID
            next_token = sorted_indices[sampled_rank]

            # Append generated token
            sequence = torch.cat(
                (
                    sequence,
                    next_token,
                )
            )

            # Stop after generating EOT
            if (
                eot_token_id is not None
                and next_token.item() == eot_token_id
            ):
                break

    # Restore previous mode
    model.train(was_training)

    return sequence