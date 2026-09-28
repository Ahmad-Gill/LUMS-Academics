import argparse

import numpy as np
import torch
import os

from src.adamw import AdamW
from src.checkpoint import load_checkpoint, save_checkpoint
from src.data import get_batch, load_token_array
from src.gradient_clipping import gradient_clipping
from src.loss import cross_entropy
from src.schedule import get_lr_cosine_schedule
from src.transformer_lm import TransformerLM


def evaluate_validation(
    model,
    val_tokens,
    batch_size,
    sequence_length,
    num_validation_batches,
    device,
    val_generator,
):
    was_training = model.training

    model.eval()

    total_loss = 0.0

    with torch.inference_mode():
        for _ in range(num_validation_batches):
            x, y = get_batch(
                val_tokens,
                batch_size,
                sequence_length,
                device,
                val_generator,
            )

            logits = model(x)

            loss = cross_entropy(
                logits,
                y,
            )

            total_loss += loss.item()

    validation_loss = (
        total_loss / num_validation_batches
    )

    model.train(was_training)

    return validation_loss


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--val-path",
        type=str,
        required=True,
    )

    # Model
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--n-q-heads",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--n-kv-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--d-ff",
        type=int,
        default=1344,
    )

    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
    )

    parser.add_argument(
        "--norm-eps",
        type=float,
        default=1e-5,
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
    )

    # Optimizer
    parser.add_argument(
        "--learning-rate-max",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--learning-rate-min",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--cosine-steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    # Evaluation / logging
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--num-validation-batches",
        type=int,
        default=10,
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    # Seeds
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    train_tokens = load_token_array(args.train_path)
    val_tokens = load_token_array(args.val_path)

    # --------------------------------------------------
    # Create independent generators
    # --------------------------------------------------

    train_generator = torch.Generator(device="cpu")
    val_generator = torch.Generator(device="cpu")

    train_generator.manual_seed(args.seed)
    val_generator.manual_seed(args.seed + 1)

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_q_heads=args.n_q_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_eps=args.norm_eps,
        device=device,
    )

    # --------------------------------------------------
    # Create optimizer
    # --------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------
    # Load checkpoint if provided
    # --------------------------------------------------

    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        next_step = load_checkpoint(
            args.checkpoint_path,
            model,
            optimizer,
            train_generator,
            val_generator,
        )

        print(f"Resuming training from step {next_step}")

    else:
        next_step = 0

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for step in range(
        next_step,
        args.num_steps,
    ):
        model.train()

        # Learning rate for THIS global step
        lr = get_lr_cosine_schedule(
            step,
            args.learning_rate_max,
            args.learning_rate_min,
            args.warmup_steps,
            args.cosine_steps,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        # Start accumulating gradients
        optimizer.zero_grad()

        train_loss = 0.0

        for _ in range(
            args.gradient_accumulation_steps
        ):
            x, y = get_batch(
                train_tokens,
                args.batch_size,
                args.context_length,
                device,
                train_generator,
            )

            logits = model(x)

            microbatch_loss = cross_entropy(
                logits,
                y,
            )

            # Divide before backward
            loss_for_backward = (
                microbatch_loss
                / args.gradient_accumulation_steps
            )

            loss_for_backward.backward()

            train_loss += microbatch_loss.detach().item()

        train_loss /= args.gradient_accumulation_steps

        # Clip ONCE per global step
        grad_norm = gradient_clipping(
            model.parameters(),
            args.max_grad_norm,
        )

        # Update ONCE per global step
        optimizer.step()

        completed_steps = step + 1

        final_step = (
            completed_steps == args.num_steps
        )

        should_validate = (
            final_step
            or completed_steps % args.eval_interval == 0
        )

        should_log = (
            should_validate
            or completed_steps % args.log_interval == 0
        )

        should_checkpoint = (
            final_step
            or completed_steps % args.checkpoint_interval == 0
        )

        validation_loss = None

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if should_validate:
            validation_loss = evaluate_validation(
                model,
                val_tokens,
                args.batch_size,
                args.context_length,
                args.num_validation_batches,
                device,
                val_generator,
            )

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        if should_log:
            message = (
                f"step={completed_steps} "
                f"train_loss={train_loss:.4f} "
                f"lr={lr:.6g} "
                f"grad_norm={grad_norm:.4f}"
            )

            if validation_loss is not None:
                message += (
                    f" val_loss={validation_loss:.4f}"
                )

            print(message)

        # --------------------------------------------------
        # Checkpoint
        # --------------------------------------------------

        if should_checkpoint:
            if args.checkpoint_path is None:
                raise ValueError(
                    "checkpoint path is required when "
                    "checkpointing is enabled"
                )

            save_checkpoint(
                model,
                optimizer,
                next_step=completed_steps,
                train_generator=train_generator,
                val_generator=val_generator,
                out=args.checkpoint_path,
            )

            print(
                f"Saved checkpoint at step {completed_steps}"
            )


if __name__ == "__main__":
    main()