import math
import torch

from src.data import load_token_array, get_batch
from src.transformer_lm import TransformerLM
from src.loss import cross_entropy


CHECKPOINT_PATH = "training/checkpoints/final_checkpoint.pt"
VAL_PATH = "data/tinystories/data/validation.bin"

BATCH_SIZE = 16
SEQUENCE_LENGTH = 256
NUM_BATCHES = 100
SEED = 42

DEVICE = "mps"


def main():
    val_data = load_token_array(VAL_PATH)

    model = TransformerLM(
        vocab_size=8192,
        context_length=256,
        d_model=512,
        num_layers=4,
        n_q_heads=16,
        n_kv_heads=4,
        d_ff=1344,
        rope_theta=10000.0,
        norm_eps=1e-5,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    # Fresh generator required by the assignment
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)

    total_loss = 0.0

    with torch.no_grad():
        for _ in range(NUM_BATCHES):
            inputs, targets = get_batch(
                val_data,
                BATCH_SIZE,
                SEQUENCE_LENGTH,
                DEVICE,
                generator,
            )

            logits = model(inputs)

            loss = cross_entropy(
                logits,
                targets,
            )

            total_loss += loss.item()

    mean_ce = total_loss / NUM_BATCHES
    perplexity = math.exp(mean_ce)

    print(f"Validation batches: {NUM_BATCHES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Validation seed: {SEED}")
    print(f"Mean cross-entropy: {mean_ce:.4f}")
    print(f"Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()