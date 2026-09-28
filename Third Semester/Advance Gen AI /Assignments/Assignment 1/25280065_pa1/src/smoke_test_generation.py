import torch

from src.transformer_lm import TransformerLM
from src.generation import generate


CHECKPOINT_PATH = "training/smoke_test/checkpoint.pt"

DEVICE = "mps"
CONTEXT_LENGTH = 256
VOCAB_SIZE = 8192


def main():
    # Create model
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

    # Load checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    # Simple token prompt
    prompt_ids = torch.tensor(
        [1, 100, 200, 300],
        dtype=torch.long,
        device=DEVICE,
    )

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(42)

    output = generate(
        model,
        prompt_ids,
        max_new_tokens=50,
        context_length=CONTEXT_LENGTH,
        temperature=1.0,
        top_p=1.0,
        eot_token_id=None,
        generator=generator,
    )

    print("Prompt token IDs:")
    print(prompt_ids.tolist())

    print("\nGenerated token IDs:")
    print(output.tolist())

    print("\nNumber of tokens:")
    print(output.numel())


if __name__ == "__main__":
    main()