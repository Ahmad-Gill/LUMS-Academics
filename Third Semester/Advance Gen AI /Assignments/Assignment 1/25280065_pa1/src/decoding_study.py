import torch
from tokenizers import Tokenizer

from src.generation import generate
from src.transformer_lm import TransformerLM


CHECKPOINT_PATH = "training/checkpoints/final_checkpoint.pt"
TOKENIZER_PATH = "data/tinystories/tokenizer/tokenizer.json"

DEVICE = "mps"
CONTEXT_LENGTH = 256
MAX_NEW_TOKENS = 100
SEED = 42

PROMPTS = [
    "Once upon a time",
    "The little girl",
    "One day, a small",
]

SETTINGS = [
    ("temperature_0.7", 0.7, 1.0),
    ("temperature_1.0", 1.0, 1.0),
    ("temperature_1.3", 1.3, 1.0),
    ("top_p_0.8", 1.0, 0.8),
    ("top_p_0.9", 1.0, 0.9),
    ("top_p_1.0", 1.0, 1.0),
]


def load_model():
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

    return model


def main():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    model = load_model()

    eot_token_id = tokenizer.token_to_id("<|endoftext|>")

    print("Decoding study")
    print("=" * 80)

    for setting_name, temperature, top_p in SETTINGS:
        print()
        print(f"Setting: {setting_name}")
        print(f"Temperature: {temperature}")
        print(f"Top-p: {top_p}")
        print("-" * 80)

        for prompt_number, prompt in enumerate(PROMPTS):
            encoded = tokenizer.encode(prompt)

            prompt_ids = torch.tensor(
                encoded.ids,
                dtype=torch.long,
                device=DEVICE,
            )

            generator = torch.Generator(device=DEVICE)
            generator.manual_seed(SEED + prompt_number)

            output_ids = generate(
                model,
                prompt_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                context_length=CONTEXT_LENGTH,
                temperature=temperature,
                top_p=top_p,
                eot_token_id=eot_token_id,
                generator=generator,
            )

            output_text = tokenizer.decode(
                output_ids.detach().cpu().tolist()
            )

            print(f"\nPrompt {prompt_number + 1}: {prompt}")
            print(output_text)


if __name__ == "__main__":
    main()