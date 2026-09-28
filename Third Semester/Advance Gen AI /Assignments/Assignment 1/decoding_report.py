"""Decoding study with quantitative metrics for REPORT.md Section 4.

Run from the repository root:
    uv run python decoding_report.py

Writes:
    decoding_results.md                     tables + full generated samples
    report_assets/decoding_metrics.png      figure for the report
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

from src.generation import generate
from src.transformer_lm import TransformerLM

MODEL_PATH = "final_model.pt"  # same FP16 artifact that is submitted
TOKENIZER_PATH = "data/tinystories/tokenizer/tokenizer.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CONTEXT_LENGTH = 256
MAX_NEW_TOKENS = 100
SEEDS = [42, 43, 44, 45, 46]  # 5 samples per prompt per setting

PROMPTS = [
    "Once upon a time",
    "The little girl",
    "One day, a small",
]

# (name, temperature, top_p). temperature=1.0/top_p=1.0 is the shared reference.
SETTINGS = [
    ("T=0.7, p=1.0", 0.7, 1.0),
    ("T=1.0, p=1.0", 1.0, 1.0),
    ("T=1.3, p=1.0", 1.3, 1.0),
    ("T=1.0, p=0.8", 1.0, 0.8),
    ("T=1.0, p=0.9", 1.0, 0.9),
]


def load_model():
    model = TransformerLM(
        vocab_size=8192,
        context_length=CONTEXT_LENGTH,
        d_model=512,
        num_layers=4,
        n_q_heads=16,
        n_kv_heads=4,
        d_ff=1344,
        rope_theta=10000.0,
        norm_eps=1e-5,
    )
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.to(DEVICE).eval()


def ngrams(ids, n):
    return [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]


def metrics(new_ids, eot_id):
    """Token-level statistics of one completion (prompt excluded)."""
    ended = eot_id in new_ids
    body = new_ids[: new_ids.index(eot_id)] if ended else new_ids
    bigrams = ngrams(body, 2)
    fourgrams = ngrams(body, 4)
    distinct2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    # fraction of 4-grams that already appeared earlier in the same completion
    seen, repeats = set(), 0
    for g in fourgrams:
        repeats += g in seen
        seen.add(g)
    repeat4 = repeats / len(fourgrams) if fourgrams else 0.0
    return {
        "length": len(new_ids),
        "eot": float(ended),
        "distinct2": distinct2,
        "repeat4": repeat4,
    }


def mean(values):
    return sum(values) / len(values)


def main():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    model = load_model()

    rows, samples = [], {}
    for name, temperature, top_p in SETTINGS:
        stats = []
        for p_idx, prompt in enumerate(PROMPTS):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE)
            for seed in SEEDS:
                generator = torch.Generator(device=DEVICE)
                generator.manual_seed(seed)
                out = generate(
                    model,
                    prompt_tensor,
                    max_new_tokens=MAX_NEW_TOKENS,
                    context_length=CONTEXT_LENGTH,
                    temperature=temperature,
                    top_p=top_p,
                    eot_token_id=eot_id,
                    generator=generator,
                ).tolist()
                new_ids = out[len(prompt_ids) :]
                stats.append(metrics(new_ids, eot_id))
                if seed == SEEDS[0]:
                    text = tokenizer.decode(out, skip_special_tokens=False)
                    samples[(name, p_idx)] = text
        rows.append(
            {
                "setting": name,
                "length": mean([s["length"] for s in stats]),
                "eot": mean([s["eot"] for s in stats]),
                "distinct2": mean([s["distinct2"] for s in stats]),
                "repeat4": mean([s["repeat4"] for s in stats]),
                "n": len(stats),
            }
        )
        print(f"done: {name}")

    # ---------- markdown ----------
    lines = [
        f"Model: `{MODEL_PATH}` | device: {DEVICE} | max_new_tokens={MAX_NEW_TOKENS} | "
        f"seeds={SEEDS} | {len(PROMPTS)} prompts -> {len(PROMPTS) * len(SEEDS)} samples per setting",
        "",
        "| Setting | Mean new tokens | Ended with EOT | Distinct-2 (higher = more varied) | Repeated 4-grams (higher = more looping) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['setting']} | {r['length']:.1f} | {r['eot'] * 100:.0f}% | "
            f"{r['distinct2']:.3f} | {r['repeat4'] * 100:.1f}% |"
        )
    lines += ["", "## Samples (seed 42, not cherry-picked)", ""]
    for name, _, _ in SETTINGS:
        lines.append(f"### {name}")
        lines.append("")
        for p_idx, prompt in enumerate(PROMPTS):
            text = samples[(name, p_idx)].replace("\n", " ")
            lines.append(f"**Prompt {p_idx + 1} ({prompt}):**")
            lines.append("")
            lines.append(f"> {text}")
            lines.append("")
    with open("decoding_results.md", "w") as f:
        f.write("\n".join(lines))

    # ---------- figure ----------
    os.makedirs("report_assets", exist_ok=True)
    names = [r["setting"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(names, [r["distinct2"] for r in rows], color="#1f77b4")
    axes[0].set_title("Distinct-2 (vocabulary variety)")
    axes[1].bar(names, [r["repeat4"] * 100 for r in rows], color="#1f77b4")
    axes[1].set_title("Repeated 4-grams (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Decoding metrics, {len(PROMPTS) * len(SEEDS)} samples per setting")
    fig.tight_layout()
    fig.savefig("report_assets/decoding_metrics.png", dpi=200)

    print("\nWrote decoding_results.md and report_assets/decoding_metrics.png")


if __name__ == "__main__":
    main()