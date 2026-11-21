"""
src/generation.py - Prompt formatting and baseline generation routines.
"""
import time
from typing import Any, Callable

from .extraction import extract_answer, parse_steps

GenerateFn = Callable[[str, int, dict], list[dict]]

GREEDY_KWARGS = {
    "do_sample": False, "temperature": None, "top_p": None, "top_k": None,
    "max_new_tokens": 512,
}


def sampling_kwargs(temperature: float = 0.7) -> dict:
    """Pure temperature sampling (overrides Qwen's default top_p/top_k/repetition_penalty)."""
    return {
        "do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0,
        "repetition_penalty": 1.0, "max_new_tokens": 512,
    }


def _get_generate_fn(model: Any, tokenizer: Any, generate_fn: GenerateFn | None) -> GenerateFn:
    """Use the given generate_fn, or build a Hugging Face one from model + tokenizer."""
    if generate_fn is not None:
        return generate_fn
    from .utils import make_hf_generate_fn
    return make_hf_generate_fn(model, tokenizer)


def format_step_prompt(question: str) -> str:
    """Zero-shot prompt that forces numbered steps and a 'Final Answer: <int>' line."""
    return (
        "Solve the following math word problem step by step.\n\n"
        "Rules:\n"
        "- Write each step on its own line, starting with \"Step 1:\", \"Step 2:\", and so on.\n"
        "- Each step should do one small deduction or calculation and show the arithmetic.\n"
        "- Use plain text only. Do not use markdown, headings, bold, LaTeX, or \\boxed{}.\n"
        "- After the last step, write one final line in exactly this form: Final Answer: <integer>\n"
        "- The final line must contain only the number (no units, no $ sign, no words).\n\n"
        f"Problem: {question}\n\n"
        "Use this format:\n"
        "Step 1: ...\n"
        "Step 2: ...\n"
        "Final Answer: <integer>"
    )


def compute_majority_vote(answers: list[int | None]) -> int | None:
    """Most common non-None answer; ties go to the answer that appeared first."""
    counts: dict[int, int] = {}
    first_seen: list[int] = []
    for a in answers:
        if a is None:
            continue
        if a not in counts:
            counts[a] = 0
            first_seen.append(a)
        counts[a] += 1

    if not counts:
        return None
    best_count = max(counts.values())
    for a in first_seen:  # earliest answer with the top count wins
        if counts[a] == best_count:
            return a
    return None


def _make_candidate(output: dict) -> dict:
    text = output["text"]
    return {
        "completion": text,
        "steps": parse_steps(text),
        "answer": extract_answer(text),
        "num_tokens": int(output["num_tokens"]),
    }


def run_greedy(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    generate_fn: GenerateFn | None = None,
) -> dict[str, Any]:
    """One deterministic completion (N = 1)."""
    gen = _get_generate_fn(model, tokenizer, generate_fn)
    prompt = format_step_prompt(question)

    start = time.perf_counter()
    output = gen(prompt, 1, dict(GREEDY_KWARGS))[0]
    elapsed = time.perf_counter() - start

    cand = _make_candidate(output)
    return {
        "prompt": prompt,
        "completion": cand["completion"],
        "steps": cand["steps"],
        "answer": cand["answer"],
        "generation_calls": 1,
        "generated_tokens": cand["num_tokens"],
        "time_seconds": elapsed,
    }


def sample_candidates(gen: GenerateFn, question: str, n: int, temperature: float) -> list[dict]:
    """Draw n sampled completions and turn them into candidate dicts."""
    if n <= 0:
        return []
    outputs = gen(format_step_prompt(question), n, sampling_kwargs(temperature))
    return [_make_candidate(o) for o in outputs]


def run_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    temperature: float = 0.7,
    generate_fn: GenerateFn | None = None,
) -> dict[str, Any]:
    """Sample n paths and majority-vote over their answers."""
    gen = _get_generate_fn(model, tokenizer, generate_fn)

    start = time.perf_counter()
    candidates = sample_candidates(gen, question, n, temperature)
    elapsed = time.perf_counter() - start

    return {
        "candidates": candidates,
        "answer": compute_majority_vote([c["answer"] for c in candidates]),
        "generation_calls": len(candidates),
        "generated_tokens": sum(c["num_tokens"] for c in candidates),
        "time_seconds": elapsed,
    }
