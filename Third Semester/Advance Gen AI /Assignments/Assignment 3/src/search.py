"""
src/search.py - Path aggregation, Best-of-N search, and compute-matched search.
"""
import time
from typing import Any, Callable

from .generation import _get_generate_fn, compute_majority_vote, sample_candidates
from .prm_judge import score_path

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]

TIE_TOL = 1e-9


def aggregate_path_score(
    step_scores: list[float],
    strategy: str = "min",
    penalty_lambda: float = 2.0,
) -> float:
    """Reduce step scores (+1 / -1) to one path score."""
    if strategy not in ("min", "product", "step_penalty"):
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
    if not step_scores:
        return -1.0

    if strategy == "min":
        return float(min(step_scores))

    if strategy == "product":
        prob = 1.0
        for r in step_scores:
            p = min(max((r + 1.0) / 2.0, 0.01), 0.99)  # clamp to [0.01, 0.99]
            prob *= p
        return prob

    # step_penalty
    mean = sum(step_scores) / len(step_scores)
    num_invalid = sum(1 for r in step_scores if r == -1.0)
    return float(mean - penalty_lambda * num_invalid)


def select_best_path(
    candidate_paths: list[dict[str, Any]],
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0,
) -> dict[str, Any]:
    """Filter -> score -> take best -> break ties by majority vote."""
    if not candidate_paths:
        raise ValueError("candidate_paths is empty")

    scores = [
        aggregate_path_score(c.get("step_scores") or [], aggregation_strategy, penalty_lambda)
        for c in candidate_paths
    ]

    # 1) eligibility: drop answer=None if any candidate has a real answer
    indices = list(range(len(candidate_paths)))
    if any(c.get("answer") is not None for c in candidate_paths):
        indices = [i for i in indices if candidate_paths[i].get("answer") is not None]

    # 2) tied set = everyone within tolerance of the best score
    best = max(scores[i] for i in indices)
    tied = [i for i in indices if abs(scores[i] - best) <= TIE_TOL]

    # 3) tie-break with majority vote over the tied answers
    chosen = tied[0]
    if len(tied) > 1:
        vote = compute_majority_vote([candidate_paths[i].get("answer") for i in tied])
        if vote is not None:
            chosen = next(i for i in tied if candidate_paths[i].get("answer") == vote)

    result = dict(candidate_paths[chosen])  # shallow copy, input untouched
    result["path_score"] = float(scores[chosen])
    result["num_tied"] = len(tied)
    return result


def run_prm_best_of_n(
    model: Any = None,
    judge_model: Any = None,
    gen_tokenizer: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0,
    candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None,
    judge_fn: JudgeFn | None = None,
) -> dict[str, Any]:
    """Sample (or reuse) n paths, judge every step, pick the best path."""
    start = time.perf_counter()

    generation_calls, generated_tokens = 0, 0
    if candidates is None:
        gen = _get_generate_fn(model, gen_tokenizer, generate_fn)
        candidates = sample_candidates(gen, question, n, temperature=0.7)
        generation_calls = len(candidates)
        generated_tokens = sum(c["num_tokens"] for c in candidates)

    if judge_fn is None:
        from .utils import make_hf_judge_fn
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)

    scored, judge_calls, judge_tokens = [], 0, 0
    for c in candidates:
        res = score_path(question=question, candidate=c, judge_fn=judge_fn)
        new_c = dict(c)
        new_c["step_scores"] = res["step_scores"]
        new_c["rationales"] = res["rationales"]
        new_c["judge_texts"] = res["judge_texts"]
        scored.append(new_c)
        judge_calls += res["judge_calls"]
        judge_tokens += res["judge_tokens"]

    selected = select_best_path(scored, aggregation_strategy, penalty_lambda)

    return {
        "selected": selected,
        "answer": selected.get("answer"),
        "candidates": scored,
        "generation_calls": generation_calls,
        "judge_calls": judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": judge_tokens,
        "time_seconds": time.perf_counter() - start,
    }


def run_compute_matched_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    target_calls: int = 15,
    temperature: float = 0.7,
    existing_candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None,
) -> dict[str, Any]:
    """Self-consistency with target_calls samples, reusing existing ones."""
    existing = list(existing_candidates or [])
    deficit = max(0, target_calls - len(existing))

    start = time.perf_counter()
    new = []
    if deficit > 0:
        gen = _get_generate_fn(model, tokenizer, generate_fn)
        new = sample_candidates(gen, question, deficit, temperature)
    elapsed = time.perf_counter() - start

    all_cands = existing + new
    return {
        "candidates": all_cands,
        "answer": compute_majority_vote([c.get("answer") for c in all_cands]),
        "generation_calls": len(new),
        "total_samples": len(all_cands),
        "generated_tokens": sum(c["num_tokens"] for c in new),
        "time_seconds": elapsed,
    }
