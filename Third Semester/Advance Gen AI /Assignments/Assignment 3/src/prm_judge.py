"""
src/prm_judge.py - PRM judge prompt construction, parsing, and step scoring.
"""
import re
from typing import Any, Callable

JudgeFn = Callable[[str], dict]

_VERDICT_LINE = re.compile(r"verdict\s*:\s*\**\s*(VALID|INVALID)\b", re.IGNORECASE)
_NEGATIVE = re.compile(r"\bINVALID\b|\bNOT\s+VALID\b", re.IGNORECASE)
_POSITIVE = re.compile(r"\bVALID\b", re.IGNORECASE)
_RATIONALE = re.compile(r"rationale\s*:\s*([^\n]*)", re.IGNORECASE)
# First sentence: stop at . ! ? only when followed by whitespace or end (so "2.5" is kept)
_FIRST_SENTENCE = re.compile(r"^(.*?[.!?])(?=\s|$)")


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    """Prompt asking the judge whether ONE step is locally valid."""
    if preceding_steps:
        previous = "\n".join(f"Step {i}: {s}" for i, s in enumerate(preceding_steps, start=1))
    else:
        previous = "(no preceding steps)"

    return (
        "You are a careful math teacher checking ONE step of a student's solution.\n\n"
        f"Problem:\n{question}\n\n"
        f"Preceding steps:\n{previous}\n\n"
        f"Current Step: {current_step}\n\n"
        "Instructions:\n"
        "- Assume all preceding steps are correct, even if they look wrong.\n"
        "- Judge ONLY the Current Step.\n"
        "- The Current Step is VALID if its arithmetic is correct and it follows logically "
        "from the problem and the preceding steps.\n"
        "- The Current Step is INVALID if it has a calculation mistake, uses a wrong number "
        "or quantity from the problem, or makes an unjustified deduction.\n"
        "- Recompute every calculation in the Current Step yourself before deciding.\n\n"
        "Answer with exactly two lines:\n"
        "Verdict: VALID   (or)   Verdict: INVALID\n"
        "Rationale: <one short sentence>"
    )


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    """Turn judge text into (+1.0 or -1.0, one-sentence rationale)."""
    if not judge_output or not isinstance(judge_output, str):
        return -1.0, ""

    # 1) explicit "Verdict: ..." line (first one wins)
    m = _VERDICT_LINE.search(judge_output)
    if m:
        score = 1.0 if m.group(1).upper() == "VALID" else -1.0
    # 2) negative words checked BEFORE "VALID" (because "INVALID" contains "VALID")
    elif _NEGATIVE.search(judge_output):
        score = -1.0
    # 3) standalone VALID
    elif _POSITIVE.search(judge_output):
        score = 1.0
    # 4) unparseable -> INVALID
    else:
        score = -1.0

    rationale = ""
    r = _RATIONALE.search(judge_output)
    if r:
        line = r.group(1)
        s = _FIRST_SENTENCE.match(line)
        rationale = (s.group(1) if s else line).strip()

    return score, rationale


def score_path(
    judge_model: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    candidate: dict[str, Any] | None = None,
    judge_fn: JudgeFn | None = None,
) -> dict[str, Any]:
    """Judge every step of a candidate. Step t sees steps[:t] as context."""
    if judge_fn is None:
        from .utils import make_hf_judge_fn
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)

    steps = list((candidate or {}).get("steps") or [])
    step_scores, rationales, judge_texts = [], [], []
    judge_tokens = 0

    for t, step in enumerate(steps):
        prompt = build_prm_judge_prompt(question, steps[:t], step)
        out = judge_fn(prompt)
        score, rationale = parse_judge_decision(out.get("text", ""))
        step_scores.append(score)
        rationales.append(rationale)
        judge_texts.append(out.get("text", ""))
        judge_tokens += int(out.get("num_tokens", 0))

    return {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_texts": judge_texts,
        "judge_calls": len(steps),
        "judge_tokens": judge_tokens,
    }
