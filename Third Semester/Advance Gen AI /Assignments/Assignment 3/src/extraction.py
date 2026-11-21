"""
src/extraction.py - Step and answer extraction utilities.
"""
import re

# "Step 3:" style label, allowing markdown like "**Step 3:**" or "### Step 3."
_STEP_LABEL = r"[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*"
_STEP_PATTERN = (
    r"(?:^|\n)\s*(?:" + _STEP_LABEL + r")(.*?)"
    r"(?=(?:\n\s*" + _STEP_LABEL + r")|(?:\n\s*[#*`\s]*Final Answer)|$)"
)

# Everything after the LAST "Final Answer" label on its line
_FINAL_LINE = re.compile(r"final\s*answer\s*[*:#`\s]*([^\n]*)", re.IGNORECASE)

# A number at the start of that text, e.g. "$1,234", "-5", "18.00", "\boxed{18}"
_NUMBER = re.compile(r"^[\s$`*(]*(?:\\boxed\{)?[\s$]*(-?\d[\d,]*(?:\.\d+)?)")


def parse_steps(completion_text: str) -> list[str]:
    """Returns the text of each 'Step k:' block, with markdown decoration removed."""
    if not completion_text or not isinstance(completion_text, str):
        return []
    matches = re.findall(_STEP_PATTERN, completion_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = []
    for m in matches:
        text = m.strip().strip("*#`").strip()
        if text:
            cleaned.append(text)
    return cleaned


def extract_answer(completion_text: str) -> int | None:
    """Returns the integer after the last 'Final Answer', or None. Never raises."""
    try:
        if not completion_text or not isinstance(completion_text, str):
            return None

        lines = _FINAL_LINE.findall(completion_text)
        if not lines:
            return None
        last = lines[-1].strip()

        match = _NUMBER.match(last)
        if not match:
            return None

        value = float(match.group(1).replace(",", ""))
        if not value.is_integer():
            return None  # e.g. 18.5 is not a valid integer answer
        return int(value)
    except Exception:
        return None
