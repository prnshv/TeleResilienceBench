from __future__ import annotations

import math
import re
from typing import List, Literal, Optional

import tiktoken

Boundary = Literal["none", "sentence", "paragraph"]
HalfBy = Literal["char", "token"]


def last_final_answer_payload(text: str) -> Optional[str]:
    """
    Return the payload from the last line matching ``Final Answer: ...`` (case-insensitive).
    """
    if not (text or "").strip():
        return None
    for line in reversed(text.splitlines()):
        m = re.match(r"^\s*final\s*answer\s*:\s*(.+?)\s*$", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def strip_trailing_final_answer_lines(text: str) -> str:
    """Remove trailing ``Final Answer:`` lines (for storing reasoning / half-traces)."""
    lines = text.splitlines()
    while lines and re.match(r"^\s*final\s*answer\s*:", lines[-1], re.IGNORECASE):
        lines.pop()
    return "\n".join(lines).rstrip()


def model_text_for_parsing(thinking: str, response: str) -> str:
    """Join thinking + response so a ``Final Answer:`` line in either part is visible to parsers."""
    parts = [(thinking or "").strip(), (response or "").strip()]
    return "\n\n".join(p for p in parts if p)


def _choice_index_from_segment(text: str, num_choices: int) -> Optional[int]:
    """Extract 0-based MC index from one string segment."""
    text = (text or "").strip()
    if not text or num_choices < 1:
        return None
    max_digit = min(9, num_choices)
    digit_class = f"[1-{max_digit}]"
    tail = text[-120:] if len(text) > 120 else text
    for pat in [
        r"(?:answer|option|choice)\s*[:\s]+(" + digit_class + r")\b",
        r"\b(" + digit_class + r")\.?\s*$",
        r"(?:is|:)\s*(" + digit_class + r")\s*\.?\s*$",
        r"\b(" + digit_class + r")\s*$",
    ]:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < num_choices:
                return idx
    for content in (tail, text):
        matches = list(re.finditer(rf"\b({digit_class})\b", content))
        if matches:
            idx = int(matches[-1].group(1)) - 1
            if 0 <= idx < num_choices:
                return idx
    return None


def extract_choice_index(answer_text: str, num_choices: int) -> Optional[int]:
    """
    Extract 0-based option index. Prefer ``Final Answer:`` line, then digits / keywords.
    """
    text = (answer_text or "").strip()
    if not text or num_choices < 1:
        return None
    payload = last_final_answer_payload(text)
    for segment in ([payload] if payload else []) + [text]:
        if not segment:
            continue
        idx = _choice_index_from_segment(segment, num_choices)
        if idx is not None:
            return idx
    return None


FLOAT_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def extract_float_answer(text: str) -> Optional[float]:
    """Parse a numeric final answer (TeleMath). Prefer ``Final Answer:`` then \\boxed{}, then tail heuristics."""
    if not (text or "").strip():
        return None
    s = text.strip()
    payload = last_final_answer_payload(s)
    if payload:
        m0 = re.search(r"^(" + FLOAT_RE + r")\s*$", payload.strip(), re.IGNORECASE)
        if m0:
            try:
                return float(m0.group(1))
            except ValueError:
                pass
        m0 = re.search(FLOAT_RE, payload)
        if m0:
            try:
                return float(m0.group(0))
            except ValueError:
                pass
    m = re.search(r"\\boxed\{\s*([^}]*?)\s*\}", s)
    if m:
        inner = m.group(1).strip().strip("{}")
        try:
            return float(inner)
        except ValueError:
            pass
    tail = s[-800:] if len(s) > 800 else s
    for pat in [
        r"(?:answer|final)\s*[:\s]+(" + FLOAT_RE + r")\s*\.?\s*$",
        r"(?:=\s*)(" + FLOAT_RE + r")\s*\.?\s*$",
        r"\b(" + FLOAT_RE + r")\s*\.?\s*$",
    ]:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    matches = list(re.finditer(FLOAT_RE, tail))
    if matches:
        try:
            return float(matches[-1].group(0))
        except ValueError:
            return None
    return None


def floats_match(
    a: float,
    b: float,
    *,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-5,
) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _label_choice_index(text: str, choices: List[str]) -> Optional[int]:
    text = text or ""
    ranked = sorted(enumerate(choices), key=lambda ic: len(str(ic[1])), reverse=True)
    for i, c in ranked:
        label = str(c).strip()
        if len(label) < 1:
            continue
        if re.search(rf"\b{re.escape(label)}\b", text, re.IGNORECASE):
            return i
    return None


def parse_mc_output(answer_text: str, choices: List[str]) -> Optional[int]:
    """
    Parse 0-based MC index: ``Final Answer:`` + digits, then labels on payload, then full text.
    """
    n = len(choices)
    if n < 1:
        return None
    text = (answer_text or "").strip()
    payload = last_final_answer_payload(text)
    for segment in ([payload] if payload else []) + [text]:
        if not segment:
            continue
        idx = extract_choice_index(segment, n)
        if idx is not None:
            return idx
        idx = _label_choice_index(segment, choices)
        if idx is not None:
            return idx
    return None


def _encoding():
    return tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_encoding().encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = _encoding()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


def _last_sentence_boundary_before(s: str, end: int) -> int:
    """Largest index in [0, end] that ends a sentence (.!? followed by space or EOS)."""
    if end <= 0:
        return 0
    chunk = s[:end]
    best = 0
    for m in re.finditer(r"[.!?](?:\s|$)", chunk):
        best = m.end()
    return best if best > 0 else end


def _last_paragraph_boundary_before(s: str, end: int) -> int:
    if end <= 0:
        return 0
    chunk = s[:end]
    idx = chunk.rfind("\n\n")
    if idx != -1 and idx + 2 < end:
        return idx + 2
    return end


def compute_reasoning_half(
    reasoning_full: str,
    *,
    half_by: HalfBy = "token",
    boundary: Boundary = "sentence",
) -> str:
    """
    Take the first half of the reasoning trace, optionally snapping to a boundary
    before the midpoint.
    """
    s = reasoning_full or ""
    if not s.strip():
        return ""

    if half_by == "char":
        mid = max(1, len(s) // 2)
        cut = mid
    else:
        ntok = max(1, _token_len(s) // 2)
        cut = len(_truncate_to_tokens(s, ntok))

    if boundary == "paragraph":
        cut = _last_paragraph_boundary_before(s, cut)
    elif boundary == "sentence":
        cut = _last_sentence_boundary_before(s, cut)

    cut = max(0, min(cut, len(s)))
    return s[:cut].rstrip()


def extract_delimited_reasoning(text: str, tag: str = "redacted_thinking") -> tuple[str, str]:
    """
    Parse ``<tag>...</tag>`` block as reasoning; remainder is treated as tail (answer).
    """
    open_t = f"<{tag}>"
    close_t = f"</{tag}>"
    if open_t not in text or close_t not in text:
        return "", text
    start = text.find(open_t) + len(open_t)
    end = text.find(close_t, start)
    if end == -1:
        return "", text
    reasoning = text[start:end].strip()
    tail = (text[end + len(close_t) :]).strip()
    return reasoning, tail
