from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class OllamaGenerateResult:
    thinking: str
    response: str
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    raw: Dict[str, Any]


def ollama_version(host: str = "http://localhost:11434") -> Dict[str, Any]:
    url = f"{host.rstrip('/')}/api/version"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def ollama_generate(
    model: str,
    prompt: str,
    *,
    host: str = "http://localhost:11434",
    think: bool = True,
    num_predict: int = 2048,
    temperature: float = 0.0,
    timeout: int = 600,
    num_ctx: Optional[int] = None,
    num_batch: Optional[int] = None,
    num_gpu: Optional[int] = None,
    num_thread: Optional[int] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> OllamaGenerateResult:
    """
    POST /api/generate with ``think`` as a top-level field (not inside options).

    Pass ``num_ctx`` to cap context per request (smaller often = faster; overrides a huge GUI default).
    ``num_batch`` affects prompt-batch sizing (llama.cpp); tune if you chase prefill throughput.
    """
    url = f"{host.rstrip('/')}/api/generate"
    opts: Dict[str, Any] = {
        "num_predict": num_predict,
        "temperature": temperature,
    }
    if num_ctx is not None:
        opts["num_ctx"] = int(num_ctx)
    if num_batch is not None:
        opts["num_batch"] = int(num_batch)
    if num_gpu is not None:
        opts["num_gpu"] = int(num_gpu)
    if num_thread is not None:
        opts["num_thread"] = int(num_thread)
    if extra_options:
        for k, v in extra_options.items():
            if v is not None:
                opts[k] = v
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": opts,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    thinking = str(data.get("thinking") or "").strip()
    response = str(data.get("response") or "").strip()
    pt = data.get("prompt_eval_count")
    ot = data.get("eval_count")
    prompt_tokens = int(pt) if pt is not None else None
    output_tokens = int(ot) if ot is not None else None
    return OllamaGenerateResult(
        thinking=thinking,
        response=response,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        raw=data,
    )


def build_base_mc_prompt(question: str, choices: list[str], use_delimiter_fallback: bool) -> str:
    lines = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(choices))
    if use_delimiter_fallback:
        return f"""You are a telecom engineering assistant. Solve this multiple-choice question.

Question:
{question}

Options:
{lines}

Put your detailed reasoning inside <redacted_thinking></redacted_thinking> tags.
After the closing tag, output exactly one line and nothing else:
Final Answer: <n>
where <n> is the option number from 1 to {len(choices)} (integer only)."""
    return f"""You are a telecom engineering assistant. Solve this multiple-choice question.

Question:
{question}

Options:
{lines}

Use the model's thinking stream for detailed reasoning. When you are done reasoning, your final visible reply MUST be exactly one line and nothing else:

Final Answer: <n>

where <n> is the option number from 1 to {len(choices)} (integer only). Do not add any other text before or after that line in the final reply."""


def build_continuation_mc_prompt(question: str, choices: list[str], reasoning_half: str) -> str:
    lines = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(choices))
    half = reasoning_half.strip() or "(no prior reasoning provided)"
    return f"""You are a telecom engineering assistant. You started reasoning about a multiple-choice question but stopped mid-way. Continue from the partial reasoning in your thinking stream, finish your analysis, and give the final answer.

Question:
{question}

Options:
{lines}

Partial reasoning (incomplete — continue from here):
{half}

Use the model's thinking stream to extend this partial trace. When you are done reasoning, your final visible reply MUST be exactly one line and nothing else:

Final Answer: <n>

where <n> is the option number from 1 to {len(choices)} (integer only). Do not add any other text before or after that line in the final reply."""


def build_base_telemath_prompt(question: str, use_delimiter_fallback: bool) -> str:
    if use_delimiter_fallback:
        return f"""You are an expert in telecommunications mathematics. Solve the problem.

Question:
{question}

Put your detailed reasoning inside <redacted_thinking></redacted_thinking> tags.
After the closing tag, output exactly one line and nothing else:
Final Answer: <number>
Use decimal or scientific notation (e.g. Final Answer: 3.14 or Final Answer: 1.5e-4)."""
    return f"""You are an expert in telecommunications mathematics. Solve the problem.

Question:
{question}

Use the thinking stream for detailed work. When done, your final visible reply MUST be exactly one line and nothing else:

Final Answer: <number>

where <number> is the numeric result in decimal or scientific notation. No other text in the final reply."""


def build_continuation_telemath_prompt(question: str, reasoning_half: str) -> str:
    half = reasoning_half.strip() or "(no prior reasoning provided)"
    return f"""You are an expert in telecommunications mathematics. You started solving a problem but stopped mid-way. Continue from the partial work in your thinking stream, finish the derivation, and output the final numeric answer.

Question:
{question}

Partial reasoning (incomplete — continue from here):
{half}

Use the thinking stream for detailed work. When done, your final visible reply MUST be exactly one line and nothing else:

Final Answer: <number>

where <number> is in decimal or scientific notation. No other text in the final reply."""
