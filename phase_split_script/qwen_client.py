"""Qwen-VL API client for phase annotation.

Supports two backends:
1. DashScope (cloud): default, uses ``DASHSCOPE_API_KEY``.
2. Local vLLM/OpenAI-compatible server: selected via ``QWEN_API_BASE`` env var.

For local vLLM, set e.g.:
    export QWEN_API_BASE=http://localhost:8001/v1
    export QWEN_MODEL=qwen-local
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any, Callable

import requests
from PIL import Image


DASHSCOPE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
DEFAULT_MODEL = "qwen3-vl-flash"
MAX_RETRIES = 3


def _get_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY environment variable is not set. "
            "Get a key from https://dashscope.aliyun.com and export it, "
            "or set QWEN_API_BASE to use a local vLLM server."
        )
    return key


def _api_base() -> str:
    return os.environ.get("QWEN_API_BASE", DASHSCOPE_URL).rstrip("/")


def _is_local_mode() -> bool:
    base = _api_base()
    return "localhost" in base or "127.0.0.1" in base or ":800" in base


def _pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image to base64 string."""
    buffer = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_local_messages(
    images: list[Image.Image],
    prompt: str,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible message content with images."""
    content: list[dict[str, Any]] = []
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_base64(img)}"},
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _build_dashscope_content(
    images: list[Image.Image],
    prompt: str,
) -> list[dict[str, Any]]:
    """Build DashScope-compatible message content with images."""
    content: list[dict[str, Any]] = []
    for img in images:
        content.append({"image": f"data:image/jpeg;base64,{_pil_to_base64(img)}"})
    content.append({"text": prompt})
    return content


def _join_text_parts(value: Any) -> str:
    """Flatten text-like response fields into a single string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_join_text_parts(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if value.get("type") in {"image_url", "image"}:
            return ""
        parts = [
            _join_text_parts(value.get("text")),
            _join_text_parts(value.get("content")),
            _join_text_parts(value.get("output_text")),
        ]
        return "\n".join(part for part in parts if part)
    return ""


def _summarize_response(response_json: dict[str, Any], limit: int = 1200) -> str:
    """Build a compact response summary for parse/debug errors."""
    text = json.dumps(response_json, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated)"


def _extract_text(response_json: dict[str, Any]) -> str:
    """Extract assistant text from either DashScope or OpenAI response shapes."""
    # DashScope shape: output.choices[0].message.content
    choices = response_json.get("output", {}).get("choices", [])
    if not choices:
        # OpenAI / vLLM shape: choices[0].message.content
        choices = response_json.get("choices", [])
    if not choices:
        raise ValueError(f"No choices in response: {response_json}")

    choice = choices[0]
    message = choice.get("message", {})
    candidates = [
        _join_text_parts(message.get("content")),
        _join_text_parts(message.get("text")),
        _join_text_parts(choice.get("text")),
        _join_text_parts(choice.get("output_text")),
        _join_text_parts(response_json.get("text")),
        _join_text_parts(response_json.get("output_text")),
    ]
    text = "\n".join(part for part in candidates if part).strip()
    if text:
        return text

    reasoning = "\n".join(
        part
        for part in (
            _join_text_parts(message.get("reasoning_content")),
            _join_text_parts(message.get("reasoning")),
            _join_text_parts(choice.get("reasoning_content")),
        )
        if part
    ).strip()
    if reasoning:
        raise ValueError(
            "Assistant content missing in response; found reasoning text only. "
            f"Response summary: {_summarize_response(response_json)}"
        )

    raise ValueError(
        "No assistant text found in response. "
        f"Response summary: {_summarize_response(response_json)}"
    )


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> tags if the model emitted them inline."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _parse_json_phase(text: str, num_phases: int) -> dict[str, Any]:
    """Parse a phase number from model output text.

    First strips inline thinking tags, then tries a single phase number,
    then JSON formats for backward compatibility.
    """
    text = _strip_thinking(text)

    # Primary: the model was asked to output ONLY a single number.
    stripped = text.strip()
    if re.fullmatch(r"\d+", stripped):
        phase = int(stripped)
        if 0 <= phase < num_phases:
            return {"phase": phase}

    # Backward compatibility: try whole-text JSON.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "phase" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Backward compatibility: try extracting the first JSON object.
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "phase" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: search for any integer that looks like a phase label.
    numbers = re.findall(r"\b(\d+)\b", text)
    for n in numbers:
        phase = int(n)
        if 0 <= phase < num_phases:
            return {"phase": phase}

    raise ValueError(f"Could not parse phase from Qwen response: {text!r}")


def call_qwen_vl(
    images: list[Image.Image],
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
    enable_reasoning: bool = True,
    response_parser: Callable[[str], dict[str, Any]] | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Call Qwen-VL with a list of PIL images and a text prompt.

    Args:
        images: List of PIL images (e.g. main view + wrist view).
        prompt: Text prompt describing the task and phase definitions.
        model: Model name (DashScope model or local vLLM served model).
        max_retries: Number of retries on transient failures.
        enable_reasoning: Whether to request reasoning/thinking from the model.
            For local Qwen3 models this is passed via ``chat_template_kwargs`` and
            reinforced in the prompt.
        response_parser: Optional callable that parses raw model text into a dict.
            If None, the default per-frame phase parser is used.
        max_tokens: Maximum number of output tokens. Increase this when the model
            needs to emit long reasoning plus a structured JSON answer.

    Returns:
        Parsed response dict, e.g. {"phase": 2, "confidence": "high"}.
    """
    api_base = _api_base()
    local_mode = _is_local_mode()

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if local_mode:
        # Local vLLM usually does not require an API key, but accept one if set.
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if api_key and api_key != "local":
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{api_base}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": _build_local_messages(images, prompt),
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_reasoning},
        }
    else:
        api_key = _get_api_key()
        headers["Authorization"] = f"Bearer {api_key}"
        url = api_base
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": _build_dashscope_content(images, prompt),
                    }
                ]
            },
        }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=180,
            )
            if response.status_code == 400:
                print(f"[debug] 400 response body: {response.text}")
            response.raise_for_status()
            response_json = response.json()

            # DashScope sometimes returns errors with HTTP 200.
            if not local_mode and response_json.get("code") not in (None, ""):
                raise RuntimeError(f"DashScope API error: {response_json}")

            text = _extract_text(response_json)
            if response_parser is None:
                return _parse_json_phase(text, num_phases=10)
            return response_parser(text)
        except Exception as e:
            last_error = e
            if attempt + 1 < max_retries:
                wait = 2 ** attempt
                time.sleep(wait)

    raise RuntimeError(
        f"Qwen-VL API call failed after {max_retries} retries: {last_error}"
    )
