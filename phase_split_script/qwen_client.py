"""DashScope Qwen-VL API client for phase annotation.

Requires the environment variable ``DASHSCOPE_API_KEY`` to be set.
Uses the standard HTTP API so no extra SDK installation is needed.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any

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
            "Get a key from https://dashscope.aliyun.com and export it."
        )
    return key


def _pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image to base64 string for the DashScope API."""
    buffer = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_text(response_json: dict[str, Any]) -> str:
    """Extract assistant text from a DashScope response, handling several shapes."""
    choices = response_json.get("output", {}).get("choices", [])
    if not choices:
        raise ValueError(f"No choices in response: {response_json}")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "\n".join(texts)
    return str(content)


def _parse_json_phase(text: str, num_phases: int) -> dict[str, Any]:
    """Parse a phase number from model output text.

    First tries to parse the whole text as JSON, then falls back to extracting
    the first JSON object, and finally to regex search for a phase number.
    """
    text = text.strip()

    # Try whole-text JSON.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "phase" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Try extracting the first JSON object from the text.
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
) -> dict[str, Any]:
    """Call Qwen-VL with a list of PIL images and a text prompt.

    Args:
        images: List of PIL images (e.g. main view + wrist view).
        prompt: Text prompt describing the task and phase definitions.
        model: DashScope model name.
        max_retries: Number of retries on transient failures.

    Returns:
        Parsed response dict, e.g. {"phase": 2, "confidence": "high"}.
    """
    api_key = _get_api_key()

    content: list[dict[str, Any]] = []
    for img in images:
        content.append({"image": _pil_to_base64(img)})
    content.append({"text": prompt})

    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                DASHSCOPE_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            if response.status_code == 400:
                print(f"[debug] 400 response body: {response.text}")
            response.raise_for_status()
            response_json = response.json()

            if response_json.get("code") is not None and response_json.get("code") != "":
                # DashScope sometimes returns errors with HTTP 200.
                raise RuntimeError(f"DashScope API error: {response_json}")

            text = _extract_text(response_json)
            return _parse_json_phase(text, num_phases=10)
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            time.sleep(wait)

    raise RuntimeError(
        f"Qwen-VL API call failed after {max_retries} retries: {last_error}"
    )
