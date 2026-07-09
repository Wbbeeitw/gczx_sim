# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

import pytest
from PIL import Image

from phase_split_script import qwen_client


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_extract_text_uses_choice_text_when_message_content_is_none():
    response = {
        "choices": [
            {
                "message": {"content": None},
                "text": '{"phase": 3, "confidence": "high"}',
            }
        ]
    }

    assert qwen_client._extract_text(response) == '{"phase": 3, "confidence": "high"}'


def test_extract_text_uses_output_text_fallback():
    response = {
        "choices": [
            {
                "message": {"content": None},
                "output_text": '{"phase": 4, "confidence": "high"}',
            }
        ]
    }

    assert qwen_client._extract_text(response) == '{"phase": 4, "confidence": "high"}'


def test_extract_text_raises_helpful_error_on_reasoning_only_response():
    response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning_content": "I am still thinking about the answer.",
                }
            }
        ]
    }

    with pytest.raises(ValueError, match="Assistant content missing in response"):
        qwen_client._extract_text(response)


def test_call_qwen_vl_local_mode_uses_chat_template_kwargs(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        })
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": None},
                        "text": "3",
                    }
                ]
            }
        )

    monkeypatch.setenv("QWEN_API_BASE", "http://localhost:8001/v1")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(qwen_client.requests, "post", fake_post)

    img = Image.new("RGB", (16, 16), color=(127, 127, 127))
    result = qwen_client.call_qwen_vl(
        images=[img],
        prompt="Classify the phase.",
        model="qwen-local",
        enable_reasoning=False,
        max_retries=1,
    )

    assert result == {"phase": 3}
    assert len(calls) == 1
    assert calls[0]["url"] == "http://localhost:8001/v1/chat/completions"
    assert calls[0]["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "enable_thinking" not in calls[0]["json"]


def test_call_qwen_vl_does_not_sleep_after_final_failed_attempt(monkeypatch):
    sleep_calls = []

    def fake_post(url, headers, json, timeout):
        return _FakeResponse({"choices": [{"message": {"content": None}}]})

    monkeypatch.setenv("QWEN_API_BASE", "http://localhost:8001/v1")
    monkeypatch.setattr(qwen_client.requests, "post", fake_post)
    monkeypatch.setattr(qwen_client.time, "sleep", sleep_calls.append)

    img = Image.new("RGB", (8, 8), color=(0, 0, 0))
    with pytest.raises(RuntimeError, match="No assistant text found in response"):
        qwen_client.call_qwen_vl(
            images=[img],
            prompt="Classify the phase.",
            model="qwen-local",
            max_retries=1,
        )

    assert sleep_calls == []


def test_task1_vlm_imports_without_pyav_installed():
    module = importlib.import_module("phase_split_script.task1_vlm")

    assert module.NUM_PHASES == 6
