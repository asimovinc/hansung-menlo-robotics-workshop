# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any


TOKAMAK_URL = "https://api.tokamak.sh/v1/chat/completions"
DEFAULT_LLM_MODEL = "minimaxai/minimax-m3"
DEFAULT_VLM_MODEL = "qwen/qwen3.6-35b-a3b"
ALLOWED_AI_MODELS = (
    "minimaxai/minimax-m3",
    "qwen/qwen3.6-35b-a3b",
)
ALLOWED_LLM_MODELS = ALLOWED_AI_MODELS
ALLOWED_VLM_MODELS = ALLOWED_AI_MODELS


def get_llm_model(default: str = DEFAULT_LLM_MODEL) -> str:
    """Return the configured text model for scripts and notebooks."""
    model = os.environ.get("MENLO_LLM_MODEL", default).strip()
    if model not in ALLOWED_AI_MODELS:
        allowed = ", ".join(ALLOWED_AI_MODELS)
        raise ValueError(f"MENLO_LLM_MODEL must be one of: {allowed}")
    return model


def get_vlm_model(default: str = DEFAULT_VLM_MODEL) -> str:
    """Return the configured vision model for scripts and notebooks."""
    model = os.environ.get("MENLO_VLM_MODEL", default).strip()
    if model not in ALLOWED_AI_MODELS:
        allowed = ", ".join(ALLOWED_AI_MODELS)
        raise ValueError(f"MENLO_VLM_MODEL must be one of: {allowed}")
    return model


def call_llm(
    messages: list[dict[str, Any]],
    *,
    api_key: str,
    model: str | None = None,
    timeout_s: int = 120,
) -> str:
    import requests

    selected_model = model or get_llm_model()
    response = requests.post(
        TOKAMAK_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": selected_model, "messages": messages},
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def ask_vlm(
    jpeg_bytes: bytes,
    prompt: str,
    *,
    api_key: str,
    model: str | None = None,
) -> str:
    b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{b64_image}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return call_llm(messages, api_key=api_key, model=model or get_vlm_model())


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Find a JSON tool call in model text."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = match.group(1) if match else None

    if blob is None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        blob = match.group(0) if match else None

    if blob is None:
        return None
    try:
        call = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if "tool" not in call:
        return None
    call.setdefault("args", {})
    return call


def build_system_prompt(tools: dict[str, dict[str, str]]) -> str:
    lines = [
        "You control a humanoid warehouse robot by calling tools.",
        "To call a tool, reply with ONLY a single JSON object in a fenced code block:",
        "```json",
        '{"tool": "<tool name>", "args": {...}}',
        "```",
        "Write nothing else outside the code block.",
        "When the task is complete or impossible, call the special tool 'done':",
        '{"tool": "done", "args": {"summary": "<one sentence about what happened>"}}',
        "",
        "Available tools:",
    ]
    for name, spec in tools.items():
        lines.append(f"- {name}: {spec['description']}")
    return "\n".join(lines)

