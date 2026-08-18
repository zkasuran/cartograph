"""Shared LLM client for the Hack Hydra builds.

OpenAI-compatible. Reads OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL from the
environment (load .env with override=True so the lane file wins over any stale
shell export). Outward-facing code and docs describe this only as "a top-tier
OpenAI model over an OpenAI-compatible endpoint" -- never name the host or id.
"""
from __future__ import annotations
import json, os
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

from openai import OpenAI


def _client() -> OpenAI:
    kw: dict[str, Any] = {}
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        kw["base_url"] = base
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""), **kw)


DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.0,
         tools: list | None = None, tool_choice: Any = None, max_tokens: int | None = None):
    """Return the raw first-choice message object."""
    kw: dict[str, Any] = {"model": model or DEFAULT_MODEL, "messages": messages,
                          "temperature": temperature}
    if tools:
        kw["tools"] = tools
    if tool_choice is not None:
        kw["tool_choice"] = tool_choice
    if max_tokens:
        kw["max_tokens"] = max_tokens
    return _client().chat.completions.create(**kw).choices[0].message


def ask(prompt: str, system: str | None = None, **kw) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    return (chat(msgs, **kw).content or "").strip()


def ask_json(prompt: str, system: str | None = None, **kw) -> Any:
    """Ask for strict JSON and parse it (tolerates ```json fences)."""
    txt = ask(prompt, system, **kw)
    if "```" in txt:
        txt = txt.split("```", 2)[1]
        if txt.startswith("json"):
            txt = txt[4:]
    return json.loads(txt.strip())
