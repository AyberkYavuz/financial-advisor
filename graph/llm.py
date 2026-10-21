"""Provider-agnostic LLM access for the multi-agent financial advisor.

This module is the single integration point for every language-model provider
the system supports (DeepSeek, OpenAI, Anthropic, Gemini). It exposes:

* :func:`get_provider` — resolve the active provider from configuration.
* :func:`get_llm` — return a lazily-initialized, cached chat model for the
  active provider, or ``None`` when that provider is not configured.
* :func:`ask_structured` — a provider-agnostic structured-output helper that
  asks the model for JSON matching a Pydantic schema and parses it back.

Structured output deliberately uses plain JSON mode + Pydantic parsing rather
than native tool/function calling. Not every provider supports native
structured output (notably DeepSeek's reasoning models reject both OpenAI
function calling and JSON-schema ``response_format``); plain JSON is the one
capability all four providers share, so a single code path works everywhere.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

logger = logging.getLogger(__name__)

#: Absolute path to the project root (``graph/llm.py`` -> root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load the root ``.env`` so provider keys/models are available even when this
# module is imported outside of ``langgraph dev`` (e.g. unit tests or scripts).
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - python-dotenv is a project dependency
    pass

Provider = Literal["deepseek", "openai", "anthropic", "gemini"]

#: Providers the system can route to.
PROVIDERS: tuple[Provider, ...] = ("deepseek", "openai", "anthropic", "gemini")

#: Default model per provider when no ``<PROVIDER>_MODEL`` env var is set.
DEFAULT_MODELS: dict[Provider, str] = {
    "deepseek": "deepseek-v4-pro",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-2.0-flash",
}

#: Custom base URL per provider (``None`` means the client's own default).
BASE_URLS: dict[Provider, str | None] = {
    "deepseek": "https://api.deepseek.com",
    "openai": None,
    "anthropic": None,
    "gemini": None,
}

#: Cache of initialized chat models keyed by ``(provider, model)``.
_llm_cache: dict[tuple[str, str], BaseChatModel] = {}


def get_provider() -> Provider:
    """Return the active provider from ``LLM_PROVIDER`` (default ``deepseek``)."""
    raw = (os.getenv("LLM_PROVIDER") or "deepseek").strip().lower()
    if raw not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER {raw!r}; expected one of {PROVIDERS}.")
    return raw  # type: ignore[return-value]


def _api_key(provider: Provider) -> str | None:
    """Return the API key for ``provider``, or ``None`` if not configured."""
    value = os.getenv(f"{provider.upper()}_API_KEY")
    return value.strip() if value else None


def get_llm(provider: Provider | None = None) -> BaseChatModel | None:
    """Return a lazily-initialized, cached chat model for the active provider.

    Returns ``None`` when the provider's API key is missing so callers can fall
    back to deterministic behavior.
    """
    provider = provider or get_provider()
    api_key = _api_key(provider)
    if not api_key:
        return None

    model = os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]
    cache_key = (provider, model)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI

        llm: BaseChatModel = ChatOpenAI(
            model=model,
            base_url=BASE_URLS["deepseek"],
            api_key=api_key,
            temperature=0,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=model, api_key=api_key, temperature=0)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model, api_key=api_key, temperature=0)
    else:  # pragma: no cover - guarded by the Provider literal
        raise ValueError(f"Unsupported provider {provider!r}")

    _llm_cache[cache_key] = llm
    return llm


T = TypeVar("T", bound=BaseModel)


def _strip_code_fences(text: str) -> str:
    """Strip surrounding Markdown code fences (`` ```json ... ``` ``) from a response."""
    cleaned = text.strip()
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    return cleaned


def _extract_json_object(text: str) -> str:
    """Extract the first ``{...}`` JSON object from a possibly-noisy response."""
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _as_text(response: Any) -> str:
    """Coerce a chat-model response into a plain string."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def ask_structured(
    llm: BaseChatModel,
    schema: type[T],
    system: str,
    user: str,
) -> T:
    """Ask ``llm`` to produce JSON matching ``schema`` and parse it.

    The JSON Schema for ``schema`` is embedded in the system prompt so the model
    knows the exact field names, types, and enum values to emit. This works on
    every provider because it relies only on plain JSON output.

    Raises:
        ValueError: If the response cannot be parsed into ``schema``.
    """
    schema_json = json.dumps(schema.model_json_schema())
    instruction = (
        f"{system}\n\n"
        "Respond with a single JSON object and nothing else. Do not wrap the "
        "JSON in Markdown code fences. The JSON object must conform to the "
        "following JSON Schema:\n"
        f"{schema_json}"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user},
    ]

    text = _as_text(llm.invoke(messages)).strip()

    try:
        return schema.model_validate_json(text)
    except Exception:
        pass

    # Retry on a best-effort extraction of the leading JSON object.
    extracted = _extract_json_object(text)
    try:
        return schema.model_validate_json(extracted)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear ValueError
        raise ValueError(f"Failed to parse {schema.__name__} from LLM output: {exc}") from exc


__all__ = [
    "Provider",
    "PROVIDERS",
    "DEFAULT_MODELS",
    "get_provider",
    "get_llm",
    "ask_structured",
]
