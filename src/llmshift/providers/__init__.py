"""Provider registry: turn a spec like "openai:gpt-5" into a Provider.

Spec forms:
  provider:model         openai:gpt-5, anthropic:claude-sonnet-4-5, gemini:gemini-2.5-flash,
                         azure:<deployment>, ollama:llama3.1:8b, openrouter:meta-llama/llama-3.3-70b,
                         mock:legacy, python:my_app.llm:answer, <custom-name>:model
  bare model id          gpt-4o, claude-3-5-haiku-20241022, gemini-2.0-flash (provider inferred)
"""

from __future__ import annotations

import difflib
import ipaddress
import os
import re
from typing import Any

from .anthropic import AnthropicProvider
from .base import Provider, ProviderError, Response
from .custom import CommandProvider, PythonProvider
from .gemini import GeminiProvider
from .mock import MockProvider
from .openai import AzureOpenAIProvider, OpenAIProvider

TYPES: dict[str, type] = {
    "openai": OpenAIProvider,
    "azure": AzureOpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "mock": MockProvider,
    "python": PythonProvider,
    "command": CommandProvider,
}

PRESETS: dict[str, dict] = {
    "openai": {"type": "openai", "base_url_env": "OPENAI_BASE_URL", "api_key_env": "OPENAI_API_KEY"},
    "azure": {"type": "azure"},
    "anthropic": {"type": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "gemini": {"type": "gemini", "api_key_env": "GEMINI_API_KEY"},
    "openrouter": {"type": "openai", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "together": {"type": "openai", "base_url": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    "groq": {"type": "openai", "base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    "mistral": {"type": "openai", "base_url": "https://api.mistral.ai/v1", "api_key_env": "MISTRAL_API_KEY"},
    "deepseek": {"type": "openai", "base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY"},
    "xai": {"type": "openai", "base_url": "https://api.x.ai/v1", "api_key_env": "XAI_API_KEY"},
    "fireworks": {"type": "openai", "base_url": "https://api.fireworks.ai/inference/v1", "api_key_env": "FIREWORKS_API_KEY"},
    "perplexity": {"type": "openai", "base_url": "https://api.perplexity.ai", "api_key_env": "PERPLEXITY_API_KEY"},
    "cohere": {"type": "openai", "base_url": "https://api.cohere.ai/compatibility/v1", "api_key_env": "COHERE_API_KEY"},
    "ollama": {"type": "openai", "base_url": "http://localhost:11434/v1", "api_key_env": None, "requires_key": False},
    "lmstudio": {"type": "openai", "base_url": "http://localhost:1234/v1", "api_key_env": None, "requires_key": False},
    "vllm": {"type": "openai", "base_url": "http://localhost:8000/v1", "api_key_env": "VLLM_API_KEY", "requires_key": False},
    "mock": {"type": "mock"},
    "python": {"type": "python"},
}
ALIASES = {"google": "gemini", "claude": "anthropic", "azure-openai": "azure", "azure_openai": "azure", "vertex": "gemini"}

_INFER_RULES = [
    ("openai", r"^(gpt-|o\d(\b|-)|chatgpt-|text-davinci|davinci|babbage|ft:|text-embedding-(?!00)|codex|computer-use|omni-moderation|text-moderation|dall-e|whisper|tts-)"),
    ("anthropic", r"^claude-"),
    ("gemini", r"^(models/)?(gemini-|gemma-|text-embedding-00|gemini-embedding)"),
    ("mistral", r"^(mistral-|open-mistral|open-mixtral|codestral|ministral|pixtral|magistral|devstral)"),
    ("deepseek", r"^deepseek-"),
    ("xai", r"^grok-"),
    ("cohere", r"^(command(-|$)|c4ai-)"),
]


def known_providers(config: dict | None) -> set[str]:
    names = set(PRESETS) | set(ALIASES)
    names |= {str(k).lower() for k in ((config or {}).get("providers") or {})}
    return names


def infer_provider(model: str) -> str | None:
    for name, pattern in _INFER_RULES:
        if re.match(pattern, model.strip(), re.I):
            return name
    return None


def _user_config(config: dict | None, *names: str) -> dict | None:
    """The user's settings for the first of `names` configured under `providers:`."""
    providers = (config or {}).get("providers") or {}
    if not isinstance(providers, dict):
        return None
    lowered = {str(k).lower(): v for k, v in providers.items()}
    for name in names:
        if name in lowered:
            value = lowered[name]
            return dict(value) if isinstance(value, dict) else {}
    return None


def _unspecified(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_unspecified
    except ValueError:  # a name, not an address
        return False


def ollama_url(value: str) -> str:
    """OLLAMA_HOST read the way Ollama reads it: "0.0.0.0", "localhost", ":11434",
    "box:8080", "https://ollama.example.com". Without a scheme the port defaults to 11434
    (80 or 443 with one). 0.0.0.0 and :: mean "every interface" to a server, which is no
    address to call (Windows refuses it): the local machine is meant."""
    text = value.strip()
    scheme, sep, rest = text.partition("://")
    if sep:
        scheme = scheme.lower()
        port = {"http": "80", "https": "443"}.get(scheme, "11434")
    else:
        scheme, rest, port = "http", text, "11434"
    hostport, _, path = rest.partition("/")
    if hostport.startswith("["):  # [::1]:11434
        host, _, after = hostport[1:].partition("]")
        given = after[1:] if after.startswith(":") else ""
    elif hostport.count(":") == 1:
        host, given = hostport.split(":")
    else:  # a name, or an IPv6 address without a port
        host, given = hostport, ""
    if re.fullmatch(r"[0-9]{1,5}", given) and int(given) <= 65535:
        port = given
    if not host or _unspecified(host):  # a server's listen address: connect to this machine
        host = "127.0.0.1"
    if ":" in host:  # an IPv6 address
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}" + (f"/{path.strip('/')}" if path.strip("/") else "")


def parse_spec(spec: str, config: dict | None = None) -> tuple[str | None, str]:
    spec = (spec or "").strip()
    if ":" in spec:
        prefix, rest = spec.split(":", 1)
        lower = prefix.lower()
        if lower in known_providers(config):
            # A provider configured under this exact name wins over a built-in alias.
            if _user_config(config, lower) is not None:
                return lower, rest.strip()
            return ALIASES.get(lower, lower), rest.strip()
    return None, spec


def resolve(spec: str, config: dict | None = None) -> Provider:
    """Create a Provider for `spec`. Raises ProviderError with a helpful message."""
    config = config or {}
    name, model = parse_spec(spec, config)
    if not model:
        raise ProviderError(f"model spec {spec!r} has no model name", code="invalid_spec")
    if name is None:
        name = infer_provider(model)
        if name is None:
            prefix, _, rest = model.partition(":")
            close = difflib.get_close_matches(prefix.lower(), sorted(known_providers(config)), n=1, cutoff=0.75) if rest else []
            if close:  # opneai:gpt-4o
                hint = f"unknown provider {prefix!r} in {spec!r}; did you mean {close[0]}:{rest}?"
            elif prefix.lower() == "command" and rest:
                hint = (f"{spec!r}: a command provider is set up under a name of its own in llmshift.yaml "
                        "(providers: {my-app: {type: command, command: ...}}) and used by that name, e.g. my-app:model")
            else:
                hint = f"cannot tell which provider serves {model!r}; write it as provider:model, e.g. openai:{model} or ollama:{model}"
            raise ProviderError(hint, code="unknown_provider")
    canonical = ALIASES.get(name, name)
    alias_names = sorted(a for a, target in ALIASES.items() if target == canonical)
    user_cfg = _user_config(config, name, canonical, *alias_names) or {}
    options: dict[str, Any] = dict(PRESETS.get(name) or PRESETS.get(canonical) or {})
    options["timeout"] = config.get("timeout", 120)
    options["max_retries"] = config.get("max_retries", 4)
    options.update(user_cfg)
    if "timeout" in config or "timeout" in user_cfg:  # python: functions get a time limit only when one is set
        options["timeout_set"] = True
    if name == "ollama" and os.environ.get("OLLAMA_HOST", "").strip() and "base_url" not in user_cfg:
        options["base_url"] = ollama_url(os.environ["OLLAMA_HOST"]) + "/v1"
    ptype = options.get("type") or canonical
    cls = TYPES.get(ptype)
    if cls is None:
        raise ProviderError(f"provider {name!r} has unknown type {ptype!r}; use one of {', '.join(sorted(TYPES))}", code="invalid_spec")
    return cls(model, name=name, options=options)


__all__ = ["Provider", "ProviderError", "Response", "resolve", "parse_spec", "infer_provider", "TYPES", "PRESETS"]
