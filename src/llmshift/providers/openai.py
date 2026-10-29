"""OpenAI (and OpenAI-compatible) provider: Chat Completions, Responses API,
embeddings, and Azure OpenAI.

Requests adapt automatically to model differences that commonly break a model
switch: reasoning models that reject `temperature` or `max_tokens`, servers that
do not know `max_completion_tokens` or the `developer` role, and models that are
only served by the Responses API. Only genuine "unsupported parameter/value"
errors are adapted to (and learned for later requests); ordinary value errors
are reported as they are. Each adaptation is listed in `adaptations`.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import urllib.parse
import uuid
from typing import Any

from .. import messages as msgs
from ..textutil import json_loads, stable_hash
from . import http
from .base import Provider, ProviderError, Response, base_url_from, clean_key, option_mapping, params_of, value_key

DROPPABLE = {
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "seed",
    "stop",
    "parallel_tool_calls",
    "reasoning_effort",
    "verbosity",
    "n",
    "user",
}
RESPONSES_ONLY_RE = re.compile(r"(^o1-pro|^o3-pro|-pro$|-pro-\d{4}|codex|^computer-use|deep-research|-cyber|^gpt-oss)")
_UNSUPPORTED_RE = re.compile(
    r"(?i)unsupported (?:parameter|value)|not supported|does not support|is not available for this model|only the default|"
    r"unrecognized request argument|unknown parameter|extra (?:fields|inputs) (?:are )?not permitted|unexpected keyword"
)
_GENERIC_CODES = {"", "none", "invalid_request_error", "invalid_argument", "bad_request", "400", "422"}
MAX_ADAPTATIONS = 8


def _env(name: str | None) -> str | None:
    return os.environ.get(name) if name else None


def _is_local(url: str) -> bool:
    return bool(re.match(r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|host\.docker\.internal)(:|/|$)", url))


def _arguments_str(args: Any) -> str:
    if isinstance(args, str):
        return args
    return json.dumps(args if args is not None else {}, ensure_ascii=False)


def _usage(usage: dict, inp: str, out: str, details: str) -> dict:
    """Input and output tokens as reported (reasoning tokens are in the output), and the
    cached share of the input, which costs the cache price."""
    if not usage:
        return {}
    found = {"input_tokens": usage.get(inp), "output_tokens": usage.get(out)}
    cached = usage.get(details).get("cached_tokens") if isinstance(usage.get(details), dict) else None
    if isinstance(cached, int) and not isinstance(cached, bool) and cached > 0:
        found["cached_input_tokens"] = cached
    return found


def _parse_arguments(args: Any) -> Any:
    if isinstance(args, str):
        try:
            return json_loads(args) if args.strip() else {}
        except (ValueError, RecursionError):
            return args
    return args if args is not None else {}


def _unsupported(err: ProviderError) -> bool:
    """A capability error ('Unsupported parameter: ...'), not a bad value ('above maximum')."""
    code = str(err.code or "").lower()
    if code in ("unsupported_parameter", "unsupported_value"):
        return True
    if code not in _GENERIC_CODES:
        return False  # a specific error such as decimal_above_max_value is a real problem
    return bool(_UNSUPPORTED_RE.search(err.message or ""))


class OpenAIProvider(Provider):
    kind = "openai"
    supports_embeddings = True
    default_base_url = "https://api.openai.com/v1"

    def __init__(self, model: str, *, name: str | None = None, options: dict | None = None):
        super().__init__(model, name=name or "openai", options=options)
        o = self.options
        self.base_url = base_url_from(o, o.get("base_url_env"), self.default_base_url, self.spec)
        key_env = o.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = clean_key(o.get("api_key")) or clean_key(_env(key_env))
        self.key_env = key_env
        # By host name: a substring test would also accept api.openai.com.example.net.
        self.official = (urllib.parse.urlsplit(self.base_url).hostname or "").lower() == "api.openai.com"
        self.requires_key = bool(o.get("requires_key", key_env is not None and not _is_local(self.base_url)))
        self.api = str(o.get("api", "auto"))
        self.extra_headers = option_mapping(o, "headers", self.spec, text_values=True)
        self.extra_body = option_mapping(o, "extra_body", self.spec)
        if self.official:
            org = o.get("organization") or os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION")
            project = o.get("project") or os.environ.get("OPENAI_PROJECT_ID")
            if org:
                self.extra_headers.setdefault("OpenAI-Organization", org)
            if project:
                self.extra_headers.setdefault("OpenAI-Project", project)
        self._lock = threading.Lock()
        # refused: setting -> values the model rejected (reasoning models take temperature 1, not 0)
        self._learned: dict[str, Any] = {"drop": set(), "refused": {}, "max_tokens_field": None, "api": None, "developer_role": None}
        self.adaptations: list[str] = []

    # ------------------------------------------------------------- plumbing

    def identity(self) -> dict:
        shaping = {k: v for k, v in self.options.items() if k not in ("api_key", "timeout", "max_retries", "api_key_env", "requires_key", "timeout_set")}
        shaping["headers"] = {k: v for k, v in self.extra_headers.items() if k.lower() not in ("authorization", "api-key")}
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url, "options": stable_hash(shaping, 16)}

    def _headers(self) -> dict:
        headers = dict(self.extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.requires_key:
            raise ProviderError(
                f"no API key for {self.spec}: set the {self.key_env} environment variable",
                code="missing_api_key",
            )
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post(self, path: str, body: dict) -> dict:
        return http.post_json(self._url(path), body, headers=self._headers(), timeout=self.timeout, max_retries=self.max_retries)

    def _learn(self, note: str, **updates: Any) -> None:
        with self._lock:
            for key, value in updates.items():
                if key == "drop":
                    self._learned["drop"].add(value)
                elif key == "refuse":  # (setting, value)
                    self._learned["refused"].setdefault(value[0], set()).add(value_key(value[1]))
                else:
                    self._learned[key] = value
            if note not in self.adaptations:
                self.adaptations.append(note)

    def _note(self, note: str) -> None:
        with self._lock:
            if note not in self.adaptations:
                self.adaptations.append(note)

    # ---------------------------------------------------------- conversions

    def _chat_messages(self, messages: list[dict]) -> list[dict]:
        out = []
        developer_ok = self._learned.get("developer_role")
        for m in msgs.link_tool_calls(messages):
            role = m.get("role")
            if role == "developer" and (not self.official or developer_ok is False):
                role = "system"
            msg: dict = {"role": role, "content": m.get("content", "")}
            if role == "assistant" and m.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc.get("name"), "arguments": _arguments_str(tc.get("arguments"))}}
                    for tc in m["tool_calls"]
                ]
                if not msg["content"]:
                    msg["content"] = None
            if role == "tool":
                msg["tool_call_id"] = m.get("tool_call_id") or f"call_{uuid.uuid4().hex[:24]}"
                msg["content"] = msgs.content_text(m.get("content"))
            if m.get("name") and role == "user":
                msg["name"] = m["name"]
            out.append(msg)
        return out

    def _drop_learned(self, body: dict) -> None:
        """Leave out what this model rejected before: whole settings, or rejected values."""
        with self._lock:
            drop = set(self._learned["drop"])
            refused = {k: set(v) for k, v in self._learned["refused"].items()}
        for key in drop:
            body.pop(key, None)
            if key == "reasoning_effort":
                body.pop("reasoning", None)
        for key, values in refused.items():
            if key in body and value_key(body[key]) in values:
                body.pop(key)

    def _chat_body(self, request: dict, *, drops: bool = True) -> dict:
        params = params_of(request)
        body: dict = {"model": self.model, "messages": self._chat_messages(request.get("messages") or [])}
        for key in msgs.PARAM_KEYS:
            if params.get(key) is not None:
                body[key] = copy.deepcopy(params[key])
        if self.official and "max_tokens" in body and "max_completion_tokens" not in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
        with self._lock:
            mt_field = self._learned["max_tokens_field"]
        if drops:
            self._drop_learned(body)
        if mt_field == "max_tokens" and "max_completion_tokens" in body:
            body["max_tokens"] = body.pop("max_completion_tokens")
        elif mt_field == "max_completion_tokens" and "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
        if not body.get("tools"):
            body.pop("tool_choice", None)
            body.pop("parallel_tool_calls", None)
        body.update(copy.deepcopy(self.extra_body))
        return body

    def _responses_body(self, request: dict, *, drops: bool = True) -> dict:
        params = params_of(request)
        items: list[dict] = []
        for m in msgs.link_tool_calls(request.get("messages") or []):
            role = m.get("role")
            content = m.get("content")
            if role in ("system", "developer", "user"):
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            url = (part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict) else part.get("image_url")
                            parts.append({"type": "input_image", "image_url": url})
                        else:
                            parts.append({"type": "input_text", "text": msgs.content_text([part])})
                    items.append({"role": role, "content": parts})
                else:
                    items.append({"role": role, "content": msgs.content_text(content)})
            elif role == "assistant":
                text = msgs.content_text(content)
                if text:
                    items.append({"role": "assistant", "content": text})
                for tc in m.get("tool_calls") or []:
                    items.append({"type": "function_call", "call_id": tc["id"], "name": tc.get("name"), "arguments": _arguments_str(tc.get("arguments"))})
            elif role == "tool":
                items.append({"type": "function_call_output", "call_id": m.get("tool_call_id") or "", "output": msgs.content_text(content)})
        body: dict = {"model": self.model, "input": items, "store": False}
        for key in ("temperature", "top_p", "parallel_tool_calls"):
            if params.get(key) is not None:
                body[key] = params[key]
        max_out = params.get("max_completion_tokens") or params.get("max_tokens")
        if max_out:
            body["max_output_tokens"] = max_out
        text_cfg: dict = {}
        rf = params.get("response_format")
        if isinstance(rf, dict):
            if rf.get("type") == "json_schema":
                js = rf.get("json_schema") or {}
                fmt = {"type": "json_schema", "name": js.get("name") or "output", "schema": js.get("schema") or {}}
                if js.get("strict") is not None:
                    fmt["strict"] = js["strict"]
                text_cfg["format"] = fmt
            elif rf.get("type") == "json_object":
                text_cfg["format"] = {"type": "json_object"}
        if params.get("verbosity"):
            text_cfg["verbosity"] = params["verbosity"]
        if text_cfg:
            body["text"] = text_cfg
        if params.get("reasoning_effort"):
            body["reasoning"] = {"effort": params["reasoning_effort"]}
        tools = []
        for t in params.get("tools") or []:
            fn = t.get("function") if isinstance(t, dict) else None
            if fn:
                tool = {"type": "function", "name": fn.get("name"), "parameters": fn.get("parameters") or {"type": "object", "properties": {}}}
                if fn.get("description"):
                    tool["description"] = fn["description"]
                tool["strict"] = bool(fn.get("strict", False))
                tools.append(tool)
        if tools:
            body["tools"] = tools
            tc = params.get("tool_choice")
            if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                body["tool_choice"] = {"type": "function", "name": tc["function"].get("name")}
            elif isinstance(tc, str):
                body["tool_choice"] = tc
        if drops:
            self._drop_learned(body)
        body.update(copy.deepcopy(self.extra_body))
        return body

    def _rejected_in(self, request: dict, api: str, sent: dict) -> list[str]:
        """The request settings this model refused, so the answer came from a changed request."""
        full = self._responses_body(request, drops=False) if api == "responses" else self._chat_body(request, drops=False)
        out = [key for key in sorted(DROPPABLE) if key in full and key not in sent]
        if api == "responses":
            if "reasoning" in full and "reasoning" not in sent:
                out.append("reasoning_effort")
            if (full.get("text") or {}).get("verbosity") and not (sent.get("text") or {}).get("verbosity"):
                out.append("verbosity")
        return out

    # -------------------------------------------------------------- parsing

    @staticmethod
    def parse_chat(data: Any) -> Response:
        data = http.require_object(data, "chat completion", "choices")
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"response has no choices: {json.dumps(data)[:200]}", code="invalid_response")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        content = message.get("content")
        text = msgs.content_text(content) if not isinstance(content, str) else content
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            if fn.get("name"):
                call = {"name": fn["name"], "arguments": _parse_arguments(fn.get("arguments"))}
                if tc.get("id"):
                    call["id"] = tc["id"]
                tool_calls.append(call)
        fc = message.get("function_call")
        if isinstance(fc, dict) and fc.get("name"):
            tool_calls.append({"name": fc["name"], "arguments": _parse_arguments(fc.get("arguments"))})
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return Response(
            text=text or "",
            tool_calls=tool_calls,
            model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
            usage=_usage(usage, "prompt_tokens", "completion_tokens", "prompt_tokens_details"),
            fingerprint=data.get("system_fingerprint"),
            refusal=message.get("refusal") or None,
        )

    @staticmethod
    def parse_responses(data: Any) -> Response:
        data = http.require_object(data, "responses", "output", "output_text", "status")
        texts, tool_calls, refusal = [], [], None
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "message":
                for part in item.get("content") or []:
                    if part.get("type") in ("output_text", "text"):
                        texts.append(part.get("text") or "")
                    elif part.get("type") == "refusal":
                        refusal = part.get("refusal") or "refused"
            elif itype == "function_call":
                call = {"name": item.get("name"), "arguments": _parse_arguments(item.get("arguments"))}
                if item.get("call_id"):
                    call["id"] = item["call_id"]
                tool_calls.append(call)
        text = "".join(texts) if texts else (data.get("output_text") or "")
        status = data.get("status")
        finish = "stop"
        if status == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason")
            finish = "length" if reason == "max_output_tokens" else (reason or "incomplete")
        elif tool_calls:
            finish = "tool_calls"
        if status == "failed":
            err = data.get("error") or {}
            raise ProviderError(err.get("message") or "response failed", code=err.get("code") or "response_failed")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return Response(
            text=text,
            tool_calls=tool_calls,
            model=data.get("model"),
            finish_reason=finish,
            usage=_usage(usage, "input_tokens", "output_tokens", "input_tokens_details"),
            refusal=refusal,
        )

    # ------------------------------------------------------------- adaptation

    def _wants_responses(self) -> bool:
        if self.api == "responses":
            return True
        if self.api == "chat":
            return False
        learned = self._learned.get("api")
        if learned:
            return learned == "responses"
        return self.official and bool(RESPONSES_ONLY_RE.search(self.model))

    def _adapt(self, body: dict, err: ProviderError, api: str, state: dict) -> str | None:
        """Change `body` to work around err. Returns "retry", "switch" (rebuild for
        another API) or None (not something to adapt to: raise it)."""
        if err.status not in (400, 404, 422):
            return None
        message = err.message or ""
        lower = message.lower()
        if api == "chat" and self.api != "chat":
            if "only supported in v1/responses" in lower or "not supported in the v1/chat/completions" in lower or "only available in the responses api" in lower:
                self._learn(f"{self.model} is only served by the Responses API; switched to /v1/responses", api="responses")
                return "switch"
            if "v1/responses" in lower and "tool" in lower:
                # e.g. function tools + reasoning effort: send just this request through Responses.
                state["api"] = "responses"
                self._note(f"requests with tools for {self.model} were sent through the Responses API")
                return "switch"
        if err.status == 404:
            return None
        param = err.param or ""
        quoted = re.findall(r"'([a-z_.]+)'", message)
        if "must contain the word 'json'" in lower and api == "chat":
            messages = body.get("messages") or []
            sys_msgs = [m for m in messages if m.get("role") in ("system", "developer")]
            if sys_msgs and isinstance(sys_msgs[0].get("content"), str):
                sys_msgs[0]["content"] = sys_msgs[0]["content"].rstrip() + "\n\nRespond in JSON."
            else:
                messages.insert(0, {"role": "system", "content": "Respond in JSON."})
            self._note("added 'Respond in JSON.' (JSON mode requires the word json in the prompt)")
            return "retry"
        if not _unsupported(err):
            return None  # e.g. "max_tokens is too large", "temperature above maximum": a real error
        tried = state.setdefault("swapped", set())
        if (param == "max_tokens" or "'max_tokens'" in message) and "max_tokens" in body and "max_completion_tokens" not in tried:
            body["max_completion_tokens"] = body.pop("max_tokens")
            tried.add("max_completion_tokens")
            self._learn("used max_completion_tokens instead of max_tokens", max_tokens_field="max_completion_tokens")
            return "retry"
        if (param == "max_completion_tokens" or "'max_completion_tokens'" in message) and "max_completion_tokens" in body and "max_tokens" not in tried:
            body["max_tokens"] = body.pop("max_completion_tokens")
            tried.add("max_tokens")
            self._learn("used max_tokens instead of max_completion_tokens", max_tokens_field="max_tokens")
            return "retry"
        if param.startswith("reasoning") and ("reasoning" in body or "reasoning_effort" in body):
            body.pop("reasoning", None)
            body.pop("reasoning_effort", None)
            self._learn(f"dropped unsupported reasoning effort for {self.model} ({message[:80]})", drop="reasoning_effort")
            return "retry"
        if param.startswith("text.") and isinstance(body.get("text"), dict):
            sub = param.split(".", 1)[1]
            if sub in body["text"]:
                body["text"].pop(sub, None)
                self._learn(f"dropped unsupported parameter '{param}' for {self.model}", drop="verbosity" if sub == "verbosity" else sub)
                return "retry"
        candidates = [param] if param else []
        candidates += [q for q in quoted if q not in candidates]
        for name in candidates:
            if name in DROPPABLE and name in body:
                value = body.pop(name, None)
                if name in ("temperature", "top_p"):  # often only the default is taken: learn the value
                    self._learn(f"dropped unsupported parameter '{name}' for {self.model}", refuse=(name, value))
                else:
                    self._learn(f"dropped unsupported parameter '{name}' for {self.model}", drop=name)
                return "retry"
            if name == "reasoning_effort" and "reasoning" in body:
                body.pop("reasoning", None)
                self._learn(f"dropped unsupported parameter 'reasoning_effort' for {self.model}", drop="reasoning_effort")
                return "retry"
        if "developer" in lower and "role" in lower and any(m.get("role") == "developer" for m in body.get("messages", [])):
            for m in body["messages"]:
                if m.get("role") == "developer":
                    m["role"] = "system"
            self._learn("sent 'developer' messages as 'system'", developer_role=False)
            return "retry"
        if "system" in lower and "role" in lower and any(m.get("role") == "system" for m in body.get("messages", [])):
            body["messages"] = [{"role": "user", "content": msgs.content_text(m.get("content"))} if m.get("role") == "system" else m for m in body["messages"]]
            self._note(f"{self.model} rejects system messages; sent them as user messages")
            return "retry"
        return None

    # ------------------------------------------------------------------- API

    def _complete(self, request: dict) -> Response:
        state: dict = {}
        adaptations = 0
        while True:
            api = state.get("api") or ("responses" if self._wants_responses() else "chat")
            body = self._responses_body(request) if api == "responses" else self._chat_body(request)
            while True:
                try:
                    if api == "responses":
                        response = self.parse_responses(self._post("/responses", body))
                    else:
                        response = self.parse_chat(self._post("/chat/completions", body))
                    response.rejected = self._rejected_in(request, api, body)
                    return response
                except ProviderError as err:
                    if adaptations >= MAX_ADAPTATIONS:
                        raise
                    outcome = self._adapt(body, err, api, state)
                    if outcome is None:
                        raise
                    adaptations += 1
                    if outcome == "switch":
                        break  # rebuild the body for the other API

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), 256):
            chunk = texts[start : start + 256]
            data = http.require_object(self._post("/embeddings", {"model": self.model, "input": chunk}), "embeddings", "data")
            rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
            out.extend(r.get("embedding") or [] for r in rows)
        return out


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI. `model` is the deployment name.

    Uses the deployments route with api-version (default from
    AZURE_OPENAI_API_VERSION or 2024-10-21), or the v1 route when
    api_version is "v1".
    """

    kind = "azure"

    def __init__(self, model: str, *, name: str | None = None, options: dict | None = None):
        o = dict(options or {})
        endpoint = str(o.get("endpoint") or _env(o.get("endpoint_env", "AZURE_OPENAI_ENDPOINT")) or "").strip().rstrip("/")
        if not endpoint and "base_url" not in o:
            raise ProviderError(
                f"{name or 'azure'}:{model}: set AZURE_OPENAI_ENDPOINT (e.g. https://my-resource.openai.azure.com) or providers.azure.endpoint",
                code="missing_endpoint",
            )
        if endpoint and "://" not in endpoint:
            endpoint = "https://" + endpoint  # the resource's host name alone, as the Azure portal shows it
        self.api_version = str(o.get("api_version") or os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-10-21")
        if self.api_version == "v1":
            o.setdefault("base_url", f"{endpoint}/openai/v1")
        else:
            o.setdefault("base_url", f"{endpoint}/openai/deployments/{model}")
        o.setdefault("api_key_env", "AZURE_OPENAI_API_KEY")
        o.setdefault("api", "chat" if self.api_version != "v1" else "auto")
        super().__init__(model, name=name or "azure", options=o)
        self.endpoint = endpoint
        self.official = False
        self.ad_token = clean_key(o.get("ad_token") or os.environ.get("AZURE_OPENAI_AD_TOKEN"))

    def identity(self) -> dict:
        ident = super().identity()
        ident["api_version"] = self.api_version
        return ident

    def _headers(self) -> dict:
        headers = dict(self.extra_headers)
        if self.api_key:
            headers["api-key"] = self.api_key
        elif self.ad_token:
            headers["Authorization"] = f"Bearer {self.ad_token}"
        else:
            raise ProviderError(f"no API key for {self.spec}: set {self.key_env} or AZURE_OPENAI_AD_TOKEN", code="missing_api_key")
        return headers

    def _url(self, path: str) -> str:
        if self.api_version == "v1":
            return f"{self.base_url}{path}"
        return f"{self.base_url}{path}?api-version={self.api_version}"

    def _chat_body(self, request: dict, *, drops: bool = True) -> dict:
        body = super()._chat_body(request, drops=drops)
        if self.api_version != "v1":
            body.pop("model", None)
        return body
