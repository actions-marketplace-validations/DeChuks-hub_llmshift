"""Record real LLM traffic from your app so `llmshift snapshot` can turn it into tests.

    from openai import OpenAI
    from llmshift.record import wrap

    client = wrap(OpenAI(), path="logs/llm.jsonl")   # also works with AsyncOpenAI, Anthropic, AsyncAnthropic
    client.chat.completions.create(...)               # behaves exactly as before

Agents: run each task inside `agent_run` so its calls (and the tool results they
carry) become one multi-step test case:

    from llmshift.record import agent_run
    with agent_run("refund-flow"):
        my_agent.run(task)

Or log calls yourself (any SDK or raw HTTP):

    from llmshift.record import log
    log(request={"model": ..., "messages": [...]}, response=resp, path="logs/llm.jsonl")

Every call appends one JSON line. Recording never raises into your app: if
writing fails, a single warning is emitted and the call result is returned
unchanged. Secrets (API keys, tokens) are redacted before writing.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import datetime as _dt
import functools
import inspect
import json
import os
import reprlib
import secrets
import sys
import threading
import time
import uuid
import warnings
import weakref
from pathlib import Path
from typing import Any, Callable

from . import fileio
from .redact import Redactor
from .textutil import json_loads

_ENDPOINTS = {
    ("chat", "completions", "create"): "chat",
    ("chat", "completions", "parse"): "chat",
    ("responses", "create"): "responses",
    ("responses", "parse"): "responses",
    ("messages", "create"): "messages",
    ("messages", "parse"): "messages",
}
# SDK stream helpers (`with client.messages.stream(...) as s:`): recorded when the block ends.
_STREAM_HELPERS = {
    ("chat", "completions", "stream"): "chat",
    ("responses", "stream"): "responses",
    ("messages", "stream"): "messages",
}
# Attribute paths that lead to an endpoint; everything else is returned untouched.
_ROUTES = {key[:i] for key in (*_ENDPOINTS, *_STREAM_HELPERS) for i in range(1, len(key))}
_ROUTES |= {("beta",)} | {("beta",) + route for route in _ROUTES}
# client.chat.completions.with_raw_response.create(...) (or client.with_raw_response.chat...)
# answers with the HTTP response; the call is recorded when the app parses it.
_RESPONSE_WRAPPERS = ("with_raw_response", "with_streaming_response")
_FORMAT_ARGS = ("response_format", "text_format", "output_format")  # may be a pydantic class
_CLIENT_FACTORIES = ("with_options", "copy")  # methods that return a configured copy of the client

_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
_SAMPLER = secrets.SystemRandom()  # which calls sample_rate records


def _path_key(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(path))


def _path_lock(path: str | os.PathLike) -> threading.Lock:
    """One lock per log file, shared by every Recorder writing to it."""
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(_path_key(path), threading.Lock())


def _dump(model_dump: Any) -> Any:
    try:
        return model_dump(warnings=False)  # pydantic v2: no serializer warnings in the app's output
    except TypeError:
        return model_dump()


_MAX_DEPTH = 64
_MAX_OBJECTS = 200_000  # objects converted: a Mock in an app's tests invents attributes without end
_MAX_VALUES = 5_000_000  # values in all, so data that shares itself heavily still ends
_PLAIN = (dict, list, tuple, str, int, float, bool, type(None))


def _sentinel(value: Any) -> bool:
    """The SDKs' "argument not given" markers (openai.NOT_GIVEN, omit; the same in
    anthropic and other generated SDKs): no value at all, never part of a request."""
    return type(value).__name__ in ("NotGiven", "Omit")


def to_dict(obj: Any) -> Any:
    """Best-effort conversion of SDK objects (pydantic models etc.) to plain data.
    "Not given" sentinels are dropped. Past 64 levels, past 200,000 objects to convert
    (a Mock invents attributes without end) or where data contains itself, the rest
    becomes text; plain data of any size (a response with logprobs) stays data."""
    return _plain(obj, 0, [_MAX_OBJECTS, _MAX_VALUES], set())


def _plain(obj: Any, depth: int, budget: list, path: set) -> Any:
    budget[1] -= 1
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if depth > _MAX_DEPTH or budget[1] <= 0 or id(obj) in path:  # id in path: it contains itself
        return _text(obj)
    path.add(id(obj))
    try:
        if isinstance(obj, dict):
            return {k: _plain(v, depth + 1, budget, path) for k, v in obj.items() if not _sentinel(v)}
        if isinstance(obj, (list, tuple)):
            return [_plain(v, depth + 1, budget, path) for v in obj if not _sentinel(v)]
        budget[0] -= 1
        if budget[0] <= 0:
            return _text(obj)
        for method in ("model_dump", "to_dict", "dict"):
            fn = getattr(obj, method, None)
            if callable(fn):
                try:
                    data = _dump(fn) if method == "model_dump" else fn()
                except Exception:  # nosec B112  # try the next way to read the object
                    continue
                if isinstance(data, _PLAIN):  # a Mock's model_dump() returns another Mock
                    return _plain(data, depth + 1, budget, path)
        if hasattr(obj, "__dict__") and not type(obj).__module__.startswith("unittest.mock"):
            return {k: _plain(v, depth + 1, budget, path) for k, v in vars(obj).items() if not k.startswith("_") and not _sentinel(v)}
        return _text(obj)
    finally:
        path.discard(id(obj))


_SHORT = reprlib.Repr()  # a container shown as text is shown in part: str() would render all of it
_SHORT.maxlevel, _SHORT.maxstring, _SHORT.maxother = 3, 300, 300


def _text(obj: Any) -> str:
    try:
        return (_SHORT.repr(obj) if isinstance(obj, (dict, list, tuple, set)) else str(obj))[:1000]
    except Exception:
        return f"<{type(obj).__name__}>"


_AGENT_RUN: contextvars.ContextVar = contextvars.ContextVar("llmshift_agent_run", default=None)


@contextlib.contextmanager
def agent_run(name: str | None = None, run_id: str | None = None) -> Any:
    """Link every call recorded inside the block (also in asyncio tasks started there)
    into one agent run. Yields the run id."""
    state = {"id": str(run_id or uuid.uuid4().hex[:16]), "name": name, "step": 0, "lock": threading.Lock()}
    token = _AGENT_RUN.set(state)
    try:
        yield state["id"]
    finally:
        _AGENT_RUN.reset(token)


class Recorder:
    def __init__(self, path: str | os.PathLike = "logs/llmshift.jsonl", *, sample_rate: float = 1.0, redact_secrets: bool = True, metadata: dict | None = None):
        self.path = Path(path).expanduser()
        rate = float(sample_rate)
        if rate != rate:  # NaN would otherwise record every call
            raise ValueError("sample_rate must be a number between 0 and 1")
        self.sample_rate = max(0.0, min(1.0, rate))
        self.redactor = Redactor(secrets=True, pii=False) if redact_secrets else None
        self.metadata = dict(metadata or {})
        self._lock = _path_lock(self.path)
        self._warned = False
        _ALL_RECORDERS.add(self)

    def sampled(self) -> bool:
        return self.sample_rate >= 1.0 or _SAMPLER.random() < self.sample_rate

    def log(self, request: dict, response: Any = None, *, provider: str | None = None, api: str = "chat", latency_ms: float | None = None, error: str | None = None, metadata: dict | None = None) -> None:
        try:
            req = to_dict(request) or {}
            req = {k: v for k, v in req.items() if k not in ("extra_headers", "extra_query", "timeout", "api_key")}
            rec: dict = {
                "llmshift_record": 1,
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
                "provider": provider,
                "api": api,
                "request": req,
                "response": to_dict(response),
            }
            if latency_ms is not None:
                rec["latency_ms"] = round(latency_ms, 1)
            if error:
                rec["error"] = error
            meta = to_dict({**self.metadata, **(metadata or {})})  # plain data: an app's own objects too
            if meta:
                rec["metadata"] = meta
            run = _AGENT_RUN.get()
            if run is not None:
                with run["lock"]:
                    run["step"] += 1
                    step = run["step"]
                rec["agent_run"] = {"id": run["id"], "step": step, **({"name": run["name"]} if run["name"] else {})}
            if self.redactor is not None:
                rec = self.redactor.obj(rec)
            data = memoryview((json.dumps(rec, ensure_ascii=False, default=str) + "\n").encode("utf-8", "replace"))
            with self._lock:
                # One locked append per record: lines stay whole even when several
                # processes (web workers, jobs) log to the same file.
                fileio.append(self.path, bytes(data))
        except Exception as exc:  # recording must never break the app
            if not self._warned:
                self._warned = True
                _warn(f"llmshift: could not record an LLM call ({exc})")


def _warn(message: str) -> None:
    """A warning that can never break the app: under -W error (or pytest's
    filterwarnings = error) warnings.warn would raise into the app's own call."""
    with contextlib.suppress(Exception):
        warnings.warn(message, RuntimeWarning, stacklevel=3)


_recorders: dict[str, Recorder] = {}
_recorders_lock = threading.Lock()
_ALL_RECORDERS: weakref.WeakSet = weakref.WeakSet()


def _release_after_fork() -> None:
    # Only the forking thread survives a fork; a lock another thread held at that moment
    # would stay locked in the child forever.
    redactor_locks = [r.redactor._lock for r in list(_ALL_RECORDERS) if r.redactor is not None]
    for lock in (_PATH_LOCKS_GUARD, _recorders_lock, *_PATH_LOCKS.values(), *redactor_locks):
        if lock.locked():
            lock.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_release_after_fork)


def log(request: dict, response: Any = None, *, path: str | os.PathLike | None = None, **kwargs: Any) -> None:
    """Append one request/response pair to the log file."""
    if path is None:
        path = os.environ.get("LLMSHIFT_LOG", "logs/llmshift.jsonl")
    key = _path_key(path)
    with _recorders_lock:
        recorder = _recorders.get(key)
        if recorder is None:
            recorder = _recorders[key] = Recorder(path)
    recorder.log(request, response, **kwargs)


def _get(obj: Any, *path: Any) -> Any:
    for key in path:
        if obj is None:
            return None
        if isinstance(key, int):
            try:
                obj = obj[key]
            except (IndexError, KeyError, TypeError):
                return None
        elif isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
    return obj


def _arguments(text: str) -> Any:
    try:
        return json_loads(text) if text.strip() else {}
    except (ValueError, RecursionError):
        return {"_raw": text}


class _StreamAccumulator:
    """Rebuilds a complete response from the streamed chunks of the major SDKs:
    text, tool calls (with arguments split across chunks), finish reason and usage."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.model: str | None = None
        self.finish: str | None = None
        self.usage: dict = {}
        self.calls: dict[tuple, dict] = {}
        self.choices: dict[int, dict] = {}  # chat: each choice (n > 1) on its own
        self.final: dict | None = None  # Responses API: the whole response arrives at the end

    @property
    def complete(self) -> bool:
        """Whether the stream reached its end: a finish reason, or the final response."""
        return self.final is not None or self.finish is not None

    def add(self, chunk: Any) -> None:
        with contextlib.suppress(Exception):
            self._add(chunk)

    def _usage(self, usage: Any) -> None:
        usage = to_dict(usage)
        if isinstance(usage, dict):
            self.usage.update({k: v for k, v in usage.items() if v is not None})

    def _add(self, chunk: Any) -> None:
        self.model = self.model or _get(chunk, "model") or _get(chunk, "message", "model") or _get(chunk, "modelVersion")
        ctype = _get(chunk, "type")
        if ctype is None and _get(chunk, "candidates") is not None:  # Gemini
            cand = _get(chunk, "candidates", 0)
            for part in _get(cand, "content", "parts") or []:
                text = _get(part, "text")
                if isinstance(text, str) and not _get(part, "thought"):
                    self.text.append(text)
                call = _get(part, "functionCall") or _get(part, "function_call")
                if _get(call, "name"):
                    args = to_dict(_get(call, "args"))
                    self.calls[("gemini", len(self.calls))] = {"id": _get(call, "id"), "name": _get(call, "name"), "arguments": json.dumps(args or {})}
            self.finish = _get(cand, "finishReason") or _get(cand, "finish_reason") or self.finish
            if (_get(chunk, "usageMetadata") or _get(chunk, "usage_metadata")) is not None:
                self._usage(_get(chunk, "usageMetadata") or _get(chunk, "usage_metadata"))
        elif ctype is None:  # OpenAI chat completion chunks: each choice (n > 1) by its index
            for choice in _get(chunk, "choices") or []:
                index = _get(choice, "index")
                index = index if isinstance(index, int) else 0
                slot = self.choices.setdefault(index, {"text": [], "refusal": [], "calls": {}, "finish": None})
                delta = _get(choice, "delta")
                for key in ("text", "refusal"):
                    piece = _get(delta, "content" if key == "text" else "refusal")
                    if isinstance(piece, str):
                        slot[key].append(piece)
                for tc in _get(delta, "tool_calls") or []:
                    tindex = _get(tc, "index")
                    call = slot["calls"].setdefault(tindex if isinstance(tindex, int) else len(slot["calls"]), {"id": None, "name": "", "arguments": ""})
                    call["id"] = call["id"] or _get(tc, "id")
                    call["name"] = call["name"] or _get(tc, "function", "name") or ""
                    args = _get(tc, "function", "arguments")
                    if isinstance(args, str):
                        call["arguments"] += args
                fin = _get(choice, "finish_reason")
                if fin:
                    slot["finish"] = fin
                    self.finish = self.finish or fin
            if _get(chunk, "usage") is not None:  # sent last with stream_options.include_usage
                self._usage(_get(chunk, "usage"))
        elif ctype == "message_start":  # Anthropic
            self._usage(_get(chunk, "message", "usage"))
        elif ctype == "content_block_start":
            block = _get(chunk, "content_block")
            if _get(block, "type") == "tool_use":
                self.calls[("msg", _get(chunk, "index"))] = {"id": _get(block, "id"), "name": _get(block, "name") or "", "arguments": ""}
        elif ctype == "content_block_delta":
            dtype = _get(chunk, "delta", "type")
            if dtype == "text_delta":
                self.text.append(_get(chunk, "delta", "text") or "")
            elif dtype == "input_json_delta":
                slot = self.calls.get(("msg", _get(chunk, "index")))
                if slot is not None:
                    slot["arguments"] += _get(chunk, "delta", "partial_json") or ""
        elif ctype == "message_delta":
            self.finish = _get(chunk, "delta", "stop_reason") or self.finish
            self._usage(_get(chunk, "usage"))
        elif ctype == "response.output_text.delta":  # OpenAI Responses
            self.text.append(_get(chunk, "delta") or "")
        elif ctype in ("response.completed", "response.incomplete", "response.failed"):
            final = to_dict(_get(chunk, "response"))
            if isinstance(final, dict):
                self.final = final

    def response(self, api: str) -> dict:
        if api == "responses" and self.final:
            return self.final
        text = "".join(self.text)
        calls = [self.calls[k] for k in sorted(self.calls, key=lambda k: (k[0], k[1] if isinstance(k[1], int) else 0))]
        out: dict
        if api == "gemini":
            parts: list[dict] = [{"text": text}] if text or not calls else []
            for c in calls:
                fc = {"name": c["name"], "args": _arguments(c["arguments"])}
                if c["id"]:
                    fc["id"] = c["id"]
                parts.append({"functionCall": fc})
            out = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": self.finish}], "modelVersion": self.model}
            if self.usage:
                out["usageMetadata"] = dict(self.usage)
            return out
        if api == "messages":
            content: list[dict] = [{"type": "text", "text": text}] if text or not calls else []
            content += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": _arguments(c["arguments"])} for c in calls]
            out = {"type": "message", "role": "assistant", "model": self.model, "content": content, "stop_reason": self.finish}
        elif api == "responses":
            out = {"object": "response", "model": self.model, "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        else:
            choices = []
            for index in sorted(self.choices) or [0]:
                slot = self.choices.get(index) or {"text": [], "refusal": [], "calls": {}, "finish": self.finish}
                said = "".join(slot["text"])
                chat_calls = [slot["calls"][k] for k in sorted(slot["calls"])]
                message: dict = {"role": "assistant", "content": said if said or not chat_calls else None}
                if slot["refusal"]:
                    message["refusal"] = "".join(slot["refusal"])
                if chat_calls:
                    message["tool_calls"] = [
                        {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for c in chat_calls
                    ]
                choices.append({"index": index, "message": message, "finish_reason": slot["finish"]})
            out = {"object": "chat.completion", "model": self.model, "choices": choices}
        if self.usage:
            out["usage"] = dict(self.usage)
        return out


def _failure(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}"[:500]


class _StreamProxy:
    """Passes a stream through unchanged and records the whole response once it ends.
    A stream that failed, was closed or abandoned part-way, or ended without a finish
    reason is recorded with an error, so it never becomes a test's expected answer."""

    def __init__(self, stream: Any, done: Any):
        self._stream = stream
        self._done = done
        self._acc = _StreamAccumulator()
        self._finished = False
        self._it: Any = None
        self._ait: Any = None

    def _finish(self, error: str | None = None) -> None:
        if not self._finished:
            self._finished = True
            if self._acc.complete:  # the answer reached its finish reason: whole, however the app left
                error = None        # (apps often stop at finish_reason, before a trailing usage chunk)
            elif error is None:
                error = "the stream ended without a finish reason (cut off?)"
            self._done(self._acc, error)

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        if self._it is None:
            self._it = iter(self._stream)
        try:
            chunk = next(self._it)
        except StopIteration:
            self._finish()
            raise
        except BaseException as exc:  # the connection failed, or the SDK raised an error event
            self._finish(_failure(exc))
            raise
        self._acc.add(chunk)
        return chunk

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if self._ait is None:
            self._ait = self._stream.__aiter__()
        try:
            chunk = await self._ait.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._finish(_failure(exc))
            raise
        self._acc.add(chunk)
        return chunk

    def close(self) -> Any:
        self._finish("the app closed the stream before its end")
        closer = getattr(self._stream, "close", None)
        return closer() if callable(closer) else None

    async def aclose(self) -> Any:
        self._finish("the app closed the stream before its end")
        closer = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        if callable(closer):
            result = closer()
            if inspect.isawaitable(result):
                return await result
            return result
        return None

    def __enter__(self) -> Any:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        self._finish(_failure(exc[1]) if exc[1] is not None else "the app left the stream before its end")
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*exc)
        return None

    async def __aenter__(self) -> Any:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        self._finish(_failure(exc[1]) if exc[1] is not None else "the app left the stream before its end")
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*exc)
        return None

    def __del__(self) -> None:  # a stream abandoned part-way is still recorded, as incomplete
        # A plain try, not contextlib.suppress: __del__ can run while the interpreter shuts
        # down, after module globals such as contextlib are gone.
        try:
            self._finish("the stream was abandoned before its end")
        except Exception:  # nosec B110  # a finalizer must never raise
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _openai_converter(api: str) -> Any:
    """The OpenAI SDK's own conversion of a response-format class, when it is installed."""
    try:
        if api == "chat":
            from openai.lib._parsing._completions import type_to_response_format_param as convert
        else:
            from openai.lib._parsing._responses import type_to_text_format_param as convert
    except Exception:
        return None
    return convert


def _json_schema(cls: Any, converter: Any = None) -> Any:
    """What the SDK sends for a pydantic class given as a response format: its own
    conversion when the SDK provides one, else the class's JSON schema."""
    if converter is not None:
        with contextlib.suppress(Exception):
            return converter(cls)
    try:
        return cls.model_json_schema()
    except Exception:
        return None


def _wire_request(api: str, kwargs: dict) -> dict:
    """The request as sent on the wire: pydantic classes passed as `response_format`
    (chat), `text_format` (responses) or `output_format` (messages) become the JSON
    schema the SDK derives from them, so recorded cases replay with the same format."""
    if not any(isinstance(kwargs.get(k), type) for k in _FORMAT_ARGS):
        return kwargs
    try:
        return _wire_formats(api, kwargs)
    except Exception:  # recording never breaks the app
        return kwargs


def _wire_formats(api: str, kwargs: dict) -> dict:
    out = dict(kwargs)
    cls = out.pop("text_format", None) if api == "responses" else out.get("output_format") if api == "messages" else out.get("response_format")
    if not isinstance(cls, type):
        return out
    name = getattr(cls, "__name__", "response")
    if api == "chat":
        param = _json_schema(cls, _openai_converter("chat"))
        if isinstance(param, dict) and param.get("type") == "json_schema":
            out["response_format"] = param
        elif isinstance(param, dict):
            out["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": param, "strict": True}}
    elif api == "responses":
        param = _json_schema(cls, _openai_converter("responses"))
        if isinstance(param, dict) and param.get("type") != "json_schema":
            param = {"type": "json_schema", "name": name, "schema": param, "strict": True}
        if isinstance(param, dict):
            out["text"] = {**(out.get("text") or {}), "format": param}
    elif api == "messages":
        out.pop("output_format", None)
        schema = _json_schema(cls)
        if isinstance(schema, dict):
            out["output_config"] = {**(out.get("output_config") or {}), "format": {"type": "json_schema", "schema": schema}}
    return out


def _complete(snapshot: Any) -> bool:
    """Whether a stream helper's answer reached its end (a stop or finish reason)."""
    data = to_dict(snapshot)
    if not isinstance(data, dict):
        return False
    if data.get("stop_reason") or data.get("status") in ("completed", "incomplete", "failed"):
        return True  # Anthropic; OpenAI Responses (the helper keeps it only once done)
    choices = data.get("choices")
    return bool(choices) and all(isinstance(c, dict) and c.get("finish_reason") for c in choices)


def _request_copy(api: str, kwargs: dict) -> dict:
    """The request as sent, copied to plain data when the call starts: an app that
    appends to its message list before a stream ends must not change the record, and
    the shadow model gets plain messages, not SDK objects or "not given" markers."""
    try:
        data = to_dict(_wire_request(api, kwargs))
        return data if isinstance(data, dict) else {}
    except Exception:  # recording never breaks the app
        return {}


def _final_snapshot(stream: Any) -> Any:
    """The answer an SDK stream helper has accumulated so far (reads nothing more)."""
    readers = (
        lambda s: s.current_message_snapshot,  # Anthropic MessageStream
        lambda s: s.current_completion_snapshot,  # OpenAI ChatCompletionStream
        lambda s: s._state._completed_response,  # OpenAI ResponseStream (set once completed)
    )
    for read in readers:
        try:
            snapshot = read(stream)
        except Exception:  # nosec B112  # nothing received yet, or another SDK version
            continue
        if snapshot is not None:
            return snapshot
    return None


_UNSEEN = object()


def _on_parse(response: Any, handle: Callable[[Any], Any], fail: Callable[[BaseException], None]) -> Any:
    """An SDK HTTP response (with_raw_response) whose parse() hands what it returns to
    `handle` once: the answer is recorded, a stream wrapped. A with_streaming_response
    context manager gets the same hook on the response it yields."""
    if not hasattr(response, "parse") and (hasattr(response, "__enter__") or hasattr(response, "__aenter__")):
        return _ParseOnEnter(response, handle, fail)
    original = getattr(response, "parse", None)
    if not callable(original):
        return response
    lock = threading.Lock()
    state = {"seen": _UNSEEN, "value": None}

    def once(parsed: Any) -> Any:
        with lock:
            if state["seen"] is not _UNSEEN:  # the SDK caches the parsed answer: the same object again
                return state["value"] if parsed is state["seen"] else parsed
            state["seen"] = parsed
            try:
                state["value"] = handle(parsed)
            except Exception:  # recording never breaks the app
                state["value"] = parsed
            return state["value"]

    def parse(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        if inspect.isawaitable(out):  # AsyncAPIResponse.parse() is a coroutine

            async def later() -> Any:
                return once(await out)

            return later()
        return once(out)

    with contextlib.suppress(Exception):
        response.parse = parse
    return response


class _ParseOnEnter:
    """A with_streaming_response context manager: the response it yields is hooked."""

    def __init__(self, manager: Any, handle: Callable[[Any], Any], fail: Callable[[BaseException], None]):
        self._manager, self._handle, self._fail = manager, handle, fail

    def __enter__(self) -> Any:
        try:
            response = self._manager.__enter__()
        except Exception as exc:  # an error status raises here
            self._fail(exc)
            raise
        return _on_parse(response, self._handle, self._fail)

    def __exit__(self, *exc: Any) -> Any:
        return self._manager.__exit__(*exc)

    async def __aenter__(self) -> Any:
        try:
            response = await self._manager.__aenter__()
        except Exception as exc:
            self._fail(exc)
            raise
        return _on_parse(response, self._handle, self._fail)

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._manager.__aexit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


class _HelperStream:
    """What an SDK stream helper returns (a context manager that makes the request on
    entering). Passed through unchanged; the final answer is recorded when the block ends."""

    def __init__(self, manager: Any, done: Any):
        self._manager = manager
        self._done = done
        self._stream: Any = None
        self._start = time.perf_counter()

    def _record(self, exc: BaseException | None) -> None:
        with contextlib.suppress(Exception):  # recording never breaks the app
            latency = (time.perf_counter() - self._start) * 1000
            snapshot = _final_snapshot(self._stream) if self._stream is not None else None
            if snapshot is not None and _complete(snapshot):
                self._done(snapshot, None, latency)  # the answer is whole, whatever the app did afterwards
            elif exc is not None:
                self._done(snapshot, _failure(exc), latency)
            elif snapshot is not None:
                self._done(snapshot, "the stream block ended before the answer was complete", latency)

    def __enter__(self) -> Any:
        self._start = time.perf_counter()
        try:
            self._stream = self._manager.__enter__()
        except Exception as exc:
            self._record(exc)
            raise
        return self._stream

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self._manager.__exit__(*exc)
        finally:
            self._record(exc[1])

    async def __aenter__(self) -> Any:
        self._start = time.perf_counter()
        try:
            self._stream = await self._manager.__aenter__()
        except Exception as exc:
            self._record(exc)
            raise
        return self._stream

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            return await self._manager.__aexit__(*exc)
        finally:
            self._record(exc[1])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


_PROXY_FIELDS = ("_target", "_recorder", "_provider", "_parts", "_shadow")
# Set while a recording call runs the SDK's, so that a recording call inside it passes the
# call on. A tracer that patched create after wrap() (LangSmith) calls the recording create,
# and the proxy wrapped the tracer's create again: every call was recorded and shadowed twice.
_IN_CALL: contextvars.ContextVar = contextvars.ContextVar("llmshift_in_call", default=False)


def _trail(fn: Any) -> list:
    """fn and the functions it wraps, as functools.wraps leaves them in __wrapped__."""
    out = []
    with contextlib.suppress(Exception):  # an odd object's __wrapped__ never breaks the app
        while fn is not None and len(out) < 32:
            out.append(fn)
            fn = getattr(fn, "__wrapped__", None)
    return out


def _records(fn: Any, kind: str) -> bool:
    """Whether fn is a recording call of this kind ("call" or "stream") or wraps one: a
    tracer's create around wrap()'s. That one records, after the tracer took out its own
    arguments (langsmith_extra)."""
    for f in _trail(fn):
        with contextlib.suppress(Exception):
            if getattr(f, "_llmshift_records", None) == kind:
                return True
    return False


def _like(fn: Any, wrapper: Any, kind: str | None = None) -> Any:
    """`wrapper` with the name, signature and __wrapped__ of the SDK method fn, as
    instructor and tracers read them."""
    with contextlib.suppress(Exception):  # a callable update_wrapper cannot read: the wrapper works all the same
        functools.update_wrapper(wrapper, fn)
        if sys.version_info < (3, 10) and inspect.ismethod(fn):
            # 3.9's inspect.signature follows __wrapped__ past the method to its function,
            # and showed `self` as the first parameter.
            wrapper.__signature__ = inspect.signature(fn)
    wrapper.__dict__.pop("_llmshift_records", None)  # copied from a tracer's create around a recording one
    if kind:
        wrapper._llmshift_records = kind
    return wrapper


def _for_raw_response(kwargs: dict) -> bool:
    """Whether an SDK's with_raw_response or with_streaming_response made this call. Built on
    a create a tracer patched over wrap()'s, it calls the recording create for the HTTP
    response; the recording call around it records the answer, once parsed."""
    headers = kwargs.get("extra_headers")
    return isinstance(headers, dict) and "X-Stainless-Raw-Response" in headers


class _Proxy:
    def __init__(self, target: Any, recorder: Recorder | None, provider: str | None, parts: tuple = (), shadow: Any = None):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_parts", parts)
        object.__setattr__(self, "_shadow", shadow)

    @property  # type: ignore[misc]
    def __class__(self) -> Any:  # isinstance(wrap(OpenAI()), OpenAI) stays true: instructor and others check it
        return type(self._target)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._target)))

    def __copy__(self) -> Any:
        return _Proxy(copy.copy(self._target), self._recorder, self._provider, self._parts, self._shadow)

    def __deepcopy__(self, memo: dict) -> Any:
        return _Proxy(copy.deepcopy(self._target, memo), self._recorder, self._provider, self._parts, self._shadow)

    def __getattr__(self, name: str) -> Any:
        if name in _PROXY_FIELDS:  # not set yet (an instance being built): never ask the target
            raise AttributeError(name)
        attr = getattr(self._target, name)
        parts = self._parts + (name,)
        raw = any(p in _RESPONSE_WRAPPERS for p in parts)
        key = tuple(p for p in parts if p not in _RESPONSE_WRAPPERS)
        api = _ENDPOINTS.get(key[-3:]) or _ENDPOINTS.get(key[-2:])
        if api and callable(attr):
            return attr if not raw and _records(attr, "call") else self._wrap_call(attr, api, raw=raw)
        helper = None if raw else (_STREAM_HELPERS.get(parts[-3:]) or _STREAM_HELPERS.get(parts[-2:]))
        if helper and callable(attr):
            return attr if _records(attr, "stream") else self._wrap_helper(attr, helper)
        if not self._parts and name in _CLIENT_FACTORIES and callable(attr):
            return self._wrap_factory(attr)
        if key in _ROUTES or (raw and not key):
            return _Proxy(attr, self._recorder, self._provider, parts, self._shadow)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)

    def __enter__(self) -> Any:
        self._target.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._target.__exit__(*exc)

    async def __aenter__(self) -> Any:
        await self._target.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._target.__aexit__(*exc)

    def _wrap_factory(self, fn: Any) -> Any:
        recorder, provider, shadow = self._recorder, self._provider, self._shadow

        def call(*args: Any, **kwargs: Any) -> Any:
            return _Proxy(fn(*args, **kwargs), recorder, provider, shadow=shadow)

        return _like(fn, call)

    def _wrap_helper(self, fn: Any, api: str) -> Any:
        recorder, provider, shadow = self._recorder, self._provider, self._shadow

        def call(*args: Any, **kwargs: Any) -> Any:
            if _IN_CALL.get():  # a recording call runs this one: that one records it
                return fn(*args, **kwargs)
            logging = recorder is not None and recorder.sampled()
            token = _IN_CALL.set(True)
            try:
                if not logging and shadow is None:
                    return fn(*args, **kwargs)
                request = _request_copy(api, kwargs)
                manager = fn(*args, **kwargs)
            finally:
                _IN_CALL.reset(token)

            def done(snapshot: Any, error: str | None, latency_ms: float) -> None:
                if snapshot is None and error is None:
                    return
                if logging:
                    recorder.log(request, snapshot, provider=provider, api=api, latency_ms=latency_ms, error=error)
                if shadow is not None and error is None:
                    with contextlib.suppress(Exception):  # shadowing never breaks the app
                        shadow.observe(api, request, snapshot)

            return _HelperStream(manager, done)

        return _like(fn, call, "stream")

    def _wrap_call(self, fn: Any, api: str, raw: bool = False) -> Any:
        recorder, provider, shadow = self._recorder, self._provider, self._shadow

        def elapsed(start: float) -> float:
            return (time.perf_counter() - start) * 1000

        def observe(kwargs: dict, response: Any) -> None:
            if shadow is not None:
                with contextlib.suppress(Exception):  # shadowing never breaks the app
                    shadow.observe(api, kwargs, response)

        def failed(kwargs: dict, start: float, exc: BaseException, logging: bool) -> None:
            if logging:
                recorder.log(kwargs, None, provider=provider, api=api, error=f"{exc.__class__.__name__}: {exc}"[:500], latency_ms=elapsed(start))

        def finished(kwargs: dict, start: float, result: Any, logging: bool) -> Any:
            if raw:
                return _on_parse(result, lambda parsed: finished_parsed(kwargs, start, parsed, logging), lambda exc: failed(kwargs, start, exc, logging))
            return finished_parsed(kwargs, start, result, logging)

        def finished_parsed(kwargs: dict, start: float, result: Any, logging: bool) -> Any:
            if kwargs.get("stream"):

                def done(acc: _StreamAccumulator, error: str | None) -> None:
                    response = acc.response(api)
                    if logging:
                        recorder.log(kwargs, response, provider=provider, api=api, latency_ms=elapsed(start), error=error)
                    if error is None:
                        observe(kwargs, response)

                return _StreamProxy(result, done)
            if logging:
                recorder.log(kwargs, result, provider=provider, api=api, latency_ms=elapsed(start))
            observe(kwargs, result)
            return result

        async def finish_async(awaitable: Any, kwargs: dict, start: float, logging: bool, recording: bool) -> Any:
            token = _IN_CALL.set(True)
            try:
                result = await awaitable
            except Exception as exc:
                failed(kwargs, start, exc, logging)
                raise
            finally:
                _IN_CALL.reset(token)
            return finished(kwargs, start, result, logging) if recording else result

        def begin(kwargs: dict) -> tuple:
            """Whether the call is logged, whether it is recorded at all (logged or
            shadowed), and the request as sent."""
            logging = recorder is not None and recorder.sampled()
            recording = logging or shadow is not None
            return logging, recording, _request_copy(api, kwargs) if recording else {}

        # An async client's create is often a plain function (the SDK's argument check) around
        # a coroutine function. Its wrapper is a coroutine function, as instructor and tracers
        # check; with_streaming_response's create is not one: it returns a context manager.
        if inspect.iscoroutinefunction(fn) or (not raw and any(inspect.iscoroutinefunction(f) for f in _trail(fn))):

            async def async_call(*args: Any, **kwargs: Any) -> Any:
                if _IN_CALL.get() or (not raw and _for_raw_response(kwargs)):  # recorded around this call
                    return await fn(*args, **kwargs)
                logging, recording, request = begin(kwargs)
                start = time.perf_counter()
                token = _IN_CALL.set(True)
                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    failed(request, start, exc, logging)
                    raise
                finally:
                    _IN_CALL.reset(token)
                return finished(request, start, result, logging) if recording else result

            return _like(fn, async_call, None if raw else "call")

        def call(*args: Any, **kwargs: Any) -> Any:
            if _IN_CALL.get() or (not raw and _for_raw_response(kwargs)):  # recorded around this call
                return fn(*args, **kwargs)
            logging, recording, request = begin(kwargs)
            start = time.perf_counter()
            token = _IN_CALL.set(True)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                failed(request, start, exc, logging)
                raise
            finally:
                _IN_CALL.reset(token)
            if inspect.isawaitable(result):  # an async method that did not look like one
                return finish_async(result, request, start, logging, recording)
            return finished(request, start, result, logging) if recording else result

        return _like(fn, call, None if raw else "call")

    def __repr__(self) -> str:
        return f"<llmshift recording proxy for {self._target!r}>"


def wrap(
    client: Any,
    path: str | os.PathLike = "logs/llmshift.jsonl",
    *,
    sample_rate: float = 1.0,
    recorder: Recorder | None = None,
    metadata: dict | None = None,
    record: bool = True,
    shadow: Any = None,
    shadow_rate: float = 0.05,
    shadow_path: str | os.PathLike | None = None,
) -> Any:
    """Return a proxy of an OpenAI/Anthropic client that records every call. With
    `shadow` (a model spec or a `llmshift.shadow.Shadow`), a share of the calls is also
    copied to that model in the background (shadow mode); `record=False` keeps only that."""
    if type(client) is _Proxy:  # wrapping twice would record every call twice
        client = client._target
    module = type(client).__module__.split(".")[0]
    provider = {"openai": "openai", "anthropic": "anthropic"}.get(module, module)
    if shadow is not None and not hasattr(shadow, "observe"):
        from .shadow import DEFAULT_PATH, Shadow, shared

        if isinstance(shadow, str):  # one set of workers however many clients are wrapped
            shadow = shared(shadow, rate=shadow_rate, path=shadow_path or DEFAULT_PATH)
        else:
            shadow = Shadow(shadow, rate=shadow_rate, path=shadow_path or DEFAULT_PATH)
    rec = (recorder or Recorder(path, sample_rate=sample_rate, metadata=metadata)) if record else None
    return _Proxy(client, rec, provider, shadow=shadow)
