"""Read LLM usage logs and normalise every record into an Interaction.

Formats are detected per record, so one directory can mix them:

* request/response pairs for OpenAI Chat Completions, OpenAI Responses,
  Anthropic Messages and Gemini (keys request/response, req/res,
  request_body/response_body, Helicone-style {"request": {"body": ...}})
* flat OpenAI logs: {"model", "messages", ..., "choices": [...]}
* OpenAI Batch API input + output files (joined on custom_id)
* chat datasets: {"messages": [..., {"role": "assistant", ...}]} and ShareGPT
* Langfuse / LangSmith style exports: input/output, inputs/outputs
* generic JSON/CSV rows with prompt/input/question + output/response/answer
* LLMShift test files and `llmshift.record` logs
"""

from __future__ import annotations

import codecs
import csv
import datetime as _dt
import email.utils
import gzip
import heapq
import io
import json
import os
import random
import re
import sys
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from . import messages as msgs
from .textutil import clean_surrogates, has_surrogate_escape, json_loads, jsonl_lines

LOG_SUFFIXES = (".jsonl", ".ndjson", ".json", ".csv", ".log", ".txt")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # logged prompts easily exceed the 128 KB default


@dataclass
class Interaction:
    messages: list[dict]
    params: dict = field(default_factory=dict)
    output: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    model: str | None = None
    timestamp: str | None = None
    finish_reason: str | None = None
    source: str = ""
    error: str | None = None
    usage: dict = field(default_factory=dict)
    latency_ms: float | None = None
    fmt: str = ""
    checks: list[dict] = field(default_factory=list)
    vars: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    case_id: str | None = None
    agent_run: dict | None = None  # {"id", "step"} when the log links calls into one agent run
    agent: dict | None = None  # set by agents.link_runs: the recorded steps of the whole run

    @property
    def has_output(self) -> bool:
        return bool((self.output or "").strip()) or bool(self.tool_calls)


@dataclass
class LogStats:
    files: int = 0
    records: int = 0
    parsed: int = 0
    formats: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    examples: dict = field(default_factory=dict)
    sampled_out: int = 0  # calls left out to stay within max_records

    def skip(self, reason: str, example: str | None = None) -> None:
        self.skipped[reason] += 1
        if example and reason not in self.examples:
            self.examples[reason] = example


# ------------------------------------------------------------------- reading


MAX_DOCUMENT_BYTES = 512 * 1024 * 1024  # a log read as one JSON document
MAX_LINE_BYTES = 64 * 1024 * 1024  # one record in a JSON-lines log
DEFAULT_MAX_RECORDS = 100_000  # requests kept in memory; beyond this a fair sample is kept
_TOO_LARGE = object()


def _open_binary(path: Path) -> Any:
    return gzip.open(path, "rb") if path.suffix.lower() == ".gz" else open(path, "rb")


def _sniff(path: Path) -> bytes:
    with _open_binary(path) as fh:
        return fh.read(4096)


def _encoding(path: Path) -> str:
    """UTF-16 when the file starts with its byte-order mark (what Windows PowerShell 5.1's
    `>` writes), else UTF-8."""
    return "utf-16" if _sniff(path).startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig"


def _read_document(path: Path, ref: str, stats: "LogStats") -> Any:
    """The whole file as one JSON document (None when it is not one)."""
    with _open_binary(path) as fh:
        raw = fh.read(MAX_DOCUMENT_BYTES + 1)
    if len(raw) > MAX_DOCUMENT_BYTES:
        stats.skip("file too large to read as one JSON document (over 512 MB); export JSON Lines instead", ref)
        return _TOO_LARGE
    text = raw.decode("utf-16" if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig", errors="replace")
    try:
        data = json_loads(text)
        return clean_surrogates(data) if has_surrogate_escape(text) else data
    except RecursionError:
        stats.skip("JSON nested too deeply", ref)
        return _TOO_LARGE
    except ValueError:
        return None


def _lines(path: Path, ref: str, stats: "LogStats") -> Iterator[tuple[int, str]]:
    """The file's lines, read one at a time (any size of file, .gz included); a line
    over MAX_LINE_BYTES is skipped without being held in memory."""
    if _encoding(path) == "utf-16":  # two bytes a character: read whole, then split
        with _open_binary(path) as fh:
            raw = fh.read(MAX_DOCUMENT_BYTES + 1)
        if len(raw) > MAX_DOCUMENT_BYTES:
            stats.skip("file too large (over 512 MB as UTF-16); save it as UTF-8", ref)
            return
        for lineno, text in enumerate(jsonl_lines(raw.decode("utf-16", errors="replace")), 1):
            yield lineno, text
        return
    with _open_binary(path) as fh:
        lineno = 0
        while True:
            raw = fh.readline(MAX_LINE_BYTES + 1)
            if not raw:
                return
            lineno += 1
            if len(raw) > MAX_LINE_BYTES:
                while raw and not raw.endswith(b"\n"):
                    raw = fh.readline(MAX_LINE_BYTES + 1)
                stats.skip(f"record too large (one line over {MAX_LINE_BYTES // (1024 * 1024)} MB)", f"{ref}:{lineno}")
                continue
            text = raw.decode("utf-8", errors="replace")
            yield lineno, (text.lstrip("\ufeff") if lineno == 1 else text)


def _inner_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return os.path.splitext(name)[1]


def iter_files(sources: Iterable[str | os.PathLike]) -> Iterator[Path]:
    for src in sources:
        p = Path(src)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and (_inner_suffix(f) in LOG_SUFFIXES) and not any(part.startswith(".") for part in f.relative_to(p).parts):
                    yield f
        elif p.is_file():
            yield p
        else:
            import glob

            matches = sorted(glob.glob(str(src), recursive=True))
            if not matches:
                raise FileNotFoundError(f"no such file or directory: {src}")
            for m in matches:
                if Path(m).is_file():
                    yield Path(m)


def _csv_key(header: str) -> str:
    """"System Prompt", "systemPrompt", " Output " -> system_prompt, system_prompt, output."""
    key = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", str(header).strip())
    return re.sub(r"[\s\-]+", "_", key).lower()


def _csv_value(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in ("[", "{"):
        try:
            return json_loads(value)
        except (ValueError, RecursionError):
            return value
    return value


def iter_records(path: Path, stats: LogStats) -> Iterator[tuple[Any, str]]:
    suffix = _inner_suffix(path)
    ref = path.as_posix()  # portable "file:line" references in test files
    if suffix == ".csv":
        with _open_binary(path) as fh:
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding=_encoding(path), errors="replace", newline=""))
            for i, row in enumerate(reader, 2):
                yield {_csv_key(k): _csv_value(v) for k, v in row.items() if k is not None and str(k).strip()}, f"{ref}:{i}"
        return
    head = _sniff(path).lstrip(b"\xef\xbb\xbf").lstrip()
    if suffix == ".json" or (head.startswith(b"[") and suffix not in (".jsonl", ".ndjson")):
        data = _read_document(path, ref, stats)
        if data is _TOO_LARGE:
            return
        if data is not None:
            if isinstance(data, dict):
                for key in ("data", "records", "logs", "items", "runs", "cases", "tests", "examples", "rows"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if isinstance(data, list):
                for i, rec in enumerate(data):
                    yield rec, f"{ref}[{i}]"
            else:
                yield data, ref
            return
    for lineno, line in _lines(path, ref, stats):
        line = line.strip()
        if not line:
            continue
        if line.startswith(("data:", "event:", "id:", "retry:")) or line == "[DONE]":
            stats.skip("streamed chunk (log whole responses, e.g. with llmshift.record)", f"{ref}:{lineno}")
            continue
        try:
            rec = json_loads(line)
        except RecursionError:
            stats.skip("JSON nested too deeply", f"{ref}:{lineno}")
            continue
        except ValueError:
            stats.skip("invalid JSON line", f"{ref}:{lineno}")
            continue
        yield (clean_surrogates(rec) if has_surrogate_escape(line) else rec), f"{ref}:{lineno}"


# ---------------------------------------------------------------- helpers


def _epoch(num: float) -> _dt.datetime | None:
    """Unix seconds or milliseconds; small numbers (a year, a count) are not timestamps."""
    if isinstance(num, bool) or num < 1e8:
        return None
    try:
        return _dt.datetime.fromtimestamp(num / 1000.0 if num > 1e11 else num, tz=_dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _ts(value: Any) -> str | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        dt = _epoch(value)
        return dt.isoformat() if dt else None
    return str(value)


_TZ_NAME_RE = re.compile(r"\s*(?:UTC|GMT|Z)$", re.I)
_TZ_COMPACT_RE = re.compile(r"([+-]\d{2})(\d{2})$")


def parse_time(value: str | None) -> _dt.datetime | None:
    """Parse ISO-8601 (Z, UTC, +HH or +HHMM offsets, basic format, week dates), date-only,
    YYYYMMDD, HTTP dates and unix timestamps, the same way on every Python. Returns None
    for anything else."""
    if not value:
        return None
    s = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        if re.fullmatch(r"\d{8}", s):
            try:
                return _dt.datetime.strptime(s, "%Y%m%d").replace(tzinfo=_dt.timezone.utc)
            except ValueError:
                return None
        return _epoch(float(s))
    if _TZ_NAME_RE.search(s) and len(s) > 10:
        s = _TZ_NAME_RE.sub("", s) + "+00:00"
    s = _TZ_COMPACT_RE.sub(r"\1:\2", s) if len(s) > 10 else s
    s = s.replace(" ", "T", 1) if "T" not in s and len(s) > 10 else s
    # What Python 3.11 reads and 3.9/3.10 do not: basic format, week dates, decimal commas,
    # offsets in hours. One reading on every Python.
    date, sep, clock = s.partition("T")
    week = re.fullmatch(r"(\d{4})-?W(\d{2})(?:-?([1-7]))?", date)
    if week:  # 2026-W09-7
        try:
            date = _dt.date.fromisocalendar(int(week.group(1)), int(week.group(2)), int(week.group(3) or 1)).isoformat()
        except ValueError:
            return None
        s = date + sep + clock
    elif sep and re.fullmatch(r"\d{8}", date):  # basic format: 20260301T120000Z
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    if sep:
        basic = re.match(r"(\d{2})(\d{2})(\d{2})?(?=[.,+-]|$)", clock)
        if basic:
            clock = ":".join(g for g in basic.groups() if g) + clock[basic.end():]
        clock = re.sub(r"^([\d:]+),(\d)", r"\1.\2", clock)  # 12:00:00,123 (a decimal comma)
        clock = re.sub(r"([+-]\d{2})$", r"\1:00", clock)  # +00, -08 (hours only)
        s = f"{date}T{clock}"
    if "." in s:
        head, _, tail = s.partition(".")
        frac = ""
        rest = ""
        for i, ch in enumerate(tail):
            if not ch.isdigit():
                rest = tail[i:]
                break
            frac += ch
        s = head + ("." + frac[:6].ljust(6, "0") if frac else "") + rest
    try:
        dt = _dt.datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = _dt.datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            try:  # Sun, 01 Mar 2026 12:00:00 GMT (HTTP and e-mail dates)
                dt = email.utils.parsedate_to_datetime(str(value).strip())
            except (TypeError, ValueError, IndexError, OverflowError):
                return None
            if dt is None:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def _first(d: dict, keys: Iterable[str]) -> Any:
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return None


def _params_from_chat(req: dict) -> dict:
    params = {k: req[k] for k in msgs.PARAM_KEYS if req.get(k) is not None}
    if req.get("functions") and not params.get("tools"):
        params["tools"] = [{"type": "function", "function": f} for f in req["functions"] if isinstance(f, dict)]
        fc = req.get("function_call")
        if isinstance(fc, dict) and fc.get("name"):
            params["tool_choice"] = {"type": "function", "function": {"name": fc["name"]}}
        elif fc in ("auto", "none"):
            params["tool_choice"] = fc
    return params


# ------------------------------------------------------- request converters


def chat_request(req: dict) -> tuple[list[dict], dict]:
    return msgs.clean_messages(req.get("messages") or []), _params_from_chat(req)


def _anthropic_content_to_canonical(role: str, content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    texts, parts, tool_calls, out = [], [], [], []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            texts.append(block.get("text", ""))
        elif btype == "image":
            src = block.get("source") or {}
            if src.get("type") == "base64":
                parts.append({"type": "image_url", "image_url": {"url": f"data:{src.get('media_type')};base64,{src.get('data')}"}})
            elif src.get("url"):
                parts.append({"type": "image_url", "image_url": {"url": src["url"]}})
        elif btype == "tool_use":
            tool_calls.append({"id": block.get("id"), "name": block.get("name"), "arguments": block.get("input") or {}})
        elif btype == "tool_result":
            res = block.get("content")
            out.append({"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": msgs.content_text(res) if not isinstance(res, str) else res})
    msg: dict = {"role": role}
    if parts:
        msg["content"] = [{"type": "text", "text": "\n".join(texts)}] + parts if texts else parts
    else:
        msg["content"] = "\n".join(texts)
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if msg["content"] or tool_calls:
        out.append(msg)
    return out


def anthropic_request(req: dict) -> tuple[list[dict], dict]:
    messages: list[dict] = []
    system = req.get("system")
    if system:
        messages.append({"role": "system", "content": system if isinstance(system, str) else msgs.content_text(system)})
    for m in req.get("messages") or []:
        messages.extend(_anthropic_content_to_canonical(m.get("role", "user"), m.get("content")))
    params: dict = {}
    for k in ("temperature", "top_p", "top_k", "max_tokens"):
        if req.get(k) is not None:
            params[k] = req[k]
    if req.get("stop_sequences"):
        params["stop"] = req["stop_sequences"]
    output_config = req.get("output_config")
    fmt = output_config.get("format") if isinstance(output_config, dict) else None
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema" and isinstance(fmt.get("schema"), dict):
        params["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": fmt["schema"]}}
    # Extended thinking and its effort: a Claude app that thinks is tested thinking.
    thinking = req.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive", "disabled", "between_tools"):  # between_tools: Sonnet 5.5's off
        params["thinking"] = {k: thinking[k] for k in ("type", "budget_tokens", "display") if thinking.get(k) is not None}
    effort = output_config.get("effort") if isinstance(output_config, dict) else None
    if isinstance(effort, str) and effort:
        params["reasoning_effort"] = effort
    tools = []
    for t in req.get("tools") or []:
        if isinstance(t, dict) and t.get("name") and "input_schema" in t:
            fn = {"name": t["name"], "parameters": t.get("input_schema") or {}}
            if t.get("description"):
                fn["description"] = t["description"]
            tools.append({"type": "function", "function": fn})
    if tools:
        params["tools"] = tools
        tc = req.get("tool_choice") or {}
        if isinstance(tc, dict):
            mapped = {"auto": "auto", "any": "required", "none": "none"}.get(tc.get("type", ""))
            if tc.get("type") == "tool" and tc.get("name"):
                params["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
            elif mapped:
                params["tool_choice"] = mapped
    return msgs.clean_messages(messages), params


def responses_request(req: dict) -> tuple[list[dict], dict]:
    messages: list[dict] = []
    if req.get("instructions"):
        messages.append({"role": "system", "content": msgs.content_text(req["instructions"])})
    inp = req.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    for item in inp if isinstance(inp, list) else []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "message")
        if itype == "message" or "role" in item:
            content = item.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "input_image":
                        parts.append({"type": "image_url", "image_url": {"url": part.get("image_url") or ""}})
                    elif "text" in part:
                        parts.append({"type": "text", "text": msgs.content_text([part])})
                content = parts if any(p["type"] != "text" for p in parts) else "\n".join(p["text"] for p in parts)
            messages.append({"role": item.get("role", "user"), "content": content or ""})
        elif itype == "function_call":
            call = {"id": item.get("call_id"), "name": item.get("name"), "arguments": item.get("arguments")}
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif itype == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": msgs.content_text(item.get("output"))})
    params: dict = {}
    for k in ("temperature", "top_p", "parallel_tool_calls"):
        if req.get(k) is not None:
            params[k] = req[k]
    if req.get("max_output_tokens"):
        params["max_completion_tokens"] = req["max_output_tokens"]
    text = req.get("text") or {}
    fmt = text.get("format") if isinstance(text, dict) else None
    if isinstance(fmt, dict):
        if fmt.get("type") == "json_schema":
            params["response_format"] = {"type": "json_schema", "json_schema": {k: fmt[k] for k in ("name", "schema", "strict", "description") if k in fmt}}
        elif fmt.get("type") == "json_object":
            params["response_format"] = {"type": "json_object"}
    if isinstance(text, dict) and text.get("verbosity"):
        params["verbosity"] = text["verbosity"]
    reasoning = req.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        params["reasoning_effort"] = reasoning["effort"]
    tools = []
    for t in req.get("tools") or []:
        if isinstance(t, dict) and t.get("type") == "function" and t.get("name"):
            fn = {"name": t["name"], "parameters": t.get("parameters") or {}}
            if t.get("description"):
                fn["description"] = t["description"]
            tools.append({"type": "function", "function": fn})
    if tools:
        params["tools"] = tools
    return msgs.clean_messages(messages), params


def gemini_request(req: dict) -> tuple[list[dict], dict]:
    messages: list[dict] = []
    si = req.get("systemInstruction") or req.get("system_instruction")
    if isinstance(si, dict):
        text = "\n".join(p.get("text", "") for p in si.get("parts") or [] if isinstance(p, dict))
        if text:
            messages.append({"role": "system", "content": text})
    elif isinstance(si, str):
        messages.append({"role": "system", "content": si})
    for c in req.get("contents") or []:
        if not isinstance(c, dict):
            continue
        role = "assistant" if c.get("role") == "model" else "user"
        texts, calls = [], []
        for p in c.get("parts") or []:
            if not isinstance(p, dict):
                continue
            if "text" in p and not p.get("thought"):
                texts.append(p["text"])
            fc = p.get("functionCall") or p.get("function_call")
            if fc:
                calls.append({"name": fc.get("name"), "arguments": fc.get("args") or {}})
            fr = p.get("functionResponse") or p.get("function_response")
            if fr:
                messages.append({"role": "tool", "tool_call_id": fr.get("id") or fr.get("name"), "content": json.dumps(fr.get("response"), ensure_ascii=False)})
        if texts or calls:
            msg: dict = {"role": role, "content": "\n".join(texts)}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
    params: dict = {}
    gen = req.get("generationConfig") or req.get("generation_config") or {}
    mapping = {"temperature": "temperature", "topP": "top_p", "topK": "top_k", "maxOutputTokens": "max_tokens", "stopSequences": "stop", "seed": "seed"}
    for src, dst in mapping.items():
        if gen.get(src) is not None:
            params[dst] = gen[src]
    if gen.get("responseMimeType") == "application/json":
        params["response_format"] = {"type": "json_object"}
    # Thinking, kept as the other providers' settings are: a budget (-1 dynamic, 0 off) as
    # thinking, a level (Gemini 3) as reasoning_effort.
    tc = gen.get("thinkingConfig") or gen.get("thinking_config")
    if isinstance(tc, dict):
        budget = tc.get("thinkingBudget", tc.get("thinking_budget"))
        if isinstance(budget, int) and not isinstance(budget, bool):
            params["thinking"] = {"type": "adaptive"} if budget < 0 else {"type": "disabled"} if budget == 0 else {"type": "enabled", "budget_tokens": budget}
        level = tc.get("thinkingLevel", tc.get("thinking_level"))
        if isinstance(level, str) and level:
            params["reasoning_effort"] = level.lower()
    return msgs.clean_messages(messages), params


# ------------------------------------------------------ response converters


def chat_response(resp: dict) -> dict:
    from .providers.openai import OpenAIProvider

    r = OpenAIProvider.parse_chat(resp)
    return {"output": r.text, "tool_calls": r.tool_calls, "model": r.model, "finish_reason": r.finish_reason, "usage": r.usage, "refusal": r.refusal}


def anthropic_response(resp: dict) -> dict:
    from .providers.anthropic import AnthropicProvider

    r = AnthropicProvider.parse(resp)
    return {"output": r.text, "tool_calls": r.tool_calls, "model": r.model, "finish_reason": r.finish_reason, "usage": r.usage, "refusal": r.refusal}


def responses_response(resp: dict) -> dict:
    from .providers.openai import OpenAIProvider

    r = OpenAIProvider.parse_responses(resp)
    return {"output": r.text, "tool_calls": r.tool_calls, "model": r.model, "finish_reason": r.finish_reason, "usage": r.usage, "refusal": r.refusal}


def gemini_response(resp: dict) -> dict:
    from .providers.gemini import GeminiProvider

    r = GeminiProvider.parse(resp)
    return {"output": r.text, "tool_calls": r.tool_calls, "model": r.model, "finish_reason": r.finish_reason, "usage": r.usage, "refusal": r.refusal}


_NON_CHAT_MODEL_RE = re.compile(r"embed|moderation|whisper|^tts-|dall-e|gpt-image|rerank|transcribe", re.I)
_NON_CHAT_URL_RE = re.compile(r"/(?:embeddings|moderations|images|audio)(?:/|$)")


def non_chat_reason(req: Any, resp: Any = None, url: Any = None) -> str | None:
    """Why a logged call is not a chat/completion call (embeddings, moderation, images,
    audio), or cannot be replayed from the log alone."""
    if isinstance(url, str) and _NON_CHAT_URL_RE.search(url):
        return "embedding/moderation/image/audio call (not a chat model call)"
    stored = isinstance(req, dict) and (req.get("previous_response_id") or req.get("conversation") or isinstance(req.get("prompt"), dict))
    if stored and "messages" not in req and "contents" not in req:  # a chat request holding its own turns is not
        # Responses API: earlier turns or the prompt itself live on OpenAI's servers, and
        # replaying what the log holds would test a different request.
        return "depends on context stored by OpenAI (previous_response_id, conversation or a stored prompt)"
    if isinstance(resp, dict):
        data = resp.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict) and ("embedding" in data[0] or "b64_json" in data[0] or ("url" in data[0] and resp.get("object") != "list")):
            return "embedding/moderation/image/audio call (not a chat model call)"
        results = resp.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict) and "flagged" in results[0]:
            return "embedding/moderation/image/audio call (not a chat model call)"
    if isinstance(req, dict) and "messages" not in req and "contents" not in req:
        model = req.get("model")
        if (isinstance(model, str) and _NON_CHAT_MODEL_RE.search(model)) or "encoding_format" in req or "dimensions" in req:
            return "embedding/moderation/image/audio call (not a chat model call)"
    return None


def _request_kind(req: dict, resp: Any = None) -> str | None:
    if not isinstance(req, dict):
        return None
    if "contents" in req:
        return "gemini" if isinstance(req.get("contents"), list) else None
    if "messages" in req:
        if not isinstance(req.get("messages"), list):
            return None
        if isinstance(resp, dict) and resp.get("type") == "message" and isinstance(resp.get("content"), list):
            return "anthropic"
        if "system" in req and not any(m.get("role") == "system" for m in req.get("messages") or [] if isinstance(m, dict)):
            return "anthropic"
        for m in req.get("messages") or []:
            if isinstance(m, dict) and isinstance(m.get("content"), list):
                if any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in m["content"]):
                    return "anthropic"
        return "chat"
    if "input" in req or "instructions" in req:
        return "responses"
    return None


def _response_kind(resp: Any) -> str | None:
    if not isinstance(resp, dict):
        return None
    if "choices" in resp:
        return "chat"
    if resp.get("type") == "message" and isinstance(resp.get("content"), list):
        return "anthropic"
    if resp.get("object") == "response" or isinstance(resp.get("output"), list):
        return "responses"
    if "candidates" in resp or "promptFeedback" in resp:
        return "gemini"
    return None


_REQ_PARSERS = {"chat": chat_request, "anthropic": anthropic_request, "responses": responses_request, "gemini": gemini_request}
_RESP_PARSERS = {"chat": chat_response, "anthropic": anthropic_response, "responses": responses_response, "gemini": gemini_response}


def sse_response(text: str) -> dict | None:
    """Rebuild a complete response from a logged server-sent-events stream."""
    if not isinstance(text, str) or "data:" not in text[:4000]:
        return None
    chunks = []
    for line in re.split(r"\r\n|\r|\n", text):  # SSE lines end there only, not at U+2028 inside the JSON
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunks.append(json_loads(payload))
        except (ValueError, RecursionError):
            return None
    chunks = [c for c in chunks if isinstance(c, dict)]
    if not chunks:
        return None
    from .record import _StreamAccumulator

    acc = _StreamAccumulator()
    for chunk in chunks:
        acc.add(chunk)
    types = {str(c.get("type") or "") for c in chunks}
    if any(t.startswith(("message_", "content_block_")) for t in types):
        api = "messages"
    elif any(t.startswith("response.") for t in types):
        api = "responses"
    elif any("candidates" in c for c in chunks):
        api = "gemini"
    else:
        api = "chat"
    return acc.response(api)


def _unwrap_body(obj: Any) -> Any:
    if isinstance(obj, dict) and isinstance(obj.get("body"), dict) and len(obj) <= 6:
        return obj["body"]
    if isinstance(obj, dict) and isinstance(obj.get("body"), str) and len(obj) <= 6:
        streamed = sse_response(obj["body"])
        if streamed is not None:
            return streamed
        try:
            parsed = json_loads(obj["body"])
        except (ValueError, RecursionError):
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    if isinstance(obj, str):
        try:
            parsed = json_loads(obj)
            return parsed if isinstance(parsed, dict) else obj
        except (ValueError, RecursionError):
            streamed = sse_response(obj)
            return streamed if streamed is not None else obj
    return obj


def from_pair(req: dict, resp: Any, rec: dict, where: str) -> Interaction | None:
    kind = _request_kind(req, resp)
    if kind is None:
        return None
    messages, params = _REQ_PARSERS[kind](req)
    inter = Interaction(messages=messages, params=params, model=req.get("model"), source=where, fmt=kind)
    run = rec.get("agent_run")
    if isinstance(run, dict) and run.get("id") is not None:
        inter.agent_run = {"id": str(run["id"]), "step": run.get("step")}
    elif rec.get("trace_id") or rec.get("traceId"):  # tracing exports (Langfuse, LangSmith, OpenTelemetry)
        inter.agent_run = {"id": str(rec.get("trace_id") or rec.get("traceId"))}
    rk = _response_kind(resp)
    if rk:
        out = _RESP_PARSERS[rk](resp)
        inter.output = out["output"]
        inter.tool_calls = out["tool_calls"]
        # Keep the model the app asked for (what a switch replaces); note the exact one served.
        served = out["model"]
        if served and inter.model and str(served).lower() != str(inter.model).lower():
            inter.metadata["served_model"] = served
        inter.model = inter.model or served
        inter.finish_reason = out["finish_reason"]
        inter.usage = out["usage"] or {}
        if out.get("refusal") and not inter.output:
            inter.output = ""
            inter.metadata["refusal"] = out["refusal"]
    elif isinstance(resp, str):
        inter.output = resp
    elif isinstance(resp, dict) and resp.get("error"):
        inter.error = json.dumps(resp.get("error"))[:300]
    status = rec.get("status_code") or rec.get("status")
    if isinstance(status, int) and status >= 400:
        inter.error = inter.error or f"HTTP {status}"
    inter.timestamp = _ts(_first(rec, ("timestamp", "ts", "created_at", "createdAt", "time", "start_time", "startTime", "date"))) or (
        _ts(_first(resp, ("created", "created_at"))) if isinstance(resp, dict) else None
    )
    lat = _first(rec, ("latency_ms", "latency", "duration_ms"))
    if isinstance(lat, (int, float)):
        inter.latency_ms = float(lat)
    if isinstance(rec.get("error"), (str, dict)) and rec.get("error"):
        inter.error = inter.error or (rec["error"] if isinstance(rec["error"], str) else json.dumps(rec["error"])[:300])
    return inter


# ---------------------------------------------------------- other formats

_LC_ROLES = {"system": "system", "human": "user", "user": "user", "ai": "assistant", "assistant": "assistant", "tool": "tool", "function": "tool", "chat": "user"}


def _langchain_messages(value: Any) -> list[dict] | None:
    """LangChain serialised messages -> canonical messages."""
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        return None
    out = []
    for m in value:
        if not isinstance(m, dict):
            return None
        if "role" in m:
            out.append({"role": _LC_ROLES.get(m["role"], m["role"]), "content": m.get("content", "")})
            continue
        kwargs = m.get("kwargs") or m.get("data") or {}
        mtype = str(m.get("type") or "")
        if mtype.lower() not in _LC_ROLES and isinstance(m.get("id"), list) and m["id"]:
            # Serialised LangChain object: {"lc": 1, "type": "constructor", "id": [..., "HumanMessage"]}
            mtype = str(m["id"][-1]).replace("MessageChunk", "").replace("Message", "").lower()
        role = _LC_ROLES.get(str(mtype).lower())
        if role is None or not isinstance(kwargs, dict):
            return None
        out.append({"role": role, "content": kwargs.get("content", "")})
    return out


def _messages_from_any(value: Any) -> list[dict] | None:
    if isinstance(value, str):
        return [{"role": "user", "content": value}] if value.strip() else None
    if isinstance(value, dict):
        if "messages" in value:
            return _messages_from_any(value["messages"])
        for key in ("input", "prompt", "question", "query", "text", "content"):
            if isinstance(value.get(key), (str, list)):
                return _messages_from_any(value[key])
        return None
    if isinstance(value, list):
        if all(isinstance(m, dict) and "role" in m for m in value):
            try:
                return msgs.clean_messages(value)
            except ValueError:
                return None
        return _langchain_messages(value)
    return None


def _output_from_any(value: Any) -> tuple[str | None, list[dict]]:
    if value is None:
        return None, []
    if isinstance(value, str):
        return value, []
    if isinstance(value, dict):
        rk = _response_kind(value)
        if rk:
            out = _RESP_PARSERS[rk](value)
            return out["output"], out["tool_calls"]
        if "generations" in value:
            gens = value["generations"]
            while isinstance(gens, list) and gens and isinstance(gens[0], list):
                gens = gens[0]
            if isinstance(gens, list) and gens and isinstance(gens[0], dict):
                g = gens[0]
                text = g.get("text")
                if not text and isinstance(g.get("message"), dict):
                    text = (g["message"].get("kwargs") or {}).get("content") or g["message"].get("content")
                return (text if isinstance(text, str) else None), []
        calls = [c for c in (msgs.normalize_tool_call(tc) for tc in value.get("tool_calls") or []) if c]
        for key in ("content", "text", "output", "completion", "answer", "response", "result"):
            if isinstance(value.get(key), str):
                return value[key], calls
        if calls:
            return "", calls
        if isinstance(value.get("kwargs"), dict) and isinstance(value["kwargs"].get("content"), str):
            return value["kwargs"]["content"], []
        if value and not set(value) <= {"error", "status", "metadata", "usage"}:
            return json.dumps(value, ensure_ascii=False, default=str), []  # a structured (JSON) answer
        return None, []
    if isinstance(value, list) and value and isinstance(value[-1], dict) and value[-1].get("role") == "assistant":
        return msgs.content_text(value[-1].get("content")), []
    if isinstance(value, list) and value:
        return json.dumps(value, ensure_ascii=False, default=str), []
    if isinstance(value, (int, float, bool)):
        return json.dumps(value), []
    return None, []


_PROMPT_KEYS = ("prompt", "input", "question", "query", "instruction", "user_input", "user_message", "user", "request_text", "text")
_OUTPUT_KEYS = ("output", "completion", "response", "answer", "reply", "assistant", "generation", "generated_text", "result", "model_output", "response_text")
_SYSTEM_KEYS = ("system", "system_prompt", "systemPrompt", "instructions")


def from_generic(rec: dict, where: str) -> Interaction | None:
    inp_key = next((k for k in _PROMPT_KEYS if k in rec and rec[k] not in (None, "")), None)
    out_key = next((k for k in _OUTPUT_KEYS if k in rec and rec[k] not in (None, "")), None)
    if inp_key is None:
        return None
    messages = _messages_from_any(rec[inp_key])
    if not messages:
        return None
    system = _first(rec, _SYSTEM_KEYS)
    if isinstance(system, str) and system.strip() and not msgs.system_prompt(messages):
        messages.insert(0, {"role": "system", "content": system})
    output, calls = _output_from_any(rec.get(out_key)) if out_key else (None, [])
    params = {}
    mp = rec.get("modelParameters") or rec.get("model_parameters") or rec.get("params") or rec.get("parameters")
    if isinstance(mp, dict):
        for k in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "stop"):
            if mp.get(k) not in (None, ""):
                params[k] = mp[k]
    model = _first(rec, ("model", "model_name", "modelName", "deployment"))
    extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
    inv = extra.get("invocation_params") if isinstance(extra.get("invocation_params"), dict) else {}
    model = model or inv.get("model") or inv.get("model_name")
    try:
        messages = msgs.clean_messages(messages)
    except ValueError:
        return None
    return Interaction(
        messages=messages,
        params=params,
        output=output,
        tool_calls=calls,
        model=str(model) if model else None,
        timestamp=_ts(_first(rec, ("timestamp", "ts", "created_at", "createdAt", "time", "startTime", "start_time", "date"))),
        source=where,
        fmt="generic",
        error=str(rec["error"])[:300] if rec.get("error") else None,
    )


def from_chat_dataset(rec: dict, where: str) -> Interaction | None:
    raw = rec.get("messages")
    if raw is None and isinstance(rec.get("conversations"), list):  # ShareGPT
        role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system", "model": "assistant"}
        raw = [{"role": role_map.get(str(t.get("from")).lower(), "user"), "content": t.get("value", "")} for t in rec["conversations"] if isinstance(t, dict)]
    if not isinstance(raw, list) or not raw:
        return None
    try:
        messages = msgs.clean_messages(raw)
    except ValueError:
        return None
    if messages[-1]["role"] != "assistant":
        return Interaction(messages=messages, params=_params_from_chat(rec), model=rec.get("model"), source=where, fmt="chat_dataset")
    last = messages[-1]
    return Interaction(
        messages=messages[:-1],
        params=_params_from_chat(rec),
        output=msgs.content_text(last.get("content")),
        tool_calls=last.get("tool_calls") or [],
        model=rec.get("model"),
        source=where,
        fmt="chat_dataset",
        timestamp=_ts(_first(rec, ("timestamp", "created_at", "time"))),
    )


def from_testcase(rec: dict, where: str) -> Interaction | None:
    from .testset import TestCase

    case = TestCase.from_dict(rec, where)
    return Interaction(
        messages=case.messages,
        params=case.params,
        output=case.baseline_output or None,
        tool_calls=case.baseline_tool_calls,
        model=case.baseline_model,
        source=where,
        fmt="llmshift",
        checks=case.checks,
        vars=case.vars,
        tags=case.tags,
        metadata=case.metadata,
        case_id=case.id,
    )


# ------------------------------------------------------------- dispatcher


def parse_record(rec: Any, where: str, batch: dict, stats: LogStats) -> Interaction | None:
    if not isinstance(rec, dict):
        stats.skip("not a JSON object", where)
        return None
    # LLMShift test cases / records
    if isinstance(rec.get("input"), dict) and "messages" in rec["input"] and ("checks" in rec or "baseline" in rec):
        return from_testcase(rec, where)
    if rec.get("llmshift_shadow"):  # shadow mode: the request as sent and the production model's answer
        req, base = rec.get("request"), rec.get("baseline")
        if not isinstance(req, dict) or not isinstance(base, dict) or not isinstance(req.get("messages"), list):
            stats.skip("shadow record without its request and answer", where)
            return None
        return Interaction(
            messages=[m for m in req["messages"] if isinstance(m, dict)],
            params=dict(req["params"]) if isinstance(req.get("params"), dict) else {},
            output=base["output"] if isinstance(base.get("output"), str) else None,
            tool_calls=[c for c in base.get("tool_calls") or [] if isinstance(c, dict)] if isinstance(base.get("tool_calls"), list) else [],
            model=base["model"] if isinstance(base.get("model"), str) else None,
            usage=dict(base["usage"]) if isinstance(base.get("usage"), dict) else {},
            timestamp=_ts(rec.get("ts")),
            source=where,
            fmt="llmshift_shadow",
        )
    if rec.get("llmshift_record"):
        reason = non_chat_reason(rec.get("request"), rec.get("response"))
        if reason:
            stats.skip(reason, where)
            return None
        return from_pair(rec.get("request") or {}, rec.get("response"), rec, where)
    # OpenAI Batch API lines (joined later; custom_id is only unique within one batch)
    if "custom_id" in rec and ("body" in rec or "response" in rec):
        cid = str(rec["custom_id"])
        key = batch_file_key(where)
        if isinstance(rec.get("body"), dict):
            batch.setdefault("inputs", {}).setdefault(cid, []).append({"key": key, "request": rec["body"], "url": rec.get("url", ""), "where": where})
        if "response" in rec or "error" in rec:
            resp = rec.get("response") or {}
            batch.setdefault("outputs", {}).setdefault(cid, []).append(
                {
                    "key": key,
                    "response": resp.get("body") if isinstance(resp, dict) else None,
                    "status": resp.get("status_code") if isinstance(resp, dict) else None,
                    "error": rec.get("error"),
                    "where": where,
                }
            )
        return None
    # request/response pairs
    for rk, sk in (("request", "response"), ("req", "res"), ("request_body", "response_body"), ("requestBody", "responseBody"), ("input", "output")):
        if rk in rec and isinstance(_unwrap_body(rec[rk]), dict):
            req = _unwrap_body(rec[rk])
            resp = _unwrap_body(rec.get(sk))
            reason = non_chat_reason(req, resp, rec.get("url") or rec.get("path") or rec.get("endpoint"))
            if reason:
                stats.skip(reason, where)
                return None
            if _request_kind(req) is None:
                continue
            inter = from_pair(req, resp, rec, where)
            if inter is not None:
                return inter
    # flat OpenAI logs: request fields + choices at top level
    if "messages" in rec and ("choices" in rec or (rec.get("type") == "message" and isinstance(rec.get("content"), list))):
        return from_pair(rec, rec, rec, where)
    if isinstance(rec.get("ideal"), (str, list)) and "input" in rec:
        stats.skip("openai/evals sample (use `llmshift import openai-evals`)", where)
        return None
    if "messages" in rec or "conversations" in rec:
        inter = from_chat_dataset(rec, where)
        if inter is not None:
            return inter
    if "inputs" in rec and "outputs" in rec:
        merged = {"input": rec.get("inputs"), "output": rec.get("outputs"), **{k: v for k, v in rec.items() if k not in ("inputs", "outputs")}}
        inter = from_generic(merged, where)
        if inter is not None:
            inter.fmt = "langsmith"
            return inter
    inter = from_generic(rec, where)
    if inter is not None:
        return inter
    stats.skip("unrecognised record", f"{where}: keys {sorted(rec)[:8]}")
    return None


_BATCH_WORDS_RE = re.compile(r"(?:^|[_\-.])(?:batch_?)?(?:inputs?|outputs?|requests?|results?|responses?|errors?|in|out)(?=$|[_\-.])")


def batch_file_key(where: str) -> str:
    """Pairs a batch's input and output files: "jobs/run7_input.jsonl:3" -> "jobs/run7"."""
    path = re.sub(r"(?::\d+|\[\d+\])$", "", where)
    stem = os.path.splitext(os.path.basename(path[:-3] if path.lower().endswith(".gz") else path))[0].lower()
    return os.path.dirname(path).lower() + "/" + _BATCH_WORDS_RE.sub("", stem)


def _finish_batch(batch: dict, stats: LogStats) -> list[Interaction]:
    out = []
    inputs, outputs = batch.get("inputs", {}), batch.get("outputs", {})
    for cid in dict.fromkeys(list(inputs) + list(outputs)):
        ins, outs = inputs.get(cid, []), outputs.get(cid, [])
        if not ins:
            for o in outs:
                stats.skip("batch output without its input line", o["where"])
            continue
        claimed: set[int] = set()
        for inp in ins:
            same = [i for i, o in enumerate(outs) if o["key"] == inp["key"] and i not in claimed]
            if len(same) == 1:
                match: int | None = same[0]
            elif not same and not outs:
                match = None  # not answered (yet)
            elif not same and len(ins) == 1 and len(outs) == 1:
                match = 0  # file names do not pair up, but the id is unique
            else:
                stats.skip("ambiguous batch custom_id (the same id in several batches)", f"{inp['where']}: {cid}")
                continue
            slot = outs[match] if match is not None else {}
            if match is not None:
                claimed.add(match)
            try:  # one malformed batch line must not stop the others
                reason = non_chat_reason(inp["request"], slot.get("response"), inp.get("url"))
                if reason:
                    stats.skip(reason, inp["where"])
                    continue
                inter = from_pair(inp["request"], slot.get("response"), {}, inp["where"])
            except Exception as exc:
                stats.skip("could not parse record", f"{inp['where']}: {exc.__class__.__name__}: {exc}")
                continue
            if inter is None:
                stats.skip("unrecognised batch request", inp["where"])
                continue
            status = slot.get("status")
            if slot.get("error") or (isinstance(status, int) and status >= 400):
                inter.error = json.dumps(slot.get("error"))[:300] if slot.get("error") else f"HTTP {status}"
            inter.fmt = "openai_batch"
            inter.metadata["custom_id"] = cid
            out.append(inter)
    return out


class _Sample:
    """A bounded sample of interactions: up to `limit` are kept. Past that, the places are
    shared fairly between prompt templates, however late one first appears: a template
    with fewer kept calls takes a place from the one with the most, so small templates
    keep every call and large ones share the rest equally. Within a template each call
    has the same chance to stay (reservoir sampling)."""

    def __init__(self, limit: int | None, seed: int = 7):
        self.limit = limit
        self.rng = random.Random(seed)  # nosec B311  # seeded on purpose: the same sample on every run; nothing secret
        self.groups: dict[str, list] = {}
        self.seen: Counter = Counter()
        self.heap: list = []  # (-kept, template), largest first; entries go stale as groups change
        self.size = 0
        self.order = 0

    @property
    def dropped(self) -> int:
        return sum(self.seen.values()) - self.size

    def _push(self, key: str) -> None:
        heapq.heappush(self.heap, (-len(self.groups[key]), key))
        if len(self.heap) > 4 * len(self.groups) + 1024:  # drop the stale entries
            self.heap = [(-len(g), k) for k, g in self.groups.items() if g]
            heapq.heapify(self.heap)

    def _evict(self, taker: str, have: int) -> bool:
        """Free a place for `taker` (holding `have`) from the largest template, when that
        one holds at least two more."""
        while self.heap:
            neg, largest = self.heap[0]
            group = self.groups[largest]
            if -neg != len(group):
                heapq.heappop(self.heap)  # stale entry
                continue
            if largest == taker or len(group) <= have + 1:
                return False
            heapq.heappop(self.heap)
            group.pop(self.rng.randrange(len(group)))
            self._push(largest)
            return True
        return False

    def add(self, inter: Interaction) -> None:
        self.order += 1
        key = msgs.template_key(inter.messages, inter.params)
        self.seen[key] += 1
        kept = self.groups.setdefault(key, [])
        item = (self.order, inter)
        if self.limit is None or self.size < self.limit:
            kept.append(item)
            self.size += 1
            self._push(key)
            return
        if self._evict(key, len(kept)):
            kept.append(item)
            self._push(key)
            return
        if kept:
            j = self.rng.randrange(self.seen[key])
            if j < len(kept):
                kept[j] = item

    def items(self) -> list[Interaction]:
        return [inter for _, inter in sorted((it for g in self.groups.values() for it in g), key=lambda it: it[0])]


def parse_logs(
    sources: Iterable[str | os.PathLike],
    *,
    admit: Callable[[Interaction], bool] | None = None,
    max_records: int | None = DEFAULT_MAX_RECORDS,
) -> tuple[list[Interaction], LogStats]:
    """Every LLM call in the logs. `admit` filters calls as they are read, and at most
    `max_records` are kept in memory (a fair sample per prompt template beyond that)."""
    stats = LogStats()
    sample = _Sample(max_records)
    batch: dict = {}
    for path in iter_files(sources):
        stats.files += 1
        try:
            records = iter_records(path, stats)
            for rec, where in records:
                stats.records += 1
                try:
                    inter = parse_record(rec, where, batch, stats)
                except Exception as exc:  # a malformed record must not stop the import
                    stats.skip("could not parse record", f"{where}: {exc.__class__.__name__}: {exc}")
                    continue
                if inter is not None:
                    if not inter.messages:
                        stats.skip("no messages", where)
                        continue
                    try:
                        keep = admit is None or admit(inter)
                        if keep:
                            sample.add(inter)
                    except Exception as exc:  # odd fields that only grouping touches: skip this one call
                        stats.skip("could not parse record", f"{where}: {exc.__class__.__name__}: {exc}")
                        continue
                    stats.parsed += 1
                    stats.formats[inter.fmt] += 1
        except (OSError, EOFError, ValueError, csv.Error, zlib.error) as exc:  # unreadable, corrupt or truncated .gz, bad CSV
            stats.skip("unreadable file", f"{path.as_posix()}: {exc.__class__.__name__}: {exc}")
    for inter in _finish_batch(batch, stats):
        stats.parsed += 1
        stats.formats[inter.fmt] += 1
        if admit is None or admit(inter):
            sample.add(inter)
    stats.sampled_out = sample.dropped
    return sample.items(), stats
