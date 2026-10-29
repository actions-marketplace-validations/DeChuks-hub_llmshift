"""`llmshift mcp`: a Model Context Protocol server, so AI coding assistants (Claude Code,
Cursor, VS Code Copilot) can use llmshift as tools - "move this repo off gpt-4o and
prove nothing broke".

    claude mcp add llmshift -- llmshift mcp

JSON-RPC 2.0 over stdio, one message per line. Only protocol messages reach stdout:
console output, `print` and child processes are sent to stderr. Tools run one at a
time; `ping` is answered at once and `notifications/cancelled` stops the running tool.
More tools can be added with the `tool` decorator:

    @tool("llmshift_pick", "Find the cheapest model that still passes ...", {"type": "object", ...})
    def _pick(args: dict, cfg: dict) -> dict:
        ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import queue
import re
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, Callable

from . import __version__
from . import console as con
from .errors import EXIT_INTERRUPTED, EXIT_OK, UsageError
from .textutil import json_loads, plural, report_json

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")  # newest first
MAX_MESSAGE_BYTES = 32 * 1024 * 1024
EOF_GRACE = 5.0  # seconds a running tool may go on once the client has closed its end
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603

INSTRUCTIONS = (
    "llmshift checks AI model migrations. Typical flow: llmshift_scan finds the model IDs the code calls and their "
    "retirement dates; llmshift_snapshot turns usage logs into a test set of real requests; llmshift_compare runs that "
    "test set on the new model and reports exactly what changed (llmshift_pick tries several cheaper models at once, "
    "llmshift_safety checks refusals, prompt injection and PII leaks); llmshift_fix suggests system-prompt changes that "
    "make the new model match; llmshift_upgrade swaps the model IDs in the code. After editing model IDs, run "
    "llmshift_compare again to prove nothing broke; after editing prompts, llmshift_check tests just those. "
    "Model specs look like provider:model, e.g. openai:gpt-5.6-sol or anthropic:claude-sonnet-5-5."
)


class ToolError(Exception):
    """The tool could not do its job; the client receives the message as an error result."""


@dataclass
class Tool:
    name: str
    title: str
    description: str
    schema: dict
    handler: Callable[[dict, Any], Any]
    read_only: bool = True
    destructive: bool = False
    open_world: bool = False  # calls model providers over the network

    def describe(self, protocol: str) -> dict:
        out: dict = {"name": self.name, "description": self.description, "inputSchema": self.schema}
        if protocol >= "2025-06-18":
            out["title"] = self.title
        if protocol >= "2025-03-26":
            out["annotations"] = {
                "title": self.title,
                "readOnlyHint": self.read_only,
                "destructiveHint": self.destructive,
                "idempotentHint": self.read_only,
                "openWorldHint": self.open_world,
            }
        return out


TOOLS: dict[str, Tool] = {}


def _as_arguments(message: str, spec: "Tool") -> str:
    """Messages written for the command line name flags (pass --judge provider:model); a tool
    caller sets arguments, so a flag the tool takes is named as its argument."""
    names = set((spec.schema.get("properties") or {}).keys())

    def name(m: re.Match) -> str:
        arg = m.group(1).replace("-", "_")
        return f"the {arg} argument" if arg in names else m.group(0)

    return re.sub(r"(?<![\w-])--([a-z][a-z0-9-]*)", name, message)


def tool(name: str, description: str, schema: dict, *, title: str | None = None, read_only: bool = True, destructive: bool = False, open_world: bool = False) -> Callable:
    """Register a tool. The handler receives (arguments, config) and returns a JSON-able
    dict; raising ToolError (or any llmshift error) becomes an error result."""

    def decorate(fn: Callable[[dict, Any], Any]) -> Callable[[dict, Any], Any]:
        TOOLS[name] = Tool(name, title or name, description, schema, fn, read_only, destructive, open_world)
        return fn

    return decorate


class _RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def _key(value: Any) -> tuple:
    return (type(value).__name__, value)


def _error(rid: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid if _valid_id(rid) else None, "error": {"code": code, "message": message}}


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def encode(message: Any) -> bytes:
    """One protocol line. ASCII-only JSON, so no encoding or newline can break framing."""
    try:
        text = json.dumps(message, ensure_ascii=True, allow_nan=False, separators=(",", ":"), default=str)
    except (ValueError, TypeError):
        text = json.dumps(_clean(message), ensure_ascii=True, allow_nan=False, separators=(",", ":"), default=str)
    return text.encode("ascii") + b"\n"


def _is_request(message: Any, method: str) -> bool:
    return isinstance(message, dict) and message.get("method") == method and _valid_id(message.get("id"))


class Server:
    def __init__(self, root: str | os.PathLike | None = None, config_path: str | None = None):
        self.root = Path(root or os.getcwd()).resolve()
        self.config_path = config_path
        self.protocol = PROTOCOL_VERSIONS[0]
        self.initialized = False
        self.client: dict = {}
        self._write_lock = threading.Lock()
        self._state = threading.Lock()
        self._running: tuple | None = None
        self._busy = False  # a request is being handled
        self._queued: set = set()  # ids read but not started yet
        self._cancelled: set = set()
        self._closed = False  # the input ended

    # ------------------------------------------------------------- dispatch

    def handle(self, message: Any) -> Any:
        """The response to one parsed message (or batch), or None for notifications."""
        if isinstance(message, list):
            if not message:
                return _error(None, INVALID_REQUEST, "empty batch")
            replies = [self.handle(m) if not isinstance(m, list) else _error(None, INVALID_REQUEST, "nested batch") for m in message]
            replies = [r for r in replies if r is not None]
            return replies or None
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            rid = message.get("id") if isinstance(message, dict) else None
            return _error(rid, INVALID_REQUEST, 'expected a JSON-RPC 2.0 object with "jsonrpc": "2.0"')
        has_id = "id" in message
        rid = message.get("id")
        method = message.get("method")
        if not isinstance(method, str):
            if has_id and ("result" in message or "error" in message):
                return None  # a response; this server sends no requests, so there is nothing to match
            return _error(rid, INVALID_REQUEST, "a request needs a string method")
        if not has_id:
            self._notification(method, message.get("params"))
            return None
        if not _valid_id(rid):
            return _error(None, INVALID_REQUEST, "id must be a string or an integer")
        params = message.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _error(rid, INVALID_PARAMS, "params must be an object")
        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [t.describe(self.protocol) for t in TOOLS.values()]}
            elif method == "tools/call":
                result = self._call(params)
            else:
                return _error(rid, METHOD_NOT_FOUND, f"method not found: {method}")
        except _RpcError as exc:
            return _error(rid, exc.code, exc.message)
        except Exception as exc:  # pragma: no cover - a bug in the dispatcher itself
            traceback.print_exc(file=sys.stderr)
            return _error(rid, INTERNAL_ERROR, f"internal error: {exc.__class__.__name__}: {exc}")
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def _notification(self, method: str, params: Any) -> None:
        if method == "notifications/initialized":
            self.initialized = True
        elif method == "notifications/cancelled":
            self.cancel(params)

    def _initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")
        self.protocol = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        self.client = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
        info: dict = {"name": "llmshift", "version": __version__}
        if self.protocol >= "2025-06-18":
            info["title"] = "LLMShift"
        return {"protocolVersion": self.protocol, "capabilities": {"tools": {"listChanged": False}}, "serverInfo": info, "instructions": INSTRUCTIONS}

    def _call(self, params: dict) -> dict:
        from .jsonschema_lite import validate

        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise _RpcError(INVALID_PARAMS, "tools/call needs the tool name in params.name")
        spec = TOOLS.get(name)
        if spec is None:
            raise _RpcError(INVALID_PARAMS, f"unknown tool: {name} (available: {', '.join(sorted(TOOLS))})")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise _RpcError(INVALID_PARAMS, f"arguments for {name} must be an object")
        problems = validate(arguments, spec.schema)
        if problems:
            raise _RpcError(INVALID_PARAMS, f"invalid arguments for {name}: " + "; ".join(problems[:5]))
        return self.call_tool(spec, arguments)

    def call_tool(self, spec: Tool, arguments: dict) -> dict:
        from .config import ConfigError, load_config
        from .providers.base import ProviderError
        from .runner import RunAborted
        from .testset import TestFileError

        try:
            with self._workspace(arguments):
                cfg = load_config(self.config_path, stop=self.root)  # never a config from above the project
                data = spec.handler(arguments, cfg)
        except RunAborted as exc:
            return self._result(f"every request failed: {exc}. Check the model name and that the provider's API key "
                                "environment variable is set (llmshift_doctor shows which keys are visible).", error=True)
        except (ToolError, UsageError, ConfigError, TestFileError, ProviderError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
            return self._result(_as_arguments(str(exc) or exc.__class__.__name__, spec), error=True)
        except (Exception, SystemExit) as exc:
            traceback.print_exc(file=sys.stderr)
            return self._result(f"unexpected {exc.__class__.__name__}: {exc}", error=True)
        return self._result(data)

    def _check_paths(self, arguments: dict, base: Path) -> None:
        """Every path a tool is given stays inside the project the server was started on:
        an assistant (or text it was shown) must not point llmshift at ~/.ssh or ../other."""
        root = self.root.resolve()
        for key in PATH_ARGS:
            value = arguments.get(key)
            for item in value if isinstance(value, list) else [value] if isinstance(value, str) else []:
                fixed = re.split(r"[*?\[]", str(item), maxsplit=1)[0] or "."  # the part of a glob before its wildcards
                path = Path(fixed) if Path(fixed).is_absolute() else base / fixed
                try:
                    path.resolve().relative_to(root)
                except ValueError:
                    raise ToolError(f"{key} {item!r} is outside the project ({root}); the tools only read and write inside it") from None

    @contextlib.contextmanager
    def _workspace(self, arguments: dict) -> Any:
        """Run in the project directory with console output on stderr and no colour."""
        target = self.root
        self._check_paths({"cwd": arguments["cwd"]} if arguments.get("cwd") else {}, self.root)
        if arguments.get("cwd"):
            target = (self.root / str(arguments["cwd"])).resolve()
            if not target.is_dir():
                raise ToolError(f"cwd {arguments['cwd']!r} is not a directory (relative paths start at {self.root})")
        self._check_paths({k: v for k, v in arguments.items() if k != "cwd"}, target)
        previous = os.getcwd()
        os.chdir(target)
        try:
            with contextlib.redirect_stdout(sys.stderr), con.no_color():
                yield target
        finally:
            os.chdir(previous)

    def _result(self, data: Any, error: bool = False) -> dict:
        text = data if isinstance(data, str) else json.dumps(_clean(data), ensure_ascii=False, default=str)
        result: dict = {"content": [{"type": "text", "text": text}], "isError": error}
        if not error and isinstance(data, dict) and self.protocol >= "2025-06-18":
            result["structuredContent"] = json.loads(text)  # exactly what the text says
        return result

    # ------------------------------------------------------------ transport

    def cancel(self, params: Any) -> None:
        rid = params.get("requestId") if isinstance(params, dict) else None
        if not _valid_id(rid):
            return
        with self._state:
            running = self._running == _key(rid)
            if not running and _key(rid) not in self._queued:
                return  # already answered (a late notice): a later request may reuse the id
            self._cancelled.add(_key(rid))
        if running:
            from .providers import http
            from .providers.custom import kill_running

            http.CANCEL.set()  # in-flight model calls stop retrying
            kill_running()

    def send(self, writer: Any, message: Any) -> bool:
        if message is None:
            return True
        data = encode(message)
        with self._write_lock:
            try:
                writer.write(data)
                writer.flush()
            except (OSError, ValueError):
                return False  # the client went away
        return True

    def _messages(self, reader: Any, writer: Any) -> Any:
        """Parsed messages from the input stream; framing errors are answered here."""
        while True:
            line = reader.readline(MAX_MESSAGE_BYTES + 1)
            if not line:
                return
            if isinstance(line, str):
                line = line.encode("utf-8", "replace")
            if len(line) > MAX_MESSAGE_BYTES and not line.endswith(b"\n"):
                while True:  # skip the rest of the oversized line
                    rest = reader.readline(1 << 20)
                    if not rest or rest.endswith(b"\n" if isinstance(rest, bytes) else "\n"):
                        break
                self.send(writer, _error(None, INVALID_REQUEST, f"message larger than {MAX_MESSAGE_BYTES // (1024 * 1024)} MB"))
                continue
            text = line.decode("utf-8", errors="replace").strip().lstrip("\ufeff")
            if not text:
                continue
            try:
                yield json_loads(text)
            except (ValueError, RecursionError) as exc:
                self.send(writer, _error(None, PARSE_ERROR, f"parse error: {exc}"))

    def serve(self, reader: BinaryIO, writer: BinaryIO) -> None:
        """Answer messages until the input ends. A reader thread keeps `ping` and
        cancellation responsive while a tool runs on this thread."""
        from .providers import http

        jobs: queue.Queue = queue.Queue()
        end = object()
        done = threading.Event()  # serve returned: the grace period has nothing left to stop
        closing: list[threading.Thread] = []

        def read_loop() -> None:
            try:
                for message in self._messages(reader, writer):
                    if _is_request(message, "ping"):
                        self.send(writer, self.handle(message))
                    elif isinstance(message, dict) and message.get("method") == "notifications/cancelled" and "id" not in message:
                        self.cancel(message.get("params"))
                    else:
                        if isinstance(message, dict) and _valid_id(message.get("id")):
                            with self._state:
                                self._queued.add(_key(message["id"]))
                        jobs.put(message)
            except Exception:  # pragma: no cover - unreadable input
                traceback.print_exc(file=sys.stderr)
            finally:
                # Started before `end` is queued (serve returns as soon as it sees it) and never
                # once serve has returned: a thread started while Python shuts down can crash it
                # (exit -11 on macOS).
                if not done.is_set():
                    grace = threading.Thread(target=wind_down, name="llmshift-mcp-eof", daemon=True)
                    grace.start()
                    closing.append(grace)
                jobs.put(end)

        def wind_down() -> None:
            """The input ended, which is how a client shuts the server down. Requests already
            read are answered, but a tool still running after EOF_GRACE seconds has its model
            calls and child processes stopped, and the requests behind it are dropped."""
            if done.wait(EOF_GRACE):
                return  # every request was answered
            with self._state:
                busy = self._busy
                if busy:
                    self._closed = True
            if busy:
                from .providers.custom import kill_running

                http.CANCEL.set()
                kill_running()

        threading.Thread(target=read_loop, name="llmshift-mcp-reader", daemon=True).start()
        try:
            while True:
                try:
                    message = jobs.get(timeout=0.5)  # a timed wait keeps Ctrl+C working on Windows
                except queue.Empty:
                    continue
                if message is end or self._closed:
                    http.CANCEL.clear()
                    return
                key = _key(message["id"]) if isinstance(message, dict) and _valid_id(message.get("id")) else None
                with self._state:
                    self._queued.discard(key)
                    if key is not None and key in self._cancelled:
                        self._cancelled.discard(key)
                        continue  # cancelled before it started
                    http.CANCEL.clear()  # a cancellation applies to one request only
                    self._running = key
                    self._busy = True
                try:
                    response = self.handle(message)
                finally:
                    with self._state:
                        self._running = None
                        self._busy = False
                        cancelled = key is not None and key in self._cancelled
                        self._cancelled.discard(key)
                        if cancelled:
                            http.CANCEL.clear()
                if not cancelled and not self.send(writer, response):
                    return
        finally:
            done.set()  # no thread is left starting or running when Python shuts down
            for grace in list(closing):
                grace.join()


def _protocol_streams() -> tuple[Any, Any]:
    """Private handles on the real stdin and stdout. fd 0 is then pointed at the null
    device and fd 1 at stderr (on Windows the standard handles too, which is what child
    processes inherit), so stray output cannot corrupt the protocol stream and no child
    process can read it, or block on it: on Windows a child that inherits a pipe the
    reader thread is waiting on hangs at startup."""
    reader = getattr(sys.stdin, "buffer", sys.stdin)
    writer = getattr(sys.stdout, "buffer", sys.stdout)
    with contextlib.suppress(AttributeError, OSError, ValueError):
        sys.stdout.flush()
        fd = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        writer = os.fdopen(fd, "wb")
    with contextlib.suppress(AttributeError, OSError, ValueError):
        fd = os.dup(sys.stdin.fileno())
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, sys.stdin.fileno())
        os.close(null)
        reader = os.fdopen(fd, "rb")
    if os.name == "nt":
        with contextlib.suppress(Exception):  # child processes still get stdin=DEVNULL wherever llmshift starts them
            import ctypes
            import msvcrt

            set_std_handle = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p)(("SetStdHandle", ctypes.windll.kernel32))
            set_std_handle(-10 & 0xFFFFFFFF, msvcrt.get_osfhandle(sys.stdin.fileno()))  # STD_INPUT_HANDLE: the null device
            set_std_handle(-11 & 0xFFFFFFFF, msvcrt.get_osfhandle(sys.stderr.fileno()))  # STD_OUTPUT_HANDLE: stderr
    return reader, writer


def run_stdio(root: str | None = None, config_path: str | None = None) -> int:
    con.setup_streams()
    con.set_color(False)
    server = Server(root=root, config_path=config_path)
    reader, writer = _protocol_streams()
    print(f"llmshift {__version__} MCP server on stdio (project: {server.root})", file=sys.stderr, flush=True)
    try:
        server.serve(reader, writer)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    return EXIT_OK


# ------------------------------------------------------------------ tools

CWD = {"type": "string", "description": "Project directory to work in, absolute or relative to where the server started (default: that directory)."}
PATH_ARGS = ("cwd", "paths", "sources", "output", "tests", "source", "write_tests", "files")  # confined to the project
TESTS = {"type": "string", "description": "Test file or directory (default: tests in llmshift.yaml, usually .llmshift/tests.jsonl)."}
STRINGS = {"type": "array", "items": {"type": "string", "minLength": 1}}


def _schema(properties: dict, required: tuple = ()) -> dict:
    props = {**properties, "cwd": CWD}
    schema: dict = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        schema["required"] = list(required)
    return schema


def _clip(text: Any, limit: int) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _registry(cfg: Any) -> Any:
    from .registry import Registry

    scan_cfg = cfg.get("scan") or {}
    configured = scan_cfg.get("registry") or []
    return Registry.load(list(configured) if isinstance(configured, list) else [configured])


def _load_cases(cfg: Any, args: dict) -> tuple[list, str]:
    from . import cli

    ns = SimpleNamespace(tests=args.get("tests"), group=args.get("groups"), tag=args.get("tags"), case=args.get("case_ids"), limit=args.get("limit"), drop_param=args.get("drop_params"))
    return cli._load_cases(cfg, ns)


def _model(args: dict, cfg: Any) -> str:
    spec = args.get("to") or cfg.get("candidate")
    if not spec:
        raise ToolError("which model? Pass `to`, e.g. openai:gpt-5.6-sol or anthropic:claude-sonnet-5-5 (or set candidate in llmshift.yaml)")
    return str(spec)


def _target(fix_cfg: dict) -> float:
    """fix.target_pass_rate from llmshift.yaml; null means the default, 100%."""
    value = fix_cfg.get("target_pass_rate")
    return float(value) if value is not None else 1.0


def _cache(cfg: Any, args: dict) -> Any:
    from . import cli

    return cli._cache(cfg, SimpleNamespace(no_cache=bool(args.get("no_cache"))))


@tool(
    "llmshift_scan",
    "Find every AI model ID the code in this project calls (Python, JS/TS, Go, Java, configs, .env, notebooks) and report "
    "each one's lifecycle: retired, retiring (date and days left), deprecated or active, the recommended replacement, and every "
    "file:line where it is used. Use it first when asked to migrate off a model, to find all the places to change, and again "
    "afterwards to confirm that no retired model remains.",
    _schema(
        {
            "paths": {**STRINGS, "description": "Files, directories or globs to scan (default: scan.paths in llmshift.yaml, else the project)."},
            "exclude": {**STRINGS, "description": "Glob patterns to skip, e.g. tests/fixtures or **/*.min.js."},
            "fail_on": {"type": "string", "enum": ["retired", "retiring", "deprecated", "unknown", "never"], "description": "Which states make would_fail_ci true (default retired)."},
            "warn_days": {"type": "integer", "minimum": 0, "maximum": 3650, "description": "Models retiring within this many days are urgent (default 180)."},
            "include_docs": {"type": "boolean", "description": "Also scan Markdown and text files."},
        }
    ),
    title="Find model calls and retirement dates",
)
def _scan_tool(args: dict, cfg: Any) -> dict:
    from .scanner import scan_paths, should_fail

    sc = cfg.get("scan") or {}
    warn_days = int(args["warn_days"]) if args.get("warn_days") is not None else int(sc.get("warn_days", 180))
    result = scan_paths(
        list(args.get("paths") or sc.get("paths") or ["."]),
        registry=_registry(cfg),
        include_docs=bool(args.get("include_docs", sc.get("include_docs", False))),
        exclude=list(sc.get("exclude") or []) + list(args.get("exclude") or []),
        warn_days=warn_days,
    )
    fail_on = args.get("fail_on") or sc.get("fail_on") or "retired"
    models = []
    for m in result.models():
        lk = m["lookup"]
        entry: dict = {"model": m["model"]}
        if lk is not None:
            info = lk.to_dict()
            info.pop("query", None)
            registry_id = info.pop("model", None)
            if registry_id and registry_id != m["model"]:
                entry["registry_id"] = registry_id
            entry.update(info)
        locations = [f.location for f in m["findings"]]
        entry["locations"] = locations[:25]
        if len(locations) > 25:
            entry["more_locations"] = len(locations) - 25
        models.append(entry)
    urgent = [m["model"] for m in models if m.get("status") == "retired" or (m.get("status") == "retiring" and (m.get("days_left") or 0) <= warn_days)]
    out = {
        "project": os.getcwd(),
        "files_scanned": result.files_scanned,
        "registry_updated": result.registry_updated,
        "warn_days": warn_days,
        "summary": result.counts(),
        "fail_on": fail_on,
        "would_fail_ci": should_fail(result, fail_on),
        "models": models,
        "errors": result.errors[:50],
    }
    if urgent:
        out["next"] = (f"Replace {', '.join(urgent[:5])} (see replacement), then run llmshift_compare with to=<replacement> "
                       "on the project's test set (llmshift_snapshot creates one from usage logs).")
    return out


@tool(
    "llmshift_models",
    "Look up retirement dates and replacements in llmshift's model lifecycle registry (OpenAI, Anthropic, Google, Mistral, "
    "Cohere, xAI, DeepSeek). Pass `model` for one exact ID (Bedrock, Vertex, Azure and OpenRouter forms are understood) or "
    "`query` to search, optionally filtered by provider, status or days until retirement.",
    _schema(
        {
            "model": {"type": "string", "minLength": 1, "description": "One model ID to look up exactly, e.g. gpt-4o-2024-05-13."},
            "query": {"type": "string", "description": "Substring of model IDs to search for, e.g. gpt-4o or claude-3."},
            "provider": {"type": "string", "description": "openai, anthropic, google, google-vertex, mistral, cohere, xai or deepseek."},
            "status": {"type": "string", "enum": ["active", "retiring", "deprecated", "retired"]},
            "retiring_within_days": {"type": "integer", "minimum": 0, "description": "Only models retiring within this many days."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Maximum models returned (default 50)."},
        }
    ),
    title="Look up model retirement dates",
)
def _models_tool(args: dict, cfg: Any) -> dict:
    from .registry import today

    reg = _registry(cfg)
    if args.get("model"):
        return {"registry_updated": reg.updated, "today": today().isoformat(), "lookup": reg.lookup(str(args["model"])).to_dict()}
    found = reg.search(args.get("query") or "", provider=args.get("provider"), status=args.get("status"))
    within = args.get("retiring_within_days")
    if within is not None:
        found = [lk for lk in found if lk.status == "retiring" and lk.days_left is not None and lk.days_left <= within]
    limit = int(args.get("limit") or 50)
    models = []
    for lk in found[:limit]:
        info = lk.to_dict()
        info.pop("query", None)
        models.append(info)
    return {"registry_updated": reg.updated, "today": today().isoformat(), "matches": len(found), "models": models, "not_shown": max(0, len(found) - limit)}


@tool(
    "llmshift_snapshot",
    "Turn LLM usage logs (JSONL, JSON or CSV, optionally .gz: OpenAI chat/responses/batch, Anthropic, Gemini, Helicone, "
    "Langfuse/LangSmith exports, llmshift.record logs) into a regression test set: real requests, the current model's "
    "answers, and checks inferred from them. Run it once before switching models; llmshift_compare and llmshift_fix use it.",
    _schema(
        {
            "sources": {**STRINGS, "minItems": 1, "description": "Log files, directories or globs."},
            "output": {"type": "string", "description": "Test file to write (default: tests in llmshift.yaml)."},
            "mode": {"type": "string", "enum": ["create", "append", "overwrite"], "description": "create (default) refuses to replace an existing file; append adds new cases only."},
            "models": {**STRINGS, "description": "Only calls to these models (globs, e.g. gpt-4o*)."},
            "since": {"type": "string", "description": "Only calls on/after this date (YYYY-MM-DD or ISO-8601)."},
            "until": {"type": "string", "description": "Only calls on/before this date."},
            "max_cases": {"type": "integer", "minimum": 1, "maximum": 100000},
            "per_group": {"type": "integer", "minimum": 1, "maximum": 100000, "description": "Maximum cases per prompt template."},
            "redact_pii": {"type": "boolean", "description": "Also replace e-mails, phone and card numbers with placeholders."},
            "baseline_model": {"type": "string", "minLength": 1, "description": "Generate answers with this model for logs that have none (makes model calls)."},
        },
        required=("sources",),
    ),
    title="Turn usage logs into a test set",
    read_only=False,
    destructive=True,
)
def _snapshot_tool(args: dict, cfg: Any) -> dict:
    from . import cli
    from .providers import resolve
    from .snapshot import build_snapshot
    from .testset import append_tests, load_tests, save_tests

    sc = cfg.get("snapshot") or {}
    out_path = Path(args.get("output") or cfg.get("tests") or ".llmshift/tests.jsonl")
    if out_path.is_dir():
        out_path = out_path / "tests.jsonl"
    mode = args.get("mode") or "create"
    if mode == "create" and out_path.exists():
        raise ToolError(f"{cli._rel(out_path)} already exists; pass mode 'append' to add new cases or 'overwrite' to replace it")
    result = build_snapshot(
        list(args["sources"]),
        models=args.get("models"),
        since=args.get("since"),
        until=args.get("until"),
        max_cases=int(args.get("max_cases") or sc.get("max_cases", 200)),
        per_group=int(args.get("per_group") or sc.get("per_group", 25)),
        max_records=int(sc.get("max_records", 100_000)),
        seed=int(sc.get("seed", 7)),
        redact_secrets=bool(sc.get("redact_secrets", True)),
        redact_pii=bool(args.get("redact_pii", sc.get("redact_pii", False))),
        baseline_provider=resolve(args["baseline_model"], cfg) if args.get("baseline_model") else None,
        concurrency=int(cfg.get("concurrency", 4)),
        cache=_cache(cfg, args),
    )
    st = result.stats
    skipped = {**dict(st.skipped), **dict(result.filtered)}
    if not result.cases:
        reasons = "; ".join(f"{n} {reason}" for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1])[:5])
        raise ToolError(f"no test cases were created from {st.records} record(s) in {st.files} file(s)" + (f" (skipped: {reasons})" if reasons else ""))
    if mode == "append" and out_path.exists():
        existing = load_tests(out_path)
        known = {c.id for c in existing}
        new = [c for c in result.cases if c.id not in known]
        append_tests(new, out_path, existing)
        written, total = len(new), len(existing) + len(new)
    else:
        save_tests(result.cases, out_path)
        written = total = len(result.cases)
    return {
        "tests_file": cli._rel(out_path),
        "cases_written": written,
        "total_cases": total,
        "files_read": st.files,
        "records_read": st.records,
        "llm_calls_recognised": st.parsed,
        "formats": dict(st.formats),
        "skipped": skipped,
        "unique_requests": result.unique,
        "duplicates_removed": result.duplicates,
        "prompt_templates": result.groups,
        "checks_inferred": dict(result.check_counts),
        "redactions": dict(result.redactions),
        "baselines_generated": result.generated,
        "next": f"llmshift_compare with to=<new model> (tests={cli._rel(out_path)})",
    }


_STATUS_ORDER = {"error": 0, "fail": 1, "warn": 2}


def _case_summary(c: Any) -> dict:
    out = {"id": c.id, "group": c.group, "status": c.status, "input": _clip(c.input_preview, 300)}
    problems = [
        {"check": r.type, "severity": r.severity, "message": _clip(r.message, 300)}
        for r in c.checks
        if r.status in ("fail", "error") and r.severity in ("error", "warn")
    ]
    if problems:
        out["problems"] = problems[:6]
    if c.error:
        out["error"] = _clip(c.error, 400)
    else:
        out["baseline_output"] = _clip(c.baseline_output, 800)
        out["candidate_output"] = _clip(c.candidate_output, 800)
        if c.candidate_tool_calls:
            out["candidate_tool_calls"] = c.candidate_tool_calls[:5]
    return out


@tool(
    "llmshift_compare",
    "Run the project's test set (made by llmshift_snapshot) on a model and report exactly what changed versus the current "
    "model's recorded answers: broken formats (invalid JSON, code fences), wrong or missing facts and numbers, tone and "
    "length changes, tool-call differences and refusals. Returns the pass rate, whether it meets the threshold, the changed "
    "cases with their failing checks, and HTML/JSON report files. Use it after changing a model ID or a prompt to prove "
    "nothing broke. It calls the model once per test case (cached on re-runs).",
    _schema(
        {
            "to": {"type": "string", "minLength": 1, "description": "The model to test, e.g. openai:gpt-5.6-sol (default: candidate in llmshift.yaml)."},
            "tests": TESTS,
            "baseline_model": {"type": "string", "minLength": 1, "description": "Also run the old model live; failures it shares are treated as noise."},
            "judge": {"type": "string", "minLength": 1, "description": "Model for LLM-judged checks (default: judge in llmshift.yaml)."},
            "min_pass_rate": {"type": "number", "minimum": 0, "maximum": 1, "description": "Threshold for passed_threshold (default: min_pass_rate in llmshift.yaml)."},
            "groups": {**STRINGS, "description": "Only these prompt templates (substring match)."},
            "tags": {**STRINGS, "description": "Only cases with these tags."},
            "case_ids": {**STRINGS, "description": "Only these case IDs."},
            "limit": {"type": "integer", "minimum": 1, "description": "Only the first N cases."},
            "max_cases_shown": {"type": "integer", "minimum": 0, "maximum": 200, "description": "Changed cases included in the result (default 10)."},
            "repeat": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Run each case N times to find flaky cases and narrow the pass-rate range (default 1)."},
            "no_cache": {"type": "boolean", "description": "Call the model even for requests answered before."},
            "drop_params": {**STRINGS, "description": "Leave these settings out of every test request (e.g. temperature, once the code no longer sends a setting the model rejects)."},
        }
    ),
    title="Compare a model on the test set",
    read_only=False,
    open_world=True,
)
def _compare_tool(args: dict, cfg: Any) -> dict:
    from . import cli
    from .compare import run_compare

    cases, path = _load_cases(cfg, args)
    candidate = _model(args, cfg)
    min_rate = args.get("min_pass_rate", cfg.get("min_pass_rate"))
    result = run_compare(
        cases,
        candidate,
        baseline=args.get("baseline_model"),
        judge=args.get("judge") or cfg.get("judge"),
        embedder=cfg.get("embedder"),
        config=cfg,
        concurrency=int(cfg.get("concurrency", 4)),
        cache=_cache(cfg, args),
        allow_python=bool(cfg.get("allow_python_checks")),
        severity=dict(cfg.get("severity") or {}),
        tests_path=path,
        min_pass_rate=float(min_rate) if min_rate is not None else None,
        show_progress=False,
        repeat=int(args.get("repeat") or cfg.get("repeat") or 1),
        judge_mode=cfg.get("judge_mode") or "auto",
    )
    reports = cli._write_reports(result, SimpleNamespace(html=None, json=None, markdown=None, junit=None, no_report=False, report_dir=None), cfg, "compare")
    s = result.summary
    changed = sorted((c for c in result.cases if c.status != "pass"), key=lambda c: (_STATUS_ORDER.get(c.status, 3), c.group, c.id))
    shown = changed[: int(args.get("max_cases_shown", 10))]
    unchecked = s.get("unchecked", 0)
    if s.get("evaluated", 0) == 0:
        if not unchecked:
            verdict = "every request failed; see the cases for the errors"
        elif not s.get("errors"):
            verdict = "no case could be checked: a check could not run; see the cases"
        else:
            verdict = f"no case could be checked: {plural(s['errors'], 'request')} failed and the other cases' checks could not run; see the cases"
        ok = False
    elif result.min_pass_rate is not None:
        ok = result.passed and not unchecked
        verdict = f"pass rate {s['pass_rate']:.0%} {'meets' if result.passed else 'is below'} the {result.min_pass_rate:.0%} threshold"
    else:
        ok = not (s.get("failed") or s.get("errors") or unchecked)
        verdict = f"{s.get('passed', 0)} unchanged, {s.get('warned', 0)} changed, {s.get('failed', 0)} failed, {plural(s.get('errors', 0), 'error')}"
    if unchecked and s.get("evaluated", 0):
        verdict += f"; {unchecked} case{'s' if unchecked != 1 else ''} could not be checked (a check could not run)"
        verdict += ", so the run is inconclusive" if result.inconclusive else ""
    out = {
        "candidate": result.candidate,
        "baseline": result.baseline,
        "tests": cli._rel(path),
        "ok": ok,
        "verdict": verdict,
        "min_pass_rate": result.min_pass_rate,
        "summary": {
            k: s.get(k)
            for k in ("total", "evaluated", "passed", "warned", "failed", "errors", "unchecked", "pass_rate", "pass_rate_range", "cases_needed",
                      "repeats", "flaky", "flaky_cases", "unchanged_rate", "latency_ms", "tokens", "agent", "vs_baseline")
            if s.get(k) is not None
        },
        "what_changed": {
            cat: {"failed": v.get("fail", 0), "changed": v.get("warn", 0), "patterns": v.get("patterns", {})}
            for cat, v in (s.get("by_category") or {}).items()
        },
        "insights": result.insights,
        "adaptations": result.adaptations,
        "cases": [_case_summary(c) for c in shown],
        "cases_not_shown": len(changed) - len(shown),
        "reports": [cli._rel(p) for p in reports],
    }
    refused, other = cli.rejected_and_other(result)
    if refused:
        out["next"] = (
            f"{result.candidate} rejects {', '.join(refused)} as the requests send them, so those calls fail: remove them from the code "
            "(llmshift_upgrade lists where), then run llmshift_compare again with drop_params"
        )
    elif other:
        out["next"] = f"llmshift_fix with to={result.candidate} suggests prompt changes for the changed cases"
    return out


@tool(
    "llmshift_fix",
    "Search for system-prompt changes that make a new model reproduce the current model's behaviour on the test set, "
    "validated by re-running the tests on the new model. Returns, per prompt template, the instructions to add or the full "
    "suggested prompt, where that prompt lives in the code (file:line), and the pass rate before and after. It never edits "
    "code: apply the prompts you agree with, then call llmshift_compare again.",
    _schema(
        {
            "to": {"type": "string", "minLength": 1, "description": "The new model, e.g. openai:gpt-5.6-sol (default: candidate in llmshift.yaml)."},
            "tests": TESTS,
            "optimizer": {"type": "string", "minLength": 1, "description": "Optional model that rewrites prompts, e.g. openai:gpt-5.6-sol (rule-based fixes work without it)."},
            "rounds": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Improvement rounds per prompt template (default 3)."},
            "judge": {"type": "string", "minLength": 1, "description": "Model for LLM-judged checks."},
            "groups": {**STRINGS, "description": "Only these prompt templates (substring match)."},
            "limit": {"type": "integer", "minimum": 1, "description": "Only the first N cases."},
            "source": {**STRINGS, "description": "Where to look for the prompts in the code (default: the project)."},
            "write_tests": {"type": "string", "description": "Also write a copy of the test set with the suggested prompts to this new file."},
            "no_cache": {"type": "boolean"},
        }
    ),
    title="Suggest prompt fixes for a new model",
    read_only=False,
    open_world=True,
)
def _fix_tool(args: dict, cfg: Any) -> dict:
    from . import cli
    from .fix import fix_markdown, patched_cases, run_fix
    from .testset import save_tests

    cases, path = _load_cases(cfg, args)
    candidate = _model(args, cfg)
    write_tests = args.get("write_tests")
    if write_tests and Path(write_tests).exists():
        raise ToolError(f"{write_tests} already exists; choose a new file for write_tests")
    fx = cfg.get("fix") or {}
    result = run_fix(
        cases,
        candidate,
        config=cfg,
        judge=args.get("judge") or cfg.get("judge"),
        optimizer=args.get("optimizer") or fx.get("optimizer"),
        concurrency=int(cfg.get("concurrency", 4)),
        cache=_cache(cfg, args),
        allow_python=bool(cfg.get("allow_python_checks")),
        severity=dict(cfg.get("severity") or {}),
        rounds=int(args.get("rounds") or fx.get("rounds", 3)),
        max_cases_per_group=int(fx.get("max_cases_per_group", 12)),
        target=_target(fx),
        source_paths=list(args.get("source") or ["."]),
        show_progress=False,
    )
    reports_dir = Path(cfg.get("reports_dir") or ".llmshift/reports")
    md_path = reports_dir / f"fix-{cli._timestamp()}.md"
    cli._write(md_path, fix_markdown(result))
    json_path = md_path.with_suffix(".json")
    cli._write(json_path, report_json(result.to_dict()) + "\n")
    out: dict = {
        "candidate": result.candidate,
        "tests": cli._rel(path),
        "nothing_to_fix": not result.groups,
        "notes": result.notes,  # what no prompt can fix, such as settings the model rejects
        "before": result.before,
        "after": result.after,
        "groups": [
            {
                "group": g.group,
                "prompt_locations": g.locations,
                "improved": g.improved,
                "strategy": g.strategy,
                "added_instructions": g.added,
                "suggested_prompt": g.suggested_prompt,
                "diff": g.diff,
                "before": g.before,
                "after": g.after,
                "resolved": g.resolved,
                "still_different": g.remaining,
                "notes": g.notes,
            }
            for g in result.groups
        ],
        "reports": [cli._rel(md_path), cli._rel(json_path)],
    }
    if write_tests:
        save_tests(patched_cases(cases, result), write_tests)
        out["patched_tests"] = cli._rel(write_tests)
    out["next"] = ("Nothing to fix: every case already matches." if not result.groups else
                   "Apply the suggested prompts at prompt_locations (they are not applied automatically), then run llmshift_compare again.")
    return out


_KEY_ENV = (
    ("OPENAI_API_KEY", "openai"), ("ANTHROPIC_API_KEY", "anthropic"), ("GEMINI_API_KEY", "gemini"), ("GOOGLE_API_KEY", "gemini"),
    ("AZURE_OPENAI_API_KEY", "azure"), ("AZURE_OPENAI_ENDPOINT", "azure"), ("MISTRAL_API_KEY", "mistral"), ("OPENROUTER_API_KEY", "openrouter"),
    ("GROQ_API_KEY", "groq"), ("DEEPSEEK_API_KEY", "deepseek"), ("XAI_API_KEY", "xai"), ("TOGETHER_API_KEY", "together"),
    ("FIREWORKS_API_KEY", "fireworks"), ("PERPLEXITY_API_KEY", "perplexity"), ("COHERE_API_KEY", "cohere"),
)


@tool(
    "llmshift_pick",
    "Find the cheapest model that still passes the project's test set: runs the tests on several candidate models at once "
    "and reports each one's pass rate (with a 95% range), cost per 1,000 requests and latency, then recommends the cheapest "
    "that meets the quality bar, with the saving against today's model (per month when `volume` is given). Calls every "
    "candidate once per test case (cached on re-runs).",
    _schema(
        {
            "models": {**STRINGS, "minItems": 1, "description": "Candidate models, e.g. openai:gpt-6-luna, anthropic:claude-haiku-5-5."},
            "tests": TESTS,
            "baseline": {"type": "string", "minLength": 1, "description": "The model used today, for its cost (default: baseline in llmshift.yaml, else the recorded model)."},
            "min_pass_rate": {"type": "number", "minimum": 0, "maximum": 1, "description": "Quality bar (default: min_pass_rate in llmshift.yaml)."},
            "volume": {"type": "integer", "minimum": 1, "description": "Requests per month, to state the monthly saving."},
            "groups": {**STRINGS, "description": "Only these prompt templates (substring match)."},
            "limit": {"type": "integer", "minimum": 1, "description": "Only the first N cases."},
            "no_cache": {"type": "boolean", "description": "Call the models even for requests answered before."},
        },
        required=("models",),
    ),
    title="Find the cheapest model that still passes",
    read_only=False,
    open_world=True,
)
def _pick_tool(args: dict, cfg: Any) -> dict:
    from .pick import run_pick
    from .pricing import overrides_from_config

    cases, path = _load_cases(cfg, args)
    pick_cfg = cfg.get("pick") or {}
    min_rate = args.get("min_pass_rate", cfg.get("min_pass_rate"))
    try:
        prices = overrides_from_config(cfg)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    result = run_pick(
        cases,
        list(args["models"]),
        baseline=args.get("baseline") or cfg.get("baseline"),
        config=cfg,
        min_pass_rate=float(min_rate) if min_rate is not None else 0.95,
        volume=args.get("volume") or pick_cfg.get("volume"),
        prices=prices,
        concurrency=int(cfg.get("concurrency", 4)),
        cache=_cache(cfg, args),
        judge=cfg.get("judge"),
        embedder=cfg.get("embedder"),
        allow_python=bool(cfg.get("allow_python_checks")),
        severity=dict(cfg.get("severity") or {}),
        tests_path=path,
    )
    return result.to_dict()


@tool(
    "llmshift_safety",
    "Check whether a new model is riskier than the current one: refusals on the project's real requests, prompt-injection "
    "probes (hidden instructions in messages, quoted documents or tool results) and leaks of confidential data from the "
    "system prompt, all built from the project's own test cases and run on both models. Reports old vs new rates, the "
    "change, a paired significance test and examples. Calls both models.",
    _schema(
        {
            "to": {"type": "string", "minLength": 1, "description": "The new model (default: candidate in llmshift.yaml)."},
            "baseline_model": {"type": "string", "minLength": 1, "description": "The current model (default: baseline in llmshift.yaml)."},
            "tests": TESTS,
            "categories": {"type": "array", "items": {"type": "string", "enum": ["refusal", "injection", "pii"]}, "description": "Which checks to run (default all)."},
            "probes": {"type": "integer", "minimum": 1, "maximum": 500, "description": "Probes per category (default 20)."},
            "no_cache": {"type": "boolean", "description": "Call the models even for requests answered before."},
        }
    ),
    title="Is the new model riskier?",
    read_only=False,
    open_world=True,
)
def _safety_tool(args: dict, cfg: Any) -> dict:
    from .safety import run_safety

    cases, _path = _load_cases(cfg, args)
    new = _model(args, cfg)
    old = args.get("baseline_model") or cfg.get("baseline")
    if not old:
        raise ToolError("which model is in use today? Pass baseline_model (or set baseline in llmshift.yaml)")
    result = run_safety(
        cases, new, str(old), config=cfg, categories=tuple(args.get("categories") or ("refusal", "injection", "pii")),
        probes=int(args.get("probes") or 20), concurrency=int(cfg.get("concurrency", 4)), cache=_cache(cfg, args), show_progress=False,
    )
    return result.to_dict()


@tool(
    "llmshift_check",
    "Test edited prompts: finds the system prompts that changed since a git base (default origin/main) and runs only "
    "their test cases, with the edited text, on the model in production. Tells you whether a prompt edit changes how "
    "the model answers. Use it after editing a prompt in the code. Calls the model once per affected test case.",
    _schema(
        {
            "model": {"type": "string", "minLength": 1, "description": "Model to test with (default: baseline in llmshift.yaml, else candidate)."},
            "base": {"type": "string", "description": "Git ref to compare against (default: origin/main)."},
            "files": {**STRINGS, "description": "Only look at these changed files."},
            "all": {"type": "boolean", "description": "Run every prompt template, not only the edited ones."},
            "tests": TESTS,
            "min_pass_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "no_cache": {"type": "boolean"},
        }
    ),
    title="Test edited prompts",
    read_only=False,
    open_world=True,
)
def _check_tool(args: dict, cfg: Any) -> dict:
    from .prcheck import run_check

    cases, _path = _load_cases(cfg, args)
    model = args.get("model") or cfg.get("baseline") or cfg.get("candidate")
    if not model:
        raise ToolError("which model? Pass `model` (or set baseline in llmshift.yaml)")
    min_rate = args.get("min_pass_rate", cfg.get("min_pass_rate"))
    run = run_check(
        cases, str(model), base=args.get("base"), files=args.get("files"), all_groups=bool(args.get("all")), config=cfg,
        judge=cfg.get("judge"), concurrency=int(cfg.get("concurrency", 4)), cache=_cache(cfg, args),
        allow_python=bool(cfg.get("allow_python_checks")), severity=dict(cfg.get("severity") or {}),
        min_pass_rate=float(min_rate) if min_rate is not None else None, show_progress=False,
    )
    return run.to_dict()


@tool(
    "llmshift_upgrade",
    "Plan (and optionally apply) the code changes that move this project off a retiring model: swaps the model ID exactly "
    "where llmshift_scan finds it, renames max_tokens to max_completion_tokens where OpenAI's newer models need it, and "
    "lists settings to check by hand. Returns a unified diff. Nothing is written unless apply is true. Use `auto` for every "
    "retiring model that has a known replacement.",
    _schema(
        {
            "from": {"type": "string", "minLength": 1, "description": "The model to move off, e.g. gpt-4o-2024-05-13."},
            "to": {"type": "string", "minLength": 1, "description": "The replacement (default: the registry's recommended replacement)."},
            "auto": {"type": "boolean", "description": "Every retired or retiring model found that has a replacement."},
            "paths": {**STRINGS, "description": "Where to look (default: scan.paths in llmshift.yaml, else the project)."},
            "apply": {"type": "boolean", "description": "Write the changes to the files (default false: only the diff). A file git cannot restore is first copied to NAME.bak, listed in `backups`."},
            "force": {"type": "boolean", "description": "Allow swaps across providers (the calling code is never rewritten)."},
        }
    ),
    title="Plan a model upgrade",
    read_only=False,
    destructive=True,
)
def _upgrade_tool(args: dict, cfg: Any) -> dict:
    from .upgrade import apply_plan, plan_upgrade

    if not args.get("from") and not args.get("auto"):
        raise ToolError("pass `from` (the model to move off) or auto=true")
    sc = cfg.get("scan") or {}
    plan = plan_upgrade(
        list(args.get("paths") or sc.get("paths") or ["."]), old=args.get("from"), new=args.get("to"), auto=bool(args.get("auto")),
        registry=_registry(cfg), warn_days=int(sc.get("warn_days", 180)), exclude=list(sc.get("exclude") or []), force=bool(args.get("force")),
    )
    out = plan.to_dict()
    if args.get("apply"):
        out["written"] = apply_plan(plan)
        if plan.backups:  # files git could not give back
            out["backups"] = plan.backups
        out["next"] = "Run llmshift_compare with to=<the new model> to prove nothing broke."
    return out


@tool(
    "llmshift_doctor",
    "Check the llmshift setup: version, the config file in use, which provider credential environment variables are set "
    "(names only, never values), whether the test set exists, and whether the configured baseline, candidate and judge "
    "models resolve. Use it when another llmshift tool fails with a configuration or credentials error.",
    _schema({}),
    title="Check the llmshift setup",
)
def _doctor_tool(args: dict, cfg: Any) -> dict:
    from .providers import resolve
    from .testset import load_tests

    reg = _registry(cfg)
    tests_path = str(cfg.get("tests") or ".llmshift/tests.jsonl")
    tests: dict = {"path": tests_path, "exists": Path(tests_path).exists()}
    if tests["exists"]:
        try:
            tests["cases"] = len(load_tests(tests_path))
        except Exception as exc:
            tests["error"] = str(exc)
    names = [name for name, _ in _KEY_ENV]
    for spec in (cfg.get("providers") or {}).values():
        env = spec.get("api_key_env") if isinstance(spec, dict) else None
        if isinstance(env, str) and env and env not in names:
            names.append(env)
    models = {}
    for role in ("baseline", "candidate", "judge"):
        spec = cfg.get(role)
        if not spec:
            continue
        try:
            provider = resolve(str(spec), cfg)
            models[role] = {"spec": spec, "provider": provider.__class__.__name__, "endpoint": getattr(provider, "base_url", None) or provider.kind}
        except Exception as exc:
            models[role] = {"spec": spec, "error": str(exc)}
    path = getattr(cfg, "path", None)
    return {
        "llmshift": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "project": os.getcwd(),
        "config": str(path) if path else None,
        "registry": {"models": len(reg.entries), "updated": reg.updated},
        "tests": tests,
        "credentials_set": [n for n in names if os.environ.get(n)],
        "credentials_missing": [n for n in names if not os.environ.get(n)],
        "models": models,
    }


# -------------------------------------------------------------------- CLI


def register(sub: Any) -> None:
    p = sub.add_parser(
        "mcp",
        help="run an MCP server so AI coding assistants can use llmshift as tools",
        description="Serve llmshift's scan, models, snapshot, compare, fix, pick, safety, check, upgrade and doctor as Model "
        "Context Protocol tools over stdio, for Claude Code (claude mcp add llmshift -- llmshift mcp), Cursor, VS Code and other "
        "MCP clients.",
    )
    p.add_argument("--cwd", help="project directory the tools work in (default: the current directory)")
    p.set_defaults(func=cmd_mcp)


def cmd_mcp(args: argparse.Namespace, cfg: Any) -> int:
    root = None
    if getattr(args, "cwd", None):
        root = Path(args.cwd).resolve()
        if not root.is_dir():
            raise UsageError(f"--cwd {args.cwd} is not a directory")
    config_path = os.path.abspath(args.config) if getattr(args, "config", None) else None
    return run_stdio(root=str(root) if root else None, config_path=config_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llmshift.mcp", description="llmshift MCP server (stdio)")
    parser.add_argument("--cwd", help="project directory the tools work in (default: the current directory)")
    parser.add_argument("--config", help="path to llmshift.yaml")
    args = parser.parse_args(argv)
    try:
        return cmd_mcp(args, None)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
