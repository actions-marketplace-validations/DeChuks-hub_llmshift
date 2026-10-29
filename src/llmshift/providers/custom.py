"""Bring-your-own providers.

* `python:my_app.llm:answer` calls `answer(request)` in your code, so you can test
  your real pipeline (prompt building, post-processing) instead of a raw model.
  A file path works too: `python:scripts/call_model.py:run`. `async def` works.
* A `command` provider (configured in llmshift.yaml) runs a program, writes the
  request as JSON to its stdin and reads the answer (text or JSON) from stdout.

The function or program receives {"model": ..., "messages": [...], "params": {...}}
and may return a string, {"text": ..., "tool_calls": [...]}, or a raw OpenAI /
Anthropic / Gemini response object.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import shlex
import signal
import subprocess  # nosec B404  # command providers run the program the user configured (see SECURITY.md)
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .. import fileio
from ..textutil import json_loads, stable_hash
from . import http
from .base import Provider, ProviderError, Response, option_mapping
from .mock import coerce_response

_RUNNING: set = set()  # live command processes, killed on Ctrl+C
_RUNNING_LOCK = threading.Lock()


def resolve_awaitable(result: Any) -> Any:
    """Run `async def` results to completion (the runner calls providers from worker threads)."""
    if not inspect.isawaitable(result):
        return result

    async def wait() -> Any:
        return await result

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(wait())
    # Called from inside a running event loop (e.g. a notebook): use a separate thread.
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = asyncio.run(wait())
        except BaseException as exc:  # re-raised in the calling thread
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _file_digest(path: Any) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except (OSError, TypeError, ValueError):
        return None


_ENV_DIRS = {"site-packages", "dist-packages", ".venv", "venv", ".tox", ".nox", "node_modules"}


def _project_digest(module: Any) -> str | None:
    """A digest of the module's file and of the other modules it brought in from the same
    project (the current folder or the module's own, outside virtual environments): an
    edited helper module, where a prompt often lives, changes it too."""
    main = getattr(module, "__file__", None)
    if not main:
        return None
    try:
        main_path = Path(main).resolve()
        roots = {Path.cwd().resolve(), main_path.parent}
    except (OSError, ValueError):
        return _file_digest(main)
    files = {main_path}
    for name, mod in list(sys.modules.items()):
        where = getattr(mod, "__file__", None)
        if not where or not str(where).endswith(".py") or name.split(".")[0] == "llmshift":
            continue
        try:
            path = Path(where).resolve()
        except (OSError, ValueError):
            continue
        if not _ENV_DIRS.intersection(path.parts) and any(root in path.parents for root in roots):
            files.add(path)
    digest = hashlib.sha256()
    for path in sorted(files)[:500]:
        digest.update(f"{path}\0{_file_digest(path)}\0".encode("utf-8", "replace"))
    return digest.hexdigest()[:16]


_LOAD_LOCKS: dict[str, threading.RLock] = {}
_LOAD_LOCKS_GUARD = threading.Lock()


def _load_lock(module_name: str) -> threading.RLock:
    with _LOAD_LOCKS_GUARD:
        return _LOAD_LOCKS.setdefault(module_name, threading.RLock())


class PythonProvider(Provider):
    kind = "python"

    def __init__(self, model: str, *, name: str | None = None, options: dict | None = None):
        super().__init__(model, name=name or "python", options=options)
        target = model
        if ":" not in target:
            raise ProviderError(f"python provider needs module:function, got {target!r}", code="invalid_spec")
        module_ref, func_name = target.rsplit(":", 1)
        try:
            if module_ref.endswith(".py") or "/" in module_ref or "\\" in module_ref:
                path = Path(module_ref).resolve()
                if not path.is_file():
                    raise ProviderError(f"file not found: {path}", code="invalid_spec")
                module_name = f"llmshift_user_{path.stem}_{stable_hash(str(path), 8)}"
                # One load at a time, as Python's own import lock does for `import`: two
                # providers on one file (pick's candidates) saw the module before it had run,
                # without the function it was about to define.
                with _load_lock(module_name):
                    module = sys.modules.get(module_name)
                    if module is None:
                        spec = importlib.util.spec_from_file_location(module_name, path)
                        if spec is None or spec.loader is None:
                            raise ProviderError(f"cannot import {path}", code="invalid_spec")
                        module = importlib.util.module_from_spec(spec)
                        # Registered first, as a normal import would, so dataclasses, pickle and
                        # typing.get_type_hints can find the module while it runs.
                        sys.modules[module_name] = module
                        try:
                            spec.loader.exec_module(module)
                        except BaseException:
                            sys.modules.pop(module_name, None)
                            raise
            else:
                cwd = os.getcwd()
                if cwd not in sys.path:
                    sys.path.insert(0, cwd)
                module = importlib.import_module(module_ref)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"cannot import {module_ref!r}: {exc}", code="invalid_spec") from None
        fn = getattr(module, func_name, None)
        if not callable(fn):
            raise ProviderError(f"{module_ref!r} has no callable {func_name!r}", code="invalid_spec")
        self.fn = fn
        # Cached answers are only reused while the code that produced them is unchanged.
        self._code = _project_digest(module)
        self._stuck: list[threading.Thread] = []
        self._stuck_lock = threading.Lock()

    def identity(self) -> dict:
        shaping = {k: v for k, v in self.options.items() if k not in ("timeout", "max_retries", "timeout_set")}
        return {"kind": self.kind, "model": self.model, "code": self._code, "options": stable_hash(shaping, 16)}

    def complete(self, request: dict) -> Response:
        # What your function raises is reported as it is (a KeyError in your code is yours,
        # not "a reply llmshift cannot read").
        start = time.perf_counter()
        response = self._complete(request)
        if response.latency_ms is None:
            response.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return response

    _MAX_STUCK = 4  # calls past their timeout still running: no more are started

    def _complete(self, request: dict) -> Response:
        payload = copy.deepcopy(request)
        payload.setdefault("model", self.options.get("target_model"))
        if not self.options.get("timeout_set"):  # no timeout configured: your function takes as long as it takes
            return coerce_response(resolve_awaitable(self.fn(payload)), default_model=self.model)
        with self._stuck_lock:
            self._stuck = [t for t in self._stuck if t.is_alive()]
            if len(self._stuck) >= self._MAX_STUCK:  # each would pile up another thread (and its paid calls)
                raise ProviderError(f"{self.spec}: {len(self._stuck)} earlier calls are still running past the {self.timeout:g}s timeout; no more are started", code="timeout")
        box: dict = {}

        def call() -> None:
            try:
                box["value"] = resolve_awaitable(self.fn(payload))
            except BaseException as exc:  # re-raised in the calling thread
                box["error"] = exc

        # With a timeout set, in a thread of its own (with the caller's context variables), so
        # a call that hangs ends at the timeout; Python cannot stop the thread: it is left to finish.
        worker = threading.Thread(target=contextvars.copy_context().run, args=(call,), daemon=True, name=f"llmshift {self.spec}")
        worker.start()
        deadline = time.monotonic() + self.timeout
        while worker.is_alive():
            worker.join(max(0.0, min(0.25, deadline - time.monotonic())))
            if worker.is_alive() and (http.CANCEL.is_set() or time.monotonic() >= deadline):
                with self._stuck_lock:
                    self._stuck.append(worker)
                if http.CANCEL.is_set():
                    raise ProviderError("cancelled", code="cancelled")
                raise ProviderError(f"{self.spec} did not return within {self.timeout:g}s (raise timeout: in llmshift.yaml to allow longer)", code="timeout")
        if "error" in box:
            raise box["error"]
        return coerce_response(box.get("value"), default_model=self.model)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a command and everything it started (a shell's children would otherwise linger)."""
    with contextlib.suppress(Exception):
        if os.name == "nt":
            taskkill = fileio.windows_tool("taskkill")
            subprocess.run([taskkill, "/F", "/T", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL, capture_output=True, timeout=15)  # nosec B603  # Windows' own taskkill, by full path
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        proc.kill()


def _reap(proc: subprocess.Popen) -> None:
    with contextlib.suppress(subprocess.TimeoutExpired, OSError, ValueError):  # a grandchild still holds the pipes; it has been sent a kill already
        proc.communicate(timeout=5)


def kill_running() -> None:
    """Stop every command still running (used when the user presses Ctrl+C), all at once:
    one at a time, Windows' taskkill takes most of a second each."""
    with _RUNNING_LOCK:
        procs = list(_RUNNING)
    killers = [threading.Thread(target=_kill_tree, args=(proc,), daemon=True) for proc in procs]
    for t in killers:
        t.start()
    for t in killers:
        t.join(timeout=20)


class CommandProvider(Provider):
    kind = "command"

    def __init__(self, model: str, *, name: str | None = None, options: dict | None = None):
        super().__init__(model, name=name or "command", options=options)
        self.command = self.options.get("command")
        if not self.command:
            raise ProviderError(f"provider {self.name!r} has type 'command' but no 'command' setting", code="invalid_spec")
        if not isinstance(self.command, (str, list)) or (isinstance(self.command, list) and not all(isinstance(c, str) for c in self.command)):
            raise ProviderError(f"provider {self.name!r}: 'command' must be a string or a list of strings", code="invalid_spec")
        try:
            self.argv = self._argv()
        except ValueError as exc:  # an unclosed quote
            raise ProviderError(f"provider {self.name!r}: cannot split 'command' into words ({exc})", code="invalid_spec") from None
        self.env = option_mapping(self.options, "env", self.spec, text_values=True)

    def _argv(self) -> list[str] | str:
        # Never through a shell, so nothing in the command is expanded, piped or chained.
        # A string is split into words as a shell would split them (quotes group words);
        # Windows splits it itself when it starts the program. For shell features, name the
        # shell yourself: ["sh", "-c", "..."] or ["cmd", "/c", "..."].
        if isinstance(self.command, list):
            return list(self.command)
        return self.command if os.name == "nt" else shlex.split(self.command)

    def identity(self) -> dict:
        shaping = {k: v for k, v in self.options.items() if k not in ("timeout", "max_retries", "timeout_set")}
        return {"kind": self.kind, "model": self.model, "command": self.command, "code": self._code(), "options": stable_hash(shaping, 16)}

    def _code(self) -> dict:
        """Digests of the files the command names (python answer.py): cached answers are
        only reused while the script that produced them is unchanged."""
        words = self.command if isinstance(self.command, list) else re.findall(r'"[^"]*"|\'[^\']*\'|\S+', self.command)
        base = Path(self.options.get("cwd") or ".")
        out = {}
        for word in words:
            name = word.strip("\"'")
            path = Path(name) if Path(name).is_absolute() else base / name
            try:
                if path.is_file() and path.stat().st_size <= 10_000_000:
                    out[name] = _file_digest(path)
            except (OSError, ValueError):
                continue
        return out

    def _complete(self, request: dict) -> Response:
        # ASCII JSON reads correctly whatever encoding the program assumes for stdin.
        payload = json.dumps({"model": self.model, **request}, ensure_ascii=True, default=str)
        env = dict(os.environ)
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        env.update(self.env)
        # Its own process group on POSIX, so a timeout can kill the program and its children.
        kwargs: dict = {"start_new_session": True} if os.name != "nt" else {}
        # The request comes from a file, not a pipe: writing a pipe the program never reads
        # blocks, and on Windows before Python 3.13 that write ignored the timeout.
        # fileio.temp_file, not tempfile: mkstemp can retry without end on Windows before 3.13.
        fd, stdin_path = fileio.temp_file(tempfile.gettempdir(), prefix="llmshift-request-")
        stdin_file = os.fdopen(fd, "w+b")

        def drop_request_file() -> None:
            stdin_file.close()
            with contextlib.suppress(OSError):
                os.unlink(stdin_path)  # the request (your prompts) is not left behind

        try:
            stdin_file.write(payload.encode("utf-8"))
            stdin_file.seek(0)
            proc = subprocess.Popen(  # nosec B603  # the user's configured program, as an argument list; never a shell
                self.argv,
                stdin=stdin_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.options.get("cwd") or None,
                env=env,
                **kwargs,
            )
        except OSError as exc:
            drop_request_file()
            raise ProviderError(f"cannot run command: {exc}", code="command_error") from None
        with _RUNNING_LOCK:
            _RUNNING.add(proc)
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=max(0.05, min(0.5, deadline - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    if http.CANCEL.is_set():
                        _kill_tree(proc)
                        _reap(proc)
                        raise ProviderError("cancelled", code="cancelled") from None
                    if time.monotonic() >= deadline:
                        _kill_tree(proc)
                        _reap(proc)
                        raise ProviderError(f"command timed out after {self.timeout:.0f}s", code="timeout", retryable=True) from None
        finally:
            drop_request_file()
            with _RUNNING_LOCK:
                _RUNNING.discard(proc)
        if proc.returncode != 0:
            said = (stderr or "").strip()
            lines = [line.strip() for line in said.splitlines() if line.strip()]
            if lines and lines[0].startswith("Traceback (most recent call last)"):
                said = lines[-1]  # a Python script's exception, not the "Traceback" line before its stack
            raise ProviderError(f"command exited with {proc.returncode}: {said[:300]}", code="command_error")
        text = (stdout or "").lstrip("\ufeff")
        out = text.strip()
        try:
            data = json_loads(out)
        except (ValueError, RecursionError):  # plain text, or JSON nested too deeply to read: the text is the answer
            return Response(text=text.rstrip("\n"), model=self.model, finish_reason="stop")
        if isinstance(data, (dict, str)):
            return coerce_response(data, default_model=self.model)
        return Response(text=out, model=self.model, finish_reason="stop")
