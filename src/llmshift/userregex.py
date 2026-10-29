"""Regular expressions from the user (a regex check, a JSON schema's pattern), searched
where one that backtracks without end cannot stop the run.

Python's re cannot be interrupted: ^(\\w+\\s?)*$ takes minutes on a 141-character answer it
does not match, and \\d+x seconds on a long run of digits. So a search runs in a worker
process that is killed after TIMEOUT seconds.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import json
import re
import subprocess  # nosec B404  # user regexes run in a child Python process, so a slow one can be stopped
import sys
import threading
import time

TIMEOUT = 2.0  # seconds one search may take

# Says "ready", then answers each [pattern, flags, text] line with a [matched, error] line.
_WORKER = r"""
import json, re, sys
sys.stdout.write("ready\n")
sys.stdout.flush()
for line in sys.stdin:
    try:
        pattern, flags, text = json.loads(line)
        reply = [re.search(pattern, text, flags) is not None, None]
    except Exception as exc:
        reply = [None, type(exc).__name__ + ": " + str(exc)]
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()
"""

# Without a repeat or an alternative, a pattern cannot backtrack: it is searched here.
_MAY_BACKTRACK_RE = re.compile(r"[*+?{|]")


GIVE_UP = 2  # timeouts after which a breaker gives a pattern up (10 cases cost 21 s, 1,000 would 33 minutes)


class RegexStopped(Exception):
    """A search that could not finish: it ran over TIMEOUT seconds (timed_out), or its
    worker process died."""

    def __init__(self, message: str, timed_out: bool = False):
        super().__init__(message)
        self.timed_out = timed_out


_lock = threading.Lock()
_worker: subprocess.Popen | None = None
_no_worker = False  # this Python cannot start one: searches run here, as before
# The breaker of the run in progress: (pattern, flags) -> searches stopped after TIMEOUT. A
# compare run holds one per model whose answers it checks, so a pattern another model's odd
# answers stopped is still tried on this one, and nothing outlives the run (an MCP server).
_BREAKER: contextvars.ContextVar = contextvars.ContextVar("llmshift_regex_breaker", default=None)


@contextlib.contextmanager
def breaker(timeouts: dict):
    """Searches in this block give a pattern up after GIVE_UP timeouts recorded in `timeouts`."""
    token = _BREAKER.set(timeouts)
    try:
        yield timeouts
    finally:
        _BREAKER.reset(token)


def _start() -> subprocess.Popen | None:
    if getattr(sys, "frozen", False) or not sys.executable:
        return None  # sys.executable is the frozen app itself, or nothing at all
    try:
        proc = subprocess.Popen(  # nosec B603  # this Python, by full path, isolated (-I -S); no shell
            [sys.executable, "-I", "-S", "-c", _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, ValueError):
        return None
    timer = threading.Timer(30, proc.kill)  # a slow start is fine, no start at all is not
    timer.start()
    try:
        ready = proc.stdout.readline()
    except (OSError, ValueError):
        ready = ""
    finally:
        timer.cancel()
    if ready.strip() != "ready":
        _stop(proc)
        return None
    return proc


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    with contextlib.suppress(Exception):
        proc.kill()
        proc.wait(timeout=5)
    for pipe in (proc.stdin, proc.stdout):
        with contextlib.suppress(Exception):
            pipe.close()


def search(pattern: str, text: str, flags: int = 0, timeouts: dict | None = None) -> bool:
    """Whether re.search(pattern, text, flags) finds a match, searched in the worker.
    Raises re.error for an invalid pattern and RegexStopped for a search that ran too long.
    `timeouts` (else the block's breaker, else none) remembers patterns that ran too long."""
    global _worker, _no_worker
    compiled = re.compile(pattern, flags)  # an invalid pattern fails here, at once
    if not _MAY_BACKTRACK_RE.search(pattern):
        return compiled.search(text) is not None
    request = json.dumps([pattern, flags, text]) + "\n"
    key = (pattern, flags)
    if timeouts is None:
        timeouts = _BREAKER.get()
    if timeouts is None:
        timeouts = {}  # no breaker: every search is tried
    with _lock:
        if timeouts.get(key, 0) >= GIVE_UP:
            raise RegexStopped(f"stopped after {TIMEOUT:.3g} s on {GIVE_UP} earlier answers, so not tried again", timed_out=True)
        for _ in range(2):  # a worker that died between searches (Ctrl+C, an antivirus) is started again
            if _worker is not None and _worker.poll() is not None:
                _stop(_worker)
                _worker = None
            if _worker is None and not _no_worker:
                _worker = _start()
                _no_worker = _worker is None
            proc = _worker
            if proc is None:
                return compiled.search(text) is not None
            limit = TIMEOUT + len(request) / 20_000_000  # sending a very long answer takes time too
            timer = threading.Timer(limit, proc.kill)
            start = time.monotonic()
            timer.start()
            try:
                proc.stdin.write(request)
                proc.stdin.flush()
                line = proc.stdout.readline()
            except (OSError, ValueError):
                line = ""
            finally:
                timer.cancel()
            if line:
                break
            _worker = None
            _stop(proc)
            if time.monotonic() - start >= limit * 0.9:  # the timer stopped it
                timeouts[key] = timeouts.get(key, 0) + 1
                raise RegexStopped(f"stopped after {limit:.3g} s", timed_out=True)
        else:  # it died at once, twice: what the search does breaks it (memory?)
            raise RegexStopped("the search stopped its worker process twice")
    matched, error = json.loads(line)
    if error is not None:
        raise RegexStopped(error)
    return bool(matched)


def stopped_message(pattern: str, exc: RegexStopped) -> str:
    """Why a pattern could not be checked, and how to fix it."""
    shown = pattern if len(pattern) <= 60 else pattern[:57] + "..."
    if not exc.timed_out:
        return f"regex /{shown}/ could not be checked ({exc})"
    return (
        f"regex /{shown}/ could not be checked ({exc}): it backtracks without end on this text. "
        "A repeat inside a repeat, such as (\\w+\\s?)*, is the usual cause: write it so each "
        "character can match only one way, such as \\w+(?:\\s\\w+)*"
    )


atexit.register(lambda: _stop(_worker))
