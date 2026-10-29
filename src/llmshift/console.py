"""Terminal output helpers: colours, symbols, tables and progress.

Everything here degrades gracefully: no colour when output is piped or NO_COLOR is
set, ASCII symbols when the stream cannot encode Unicode, and no crashes on
Windows code pages.
"""

from __future__ import annotations

import contextlib
import errno
import os
import re
import shutil
import sys
import threading
import time
from typing import Any, Iterator

_STYLES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "grey": "90",
}

_UNICODE_SYMBOLS = {
    "pass": "✓",  # nosec B105  # a check status or setting name, not a password
    "fail": "✗",
    "warn": "!",
    "info": "•",
    "error": "✗",
    "skip": "-",
    "arrow": "→",
    "bar": "─",
    "block": "█",
    "light": "░",
}
_ASCII_SYMBOLS = {
    "pass": "OK",  # nosec B105  # a check status or setting name, not a password
    "fail": "X",
    "warn": "!",
    "info": "*",
    "error": "X",
    "skip": "-",
    "arrow": "->",
    "bar": "-",
    "block": "#",
    "light": ".",
}

_state = {"color": None, "unicode": None}
_lock = threading.Lock()


def _enable_windows_vt() -> bool:
    """Turn on ANSI escape processing in the Windows console (Windows 10+)."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & 0x0004:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def setup_streams() -> None:
    """Make stdout/stderr safe for Unicode output on every platform."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            if stream.isatty():
                stream.reconfigure(errors="replace")  # type: ignore[union-attr]
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    _state["color"] = None
    _state["unicode"] = None


def set_color(enabled: bool | None) -> None:
    _state["color"] = enabled


@contextlib.contextmanager
def no_color() -> Iterator[None]:
    """Render plain text (e.g. for a file) whatever the terminal supports."""
    saved = _state["color"]
    _state["color"] = False
    try:
        yield
    finally:
        _state["color"] = saved


def color_enabled() -> bool:
    if _state["color"] is not None:
        return bool(_state["color"])
    if os.environ.get("NO_COLOR"):
        enabled = False
    elif os.environ.get("FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE"):
        enabled = True
    elif os.environ.get("GITHUB_ACTIONS") == "true":
        enabled = True
    else:
        try:
            enabled = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
        except Exception:
            enabled = False
        if enabled and os.name == "nt":
            enabled = _enable_windows_vt()
    _state["color"] = enabled
    return enabled


def unicode_enabled() -> bool:
    if _state["unicode"] is not None:
        return bool(_state["unicode"])
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "✓✗→─█░•".encode(encoding)
        ok = True
    except (UnicodeEncodeError, LookupError):
        ok = False
    _state["unicode"] = ok
    return ok


def sym(name: str) -> str:
    table = _UNICODE_SYMBOLS if unicode_enabled() else _ASCII_SYMBOLS
    return table.get(name, name)


def style(text: str, *styles: str) -> str:
    if not styles or not color_enabled():
        return text
    codes = ";".join(_STYLES[s] for s in styles if s in _STYLES)
    return f"\033[{codes}m{text}\033[0m" if codes else text


STATUS_STYLE = {
    "pass": ("green",),
    "warn": ("yellow",),
    "fail": ("red", "bold"),
    "error": ("magenta", "bold"),
    "skip": ("grey",),
    "info": ("cyan",),
}


STATUS_TEXT = {"warn": "CHANGED"}


def status_label(status: str) -> str:
    text = STATUS_TEXT.get(status, status.upper())
    return style(f"{sym(status)} {text}", *STATUS_STYLE.get(status, ()))


def width() -> int:
    try:
        return max(60, min(shutil.get_terminal_size((100, 24)).columns, 160))
    except Exception:
        return 100


# Text from outside (provider errors, model output, scanned source) can carry escape
# sequences that move the cursor, retitle the window or hide a line, and bidi controls
# that reorder it. Only the colours style() adds are obeyed; the rest is shown as escapes,
# as text. Directional marks that reorder nothing (Hebrew and Arabic text has them) stay.
_SGR = r"(?:0|1|2|3[1-6]|90)"
_UNSAFE_RE = re.compile(
    rf"\033\[(?:{_SGR}(?:;{_SGR})*)?m|[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]"
)


def safe(text: str) -> str:
    """`text` as it may be written to a terminal."""
    text = str(text)
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    colour = color_enabled()

    def shown(m: re.Match) -> str:
        s = m.group()
        if len(s) > 1:
            return s if colour else ""
        # \u0085 rather than \x85: also a valid escape inside the JSON that --format json prints
        return f"\\u{ord(s):04x}"

    return _UNSAFE_RE.sub(shown, text)


def out(text: str = "", *, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    text = safe(text)
    with _lock:
        try:
            try:
                stream.write(text + "\n")
            except UnicodeEncodeError:
                enc = getattr(stream, "encoding", None) or "ascii"
                stream.write(text.encode(enc, "replace").decode(enc, "replace") + "\n")
            stream.flush()
        except OSError as exc:
            if isinstance(exc, BrokenPipeError) or exc.errno in (errno.EPIPE, errno.EINVAL):
                _closed_pipe(stream)
            raise


def finish_output() -> None:
    """Flush stdout and stderr before exiting. A reader that went away (`llmshift --help |
    head -1`) is not an error: the rest goes nowhere, instead of Python's "Exception
    ignored" message and exit code 120."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except OSError as exc:
            if isinstance(exc, BrokenPipeError) or exc.errno in (errno.EPIPE, errno.EINVAL):
                with contextlib.suppress(Exception):
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, stream.fileno())
        except Exception:  # nosec B110  # exiting anyway: a stream that cannot flush has nowhere to report to
            pass


def _closed_pipe(stream: Any) -> None:
    """The reader went away (`llmshift scan | head`): stop quietly, as Python's docs advise."""
    with contextlib.suppress(Exception):
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stream.fileno())
    raise SystemExit(1)


def err(text: str = "") -> None:
    out(text, err=True)


def warn(text: str) -> None:
    out(style(f"warning: {text}", "yellow"), err=True)


def fail(text: str) -> None:
    out(style(f"error: {text}", "red", "bold"), err=True)


def visible_len(text: str) -> int:
    import re

    return len(re.sub(r"\033\[[0-9;]*m", "", text))


def truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if limit <= 1:
        return text[:limit]
    return text if len(text) <= limit else text[: limit - 1] + ("…" if unicode_enabled() else "~")


def table(rows: list[list[str]], headers: list[str], *, max_widths: list[int] | None = None) -> str:
    """Render a plain-text table. Cells may contain ANSI styles."""
    cols = len(headers)
    plain_rows = [[str(c) for c in row] + [""] * (cols - len(row)) for row in rows]
    widths = [visible_len(h) for h in headers]
    for row in plain_rows:
        for i, cell in enumerate(row[:cols]):
            widths[i] = max(widths[i], visible_len(cell))
    if max_widths:
        widths = [min(w, m) if m else w for w, m in zip(widths, max_widths)]
    # Fit the terminal: shrink the right-most columns first (identifiers usually come first).
    total = width()
    while sum(widths) + 2 * (cols - 1) > total:
        candidates = [i for i in range(cols - 1, -1, -1) if widths[i] > 12]
        if not candidates:
            break
        widths[candidates[0]] -= 1

    def fmt(cell: str, w: int) -> str:
        if visible_len(cell) > w:
            # Styled cells are stripped before truncation to keep escapes intact.
            import re

            cell = truncate(re.sub(r"\033\[[0-9;]*m", "", cell), w)
        return cell + " " * (w - visible_len(cell))

    lines = ["  ".join(style(fmt(h, w), "bold") for h, w in zip(headers, widths)).rstrip()]
    lines.append("  ".join(sym("bar") * w for w in widths))
    for row in plain_rows:
        lines.append("  ".join(fmt(c, w) for c, w in zip(row[:cols], widths)).rstrip())
    return "\n".join(lines)


def bar(fraction: float, length: int = 20) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * length))
    return sym("block") * filled + sym("light") * (length - filled)


class Progress:
    """A minimal thread-safe progress counter written to stderr (TTY only)."""

    def __init__(self, total: int, label: str, *, enabled: bool | None = None):
        self.total = total
        self.label = label
        self.done = 0
        self.failed = 0
        self._start = time.time()
        if enabled is None:
            try:
                enabled = sys.stderr.isatty()
            except Exception:
                enabled = False
        self.enabled = enabled and total > 0
        self._lock = threading.Lock()
        self._last = 0.0

    def advance(self, ok: bool = True) -> None:
        with self._lock:
            self.done += 1
            if not ok:
                self.failed += 1
            now = time.time()
            if self.enabled and (now - self._last > 0.1 or self.done == self.total):
                self._last = now
                extra = f", {self.failed} error{'s' if self.failed != 1 else ''}" if self.failed else ""
                msg = f"\r  {self.label}: {self.done}/{self.total}{extra}"
                with contextlib.suppress(Exception):
                    sys.stderr.write(msg)
                    sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            with contextlib.suppress(Exception):
                sys.stderr.write("\r" + " " * 70 + "\r")
                sys.stderr.flush()
