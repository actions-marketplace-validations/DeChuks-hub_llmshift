"""Appending whole lines to a file that several processes may write at once.

O_APPEND alone is not enough on Windows, where the C runtime seeks to the end and
then writes, so two processes can overwrite each other's lines. Every append here
holds an exclusive OS lock: flock on Linux and macOS, and on Windows a lock on a
byte far past the end of the file, so the lock never blocks reading or writing the
data itself.
"""

from __future__ import annotations

import errno
import os
import secrets
import time
from pathlib import Path

LOCK_TIMEOUT = 15.0  # seconds to wait for other writers
_LOCK_OFFSET = 1 << 40


def long_path(path: str | os.PathLike, *, always: bool = False) -> str:
    """The path as Windows' file functions accept it past 260 characters (the \\\\?\\ form,
    absolute) when it is that long, or `always` (a folder to walk: what lies below it may
    be); elsewhere, and for shorter paths, as it is."""
    text = os.fspath(path)
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    full = os.path.abspath(text)
    if len(full) < 240 and not always:  # room for the names below a folder
        return text
    return "\\\\?\\UNC\\" + full[2:] if full.startswith("\\\\") else "\\\\?\\" + full


def find_program(name: str) -> str | None:
    """The full path of a program in a PATH folder, never one in the current directory.
    Windows looks in the current directory first when it starts "git", so a git.exe that a
    cloned repository puts there would run instead (an untrusted search path, as in
    GitPython's CVE-2023-40590). On Windows only .exe files count: batch files go through
    cmd.exe, which re-reads the arguments."""
    suffixes = (".exe",) if os.name == "nt" else ("",)
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        folder = folder.strip().strip('"')
        if not folder or not os.path.isabs(folder):  # "" and "." are the current directory
            continue
        for suffix in suffixes:
            candidate = os.path.join(folder, name + suffix)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def windows_tool(name: str) -> str:
    """A program that ships with Windows, by its full path in System32."""
    return os.path.join(os.environ.get("SystemRoot") or r"C:\Windows", "System32", name + ".exe")


if os.name == "nt":
    import msvcrt

    def _lock(fd: int) -> None:
        deadline = time.monotonic() + LOCK_TIMEOUT
        while True:
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.002)

    def _unlock(fd: int) -> None:
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def temp_file(directory: str | os.PathLike, prefix: str = ".tmp-", suffix: str = "") -> tuple[int, str]:
    """A new file in `directory`, open for writing: (fd, path), as tempfile.mkstemp
    returns. A folder that refuses new files fails at once: on Windows, mkstemp before
    Python 3.13 takes that PermissionError for a name clash and retries ~2 billion times."""
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    for _ in range(100):
        name = os.path.join(str(directory), f"{prefix}{secrets.token_hex(8)}{suffix}")
        try:
            return os.open(name, flags, 0o600), name
        except FileExistsError:
            continue
    raise FileExistsError(errno.EEXIST, "could not create a temporary file", str(directory))


def append(path: str | os.PathLike, data: bytes) -> None:
    """Append `data` (complete lines) without interleaving with other writers. When the
    file ends mid-line (a writer killed, or a full disk), a new line is started first,
    so the broken line never swallows this record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    try:  # read access, to look at the last byte
        fd = os.open(str(target), os.O_RDWR | flags, 0o600)  # they hold prompts: owner only
    except OSError:  # a write-only file or device (/dev/stdout in a container)
        fd = os.open(str(target), os.O_WRONLY | flags, 0o600)
    try:
        _lock(fd)
        try:
            try:
                end = os.lseek(fd, 0, os.SEEK_END)
                if end > 0:
                    os.lseek(fd, end - 1, os.SEEK_SET)
                    if os.read(fd, 1) != b"\n":
                        data = b"\n" + data
            except OSError:  # a pipe or a terminal cannot seek (or be read): nothing to repair there
                pass
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def ends_with_newline(path: str | os.PathLike) -> bool:
    """True for a missing or empty file, or one whose last byte is a newline."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                return True
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) == b"\n"
    except OSError:
        return True
