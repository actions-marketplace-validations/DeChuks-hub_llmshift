"""On-disk cache of model responses, so re-running compare/fix does not pay twice.

Keys hash the provider identity plus the exact request. `watch` never uses the
cache, because its whole point is to observe the live model.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from pathlib import Path

from . import fileio
from .providers.base import Response
from .textutil import stable_hash


class ResponseCache:
    def __init__(self, directory: str | os.PathLike, enabled: bool = True):
        self.dir = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    # Version 2: responses record what of the request the model refused (Response.rejected);
    # an older answer cannot say, so it is not reused. Version 3: thinking and effort go out in
    # the form each model takes; an answer given while one bad case dropped them is not reused.
    VERSION = 3

    def key(self, identity: dict, request: dict, salt: str = "") -> str:
        return stable_hash({"identity": identity, "request": request, "salt": salt, "v": self.VERSION}, 40)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Response | None:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            resp = Response.from_dict(data) if isinstance(data, dict) else None
        except Exception:  # unreadable or corrupt entry: treat as a miss
            resp = None
        with self._lock:
            if resp is None:
                self.misses += 1
            else:
                self.hits += 1
        if resp is None:
            return None
        resp.cached = True
        return resp

    def put(self, key: str, response: Response) -> None:
        if not self.enabled or not response.ok:
            return
        path = self._path(key)
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = response.to_dict()
            data.pop("cached", None)
            text = json.dumps(data, ensure_ascii=False, default=str)
            fd, tmp = fileio.temp_file(path.parent)
            with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as fh:
                fh.write(text)
            os.replace(tmp, path)
            tmp = None
        except Exception:  # nosec B110  # a cache write failure must never fail a run
            pass
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)

    def clear(self) -> int:
        removed = 0
        if self.dir.is_dir():
            for f in self.dir.rglob("*.json"):
                with contextlib.suppress(OSError):
                    f.unlink()
                    removed += 1
        return removed
