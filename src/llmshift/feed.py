"""The live retirement feed: `llmshift update`, `llmshift digest` and `llmshift calendar`.

* update   downloads the latest lifecycle data (and, with --prices, model prices) into
           the user data directory; valid, newer data is then used instead of the copy
           bundled with this release.
* digest   sends one message listing the models your repositories use that are retired
           or retire soon (made for a weekly Slack, Teams or e-mail post).
* calendar builds a static website (with a page per model and per provider, made to be found
           by search engines and AI assistants) and an iCalendar feed of retirement dates.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from . import __version__, fileio
from . import console as con
from . import registry as reglib
from .errors import EXIT_FAIL, EXIT_OK, UsageError
from .providers import http
from .providers.base import ProviderError
from .textutil import json_loads, plural, quoteattr, xml_escape

DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/DeChuks-hub/llmshift/main/src/llmshift/data/models.json"
DEFAULT_PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
MAX_DOWNLOAD_BYTES = 64_000_000
PROVIDER_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google (Gemini API)",
    "google-vertex": "Google Vertex AI",
    "mistral": "Mistral",
    "cohere": "Cohere",
    "xai": "xAI",
    "deepseek": "DeepSeek",
}


class UpdateError(Exception):
    """Data could not be downloaded, or was rejected by validation."""


# ------------------------------------------------------------------ update


@dataclass
class UpdateResult:
    what: str  # lifecycle data | prices
    status: str  # updated | current | available
    current: str | None
    latest: str | None
    path: str | None = None
    counts: dict = field(default_factory=dict)
    changes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "what": self.what,
            "status": self.status,
            "current": self.current,
            "latest": self.latest,
            "path": self.path,
            "counts": self.counts,
            "changes": self.changes,
        }


def fetch_json(url: str) -> tuple[Any, bytes]:
    """(parsed JSON, raw bytes) from an http(s) URL or a local file."""
    if re.match(r"^https?://", url, re.I):
        try:
            raw = http.request("GET", url, raw=True, timeout=30, max_retries=2)  # a silent server costs about a minute and a half, not four
        except ProviderError as exc:
            raise UpdateError(f"could not download {url}: {exc}") from None
    else:
        path = Path(urllib.request.url2pathname(urllib.parse.urlparse(url).path)) if url.lower().startswith("file:") else Path(url)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise UpdateError(f"cannot read {url}: {exc.strerror or exc}") from None
    if not isinstance(raw, (bytes, bytearray)):
        raise UpdateError(f"{url} returned no data")
    if len(raw) > MAX_DOWNLOAD_BYTES:
        raise UpdateError(f"{url} is unexpectedly large ({len(raw):,} bytes)")
    try:
        return json_loads(bytes(raw).decode("utf-8-sig")), bytes(raw)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise UpdateError(f"{url} did not return valid JSON ({exc})") from None


def save_data(name: str, raw: bytes) -> Path:
    """Write a data file into the user data directory atomically."""
    directory = reglib.user_data_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = fileio.temp_file(directory, prefix=f".{name}.")
    except OSError as exc:
        raise UpdateError(f"cannot write to {directory}: {exc.strerror or exc} (set LLMSHIFT_HOME to another folder)") from None
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, directory / name)
    except BaseException as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        if isinstance(exc, OSError):  # name the file being replaced, not the temporary one
            raise UpdateError(f"cannot write {directory / name}: {exc.strerror or exc}") from None
        raise
    return directory / name


def registry_changes(old: dict, new: dict, limit: int = 12) -> tuple[dict, list[str]]:
    """Counts and the most useful lines describing how the lifecycle data changed."""

    def index(data: dict) -> dict:
        return {(str(e.get("id", "")).lower(), e.get("provider")): e for e in data.get("models") or [] if isinstance(e, dict)}

    a, b = index(old), index(new)
    added = [b[key] for key in b.keys() - a.keys()]
    removed = [a[key] for key in a.keys() - b.keys()]
    changed = []
    for key in a.keys() & b.keys():
        before, after = a[key], b[key]
        fields = [f for f in ("status", "retires", "replacement", "deprecated") if before.get(f) != after.get(f)]
        if fields:
            changed.append((after, before, fields))

    def soonest(e: dict) -> tuple:
        return (str(e.get("retires") or "9999-99-99"), str(e.get("id", "")))

    lines = []
    for e in sorted(added, key=soonest):
        lines.append(f"new: {e.get('id')} ({e.get('provider')})" + (f", retires {e['retires']}" if e.get("retires") else f", {e.get('status', 'active')}"))
    for after, before, fields in sorted(changed, key=lambda t: soonest(t[0])):
        diffs = ", ".join(f"{f} {before.get(f) or '-'} -> {after.get(f) or '-'}" for f in fields)
        lines.append(f"changed: {after.get('id')} ({after.get('provider')}): {diffs}")
    for e in sorted(removed, key=soonest):
        lines.append(f"removed: {e.get('id')} ({e.get('provider')})")
    counts = {"before": len(a), "after": len(b), "added": len(added), "removed": len(removed), "changed": len(changed)}
    if len(lines) > limit:
        lines = lines[:limit] + [f"... and {len(lines) - limit} more"]
    return counts, lines


def update_registry(url: str | None = None, *, check: bool = False) -> UpdateResult:
    url = url or os.environ.get("LLMSHIFT_REGISTRY_URL") or DEFAULT_REGISTRY_URL
    data, raw = fetch_json(url)
    bundled = reglib.bundled_registry()
    floor = max(50, len(bundled.get("models") or []) // 2)  # a truncated file must not wipe the data
    problems = reglib.registry_problems(data, min_models=floor)
    if problems:
        raise UpdateError(f"the lifecycle data from {url} was rejected: " + "; ".join(problems[:4]))
    current, _origin = reglib.effective_registry()
    current_date, latest_date = current.get("updated"), data.get("updated")
    counts, changes = registry_changes(current, data)
    if (reglib._date(latest_date) or _dt.date.min) <= (reglib._date(current_date) or _dt.date.min):
        return UpdateResult("lifecycle data", "current", current_date, latest_date, counts=counts)
    if check:
        return UpdateResult("lifecycle data", "available", current_date, latest_date, counts=counts, changes=changes)
    path = save_data("models.json", raw)
    return UpdateResult("lifecycle data", "updated", current_date, latest_date, str(path), counts, changes)


def update_prices(url: str | None = None, *, check: bool = False) -> UpdateResult:
    from . import pricing

    url = url or os.environ.get("LLMSHIFT_PRICES_URL") or DEFAULT_PRICES_URL
    table, _raw = fetch_json(url)
    if isinstance(table, dict) and isinstance(table.get("prices"), dict):  # already in llmshift's format
        data = dict(table)
    else:
        data = {
            "about": "Model prices in USD per 1M tokens [input, output], followed where the list has them by cache "
            "prices and long-context tiers (see pricing.convert_litellm), converted from LiteLLM's model_prices_and_context_window.json "
            "(MIT License, Copyright (c) 2023 Berri AI).",
            "source": url,
            "fetched": reglib.today().isoformat(),
            "unit": "USD per 1M tokens",
            "prices": pricing.convert_litellm(table),
        }
    problems = pricing.prices_problems(data, min_prices=500)
    if problems:
        raise UpdateError(f"the prices from {url} were rejected: " + "; ".join(problems[:4]))
    current, current_date = pricing._table()
    latest = {str(k).lower(): list(v) for k, v in data["prices"].items()}  # cache prices too
    changed = sum(1 for k, v in latest.items() if list(current.get(k) or []) != v) + len(current.keys() - latest.keys())
    counts = {"prices": len(latest), "changed": changed}
    older = (reglib._date(data.get("fetched")) or _dt.date.min) < (reglib._date(current_date) or _dt.date.min)
    if not changed or older:
        return UpdateResult("prices", "current", current_date, data.get("fetched"), counts=counts)
    if check:
        return UpdateResult("prices", "available", current_date, data.get("fetched"), counts=counts)
    raw = (json.dumps(data, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    path = save_data("pricing.json", raw)
    pricing.reload()
    return UpdateResult("prices", "updated", current_date, data.get("fetched"), str(path), counts)


def _update_line(r: UpdateResult) -> str:
    if r.status == "updated":
        if r.what == "prices":
            detail = f"{r.counts.get('prices', 0):,} models, {r.counts.get('changed', 0):,} prices changed"
        else:
            c = r.counts
            detail = f"{c.get('before', 0)} -> {c.get('after', 0)} models: {c.get('added', 0)} new, {c.get('changed', 0)} changed, {c.get('removed', 0)} removed"
        return con.style(con.sym("pass"), "green") + f" {r.what} updated: {r.current} -> {r.latest} ({detail})"
    verb = "are" if r.what == "prices" else "is"
    if r.status == "available":
        return f"newer {r.what} {verb} available: {r.latest} (you have {r.current}). Run `llmshift update{' --prices' if r.what == 'prices' else ''}` to download {'them' if verb == 'are' else 'it'}."
    return f"{r.what} {verb} up to date ({r.current})"


def cmd_update(args: argparse.Namespace, cfg: dict) -> int:
    jobs = [lambda: update_registry(args.url, check=args.check)]
    if args.prices:
        jobs.append(lambda: update_prices(args.prices_url, check=args.check))
    results, failed = [], False
    for job in jobs:
        try:
            results.append(job())
        except UpdateError as exc:
            con.fail(str(exc))
            failed = True
    if args.format == "json":
        con.out(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
    else:
        for r in results:
            con.out(_update_line(r))
            for line in r.changes:
                con.out(con.style(f"    {line}", "grey"))
            if r.path:
                con.out(con.style(f"    saved to {r.path}", "grey"))
    return EXIT_FAIL if failed else EXIT_OK


# ------------------------------------------------------------------ digest


@dataclass
class DigestItem:
    model: str
    provider: str
    status: str  # retired | retiring | deprecated
    retires: str | None = None
    days_left: int | None = None
    replacement: str | None = None
    locations: list[str] = field(default_factory=list)

    def line(self) -> str:
        if self.status == "retired":
            when = f"retired on {self.retires}" if self.retires else "retired"
        elif self.status == "retiring":
            days = self.days_left or 0
            when = f"retires {self.retires} ({'today' if days == 0 else 'tomorrow' if days == 1 else f'in {days} days'})"
        else:
            when = "deprecated, no shutdown date yet"
        text = f"{self.model} ({self.provider}): {when}"
        if self.replacement:
            text += f"; replacement: {self.replacement}"
        if self.locations:
            shown = ", ".join(self.locations[:3])
            more = len(self.locations) - 3
            text += f"; used in {shown}" + (f" and {more} more place{'s' if more != 1 else ''}" if more > 0 else "")
        return text

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [])}


@dataclass
class Digest:
    items: list[DigestItem]
    within: int
    paths: list[str]
    files_scanned: int = 0
    registry_updated: str | None = None
    errors: list[str] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for i in self.items if i.status == status)

    @property
    def severity(self) -> str:
        if self.count("retired"):
            return "critical"
        return "warning" if self.items else "info"

    def title(self) -> str:
        n = len(self.items)
        if not n:
            return f"No models in your code are retired or retire within {self.within} days"
        noun = f"{n} model{'s' if n != 1 else ''} in your repos"
        if self.count("retiring") == n:
            return f"{noun} retire within {self.within} days"
        parts = [f"{c} {label}" for c, label in ((self.count("retired"), "retired"), (self.count("retiring"), f"retiring within {self.within} days"), (self.count("deprecated"), "deprecated")) if c]
        return f"{noun} need attention: {', '.join(parts)}"

    def lines(self) -> list[str]:
        return [item.line() for item in self.items]

    def to_dict(self) -> dict:
        return {
            "llmshift_version": __version__,
            "kind": "digest",
            "title": self.title(),
            "within_days": self.within,
            "paths": self.paths,
            "files_scanned": self.files_scanned,
            "registry_updated": self.registry_updated,
            "counts": {s: self.count(s) for s in ("retired", "retiring", "deprecated")},
            "items": [i.to_dict() for i in self.items],
            "errors": self.errors,
        }

    def markdown(self) -> str:
        lines = [f"## {self.title()}", ""]
        lines += [f"- {line}" for line in self.lines()] or ["Nothing to do."]
        lines += ["", f"_Scanned {self.files_scanned} files in {', '.join(self.paths)}; lifecycle data {self.registry_updated}._"]
        return "\n".join(lines) + "\n"


def build_digest(paths: list[str], *, within: int = 30, registry: Any = None, exclude: Any = (), include_deprecated: bool = False) -> Digest:
    from .scanner import scan_paths

    if within < 0:
        raise UsageError("--within must be zero or more days")
    reg = registry or reglib.Registry.load()
    found: dict[str, DigestItem] = {}
    digest = Digest(items=[], within=within, paths=list(paths), registry_updated=reg.updated)
    for path in paths:
        try:
            result = scan_paths([path], registry=reg, exclude=exclude, warn_days=within)
        except FileNotFoundError as exc:
            raise UsageError(str(exc)) from None
        digest.files_scanned += result.files_scanned
        digest.errors.extend(result.errors)
        for m in result.models():
            lk = m["lookup"]
            if lk is None:
                continue
            if lk.status == "retired":
                status = "retired"
            elif lk.status == "retiring" and lk.days_left is not None and lk.days_left <= within:
                status = "retiring"
            elif include_deprecated and lk.status == "deprecated":
                status = "deprecated"
            else:
                continue
            item = found.get(m["model"])
            if item is None:
                item = found[m["model"]] = DigestItem(
                    model=m["model"],
                    provider=lk.provider,
                    status=status,
                    retires=lk.retires.isoformat() if lk.retires else None,
                    days_left=lk.days_left,
                    replacement=lk.replacement,
                )
            for finding in m["findings"]:
                if finding.location not in item.locations:
                    item.locations.append(finding.location)
    rank = {"retired": 0, "retiring": 1, "deprecated": 2}
    digest.items = sorted(found.values(), key=lambda i: (rank[i.status], i.retires or "9999-99-99", i.model))
    return digest


def cmd_digest(args: argparse.Namespace, cfg: dict) -> int:
    from .alerts import AlertMessage, check_destinations, deliver, run_url

    section = cfg.get("digest") if isinstance(cfg.get("digest"), dict) else {}
    scan_cfg = cfg.get("scan") if isinstance(cfg.get("scan"), dict) else {}
    paths = args.paths or ["."]
    digest = build_digest(
        paths,
        within=args.within if args.within is not None else int(section.get("within", 30)),
        exclude=list(scan_cfg.get("exclude") or []) + list(args.exclude or []),
        include_deprecated=args.include_deprecated,
    )
    if args.format == "json":
        con.out(json.dumps(digest.to_dict(), indent=2, ensure_ascii=False))
    elif args.format == "markdown":
        con.out(digest.markdown())
    else:
        con.out(con.style(digest.title(), "bold"))
        for line in digest.lines():
            con.out(f"  {con.sym('info')} {line}")
        con.out(con.style(f"  scanned {digest.files_scanned} files in {', '.join(paths)}; lifecycle data {digest.registry_updated}", "grey"))
    destinations = list(section.get("alerts") or []) + list(args.alert or [])
    code = EXIT_OK
    sending = bool(destinations) and bool(digest.items or args.always)  # nothing to send: no URL is needed either
    problems = check_destinations(destinations) if sending and not args.dry_run else []
    if problems:
        raise UsageError("the digest could not be delivered: " + "; ".join(problems))
    if sending:
        if args.dry_run:
            con.out(con.style(f"  (dry run: not sending to {len(destinations)} destination(s))", "grey"), err=True)
        else:
            msg = AlertMessage(
                title=digest.title(),
                severity=digest.severity,
                model="your repositories",
                lines=digest.lines(),
                alerts=[i.to_dict() for i in digest.items],
                summary=digest.to_dict()["counts"],
                run_url=run_url(),
                issue_title="[llmshift] model retirement digest",
            )
            for name, err in deliver(destinations, msg, default_min_severity="info"):
                if err:
                    con.warn(f"digest via {name} failed: {err}")
                    code = EXIT_FAIL
                else:
                    con.out(con.style(f"  sent via {name}", "grey"), err=True)
    fail_on = args.fail_on or "never"
    if fail_on == "retired" and digest.count("retired"):
        return EXIT_FAIL
    if fail_on == "retiring" and (digest.count("retired") or digest.count("retiring")):
        return EXIT_FAIL
    return code


# ---------------------------------------------------------------- calendar


def _ics_text(value: Any) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def fold_ics_line(line: str) -> str:
    """RFC 5545 line folding: at most 75 octets per line, continuation lines start with a space."""
    parts: list[str] = []
    cur, size, limit = "", 0, 75
    for ch in line:
        n = len(ch.encode("utf-8"))
        if size + n > limit:
            parts.append(cur)
            cur, size, limit = ch, n, 74
        else:
            cur += ch
            size += n
    parts.append(cur)
    return "\r\n ".join(parts)


def calendar_entries(reg: Any, days_back: int = 90) -> dict[str, list]:
    """Lookups grouped as upcoming, recently retired and deprecated without a date."""
    now = reglib.today()
    upcoming, recent, undated = [], [], []
    for lk in reg.search(warn_days=0):
        if lk.retires:
            if lk.retires >= now:
                upcoming.append(lk)
            elif (now - lk.retires).days <= days_back:
                recent.append(lk)
        elif lk.status == "deprecated":
            undated.append(lk)
    upcoming.sort(key=lambda lk: (lk.retires, lk.provider, lk.model))
    recent.sort(key=lambda lk: (lk.retires, lk.provider, lk.model), reverse=True)
    undated.sort(key=lambda lk: (lk.provider, lk.model))
    return {"upcoming": upcoming, "recent": recent, "undated": undated}


def build_ics(entries: dict[str, list], updated: str | None) -> bytes:
    now = reglib.today()
    stamp = (reglib._date(updated) or now).strftime("%Y%m%dT000000Z")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//llmshift//AI model retirement calendar//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:AI model retirements",
        "X-WR-CALDESC:" + _ics_text(f"Shutdown dates of AI models (lifecycle data {updated}), from llmshift"),
        "REFRESH-INTERVAL;VALUE=DURATION:PT12H",
        "X-PUBLISHED-TTL:PT12H",
    ]
    uids: set[str] = set()
    for lk in entries["upcoming"] + entries["recent"]:
        start = lk.retires
        verb = "retires" if start >= now else "retired"
        uid = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{lk.provider}-{lk.model}").strip("-") or "model"
        if uid in uids:
            uid += f"-{start:%Y%m%d}"
        uids.add(uid)
        about = [f"{lk.model} ({PROVIDER_NAMES.get(lk.provider, lk.provider)}) {verb} on {start.isoformat()}."]
        if lk.replacement:
            about.append(f"Replacement: {lk.replacement}")
        if lk.note:
            about.append(f"Note: {lk.note}")
        if lk.source:
            about.append(f"Source: {lk.source}")
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}@llmshift",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{start:%Y%m%d}",
            f"DTEND;VALUE=DATE:{start + _dt.timedelta(days=1):%Y%m%d}",
            "SUMMARY:" + _ics_text(f"{lk.model} {verb} ({lk.provider})"),
            "DESCRIPTION:" + _ics_text("\n".join(about)),
        ]
        if lk.source and re.match(r"^https?://[^\s\"<>\\]+$", lk.source):
            lines.append(f"URL:{lk.source}")
        lines += ["CATEGORIES:" + _ics_text(lk.provider), "TRANSP:TRANSPARENT", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "".join(fold_ics_line(line) + "\r\n" for line in lines).encode("utf-8")


_PAGE_CSS = """
:root { color-scheme: light dark; --bg: #ffffff; --fg: #1d1d1f; --muted: #5f6368; --line: #e3e3e8; --soft: #f4f5f7; --accent: #0b57d0; --warn: #9a5b00; --bad: #b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg: #121214; --fg: #ececf1; --muted: #a4a4ad; --line: #2c2c33; --soft: #1b1b20; --accent: #8ab4f8; --warn: #f2b84b; --bad: #f28b82; } }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--fg); }
main { max-width: 1120px; margin: 0 auto; padding: 28px 16px 56px; }
h1 { font-size: 28px; margin: 0 0 6px; }
h2 { font-size: 19px; margin: 30px 0 10px; }
p { margin: 6px 0; }
.lede, .foot, .days, .count-note { color: var(--muted); }
a { color: var(--accent); }
.actions a { margin-right: 14px; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 4px; }
.filters input, .filters select { font: inherit; padding: 7px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--soft); color: var(--fg); }
.filters input { flex: 1 1 240px; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { background: var(--soft); font-weight: 600; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
code { font: 13px ui-monospace, SFMono-Regular, Consolas, monospace; word-break: break-all; }
.days { font-size: 13px; white-space: nowrap; }
.soon .days { color: var(--warn); font-weight: 600; }
.gone .days { color: var(--bad); }
.empty { padding: 14px; color: var(--muted); }
"""

_PAGE_JS = """
(function () {
  var today = new Date(); today.setHours(0, 0, 0, 0);
  document.querySelectorAll("tr[data-date]").forEach(function (row) {
    var p = row.getAttribute("data-date").split("-");
    var days = Math.round((new Date(+p[0], +p[1] - 1, +p[2]) - today) / 86400000);
    var cell = row.querySelector(".days");
    if (cell) cell.textContent = days > 1 ? "in " + days + " days" : days === 1 ? "tomorrow" : days === 0 ? "today" : days === -1 ? "yesterday" : (-days) + " days ago";
    row.className = days < 0 ? "gone" : days <= 30 ? "soon" : "";
  });
  var q = document.getElementById("q"), provider = document.getElementById("provider"), status = document.getElementById("status");
  function apply() {
    var text = q.value.trim().toLowerCase(), prov = provider.value, st = status.value;
    document.querySelectorAll("section[data-section]").forEach(function (section) {
      var name = section.getAttribute("data-section"), show = st ? st === name : name !== "undated", visible = 0;
      section.querySelectorAll("tr[data-model]").forEach(function (row) {
        var ok = show && (!prov || row.getAttribute("data-provider") === prov) && (!text || row.getAttribute("data-search").indexOf(text) !== -1);
        row.hidden = !ok;
        if (ok) visible++;
      });
      section.hidden = !show;
      var count = section.querySelector(".count");
      if (count) count.textContent = visible;
      var empty = section.querySelector(".empty");
      if (empty) empty.hidden = visible !== 0;
    });
  }
  [q, provider, status].forEach(function (el) { el.addEventListener("input", apply); el.addEventListener("change", apply); });
  var webcal = document.getElementById("webcal");
  if (webcal && /^https?:$/.test(location.protocol)) {
    webcal.href = "webcal://" + location.host + location.pathname.replace(/[^\\/]*$/, "") + "retirements.ics";
    webcal.hidden = false;
  }
  apply();
})();
"""

# GoatCounter's script, pinned to one version: browsers refuse it if its content ever changes.
_GOATCOUNTER_JS = "https://gc.zgo.at/count.v5.js"
_GOATCOUNTER_SRI = "sha384-atnOLvQb9t+jTSipvd75X2yginT4PjVbqDdlJAmxMm+wYElFmeR6EmLP5bYeoRVQ"


def goatcounter_url(value: str) -> str:
    """Where the calendar page counts visits: a GoatCounter site code or its https://.../count URL."""
    value = value.strip()
    if re.fullmatch(r"https://[A-Za-z0-9.-]+(:[0-9]+)?/count", value):
        return value
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{1,49}", value.lower()):
        return f"https://{value.lower()}.goatcounter.com/count"
    raise UsageError(f"--goatcounter takes a GoatCounter site code (CODE in https://CODE.goatcounter.com) or its https://.../count URL, not {value!r}")


def _days_text(days: int) -> str:
    if days > 1:
        return f"in {days} days"
    if days == 1:
        return "tomorrow"
    if days == 0:
        return "today"
    return "yesterday" if days == -1 else f"{-days} days ago"


def _rows(items: list, now: _dt.date, dated: bool, links: dict | None = None) -> str:
    esc = html.escape
    out = []
    for lk in items:
        provider = PROVIDER_NAMES.get(lk.provider, lk.provider)
        search = " ".join(x for x in (lk.model, lk.provider, provider, lk.replacement or "") if x).lower()
        source = f'<a href="{esc(lk.source)}" rel="noopener">source</a>' if lk.source and re.match(r"^https?://", lk.source) else ""
        when = ""
        if dated:
            days = (lk.retires - now).days
            when = f'<td><time datetime="{lk.retires.isoformat()}">{lk.retires.isoformat()}</time><div class="days">{_days_text(days)}</div></td>'
        page = (links or {}).get((lk.provider, lk.model))
        name = f'<a href="{esc(page)}"><code>{esc(lk.model)}</code></a>' if page else f"<code>{esc(lk.model)}</code>"
        out.append(
            f'<tr data-model="{esc(lk.model)}" data-provider="{esc(lk.provider)}"'
            + (f' data-date="{lk.retires.isoformat()}"' if dated else "")
            + f' data-search="{esc(search)}">{when}<td>{name}</td><td>{esc(provider)}</td>'
            f"<td>{esc(lk.replacement or '-')}</td><td>{source}</td></tr>"
        )
    return "\n".join(out)


def build_page(entries: dict[str, list], updated: str | None, *, title: str, days_back: int, count_url: str | None = None,
               links: dict | None = None, base_url: str | None = None) -> str:
    esc = html.escape
    now = reglib.today()
    foot = "Collected from the providers' official deprecation pages. Dates can move: check the source before you plan around one."
    counter = ""
    if count_url:
        foot += ' Visits are counted with <a href="https://www.goatcounter.com" rel="noopener">GoatCounter</a>, which sets no cookies.'
        counter = f'\n<script data-goatcounter="{esc(count_url)}" async src="{_GOATCOUNTER_JS}" crossorigin="anonymous" integrity="{_GOATCOUNTER_SRI}"></script>'
    providers = sorted({lk.provider for group in entries.values() for lk in group})
    options = "".join(f'<option value="{esc(p)}">{esc(PROVIDER_NAMES.get(p, p))}</option>' for p in providers)
    head_dated = "<tr><th>Date</th><th>Model</th><th>Provider</th><th>Replacement</th><th>Source</th></tr>"
    head_undated = "<tr><th>Model</th><th>Provider</th><th>Replacement</th><th>Source</th></tr>"

    def section(name: str, heading: str, items: list, dated: bool) -> str:
        head = head_dated if dated else head_undated
        empty = '<p class="empty" hidden>No matching models.</p>' if items else '<p class="empty">None.</p>'
        return (
            f'<section data-section="{name}"><h2>{esc(heading)} (<span class="count">{len(items)}</span>)</h2>'
            f'<div class="table-wrap"><table><thead>{head}</thead><tbody>\n{_rows(items, now, dated, links)}\n</tbody></table>{empty}</div></section>'
        )

    head_links = ""
    if base_url:
        head_links = (f'\n<link rel="canonical" href="{esc(base_url)}">'
                      f'\n<link rel="alternate" type="application/rss+xml" title="AI model retirements" href="{esc(base_url)}feed.xml">')
    more = '<a href="models/">Every model, by provider</a>' if links else ""
    more += '<a href="feed.xml">RSS</a>' if base_url else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="Upcoming shutdown dates of AI models from OpenAI, Anthropic, Google, Mistral, Cohere, xAI and DeepSeek.">{head_links}
<style>{_PAGE_CSS}</style>
</head>
<body>
<main>
<h1>{esc(title)}</h1>
<p class="lede">When AI models shut down, and what replaces them. Lifecycle data updated {esc(str(updated))}; page built {now.isoformat()}.</p>
<p class="actions"><a id="webcal" href="retirements.ics" hidden>Subscribe in your calendar</a><a href="retirements.ics">Download the calendar (.ics)</a><a href="retirements.json">JSON</a>{more}<a href="https://github.com/DeChuks-hub/llmshift">Find these models in your code with llmshift</a></p>
<div class="filters">
<input id="q" type="search" placeholder="Search models or replacements" aria-label="Search models or replacements">
<select id="provider" aria-label="Provider"><option value="">All providers</option>{options}</select>
<select id="status" aria-label="Which models"><option value="">Upcoming and recently retired</option><option value="upcoming">Upcoming only</option><option value="recent">Recently retired only</option><option value="undated">Deprecated, no date yet</option></select>
</div>
{section("upcoming", "Upcoming retirements", entries["upcoming"], True)}
{section("recent", f"Retired in the last {days_back} days", entries["recent"], True)}
{section("undated", "Deprecated, no shutdown date yet", entries["undated"], False)}
<p class="foot">{foot}</p>
</main>
<script>{_PAGE_JS}</script>{counter}
</body>
</html>
"""


# ------------------------------------------------------------- model pages

_MODEL_CSS = """
nav.crumbs { font-size: 13px; color: var(--muted); margin: 0 0 12px; }
nav.crumbs a { color: var(--muted); }
h1 code { font-size: inherit; word-break: break-all; }
.facts th { width: 180px; }
.status { display: inline-block; font-size: 12px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; padding: 1px 8px; border-radius: 999px; background: var(--soft); }
.status.retiring, .status.deprecated { color: var(--warn); }
.status.retired { color: var(--bad); }
ol.steps { padding-left: 22px; }
ol.steps li { margin: 12px 0; }
pre { background: var(--soft); border: 1px solid var(--line); border-radius: 8px; padding: 9px 12px; overflow-x: auto; margin: 6px 0 0; }
pre code { word-break: normal; }
.box { border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; margin: 18px 0; background: var(--soft); }
ul.links { padding-left: 20px; }
ul.links li { margin: 4px 0; }
"""

# What the providers answer once a model is gone, for people who search the error.
_GONE_ERRORS = {
    "openai": "Error code: 404 - model_not_found: The model `{model}` does not exist or you do not have access to it.",
    "anthropic": "Error code: 404 - not_found_error: model: {model}",
    "google": "404 NOT_FOUND: models/{model} is not found for API version v1beta, or is not supported for generateContent.",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", str(text).lower()).strip("-.") or "model"


def _short_name(provider: str) -> str:
    """The provider's name where it sits in parentheses (no "Google (Gemini API)" inside another pair)."""
    return {"google": "Gemini API", "google-vertex": "Vertex AI"}.get(provider) or PROVIDER_NAMES.get(provider, provider)


def model_paths(lookups: list) -> dict[tuple[str, str], str]:
    """`models/<provider>/<model>/` for each (provider, model), unique even where IDs differ only in case or punctuation
    (folders must not collide on case-insensitive disks). Sorted first, so a model keeps its address across builds."""
    taken: set[str] = set()
    paths: dict[tuple[str, str], str] = {}
    for lk in sorted(lookups, key=lambda lk: (lk.provider, lk.model.lower(), lk.model)):
        key = (lk.provider, lk.model)
        if key in paths:
            continue
        base = f"models/{_slug(lk.provider)}/{_slug(lk.model)}"
        path, n = base, 2
        while path in taken:
            path, n = f"{base}-{n}", n + 1
        taken.add(path)
        paths[key] = path + "/"
    return paths


def base_url_value(value: str) -> str:
    """The site's public https:// address, ending in a slash."""
    value = value.strip()
    if not re.fullmatch(r"https://[A-Za-z0-9.-]+(:[0-9]+)?(/[A-Za-z0-9._~%/-]*)?", value):
        raise UsageError(f"--base-url takes the site's https:// address, e.g. https://you.github.io/repo/, not {value!r}")
    return value if value.endswith("/") else value + "/"


def _long_date(day: _dt.date) -> str:
    return f"{day.day} {day:%B %Y}"


def _when_text(lk: Any, provider: str, now: _dt.date) -> str:
    if lk.retires and lk.retires >= now:
        return f"{provider} retires {lk.model} on {_long_date(lk.retires)} ({_days_text((lk.retires - now).days)})."
    if lk.retires:
        return f"{provider} retired {lk.model} on {_long_date(lk.retires)}."
    if lk.status == "deprecated":
        return f"{provider} has deprecated {lk.model} but has not set a shutdown date yet."
    if lk.not_before:
        return f"{lk.model} is active. {provider} will not retire it before {_long_date(lk.not_before)}."
    return f"{lk.model} is active. {provider} has not announced a shutdown date."


def _state(lk: Any, now: _dt.date) -> str:
    if lk.retires:
        return "retiring" if lk.retires >= now else "retired"
    return "deprecated" if lk.status == "deprecated" else "active"


def _page_head(*, title: str, description: str, url: str | None, extra_css: str = "", ld: Any = None) -> str:
    esc = html.escape
    head = [
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(title)}</title>",
        f'<meta name="description" content="{esc(description)}">',
        f'<meta property="og:title" content="{esc(title)}">',
        f'<meta property="og:description" content="{esc(description)}">',
        '<meta property="og:type" content="website">',
        '<meta property="og:site_name" content="AI model retirement calendar">',
        '<meta name="twitter:card" content="summary">',
    ]
    if url:
        head += [f'<link rel="canonical" href="{esc(url)}">', f'<meta property="og:url" content="{esc(url)}">']
    head.append(f"<style>{_PAGE_CSS}{extra_css}</style>")
    if ld is not None:
        # JSON in a <script> element: "<" is escaped so no value can close the element.
        head.append('<script type="application/ld+json">' + json.dumps(ld, ensure_ascii=False).replace("<", "\\u003c") + "</script>")
    return "\n".join(head)


def _counter(count_url: str | None) -> str:
    if not count_url:
        return ""
    return f'\n<script data-goatcounter="{html.escape(count_url)}" async src="{_GOATCOUNTER_JS}" crossorigin="anonymous" integrity="{_GOATCOUNTER_SRI}"></script>'


def _foot(updated: str | None, count_url: str | None) -> str:
    foot = (f"Lifecycle data updated {html.escape(str(updated))}, collected from the providers' official deprecation pages. "
            "Dates can move: check the source before you plan around one.")
    if count_url:
        foot += ' Visits are counted with <a href="https://www.goatcounter.com" rel="noopener">GoatCounter</a>, which sets no cookies.'
    return f'<p class="foot">{foot}</p>'


def build_model_page(lk: Any, *, path: str, paths: dict, related: list, updated: str | None, now: _dt.date,
                     base_url: str | None = None, count_url: str | None = None) -> str:
    esc = html.escape
    provider = PROVIDER_NAMES.get(lk.provider, lk.provider)
    root = "../" * path.count("/")
    state = _state(lk, now)
    when = _when_text(lk, provider, now)
    is_model = not lk.model.startswith("/")  # a few entries are API endpoints, such as /v1/answers
    replacement = lk.replacement if lk.replacement and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:@-]*", lk.replacement) else None
    title = f"{lk.model} retirement date and replacement ({_short_name(lk.provider)})"
    description = when + (f" Replacement: {lk.replacement}." if lk.replacement else "") + \
        " Find every call in your code and test the replacement before you switch."

    def link_to(model: str) -> str:
        target = paths.get((lk.provider, model))
        return f'<a href="{root}{esc(target)}"><code>{esc(model)}</code></a>' if target else f"<code>{esc(model)}</code>"

    rows = [("Provider", esc(provider) + (f" ({esc(lk.platform)})" if lk.platform else "")),
            ("Status", f'<span class="status {state}">{state}</span>' + (" (an unpinned alias)" if lk.alias else "")),
            ("Shutdown date", f'<time datetime="{lk.retires.isoformat()}">{lk.retires.isoformat()}</time>' if lk.retires else "not announced")]
    if lk.deprecated:
        rows.append(("Deprecated on", f'<time datetime="{lk.deprecated.isoformat()}">{lk.deprecated.isoformat()}</time>'))
    if lk.not_before:
        rows.append(("Not retired before", lk.not_before.isoformat()))
    rows.append(("Replacement", link_to(lk.replacement) if lk.replacement else "none named"))
    if lk.fine_tune_of:
        rows.append(("Fine-tune of", link_to(lk.fine_tune_of)))
    if lk.source and re.match(r"^https?://", lk.source):
        rows.append(("Source", f'<a href="{esc(lk.source)}" rel="noopener">{esc(lk.source)}</a>'))
    if lk.note:
        rows.append(("Note", esc(lk.note)))
    facts = "\n".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)

    to = replacement or "<new model>"
    steps = [f'<li>Find every call to it in your code (Python, JS/TS, Go, Java, configs, notebooks), with nothing to install:'
             f"<pre><code>uvx llmshift scan</code></pre></li>"]
    if is_model:
        steps += [
            f'<li>Turn your real traffic into tests, so you compare on what your users send:<pre><code>llmshift snapshot logs/ --model "{esc(lk.model)}"</code></pre></li>',
            f"<li>See exactly what changes on the {'replacement' if replacement else 'new model'}: formats, facts, tone, tool calls and settings it rejects:"
            f"<pre><code>llmshift compare --to {esc(to)}</code></pre></li>",
            f'<li>Swap the model in your code and open a pull request with the results:<pre><code>llmshift upgrade --from "{esc(lk.model)}" --apply --compare --pr</code></pre></li>',
        ]
    heading = {"retiring": f"Before {_long_date(lk.retires) if lk.retires else ''}: move off it safely",
               "retired": "Move off it now", "deprecated": "Move off it before a date is set",
               "active": "When it retires, be ready"}[state]
    error = ""
    if state == "retired" and is_model and lk.provider in _GONE_ERRORS:
        error = (f"<h2>The error you get now</h2><p>Calls to a model after its shutdown fail. The answer usually reads "
                 f"(the wording varies by SDK and endpoint):</p><pre><code>{esc(_GONE_ERRORS[lk.provider].format(model=lk.model))}</code></pre>")
    others = ""
    if related:
        items = "\n".join(
            f'<li><a href="{root}{esc(paths[(o.provider, o.model)])}"><code>{esc(o.model)}</code></a>: '
            f"{'retires' if o.retires >= now else 'retired'} {o.retires.isoformat()}</li>" for o in related)
        others = f"<h2>Other {esc(provider)} retirements</h2><ul class=\"links\">\n{items}\n</ul>"
    report = (f"https://github.com/DeChuks-hub/llmshift/issues/new?title="
              + quote(f"Lifecycle data: {lk.model} ({lk.provider})") + "&labels=data")
    answers = [
        {"@type": "Question", "name": f"When does {lk.model} retire?", "acceptedAnswer": {"@type": "Answer", "text": when}},
        {"@type": "Question", "name": f"What replaces {lk.model}?", "acceptedAnswer": {"@type": "Answer", "text":
            f"{provider} names {lk.replacement} as the replacement." if lk.replacement else f"{provider} has not named a replacement."}},
        {"@type": "Question", "name": f"How do I find {lk.model} in my code?", "acceptedAnswer": {"@type": "Answer", "text":
            "Run uvx llmshift scan in the repository. It lists every model the code calls, when each retires, and the files and lines that call it."}},
    ]
    head = _page_head(title=title, description=description, url=f"{base_url}{path}" if base_url else None, extra_css=_MODEL_CSS,
                      ld={"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": answers})
    return f"""<!doctype html>
<html lang="en">
<head>
{head}
</head>
<body>
<main>
<nav class="crumbs"><a href="{root}">AI model retirement calendar</a> / <a href="{root}{esc(_provider_path(lk.provider))}">{esc(provider)}</a></nav>
<h1><code>{esc(lk.model)}</code></h1>
<p class="lede">{esc(when)}{f" Replacement: <code>{esc(lk.replacement)}</code>." if lk.replacement else ""}</p>
<div class="table-wrap"><table class="facts"><tbody>
{facts}
</tbody></table></div>
<h2>{esc(heading)}</h2>
<ol class="steps">
{chr(10).join(steps)}
</ol>
<p>Then <code>llmshift init --github</code> keeps the check on every pull request. <a href="https://github.com/DeChuks-hub/llmshift">LLMShift</a> is free and open source (MIT) and runs on your machine.</p>
{error}
{others}
<div class="box">Wrong date or replacement? <a href="{esc(report)}" rel="noopener">Tell us on GitHub</a>, with a link to the provider's page.</div>
{_foot(updated, count_url)}
</main>{_counter(count_url)}
</body>
</html>
"""


def _provider_path(provider: str) -> str:
    return f"models/{_slug(provider)}/"


def build_provider_page(provider: str, lookups: list, *, paths: dict, updated: str | None, now: _dt.date,
                        base_url: str | None = None, count_url: str | None = None) -> str:
    esc = html.escape
    name = PROVIDER_NAMES.get(provider, provider)
    path = _provider_path(provider)
    root = "../" * path.count("/")
    groups: dict[str, list] = {"retiring": [], "deprecated": [], "retired": [], "active": []}
    for lk in lookups:
        groups[_state(lk, now)].append(lk)
    groups["retiring"].sort(key=lambda lk: (lk.retires, lk.model))
    groups["retired"].sort(key=lambda lk: (lk.retires, lk.model), reverse=True)
    groups["deprecated"].sort(key=lambda lk: lk.model)
    groups["active"].sort(key=lambda lk: lk.model)

    def table(items: list, dated: bool) -> str:
        if not items:
            return '<p class="empty">None.</p>'
        head = "<tr>" + ("<th>Date</th>" if dated else "") + "<th>Model</th><th>Replacement</th></tr>"
        body = "\n".join(
            "<tr>" + (f'<td><time datetime="{lk.retires.isoformat()}">{lk.retires.isoformat()}</time></td>' if dated else "")
            + f'<td><a href="{root}{esc(paths[(lk.provider, lk.model)])}"><code>{esc(lk.model)}</code></a></td>'
            f"<td>{esc(lk.replacement or '-')}</td></tr>" for lk in items)
        return f'<div class="table-wrap"><table><thead>{head}</thead><tbody>\n{body}\n</tbody></table></div>'

    counts = f"{len(groups['retiring'])} retiring, {len(groups['deprecated'])} deprecated without a date, {len(groups['retired'])} retired, {len(groups['active'])} active"
    head = _page_head(title=f"{name} model retirement dates and replacements",
                      description=f"When {name} models shut down, and what replaces them: {counts}.",
                      url=f"{base_url}{path}" if base_url else None, extra_css=_MODEL_CSS)
    return f"""<!doctype html>
<html lang="en">
<head>
{head}
</head>
<body>
<main>
<nav class="crumbs"><a href="{root}">AI model retirement calendar</a> / <a href="{root}models/">All providers</a></nav>
<h1>{esc(name)} model retirement dates</h1>
<p class="lede">{esc(counts)}. Find the ones your code calls with <code>uvx llmshift scan</code>.</p>
<h2>Retiring</h2>
{table(groups["retiring"], True)}
<h2>Deprecated, no shutdown date yet</h2>
{table(groups["deprecated"], False)}
<h2>Retired</h2>
{table(groups["retired"], True)}
<h2>Active</h2>
{table(groups["active"], False)}
{_foot(updated, count_url)}
</main>{_counter(count_url)}
</body>
</html>
"""


def build_models_index(by_provider: dict[str, list], *, updated: str | None, now: _dt.date, base_url: str | None = None,
                       count_url: str | None = None) -> str:
    esc = html.escape
    items = []
    for provider in sorted(by_provider, key=lambda p: PROVIDER_NAMES.get(p, p).lower()):
        states = [_state(lk, now) for lk in by_provider[provider]]
        items.append(f'<li><a href="{esc(_provider_path(provider)[len("models/"):])}">{esc(PROVIDER_NAMES.get(provider, provider))}</a>: '
                     f"{states.count('retiring')} retiring, {states.count('retired')} retired, {len(states)} models</li>")
    head = _page_head(title="AI model retirement dates by provider",
                      description="Shutdown dates and replacements of every AI model LLMShift tracks, by provider.",
                      url=f"{base_url}models/" if base_url else None, extra_css=_MODEL_CSS)
    return f"""<!doctype html>
<html lang="en">
<head>
{head}
</head>
<body>
<main>
<nav class="crumbs"><a href="../">AI model retirement calendar</a></nav>
<h1>AI model retirement dates by provider</h1>
<ul class="links">
{chr(10).join(items)}
</ul>
{_foot(updated, count_url)}
</main>{_counter(count_url)}
</body>
</html>
"""


def build_sitemap(base_url: str, paths: list[str], updated: str | None) -> bytes:
    lastmod = (reglib._date(updated) or reglib.today()).isoformat()
    urls = "".join(f"<url><loc>{xml_escape(base_url + p)}</loc><lastmod>{lastmod}</lastmod></url>\n" for p in paths)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            + urls + "</urlset>\n").encode("utf-8")


def build_rss(base_url: str, entries: dict[str, list], paths: dict, updated: str | None, *, horizon: int = 365) -> bytes:
    """RSS 2.0: retirements in the next `horizon` days and recent ones, newest announcement first."""
    now = reglib.today()
    built = reglib._date(updated) or now
    picked = [lk for lk in entries["upcoming"] if (lk.retires - now).days <= horizon] + entries["recent"]

    def rfc822(day: _dt.date) -> str:
        return format_datetime(_dt.datetime(day.year, day.month, day.day, tzinfo=_dt.timezone.utc))

    items = []
    for lk in sorted(picked, key=lambda lk: (lk.deprecated or built, lk.retires, lk.model), reverse=True):
        provider = _short_name(lk.provider)
        verb = "retires" if lk.retires >= now else "retired"
        about = f"{lk.model} ({provider}) {verb} on {lk.retires.isoformat()}." + (f" Replacement: {lk.replacement}." if lk.replacement else "")
        link = base_url + paths[(lk.provider, lk.model)]
        items.append(
            f"<item><title>{xml_escape(f'{lk.model} ({provider}) {verb} on {lk.retires.isoformat()}')}</title>"
            f"<link>{xml_escape(link)}</link>"
            f'<guid isPermaLink="false">{xml_escape(f"llmshift:{lk.provider}:{lk.model}:{lk.retires.isoformat()}")}</guid>'
            f"<pubDate>{rfc822(lk.deprecated or built)}</pubDate><category>{xml_escape(lk.provider)}</category>"
            f"<description>{xml_escape(about)}</description></item>"
        )
    feed_url = base_url + "feed.xml"
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel>\n'
            "<title>AI model retirements</title>"
            f"<link>{xml_escape(base_url)}</link><atom:link href={quoteattr(feed_url)} rel=\"self\" type=\"application/rss+xml\"/>"
            "<description>Shutdown dates of AI models and their replacements, from LLMShift's lifecycle data.</description>"
            f"<language>en</language><lastBuildDate>{rfc822(built)}</lastBuildDate>\n"
            + "\n".join(items) + "\n</channel></rss>\n").encode("utf-8")


def build_llms_txt(entries: dict[str, list], by_provider: dict[str, list], paths: dict, updated: str | None, *,
                   base_url: str | None, horizon: int = 90) -> str:
    """https://llmstxt.org: what an AI assistant needs to answer "when does model X retire?" from this site."""
    now = reglib.today()
    root = base_url or ""
    lines = [
        "# AI model retirement calendar",
        "",
        "> When AI models from OpenAI, Anthropic, Google, Mistral, Cohere, xAI and DeepSeek shut down, and what replaces them. "
        f"Built every day from the providers' deprecation pages by LLMShift. Lifecycle data updated {updated}.",
        "",
        "To find the models a codebase calls and when each retires, run `uvx llmshift scan` in the repository. "
        "LLMShift (https://github.com/DeChuks-hub/llmshift) then turns real traffic into tests and checks the replacement "
        "before the switch (`llmshift snapshot`, `llmshift compare`, `llmshift upgrade`).",
        "",
        "## Data",
        "",
        f"- [Retirement dates as JSON]({root}retirements.json): upcoming, recently retired and deprecated models, with replacements and sources",
        f"- [iCalendar feed]({root}retirements.ics): one event per shutdown date",
    ]
    if base_url:
        lines.append(f"- [RSS feed]({root}feed.xml): retirements in the next year")
    lines += ["", "## Providers", ""]
    for provider in sorted(by_provider, key=lambda p: PROVIDER_NAMES.get(p, p).lower()):
        n = len(by_provider[provider])
        lines.append(f"- [{PROVIDER_NAMES.get(provider, provider)}]({root}{_provider_path(provider)}): {n} models")
    soon = [lk for lk in entries["upcoming"] if (lk.retires - now).days <= horizon]
    lines += ["", f"## Retiring in the next {horizon} days", ""]
    lines += [f"- [{lk.model}]({root}{paths[(lk.provider, lk.model)]}) ({_short_name(lk.provider)}): "
              f"retires {lk.retires.isoformat()}" + (f", replacement {lk.replacement}" if lk.replacement else "") for lk in soon] or ["- None."]
    return "\n".join(lines) + "\n"


def build_calendar(out_dir: str | Path, *, registry: Any = None, days_back: int = 90, title: str = "AI model retirement calendar",
                   goatcounter: str | None = None, base_url: str | None = None) -> dict:
    """Write index.html, retirements.ics, retirements.json, a page per model and per provider and llms.txt; with
    `base_url`, also sitemap.xml, robots.txt and feed.xml, which need absolute addresses. Return what was written."""
    if days_back < 0:
        raise UsageError("--days-back must be zero or more")
    count_url = goatcounter_url(goatcounter) if goatcounter else None
    base = base_url_value(base_url) if base_url else None
    reg = registry or reglib.Registry.load()
    entries = calendar_entries(reg, days_back)
    out = Path(out_dir)
    if out.exists() and not out.is_dir():
        raise UsageError(f"{out} exists and is not a folder")
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "llmshift_version": __version__,
        "generated": reglib.today().isoformat(),
        "updated": reg.updated,
        "upcoming": [lk.to_dict() for lk in entries["upcoming"]],
        "recent": [lk.to_dict() for lk in entries["recent"]],
        "deprecated_without_date": [lk.to_dict() for lk in entries["undated"]],
    }
    now = reglib.today()
    every = reg.search(warn_days=0)
    paths = model_paths(every)
    by_provider: dict[str, list] = {}
    for lk in every:
        if paths.get((lk.provider, lk.model)) and all(o.model != lk.model for o in by_provider.get(lk.provider, [])):
            by_provider.setdefault(lk.provider, []).append(lk)

    def write(rel: str, content: str | bytes) -> None:
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8", "replace"))

    page = build_page(entries, reg.updated, title=title, days_back=days_back, count_url=count_url, links=paths, base_url=base)
    write("index.html", page)
    write("retirements.ics", build_ics(entries, reg.updated))
    write("retirements.json", json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    dated = sorted((lk for group in by_provider.values() for lk in group if lk.retires), key=lambda lk: lk.retires)
    for provider, lookups in by_provider.items():
        for lk in lookups:
            # Nearby retirements from the same provider: the next ones to come, else the latest ones gone.
            peers = [o for o in dated if o.provider == provider and o.model != lk.model and not o.model.startswith("/")]
            related = ([o for o in peers if o.retires >= now][:6] or peers[-6:][::-1])
            write(paths[(lk.provider, lk.model)] + "index.html",
                  build_model_page(lk, path=paths[(lk.provider, lk.model)], paths=paths, related=related, updated=reg.updated,
                                   now=now, base_url=base, count_url=count_url))
        write(_provider_path(provider) + "index.html",
              build_provider_page(provider, lookups, paths=paths, updated=reg.updated, now=now, base_url=base, count_url=count_url))
    write("models/index.html", build_models_index(by_provider, updated=reg.updated, now=now, base_url=base, count_url=count_url))
    write("llms.txt", build_llms_txt(entries, by_provider, paths, reg.updated, base_url=base))
    files = ["index.html", "retirements.ics", "retirements.json", "llms.txt"]
    if base:
        pages = [""] + ["models/"] + sorted(_provider_path(p) for p in by_provider) + sorted(
            paths[(lk.provider, lk.model)] for group in by_provider.values() for lk in group)
        write("sitemap.xml", build_sitemap(base, pages, reg.updated))
        write("robots.txt", f"User-agent: *\nAllow: /\n\nSitemap: {base}sitemap.xml\n")
        write("feed.xml", build_rss(base, entries, paths, reg.updated))
        files += ["sitemap.xml", "robots.txt", "feed.xml"]
    model_pages = sum(len(group) for group in by_provider.values())
    return {"dir": str(out), "files": files, "model_pages": model_pages, "provider_pages": len(by_provider),
            "counts": {k: len(v) for k, v in entries.items()}}


def cmd_calendar(args: argparse.Namespace, cfg: dict) -> int:
    reg = reglib.Registry.load(args.registry or [])
    written = build_calendar(args.out, registry=reg, days_back=args.days_back, title=args.title, goatcounter=args.goatcounter,
                             base_url=args.base_url)
    c = written["counts"]
    con.out(
        con.style(con.sym("pass"), "green")
        + f" wrote {', '.join(written['files'])}, {plural(written['model_pages'], 'model page')} and "
        f"{plural(written['provider_pages'], 'provider page')} to {written['dir']} "
        f"({c['upcoming']} upcoming, {c['recent']} recently retired, {c['undated']} without a date)"
    )
    if not args.base_url:
        con.out(con.style("  add --base-url https://<your site>/ for sitemap.xml, robots.txt, feed.xml and canonical links", "grey"))
    return EXIT_OK


# --------------------------------------------------------------------- cli


def register(sub: Any) -> None:
    p = sub.add_parser(
        "update",
        help="download the latest retirement data (and, with --prices, model prices)",
        description="Download the latest lifecycle data; it is validated, kept in your user data folder (LLMSHIFT_HOME), "
        "and used instead of the bundled copy while it is newer.",
    )
    p.add_argument("--prices", action="store_true", help="also download current model prices (LiteLLM's price list)")
    p.add_argument("--check", action="store_true", help="only report whether newer data is available")
    p.add_argument("--url", metavar="URL", help="where to download the lifecycle data from (default: the llmshift repository)")
    p.add_argument("--prices-url", metavar="URL", help="where to download prices from")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser(
        "digest",
        help="one message listing the models your repos use that are retired or retire soon",
        description="Scan one or more repositories and send one message (Slack, Teams, Discord, e-mail, webhook or file) "
        "listing the models they use that are retired or retire soon. Made for a weekly scheduled job.",
    )
    p.add_argument("paths", nargs="*", help="repositories or folders to scan (default: .)")
    p.add_argument("--within", type=int, metavar="DAYS", help="include models retiring within this many days (default 30)")
    p.add_argument("--alert", action="append", metavar="DEST", help="slack:<url> | teams:<url> | discord:<url> | webhook:<url> | email:<addr> | file:<path>")
    p.add_argument("--always", action="store_true", help="send even when nothing needs attention")
    p.add_argument("--dry-run", action="store_true", help="print the digest but do not send it")
    p.add_argument("--include-deprecated", action="store_true", help="also list deprecated models without a shutdown date")
    p.add_argument("--exclude", action="append", help="glob of paths to skip (repeatable)")
    p.add_argument("--fail-on", choices=["retired", "retiring", "never"], help="exit 1 when such models are found (default never)")
    p.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    p.set_defaults(func=cmd_digest)

    p = sub.add_parser(
        "calendar",
        help="build a retirement-calendar website and iCalendar feed",
        description="Write a static website (index.html), an iCalendar feed (retirements.ics) and JSON of model retirement "
        "dates, e.g. for GitHub Pages.",
    )
    p.add_argument("--out", required=True, metavar="DIR", help="folder to write index.html, retirements.ics and retirements.json")
    p.add_argument("--days-back", type=int, default=90, help="also list models retired in the last N days (default 90)")
    p.add_argument("--title", default="AI model retirement calendar")
    p.add_argument("--registry", action="append", help="extra lifecycle registry JSON (overrides bundled entries)")
    p.add_argument("--goatcounter", metavar="CODE", help="count visits with GoatCounter, which sets no cookies: your site code "
                   "(CODE.goatcounter.com) or its https://.../count URL")
    p.add_argument("--base-url", metavar="URL", help="the site's public https:// address, e.g. https://you.github.io/repo/: "
                   "adds canonical links, sitemap.xml, robots.txt and an RSS feed (feed.xml)")
    p.set_defaults(func=cmd_calendar)
