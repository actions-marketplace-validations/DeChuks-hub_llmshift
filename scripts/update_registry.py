"""Update the lifecycle data from the providers' deprecation tables (run daily by
.github/workflows/registry-update.yml).

    python scripts/update_registry.py                 # show what would change
    python scripts/update_registry.py --write         # update src/llmshift/data/models.json
    python scripts/update_registry.py --pages DIR     # read saved pages (DIR/<provider>.html)

The OpenAI, Anthropic, Gemini API and Vertex AI pages publish their schedules as
tables (HTML tables, or Markdoc tables in the Markdown that Google's AI Studio docs carry
for readers without JavaScript). Each row's model ids, shutdown date and replacement are merged conservatively:
nothing is ever removed, models that already retired keep their record, an entry changes
only where a page states a concrete date or replacement that differs, and a model is
added only when its shutdown is still ahead. A page whose tables can no longer be read
is left to the daily source watch (scripts/check_registry_sources.py), which opens an
issue for a person to review; so is everything on pages without tables.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import functools
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "llmshift" / "data" / "models.json"
STATE = ROOT / "src" / "llmshift" / "data" / "sources.json"
sys.path.insert(0, str(ROOT / "src"))

from llmshift import feed, registry  # noqa: E402

# provider, page, and the header words of its model / date / replacement columns
SOURCES = [
    ("openai", "https://developers.openai.com/api/docs/deprecations", ("model",), ("shutdown",), ("replacement", "substitute")),
    ("anthropic", "https://platform.claude.com/docs/en/about-claude/model-deprecations", ("model",), ("retirement",), ("replacement",)),
    ("google", "https://ai.google.dev/gemini-api/docs/deprecations", ("model",), ("shutdown",), ("replacement",)),
    ("google-vertex", "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions", ("model id",), ("retirement",), ("replacement", "upgrade")),
]
MIN_ROWS = 5  # fewer readable rows than this: the page layout changed; leave it to a person
MAX_CHANGES = 40  # more changes than this in one run: write nothing, ask for a review
ID_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[-._][a-z0-9]+)+\b")
_PARENS_RE = re.compile(r"\([^)]*\)")
MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_NOT_A_DATE = (
    "not sooner", "no sooner", "or later", "no shutdown", "no retirement", "not announced", "n/a", "earliest",
    "no earlier than", "not earlier than", "not before", "not until", "tentative", "estimated", "approximately", "tbd",
)


class _Tables(HTMLParser):
    """Each table's rows, with the heading that came before it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[tuple[str, list[list[str]]]] = []
        self._stack: list[tuple[str, list[list[str]]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._heading, self._in_heading = "", None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("h1", "h2", "h3", "h4"):
            self._in_heading = []
        elif tag == "table":
            self._stack.append((self._heading, []))
        elif tag == "tr" and self._stack:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif self._cell is not None:
            self._cell.append(" ")  # words in neighbouring elements stay apart

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h1", "h2", "h3", "h4") and self._in_heading is not None:
            self._heading = " ".join("".join(self._in_heading).split())
            self._in_heading = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._stack:
            self._stack[-1][1].append(self._row)
            self._row = None
        elif tag == "table" and self._stack:
            self.tables.append(self._stack.pop())
        elif self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._in_heading is not None:
            self._in_heading.append(data)
        if self._cell is not None:
            self._cell.append(data)


def parse_date(text: str) -> str | None:
    """An exact date (ISO), or None for anything vaguer ("not sooner than ...", "or later")."""
    t = (text or "").lower()
    if any(w in t for w in _NOT_A_DATE):
        return None
    m = re.search(r"\b(20\d\d)-(\d\d)-(\d\d)\b", t)
    try:
        if m:
            return _dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        m = re.search(r"\b([a-z]{3})[a-z]*\.? (\d{1,2}),? (20\d\d)\b", t)
        if m and m[1] in MONTHS:
            return _dt.date(int(m[3]), MONTHS[m[1]], int(m[2])).isoformat()
    except ValueError:  # e.g. February 30
        return None
    return None


def model_ids(cell: str) -> list[str]:
    """Model ids in a table cell; ids in parentheses win ("GPT-4o (gpt-4o-2024-08-06, ...)").
    Real ids carry a version number, which keeps out words like "fine-tuning"; an id written
    as code (`gemini-omni-flash-preview`) counts without one."""

    def is_id(word: str) -> bool:
        return any(ch.isdigit() for ch in word) or f"`{word}`" in cell

    inner = " ".join(re.findall(r"\(([^)]*)\)", cell))
    found = [i for i in ID_RE.findall(inner) if is_id(i)]
    if not found:
        found = [i for i in ID_RE.findall(cell) if is_id(i)]
    return list(dict.fromkeys(found))


def replacement_id(cell: str) -> str | None:
    """The model a replacement cell names. Ids outside parentheses come first, because a note
    in parentheses can name the retiring model itself ("gemini-3.8-flash (gemini-3.7-flash
    is no longer available ...)"); then ids inside them ("GPT-5.6 Sol (gpt-5.6-sol)")."""
    found = model_ids(_PARENS_RE.sub(" ", cell)) or model_ids(cell)
    return found[0] if found else None


def _column(header: list[str], words: tuple, exclude: tuple = ()) -> int | None:
    for i, h in enumerate(header):
        low = h.lower()
        if any(w in low for w in words) and not any(x in low for x in exclude):
            return i
    return None


@dataclass
class Row:
    provider: str
    id: str
    retires: str | None
    replacement: str | None
    deprecated: str | None
    url: str


def parse_page(html: str, provider: str, url: str, model_words: tuple, date_words: tuple, replacement_words: tuple) -> list[Row]:
    parser = _Tables()
    parser.feed(html)
    rows: list[Row] = []
    watcher = _watcher()
    for heading, table in parser.tables + watcher.markdoc_tables(watcher.noscript_document(html) or html):
        if len(table) < 2:
            continue
        header = table[0]
        mi, di = _column(header, model_words), _column(header, date_words)
        ri = _column(header, replacement_words)
        dep_i = _column(header, ("deprecat",), exclude=("model",))  # a "Deprecated" date column, not "Deprecated model"
        if mi is None or di is None:
            continue
        announced = parse_date(heading)
        for cells in table[1:]:
            if len(cells) <= max(mi, di):
                continue  # section rows ("Preview models") span the table
            repl = replacement_id(cells[ri]) if ri is not None and ri < len(cells) else None
            deprecated = parse_date(cells[dep_i]) if dep_i is not None and dep_i < len(cells) else announced
            for mid in model_ids(cells[mi]):
                rows.append(Row(provider, mid, parse_date(cells[di]), repl, deprecated, url))
    return rows


FIELD_ORDER = ("id", "provider", "status", "deprecated", "retires", "replacement", "source", "note")


def _ordered(entry: dict) -> dict:
    """Fields in the order the data uses, so an added field reads like a hand edit."""
    known = {k: entry[k] for k in FIELD_ORDER if k in entry}
    return {**known, **{k: v for k, v in entry.items() if k not in known}}


def merge(data: dict, rows: list[Row], today: str, review: list[str] | None = None) -> list[str]:
    """Apply page rows to the data in place; returns one line per change. A shutdown date
    that moved earlier, or one already past, is not applied: users would lose time they were
    promised, so it goes to `review` for a person to check against the page."""
    review = review if review is not None else []
    by_key = {(m.get("provider"), m.get("id")): m for m in data["models"]}
    latest: dict[tuple[str, str], Row] = {}
    for r in rows:  # several rows for one model (e.g. fine-tuning and inference): the latest shutdown wins
        key = (r.provider, r.id)
        if key not in latest or (r.retires or "") > (latest[key].retires or ""):
            latest[key] = r
    changes: list[str] = []
    for (provider, mid), r in sorted(latest.items()):
        entry = by_key.get((provider, mid))
        if entry is None:
            if r.retires and r.retires >= today:
                new: dict = {"id": mid, "provider": provider, "status": "deprecated"}
                if r.deprecated and r.deprecated <= r.retires:
                    new["deprecated"] = r.deprecated
                new["retires"] = r.retires
                if r.replacement and r.replacement != mid:
                    new["replacement"] = r.replacement
                new["source"] = r.url
                new["note"] = f"added by the daily data update on {today}"
                data["models"].append(new)
                changes.append(f"added {provider}/{mid}: retires {r.retires}" + (f", replacement {r.replacement}" if new.get("replacement") else ""))
            continue
        if entry.get("status") == "retired" or (entry.get("retires") or "9999") < today:
            continue  # history stays as curated
        before = len(changes)
        current = entry.get("retires")
        if r.retires and r.retires != current and (r.retires < today or (current and r.retires < current)):
            review.append(f"{provider}/{mid}: the page says it retires {r.retires}, the data says {current or 'not scheduled'} (not applied: earlier or past)")
        elif r.retires and r.retires != current:
            changes.append(f"{provider}/{mid}: retires {current or 'not scheduled'} -> {r.retires}")
            entry["retires"] = r.retires
            if entry.get("status") == "active":
                entry["status"] = "deprecated"
        if r.replacement and r.replacement != mid and r.replacement != entry.get("replacement"):
            changes.append(f"{provider}/{mid}: replacement {entry.get('replacement') or 'none'} -> {r.replacement}")
            entry["replacement"] = r.replacement
        if len(changes) > before:  # only entries that change are touched
            ordered = _ordered(entry)
            entry.clear()
            entry.update(ordered)
    return changes


def dump(data: dict) -> bytes:
    """models.json as it is laid out in the repository: one model per line."""
    head = {k: v for k, v in data.items() if k != "models"}
    lines = ["{"] + [f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in head.items()]
    models = [f"    {json.dumps(m, ensure_ascii=False)}" for m in data["models"]]
    return ("\n".join(lines) + '\n  "models": [\n' + ",\n".join(models) + "\n  ]\n}\n").encode("utf-8")


@functools.lru_cache(maxsize=None)
def _watcher():
    spec = importlib.util.spec_from_file_location("check_registry_sources", ROOT / "scripts" / "check_registry_sources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="write the updated models.json (and mark the parsed pages as seen)")
    parser.add_argument("--pages", type=Path, help="read saved pages from this folder (<provider>.html) instead of the web")
    parser.add_argument("--max-changes", type=int, default=MAX_CHANGES)
    parser.add_argument("--summary", type=Path, help="also write a Markdown summary (commit message) here")
    parser.add_argument("--today", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    today = args.today or _dt.date.today().isoformat()
    watcher = _watcher()
    data = json.loads(MODELS.read_text(encoding="utf-8"))
    before = json.loads(json.dumps(data))
    rows: list[Row] = []
    read: dict[str, str] = {}  # url -> page text, for pages whose tables were read
    lines: dict[str, list[str]] = {}  # url -> its lines about deprecations and retirements, recorded with it
    for provider, url, model_words, date_words, replacement_words in SOURCES:
        try:  # a page saved by a browser may not be UTF-8: read what can be read
            html = (args.pages / f"{provider}.html").read_bytes().decode("utf-8", "replace") if args.pages else watcher.fetch(url)
        except (OSError, RuntimeError) as exc:
            print(f"! {provider}: cannot read {url} ({exc}); skipped")
            continue
        found = parse_page(html, provider, url, model_words, date_words, replacement_words)
        if len(found) < MIN_ROWS:
            print(f"! {provider}: only {len(found)} rows readable at {url}; the layout may have changed, leaving it to the source watch")
            continue
        print(f"  {provider}: {len(found)} rows from {url}")
        rows.extend(found)
        read[url] = watcher.page_text(html)
        lines[url] = watcher.lifecycle_lines(html)
    review: list[str] = []
    changes = merge(data, rows, today, review)
    for line in review:
        print(f"! needs a person: {line}")
    if not changes:
        print("No changes: the lifecycle data matches the providers' tables." if not review else "No changes applied.")
        return 4 if review else 0
    print(f"{len(changes)} change(s):")
    for line in changes:
        print(f"  - {line}")
    if len(changes) > args.max_changes:
        print(f"Too many changes at once (more than {args.max_changes}): nothing was written. Review them and update by hand.")
        return 3
    data["updated"] = today
    problems = registry.registry_problems(data, min_models=len(before["models"]))
    if problems:
        print("The updated data is invalid, nothing was written: " + "; ".join(problems[:4]))
        return 2
    summary, details = feed.registry_changes(before, data)
    if args.summary:
        body = [f"Lifecycle data update ({today})", "", "From the providers' deprecation tables:", ""] + [f"- {c}" for c in changes]
        args.summary.write_text("\n".join(body) + "\n", encoding="utf-8")
    if review and args.summary:
        with args.summary.open("a", encoding="utf-8") as fh:
            fh.write("\nNot applied, for a person to check:\n\n" + "".join(f"- {line}\n" for line in review))
    if args.write:
        MODELS.write_bytes(dump(data))
        state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
        pages = dict(state.get("pages") or {})
        for url, text in read.items():  # these pages are now reflected in the data
            pages[url] = {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "chars": len(text)}
        STATE.write_bytes((json.dumps({"checked": today, "pages": dict(sorted(pages.items()))}, indent=2) + "\n").encode("utf-8"))
        watcher.record_lines(lines)
        shown = MODELS.relative_to(ROOT) if MODELS.is_relative_to(ROOT) else MODELS
        print(f"Wrote {shown} ({summary.get('after', len(data['models']))} models).")
    return 4 if review else 0  # 4: written, but some dates wait for a person (the workflow publishes, then fails)


if __name__ == "__main__":
    sys.exit(main())
