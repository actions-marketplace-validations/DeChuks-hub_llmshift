"""Watch the pages the lifecycle registry was built from (run daily by .github/workflows/registry-watch.yml).

    python scripts/check_registry_sources.py            # report pages that changed since sources.json
    python scripts/check_registry_sources.py --update   # record the pages as they are now (after updating the data)
    python scripts/check_registry_sources.py --issue    # also open or update a GitHub issue (needs GITHUB_TOKEN)
    python scripts/check_registry_sources.py --lines    # also print every changed line about a deprecation or retirement

Each distinct source URL in src/llmshift/data/models.json is fetched, reduced to its
readable text (scripts, styles and markup removed) and hashed. A changed hash means
a provider edited a page the data came from, so a maintainer should review it,
update models.json and run --update. Pages that cannot be fetched are listed but do
not count as changes. Only the standard library is used.

The issue shows, for each changed page, the lines about deprecations, retirements and
shutdowns that changed since the last review: lines that mention one, and the rows of
tables whose header does. --update records those lines in scripts/reviewed-lines.json.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "llmshift" / "data" / "models.json"
STATE = ROOT / "src" / "llmshift" / "data" / "sources.json"
LINES = ROOT / "scripts" / "reviewed-lines.json"  # not in the wheel: only this script reads it
USER_AGENT = "llmshift-registry-watch/1.0 (+https://github.com/DeChuks-hub/llmshift)"
ISSUE_TITLE = "Lifecycle data sources changed"
LABEL = "registry-watch"
ISSUE_LIMIT = 60000  # GitHub refuses an issue body over 65,536 characters
LINES_SHOWN = 12  # added or removed lines shown per page
LINE_CHARS = 300

_DROP_RE = re.compile(r"<(script|style|noscript|template|svg|head|nav|footer)\b.*?</\1\s*>", re.I | re.S)
_MAIN_RE = re.compile(r"<(main|article)\b[^>]*>(.*)</\1\s*>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_NOSCRIPT_DOC_RE = re.compile(r"<noscript>\s*<pre>(.*?)</pre>\s*</noscript>", re.I | re.S)
_LIFECYCLE_RE = re.compile(r"deprecat|retir|shut ?down|shutting down|sunset|legacy|end[- ]of[- ]life|discontinu", re.I)
_MARKDOC_TABLE_RE = re.compile(r"\{%\s*table\b[^%]*%\}(.*?)\{%\s*/table\s*%\}", re.S)
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*(?:\{%[^%]*%\})?\s*$")
_MD_ITEM_RE = re.compile(r"^(?:[*+-]|\d+[.)]|>)\s+")
_BLOCK_TAGS = frozenset(("p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "div", "section", "dd", "dt", "pre", "blockquote",
                         "br", "ul", "ol", "summary", "details", "header", "caption", "figcaption"))
_SKIP_TAGS = frozenset(("script", "style", "noscript", "template", "svg", "head", "nav", "footer"))


def source_urls(models_path: Path = MODELS) -> list[str]:
    data = json.loads(models_path.read_text(encoding="utf-8"))
    urls = {str(e.get("source")).split("#", 1)[0] for e in data.get("models") or [] if isinstance(e, dict) and e.get("source")}
    urls |= {str(u).split("#", 1)[0] for u in (data.get("sources") or {}).values() if u}
    return sorted(u for u in urls if re.match(r"^https?://", u))


def noscript_document(raw: str) -> str:
    """The Markdown a JavaScript docs app puts in <noscript><pre> for readers without
    scripts, or "". Google's AI Studio docs, where the Gemini API pages moved in
    November 2026, are served this way."""
    found = _NOSCRIPT_DOC_RE.search(raw)
    return html.unescape(found.group(1)) if found else ""


def _main_html(raw: str) -> str:
    """The page without scripts, styles, menus and footers: its main content when marked up."""
    text = _DROP_RE.sub(" ", raw)
    main = _MAIN_RE.search(text)
    return main.group(2) if main else text


def _document(raw: str) -> str:
    """The <noscript> document of a JavaScript page, when it holds more text than the page; else ""."""
    doc = noscript_document(raw)
    if doc and len(" ".join(doc.split())) > len(" ".join(html.unescape(_TAG_RE.sub(" ", _main_html(raw))).split())):
        return doc
    return ""


def page_text(raw: str) -> str:
    """The readable text of a page: main content when marked up, without markup. A page
    that is a JavaScript app gives the document in its <noscript> when that holds more."""
    doc = _document(raw)
    if doc:
        return " ".join(doc.split())
    return " ".join(html.unescape(_TAG_RE.sub(" ", _main_html(raw))).split())


def _markdoc_rows(lines: list[str]) -> list[list[str]]:
    """The rows of a Markdoc table body: one "* cell" line per cell (a cell may wrap onto the
    next lines) and "---" between rows."""
    rows: list[list[str]] = []
    row: list[str] = []
    for line in lines:
        line = line.strip()
        if line == "---":
            if row:
                rows.append(row)
            row = []
        elif line == "*" or line.startswith("* "):
            row.append(line[1:].strip())
        elif line and row:
            row[-1] = f"{row[-1]} {line}".strip()
    if row:
        rows.append(row)
    return rows


def markdoc_tables(text: str) -> list[tuple[str, list[list[str]]]]:
    """Tables written in Markdoc ({% table %} ... {% /table %}), each with the heading before it.
    Google's AI Studio docs publish the Gemini API's schedules this way."""
    tables: list[tuple[str, list[list[str]]]] = []
    heading, start = "", 0
    for found in _MARKDOC_TABLE_RE.finditer(text):
        for line in text[start:found.start()].splitlines():
            title = _MD_HEADING_RE.match(line.strip())
            if title:
                heading = title.group(1)
        start = found.end()
        tables.append((heading, _markdoc_rows(found.group(1).splitlines())))
    return tables


class _Lines(HTMLParser):
    """A page as lines: one per paragraph, heading or list item, and one per table row with its
    cells joined by " | ". Each line says whether it belongs to a table whose header names a
    deprecation, retirement or shutdown. A table that holds only a header row (Mistral's page
    has one above the table of its rows) lends that to the next table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[tuple[str, bool]] = []
        self._text: list[str] = []
        self._skip = 0
        self._tables: list[dict] = []  # per open table: "about" (None before its first row), "rows", "header_only", "lent"
        self._lent: bool | None = None  # what the header-only table just closed says about the next one
        self._row: list[str] | None = None
        self._row_tags: set[str] = set()
        self._cell: list[str] | None = None

    def _flush(self) -> None:
        line = " ".join("".join(self._text).split())
        if line:
            self.lines.append((line, False))
        self._text = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif self._skip:
            return
        elif tag == "table":
            self._flush()
            self._tables.append({"about": None, "rows": 0, "header_only": True, "lent": self._lent})
            self._lent = None
        elif tag == "tr" and self._tables:
            self._row, self._row_tags = [], set()
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._row_tags.add(tag)
        elif self._cell is not None:
            self._cell.append(" ")  # words in neighbouring elements of a cell stay apart
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif self._skip:
            return
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._tables:
            line = " | ".join(cell for cell in self._row if cell)
            table = self._tables[-1]
            header = "th" in self._row_tags and "td" not in self._row_tags
            if line:
                if table["about"] is None:  # the first row says what the table is about...
                    lent = table["lent"]  # ...unless it is a row of data under a header-only table
                    table["about"] = lent if lent is not None and not header else bool(_LIFECYCLE_RE.search(line))
                table["rows"] += 1
                table["header_only"] = table["header_only"] and header
                self.lines.append((line, bool(table["about"])))
            self._row = None
        elif tag == "table" and self._tables:
            table = self._tables.pop()
            self._lent = table["about"] if table["rows"] == 1 and table["header_only"] else None
        elif self._cell is not None:
            self._cell.append(" ")
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._cell is not None:
            self._cell.append(data)
        elif self._row is None:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def _markdown_lines(doc: str) -> list[tuple[str, bool]]:
    """Markdown as lines: a paragraph (its wrapped lines joined), a heading or a list item each,
    and each row of a Markdoc table with its cells joined by " | "."""
    out: list[tuple[str, bool]] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append((" ".join(" ".join(para).split()), False))
            para.clear()

    source = doc.splitlines()
    i = 0
    while i < len(source):
        line = source[i].strip()
        if re.match(r"\{%\s*table\b", line):
            flush()
            end = next((j for j in range(i + 1, len(source)) if re.match(r"\{%\s*/table\s*%\}", source[j].strip())), len(source))
            rows = _markdoc_rows(source[i + 1:end])
            about = bool(rows) and bool(_LIFECYCLE_RE.search(" ".join(rows[0])))
            out += [(" | ".join(cell for cell in row if cell), about) for row in rows if any(row)]
            i = end + 1
            continue
        heading = _MD_HEADING_RE.match(line)
        if not line or line.startswith("{%"):  # a blank line, or a Markdoc tag such as {% callout %}
            flush()
        elif heading:
            flush()
            out.append((heading.group(1), False))
        elif _MD_ITEM_RE.match(line):
            flush()
            para.append(_MD_ITEM_RE.sub("", line, count=1))
        else:
            para.append(line)
        i += 1
    flush()
    return out


def lifecycle_lines(raw: str) -> list[str]:
    """The lines of a page about deprecations, retirements and shutdowns: those that mention
    one, and the rows of tables whose header does. Each once, in page order."""
    doc = _document(raw)
    if doc:
        pairs = _markdown_lines(doc)
    else:
        parser = _Lines()
        parser.feed(_main_html(raw))
        parser.close()
        pairs = parser.lines
    out: list[str] = []
    for line, in_table in pairs:
        line = line[:400]
        if (in_table or _LIFECYCLE_RE.search(line)) and line not in out:
            out.append(line)
    return out


def load_lines() -> dict[str, list[str]]:
    """The lifecycle lines of each page as of the last review."""
    try:
        return json.loads(LINES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record_lines(lines: dict[str, list[str]]) -> None:
    """Record pages' lifecycle lines as reviewed; other pages keep theirs."""
    if not lines:
        return
    merged = {**load_lines(), **lines}
    LINES.write_bytes((json.dumps(dict(sorted(merged.items())), indent=1, ensure_ascii=False) + "\n").encode("utf-8"))


def _https(url: str) -> str:
    """Only https:// addresses (or http:// to this machine, the tests' stand-in server):
    urllib would read file:// (a local file) just as happily."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in ("127.0.0.1", "localhost", "::1")):
        return url
    raise ValueError(f"not an https:// address: {url}")


def fetch(url: str, timeout: float = 30, attempts: int = 2) -> str:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(_https(url), headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310  # https only (_https)
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset, errors="replace")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(3)
    raise RuntimeError(str(getattr(last, "reason", last)))


def check(urls: list[str], state: dict) -> dict:
    pages = state.get("pages") or {}
    result: dict = {"changed": [], "new": [], "unreachable": [], "same": [], "current": {}, "lines": {}}
    for url in urls:
        try:
            raw = fetch(url)
        except RuntimeError as exc:
            result["unreachable"].append({"url": url, "error": str(exc)[:200]})
            continue
        text = page_text(raw)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        result["current"][url] = {"sha256": digest, "chars": len(text)}
        result["lines"][url] = lifecycle_lines(raw)
        known = pages.get(url)
        if known is None:
            result["new"].append(url)
        elif known.get("sha256") != digest:
            result["changed"].append(url)
        else:
            result["same"].append(url)
    return result


def save_state(result: dict, state: dict) -> None:
    pages = dict(state.get("pages") or {})
    pages.update(result["current"])  # unreachable pages keep their last known state
    data = {"checked": _dt.date.today().isoformat(), "pages": dict(sorted(pages.items()))}
    STATE.write_bytes((json.dumps(data, indent=2) + "\n").encode("utf-8"))
    record_lines(result.get("lines") or {})


def _github(method: str, path: str, body: dict | None = None) -> object:
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    req = urllib.request.Request(
        _https(f"{api}{path}"),
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310  # https only (_https)
        raw = resp.read()
    return json.loads(raw) if raw else None


def line_changes(result: dict) -> list[tuple[str, list[str] | None, list[str], list[str]]]:
    """For each changed or new page: its url, its lifecycle lines at the last review (None if
    never recorded, or if no lines were read now), and the lines added and removed since."""
    reviewed = load_lines()
    current = result.get("lines") or {}
    changes = []
    for url in sorted(result.get("changed") or []) + sorted(result.get("new") or []):
        now, before = current.get(url), reviewed.get(url)
        if now is None or before is None:
            changes.append((url, None, now or [], []))
        else:
            changes.append((url, before, [line for line in now if line not in before], [line for line in before if line not in now]))
    return changes


def _shown(lines: list[str], sign: str) -> list[str]:
    out = [f"{sign} {line[:LINE_CHARS]}".replace("```", "'''") for line in lines[:LINES_SHOWN]]
    if len(lines) > LINES_SHOWN:
        out.append(f"{sign} ... and {len(lines) - LINES_SHOWN} more")
    return out


def issue_body(result: dict) -> str:
    """The issue: for each changed page, the lines about deprecations, retirements and shutdowns
    that changed since the last review. Pages where such a line changed come first."""
    moved, elsewhere, unrecorded, marks = [], [], [], []
    for url, before, added, removed in line_changes(result):
        marks.append(url)
        if before is None and added:
            unrecorded.append("\n".join([f"#### {url} (not recorded before)", "```text", *_shown(added, " "), "```"]))
            marks += added
        elif before is None:
            elsewhere.append(f"- {url}")
        elif added or removed:
            moved.append("\n".join([f"#### {url}", "```diff", *_shown(removed, "-"), *_shown(added, "+"), "```"]))
            marks += [f"-{line}" for line in removed] + [f"+{line}" for line in added]
        else:
            elsewhere.append(f"- {url}: No line about a deprecation, retirement or shutdown changed.")
    head = [
        "The provider pages below changed since the lifecycle data was last reviewed. Each one shows the lines about",
        "deprecations, retirements and shutdowns that changed (lines that mention one, and rows of tables whose header",
        "does): `+` added, `-` removed.",
    ]
    tail = [
        "",
        "To resolve: check these lines on the pages and update `src/llmshift/data/models.json` if a model's lifecycle changed",
        "(bump `updated`). Then run Actions > registry-watch > Run workflow with \"reviewed\" ticked: it records",
        "the pages as they are now and this issue closes.",
    ]
    if result.get("unreachable"):
        tail += ["", "Could not be fetched this time: " + ", ".join(u["url"] for u in result["unreachable"])]
    marker = hashlib.sha256("\n".join(marks).encode("utf-8")).hexdigest()[:16]
    tail += ["", f"<!-- registry-watch:{marker} -->"]
    blocks = moved[:]
    if elsewhere:
        blocks += ["Changed elsewhere on the page:", *elsewhere]
    if unrecorded:
        blocks += ["Pages whose lines were not recorded before (their lines now):", *unrecorded]
    room = ISSUE_LIMIT - len("\n".join(head + tail))
    kept: list[str] = []
    for i, block in enumerate(blocks):
        if len("\n\n".join(kept + [block])) > room - 300:
            left = sum(1 for b in blocks[i:] if b.startswith(("####", "- ")))
            kept.append(f"... and {left} more pages; the issue has a size limit. "
                        "Run `python scripts/check_registry_sources.py --lines` for the rest.")
            break
        kept.append(block)
    return "\n".join(head) + "\n\n" + "\n\n".join(kept) + "\n" + "\n".join(tail)


def report_issue(result: dict) -> str:
    """Open the issue, or update it when the pages or their changed lines differ."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo or not os.environ.get("GITHUB_TOKEN"):
        raise SystemExit("--issue needs GITHUB_TOKEN and GITHUB_REPOSITORY (both are set inside GitHub Actions)")
    changed = sorted(result["changed"] + result["new"])
    body = issue_body(result)
    marker = re.search(r"registry-watch:([0-9a-f]+)", body).group(1)
    issues = _github("GET", f"/repos/{repo}/issues?state=open&labels={urllib.parse.quote(LABEL)}&per_page=100") or []  # by label: page 1 holds it, however many issues are open
    existing = next((i for i in issues if isinstance(i, dict) and i.get("title") == ISSUE_TITLE and "pull_request" not in i), None)
    if existing is None:
        created = _github("POST", f"/repos/{repo}/issues", {"title": ISSUE_TITLE, "body": body, "labels": [LABEL]})
        return f"opened issue #{created.get('number') if isinstance(created, dict) else '?'}"
    if f"registry-watch:{marker}" in (existing.get("body") or ""):
        return f"issue #{existing['number']} already lists these pages"
    moved = sum(1 for _, before, added, removed in line_changes(result) if added or removed)
    _github("PATCH", f"/repos/{repo}/issues/{existing['number']}", {"body": body})
    _github("POST", f"/repos/{repo}/issues/{existing['number']}/comments",
            {"body": f"{len(changed)} page(s) now differ from the reviewed data, {moved} of them in lines about deprecations, "
                     "retirements or shutdowns; the issue above was updated."})
    return f"updated issue #{existing['number']}"


def close_issue() -> str:
    """Close the open issue once every page matches the reviewed data again."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo or not os.environ.get("GITHUB_TOKEN"):
        raise SystemExit("--issue needs GITHUB_TOKEN and GITHUB_REPOSITORY (both are set inside GitHub Actions)")
    issues = _github("GET", f"/repos/{repo}/issues?state=open&labels={urllib.parse.quote(LABEL)}&per_page=100") or []  # by label: page 1 holds it, however many issues are open
    existing = next((i for i in issues if isinstance(i, dict) and i.get("title") == ISSUE_TITLE and "pull_request" not in i), None)
    if existing is None:
        return "no open issue"
    _github("POST", f"/repos/{repo}/issues/{existing['number']}/comments", {"body": "Every source page matches the reviewed lifecycle data again; closing."})
    _github("PATCH", f"/repos/{repo}/issues/{existing['number']}", {"state": "closed", "state_reason": "completed"})
    return f"closed issue #{existing['number']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--update", action="store_true", help="record the current state of every page in sources.json")
    parser.add_argument("--issue", action="store_true", help="open or update a GitHub issue when pages changed")
    parser.add_argument("--lines", action="store_true", help="print every changed line about a deprecation, retirement or shutdown")
    args = parser.parse_args(argv)
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    urls = source_urls()
    result = check(urls, state)
    print(f"{len(urls)} source pages: {len(result['same'])} unchanged, {len(result['changed'])} changed, {len(result['new'])} new, {len(result['unreachable'])} unreachable")
    for url in result["changed"]:
        print(f"  changed: {url}")
    for url in result["new"]:
        print(f"  not checked before: {url}")
    for item in result["unreachable"]:
        print(f"  unreachable: {item['url']} ({item['error']})")
    if args.lines:
        for url, before, added, removed in line_changes(result):
            print(f"\n{url}" + (" (not recorded before)" if before is None else ""))
            for line in removed:
                print(f"  - {line}")
            for line in added:
                print(f"  {'+' if before is not None else ' '} {line}")
            if before is not None and not added and not removed:
                print("  (no line about a deprecation, retirement or shutdown changed)")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Lifecycle data sources\n\n{len(result['changed'])} changed, {len(result['unreachable'])} unreachable of {len(urls)} pages.\n")
    if (args.update or not state.get("pages")) and result["current"]:
        save_state(result, state)
        try:
            shown = STATE.relative_to(ROOT)
        except ValueError:
            shown = STATE
        print(f"recorded {len(result['current'])} pages in {shown}")
    elif args.issue and (result["changed"] or result["new"]):
        print(report_issue(result))
    elif args.issue and not result["unreachable"]:
        print(close_issue())
    if urls and len(result["unreachable"]) == len(urls):
        print("no source page could be fetched (network problem?)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
