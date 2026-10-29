"""Watch the pages the lifecycle registry was built from (run daily by .github/workflows/registry-watch.yml).

    python scripts/check_registry_sources.py            # report pages that changed since sources.json
    python scripts/check_registry_sources.py --update   # record the pages as they are now (after updating the data)
    python scripts/check_registry_sources.py --issue    # also open or update a GitHub issue (needs GITHUB_TOKEN)

Each distinct source URL in src/llmshift/data/models.json is fetched, reduced to its
readable text (scripts, styles and markup removed) and hashed. A changed hash means
a provider edited a page the data came from, so a maintainer should review it,
update models.json and run --update. Pages that cannot be fetched are listed but do
not count as changes. Only the standard library is used.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "llmshift" / "data" / "models.json"
STATE = ROOT / "src" / "llmshift" / "data" / "sources.json"
USER_AGENT = "llmshift-registry-watch/1.0 (+https://github.com/DeChuks-hub/llmshift)"
ISSUE_TITLE = "Lifecycle data sources changed"
LABEL = "registry-watch"

_DROP_RE = re.compile(r"<(script|style|noscript|template|svg|head|nav|footer)\b.*?</\1\s*>", re.I | re.S)
_MAIN_RE = re.compile(r"<(main|article)\b[^>]*>(.*)</\1\s*>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def source_urls(models_path: Path = MODELS) -> list[str]:
    data = json.loads(models_path.read_text(encoding="utf-8"))
    urls = {str(e.get("source")).split("#", 1)[0] for e in data.get("models") or [] if isinstance(e, dict) and e.get("source")}
    urls |= {str(u).split("#", 1)[0] for u in (data.get("sources") or {}).values() if u}
    return sorted(u for u in urls if re.match(r"^https?://", u))


def page_text(raw: str) -> str:
    """The readable text of a page: main content when marked up, without markup."""
    text = _DROP_RE.sub(" ", raw)
    main = _MAIN_RE.search(text)
    if main:
        text = main.group(2)
    text = html.unescape(_TAG_RE.sub(" ", text))
    return " ".join(text.split())


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
    result: dict = {"changed": [], "new": [], "unreachable": [], "same": [], "current": {}}
    for url in urls:
        try:
            text = page_text(fetch(url))
        except RuntimeError as exc:
            result["unreachable"].append({"url": url, "error": str(exc)[:200]})
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        result["current"][url] = {"sha256": digest, "chars": len(text)}
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


def report_issue(result: dict) -> str:
    """Open the issue, or update its body when the set of changed pages differs."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo or not os.environ.get("GITHUB_TOKEN"):
        raise SystemExit("--issue needs GITHUB_TOKEN and GITHUB_REPOSITORY (both are set inside GitHub Actions)")
    changed = sorted(result["changed"] + result["new"])
    marker = hashlib.sha256("\n".join(changed).encode("utf-8")).hexdigest()[:16]
    lines = [
        "The provider pages below changed since the lifecycle data was last reviewed.",
        "",
        *[f"- {url}" for url in changed],
        "",
        "To resolve: review the pages and update `src/llmshift/data/models.json` if a model's lifecycle changed",
        "(bump `updated`). Then run Actions > registry-watch > Run workflow with \"reviewed\" ticked: it records",
        "the pages as they are now and this issue closes.",
    ]
    if result["unreachable"]:
        lines += ["", "Could not be fetched this time: " + ", ".join(u["url"] for u in result["unreachable"])]
    lines += ["", f"<!-- registry-watch:{marker} -->"]
    body = "\n".join(lines)
    issues = _github("GET", f"/repos/{repo}/issues?state=open&labels={urllib.parse.quote(LABEL)}&per_page=100") or []  # by label: page 1 holds it, however many issues are open
    existing = next((i for i in issues if isinstance(i, dict) and i.get("title") == ISSUE_TITLE and "pull_request" not in i), None)
    if existing is None:
        created = _github("POST", f"/repos/{repo}/issues", {"title": ISSUE_TITLE, "body": body, "labels": [LABEL]})
        return f"opened issue #{created.get('number') if isinstance(created, dict) else '?'}"
    if f"registry-watch:{marker}" in (existing.get("body") or ""):
        return f"issue #{existing['number']} already lists these pages"
    _github("PATCH", f"/repos/{repo}/issues/{existing['number']}", {"body": body})
    _github("POST", f"/repos/{repo}/issues/{existing['number']}/comments", {"body": f"{len(changed)} page(s) now differ from the reviewed data; the list above was updated."})
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
