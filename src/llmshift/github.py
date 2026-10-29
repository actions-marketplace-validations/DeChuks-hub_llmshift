"""GitHub and git helpers for `llmshift check` and `llmshift upgrade` (REST API over
plain HTTPS, git through subprocess; no SDKs)."""

from __future__ import annotations

import os
import re
import subprocess  # nosec B404  # git only, by full path, with argument lists and no shell
import tempfile
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import UsageError
from .fileio import find_program
from .providers import http
from .providers.base import ProviderError
from .textutil import json_loads

API_URL = "https://api.github.com"
SERVER_URL = "https://github.com"
CHECK_MARKER = "<!-- llmshift:check -->"
MAX_COMMENT = 65000  # GitHub rejects bodies over 65,536 characters


class GitError(RuntimeError):
    pass


# ---------------------------------------------------------------------- git


def git(*args: str, cwd: str | os.PathLike | None = None, check: bool = True, env: dict | None = None, input: bytes | None = None) -> str:
    """Run git with an argument list (never through a shell) and return stdout."""
    exe = find_program("git")
    if exe is None:
        raise GitError("git is not installed or not on PATH")
    try:
        proc = subprocess.run(  # nosec B603  # git found by full path, arguments as a list, no shell
            [exe, *args],
            cwd=str(cwd) if cwd else None,
            input=input,
            stdin=None if input is not None else subprocess.DEVNULL,
            capture_output=True,
            env=env,
        )
    except FileNotFoundError:
        raise GitError("git is not installed or not on PATH") from None
    out = proc.stdout.decode("utf-8", "replace")
    if check and proc.returncode != 0:
        raise GitError((proc.stderr.decode("utf-8", "replace") or out).strip() or f"git {' '.join(args)} failed")
    return out


def git_bytes(*args: str, cwd: str | os.PathLike | None = None) -> bytes | None:
    exe = find_program("git")
    if exe is None:
        return None
    try:
        proc = subprocess.run([exe, *args], cwd=str(cwd) if cwd else None, stdin=subprocess.DEVNULL, capture_output=True)  # nosec B603  # as in git()
    except FileNotFoundError:
        return None
    return proc.stdout if proc.returncode == 0 else None


def repo_root(cwd: str | os.PathLike | None = None) -> Path | None:
    try:
        out = git("rev-parse", "--show-toplevel", cwd=cwd).strip()
    except GitError:
        return None
    return Path(out) if out else None


def no_repo_reason(cwd: str | os.PathLike | None = None) -> str:
    """Why `repo_root` found no repository, in git's words: not a repository, an
    unsafe owner ("dubious ownership"), or no git at all."""
    try:
        git("rev-parse", "--show-toplevel", cwd=cwd)
    except GitError as exc:
        lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
        return " ".join(lines[:3]) if lines else "not a git repository"
    return ""


def ref_exists(ref: str, cwd: str | os.PathLike | None = None) -> bool:
    try:
        git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", cwd=cwd)
        return True
    except GitError:
        return False


def merge_base(a: str, b: str = "HEAD", cwd: str | os.PathLike | None = None) -> str:
    return git("merge-base", a, b, cwd=cwd).strip()


def show(ref: str, path: str, cwd: str | os.PathLike | None = None) -> bytes | None:
    """A file's content at a commit, or None when it did not exist there."""
    return git_bytes("show", f"{ref}:{path}", cwd=cwd)


@dataclass
class FileDiff:
    status: str  # A added, M modified, D deleted, R renamed, C copied
    old_path: str | None
    new_path: str | None


def changed_files(base: str, cwd: str | os.PathLike | None = None) -> list[FileDiff]:
    """Files that differ between `base` and the working tree, plus untracked files."""
    raw = git("diff", "--name-status", "-M", "-z", base, cwd=cwd)
    parts = [p for p in raw.split("\0")]
    out: list[FileDiff] = []
    i = 0
    while i < len(parts):
        status = parts[i]
        if not status:
            i += 1
            continue
        kind = status[0]
        if kind in ("R", "C") and i + 2 < len(parts):
            out.append(FileDiff(kind, parts[i + 1] if kind == "R" else None, parts[i + 2]))
            i += 3
            continue
        if i + 1 >= len(parts):
            break
        path = parts[i + 1]
        if kind == "A":
            out.append(FileDiff("A", None, path))
        elif kind == "D":
            out.append(FileDiff("D", path, None))
        else:
            out.append(FileDiff("M", path, path))
        i += 2
    for path in git("ls-files", "--others", "--exclude-standard", "-z", cwd=cwd).split("\0"):
        if path:
            out.append(FileDiff("A", None, path))
    return out


def is_dirty(cwd: str | os.PathLike | None = None) -> bool:
    """Uncommitted changes to tracked files (new, untracked files do not count)."""
    return bool(git("status", "--porcelain", "--untracked-files=no", cwd=cwd).strip())


def current_branch(cwd: str | os.PathLike | None = None) -> str | None:
    out = git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd, check=False).strip()
    return None if out in ("", "HEAD") else out


def remote_repo(remote: str = "origin", cwd: str | os.PathLike | None = None, host: str = "github.com") -> str | None:
    """owner/name of a remote on `host` (https or ssh URL); None for any other host."""
    url = git("remote", "get-url", remote, cwd=cwd, check=False).strip()
    pattern = r"(?:^|[@/])" + re.escape(host) + r"(?::\d+)?[:/]+(?P<repo>[^/\s:]+/[^/\s]+?)(?:\.git)?/?$"
    m = re.search(pattern, url, re.I)
    return m.group("repo") if m else None


def has_identity(cwd: str | os.PathLike | None = None) -> bool:
    return bool(git("config", "user.email", cwd=cwd, check=False).strip()) and bool(git("config", "user.name", cwd=cwd, check=False).strip())


_LITERAL = {"GIT_LITERAL_PATHSPECS": "1"}  # paths are paths, never glob patterns


def tracked_modes(root: str | os.PathLike, paths: list[str], ref: str = "HEAD") -> dict[str, str]:
    """path -> git file mode (100644, 100755, 120000...) for the paths `ref` has."""
    out = git("ls-tree", "-z", ref, "--", *paths, cwd=root, env={**os.environ, **_LITERAL}) if paths else ""
    modes = {}
    for entry in out.split("\0"):
        meta, _, path = entry.partition("\t")
        if path:
            modes[path] = meta.split(" ", 1)[0]
    return modes


def modified_paths(root: str | os.PathLike, paths: list[str], ref: str = "HEAD") -> list[str]:
    """Which of the paths differ from `ref` in the working tree or the index."""
    out = git("diff", "--name-only", "-z", ref, "--", *paths, cwd=root, env={**os.environ, **_LITERAL}) if paths else ""
    return [p for p in out.split("\0") if p]


def commit_files(root: str | os.PathLike, files: dict[str, bytes], modes: dict[str, str], message: str, *, parent: str = "HEAD", config: list[str] | tuple = ()) -> str:
    """A commit of `parent` with `files` replaced, made without touching the working
    tree, the index or any branch. The content goes through git's clean filters
    (line endings, LFS) exactly as `git add` would. Returns the commit's id."""
    parent_sha = git("rev-parse", "--verify", f"{parent}^{{commit}}", cwd=root).strip()
    with tempfile.TemporaryDirectory(prefix="llmshift-index-") as tmp:
        env = {**os.environ, **_LITERAL, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git("read-tree", parent_sha, cwd=root, env=env)
        records = []
        for path, data in files.items():
            blob = git("hash-object", "-w", f"--path={path}", "--stdin", cwd=root, input=data).strip()
            records.append(f"{modes[path]} blob {blob}\t{path}\0")
        git("update-index", "-z", "--index-info", cwd=root, env=env, input="".join(records).encode("utf-8"))
        tree = git("write-tree", cwd=root, env=env).strip()
    # commit-tree does not read commit.gpgSign as `git commit` does: a repository that signs
    # its commits (or requires them signed) gets a signed one.
    sign = git(*config, "config", "--bool", "commit.gpgSign", cwd=root, check=False).strip() == "true"
    return git(*config, "commit-tree", *(["-S"] if sign else []), tree, "-p", parent_sha, "-F", "-", cwd=root, input=message.encode("utf-8")).strip()


def remote_branch(root: str | os.PathLike, branch: str, remote: str = "origin") -> str | None:
    """The commit a branch points to on the remote, or None when it does not exist there."""
    out = git("ls-remote", "--heads", remote, f"refs/heads/{branch}", cwd=root)
    for line in out.splitlines():
        sha, _, ref = line.partition("\t")
        if ref.strip() == f"refs/heads/{branch}":
            return sha.strip()
    return None


def commit_message(root: str | os.PathLike, sha: str) -> str | None:
    """A commit's message when the commit is in the local repository."""
    try:
        return git("log", "-1", "--format=%B", f"{sha}^{{commit}}", "--", cwd=root)
    except GitError:
        return None


# ------------------------------------------------------------------ context


@dataclass
class Context:
    """What a GitHub Actions run (or the environment) tells us."""

    token: str | None = None
    repo: str | None = None  # owner/name
    api_url: str = API_URL
    server_url: str = SERVER_URL  # web address, for links (GitHub Enterprise has its own)
    pr_number: int | None = None
    base_ref: str | None = None
    base_sha: str | None = None
    in_actions: bool = False
    event: dict = field(default_factory=dict)

    @property
    def owner(self) -> str | None:
        return self.repo.split("/", 1)[0] if self.repo else None

    @property
    def host(self) -> str:
        return urllib.parse.urlsplit(self.server_url).hostname or "github.com"

    def compare_url(self, base: str, head: str) -> str:
        """Where a person opens the pull request by hand."""
        return f"{self.server_url}/{self.repo}/compare/{urllib.parse.quote(base, safe='/')}...{urllib.parse.quote(head, safe='/')}?expand=1"


def context(env: dict | None = None) -> Context:
    env = os.environ if env is None else env
    event: dict = {}
    path = env.get("GITHUB_EVENT_PATH")
    if path:
        try:
            event = json_loads(Path(path).read_text(encoding="utf-8-sig")) or {}
        except (OSError, ValueError, RecursionError):
            event = {}
    if not isinstance(event, dict):
        event = {}
    pr = event.get("pull_request") if isinstance(event.get("pull_request"), dict) else {}
    number = pr.get("number")
    if number is None and isinstance(event.get("issue"), dict) and event["issue"].get("pull_request"):
        number = event["issue"].get("number")
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    return Context(
        token=env.get("GITHUB_TOKEN") or env.get("GH_TOKEN") or None,
        repo=env.get("GITHUB_REPOSITORY") or None,
        api_url=(env.get("GITHUB_API_URL") or API_URL).rstrip("/"),
        server_url=(env.get("GITHUB_SERVER_URL") or SERVER_URL).rstrip("/"),
        pr_number=int(number) if isinstance(number, int) or (isinstance(number, str) and number.isdigit()) else None,
        base_ref=env.get("GITHUB_BASE_REF") or base.get("ref") or None,
        base_sha=base.get("sha") or None,
        in_actions=env.get("GITHUB_ACTIONS") == "true",
        event=event,
    )


# ---------------------------------------------------------------------- API


def api(ctx: Context, method: str, path: str, body: Any = None, max_retries: int | None = None) -> Any:
    """One REST call. A POST creates something, so it is not retried blindly: a request
    that timed out may still have gone through (callers check before trying again)."""
    if not ctx.token:
        raise UsageError("no GitHub token: set GITHUB_TOKEN (in Actions: env GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }})")
    headers = {
        "Authorization": f"Bearer {ctx.token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    retries = max_retries if max_retries is not None else (0 if method.upper() == "POST" else 3)
    return http.request(method, f"{ctx.api_url}{path}", headers=headers, body=body, timeout=60, max_retries=retries)


def create_once(ctx: Context, path: str, body: Any, find: Any, attempts: int = 3) -> Any:
    """POST that is safe to retry: after a failure that may have gone through (a timeout
    or a 5xx), `find()` looks for what the POST creates before sending it again."""
    for attempt in range(attempts):
        try:
            return api(ctx, "POST", path, body)
        except ProviderError as exc:
            if not exc.retryable or attempt + 1 >= attempts:
                raise
            http._sleep(exc.retry_after if exc.retry_after is not None else http._retry_delay(None, attempt + 1), time.sleep)
            existing = find()
            if existing is not None:
                return existing
    raise AssertionError("unreachable")  # pragma: no cover


def _paged(ctx: Context, path: str, limit: int = 10_000) -> list:
    out: list = []
    page = 1
    sep = "&" if "?" in path else "?"
    while len(out) < limit:
        items = api(ctx, "GET", f"{path}{sep}per_page=100&page={page}")
        if not isinstance(items, list) or not items:
            break
        out.extend(items)
        if len(items) < 100:
            break
        page += 1
    return out


def _units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2  # GitHub counts UTF-16 code units: an emoji is two


def fit_markdown(body: str, limit: int = MAX_COMMENT, note: str = "_(truncated: the full report is in the job summary)_") -> str:
    """`body` cut at a line break to fit GitHub's size limit, with any code fence and
    <details> it leaves open closed again."""
    if _units(body) <= limit:
        return body
    budget = limit - _units(note) - 400  # room for the closing lines
    lo, hi = 0, len(body)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _units(body[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    cut = body[:lo]
    if "\n" in cut:
        cut = cut[: cut.rfind("\n")]
    tail = []
    fence = ""  # the code fence still open where the body was cut
    for line in cut.splitlines():
        m = re.match(r" {0,3}(`{3,}|~{3,})", line)
        if not m:
            continue
        if not fence:
            fence = m.group(1)
        elif m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not line[m.end():].strip():
            fence = ""
    if fence:
        tail.append(fence)
    tail += ["</details>"] * max(0, len(re.findall(r"<details\b", cut, re.I)) - len(re.findall(r"</details\s*>", cut, re.I)))
    return cut.rstrip() + "\n" + "".join(f"{line}\n" for line in tail) + "\n" + note + "\n"


def whoami(ctx: Context) -> str | None:
    """The login this token posts as. GitHub answers for personal tokens; GITHUB_TOKEN
    and app tokens get an error, and their comments come from a bot account."""
    try:
        data = api(ctx, "GET", "/user", max_retries=1)
    except (ProviderError, UsageError):
        return None
    return str(data.get("login")) if isinstance(data, dict) and data.get("login") else None


def _posted_by_us(comment: dict, me: str | None) -> bool:
    user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
    return user.get("login") == me if me else user.get("type") == "Bot"


def upsert_comment(ctx: Context, number: int, body: str, marker: str = CHECK_MARKER) -> dict:
    """Create the PR comment that carries `marker`, or edit it if it exists. Only a
    comment this token's account wrote is edited: anyone can paste the marker."""
    if marker not in body:
        body = f"{marker}\n{body}"
    body = fit_markdown(body)
    repo = urllib.parse.quote(ctx.repo or "", safe="/")
    me = whoami(ctx)

    def find() -> dict | None:
        for comment in _paged(ctx, f"/repos/{repo}/issues/{number}/comments"):
            if isinstance(comment, dict) and marker in str(comment.get("body") or "") and _posted_by_us(comment, me):
                return comment
        return None

    existing = find()
    if existing is not None:
        return api(ctx, "PATCH", f"/repos/{repo}/issues/comments/{existing['id']}", {"body": body})
    found = []

    def find_again() -> dict | None:
        hit = find()
        if hit is not None:
            found.append(hit)
        return hit

    result = create_once(ctx, f"/repos/{repo}/issues/{number}/comments", {"body": body}, find_again)
    if found:  # the first try went through after all: make sure it has this body
        return api(ctx, "PATCH", f"/repos/{repo}/issues/comments/{found[0]['id']}", {"body": body})
    return result


def open_pull_request(ctx: Context, *, head: str, base: str, title: str, body: str) -> dict:
    """Open a PR, or update the open one from the same branch."""
    repo = urllib.parse.quote(ctx.repo or "", safe="/")
    body = fit_markdown(body, note="_(truncated)_")
    owner = (ctx.owner or "").strip()

    def open_one() -> dict | None:
        existing = api(ctx, "GET", f"/repos/{repo}/pulls?state=open&head={urllib.parse.quote(f'{owner}:{head}')}")
        return existing[0] if isinstance(existing, list) and existing and isinstance(existing[0], dict) else None

    found: list[dict] = []

    def find_again() -> dict | None:
        hit = open_one()
        if hit is not None:
            found.append(hit)
        return hit

    try:
        result = create_once(ctx, f"/repos/{repo}/pulls", {"title": title, "head": head, "base": base, "body": body, "maintainer_can_modify": True}, find_again)
    except ProviderError as exc:
        if exc.status != 422 or find_again() is None:  # 422: a pull request from this branch is already open
            raise
        result = found[-1]
    if found:
        return api(ctx, "PATCH", f"/repos/{repo}/pulls/{found[-1]['number']}", {"title": title, "body": body})
    return result


def default_branch(ctx: Context) -> str | None:
    try:
        data = api(ctx, "GET", f"/repos/{urllib.parse.quote(ctx.repo or '', safe='/')}")
    except (ProviderError, UsageError):
        return None
    return data.get("default_branch") if isinstance(data, dict) else None
