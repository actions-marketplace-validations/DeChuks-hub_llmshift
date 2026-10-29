"""`llmshift check`: on a pull request, run only the tests for the prompts it edits.

The system prompt recorded in each test group is looked up in the files the change
touches. Where the base version of a file held the prompt and the new version holds
an edited one, that group's tests run with the edited prompt, so the report says
whether the model still answers the way it did before the edit.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import os
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import __version__
from . import console as con
from . import github as gh
from .errors import EXIT_FAIL, EXIT_OK, EXIT_USAGE, UsageError
from .report import _md
from .textutil import credit, md_fence, report_json, show_bidi, source_lines

MIN_PROMPT = 15  # shorter strings are not prompts
LIVES_RATIO = 0.85  # how close a string in the base version must be to the recorded prompt
EDIT_RATIO = 0.5  # how close an edited prompt must stay to its old version to count as the same prompt
MOVE_RATIO = 0.75  # ...when it also moved to another file
TEXT_SUFFIXES = {".txt", ".md", ".prompt", ".j2", ".jinja", ".jinja2", ".tmpl", ".mustache", ".hbs", ".liquid"}
MAX_FILE_BYTES = 2_000_000

_LITERAL_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|`(?:[^`\\]|\\.)*`', re.S)
_JOIN_RE = re.compile(r"\s*\+?\s*\Z")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "b": "\b", "f": "\f", "v": "\v"}


# --------------------------------------------------------------- extraction


def _unescape(body: str) -> str:
    def sub(m: re.Match) -> str:
        esc = m.group(1)
        if esc[0] in "uU" and len(esc) > 1:
            try:
                return chr(int(esc[1:], 16))
            except ValueError:
                return esc
        if esc[0] == "x" and len(esc) > 1:
            try:
                return chr(int(esc[1:], 16))
            except ValueError:
                return esc
        return _ESCAPES.get(esc, esc)

    text = re.sub(r"\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.)", sub, body, flags=re.S)
    if re.search("[\ud800-\udfff]", text):  # "😀" is one emoji written as a surrogate pair
        text = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
    return text


def _quoted_strings(text: str) -> list[str]:
    """String literals of most languages; adjacent ones joined by + or whitespace are merged."""
    out: list[str] = []
    current: str | None = None
    last_end = 0
    for m in _LITERAL_RE.finditer(text):
        value = _unescape(m.group(0)[1:-1])
        if current is not None and _JOIN_RE.match(text[last_end:m.start()]):
            current += value
        else:
            if current is not None:
                out.append(current)
            current = value
        last_end = m.end()
    if current is not None:
        out.append(current)
    return out


def _is_add(node: ast.AST) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)


def _render(node: ast.AST) -> str | None:
    """The runtime text of a Python string expression (f-string fields kept as {expr})."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                try:
                    parts.append("{" + ast.unparse(v.value) + "}")  # type: ignore[attr-defined]
                except Exception:
                    parts.append("{...}")
        return "".join(parts)
    if _is_add(node):  # "a" + "b" + "c" is ((a + b) + c): walk down the left side, not recursively
        pieces = []
        while _is_add(node):
            right = _render(node.right)  # type: ignore[attr-defined]
            if right is None:
                return None
            pieces.append(right)
            node = node.left  # type: ignore[attr-defined]
        left = _render(node)
        if left is None:
            return None
        pieces.append(left)
        return "".join(reversed(pieces))
    return None


def _python_nodes(text: str) -> list[tuple[str, int, int]] | None:
    """(string, first line, last line) for each string expression in Python source."""
    from .textutil import parse_python

    tree = parse_python(text)
    if tree is None:
        return None
    inner = {id(n.left) for n in ast.walk(tree) if _is_add(n)}  # a chain is read once, from its top
    out: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if _is_add(node) and id(node) in inner:
            continue
        try:
            found = _chain_start(node) if _is_add(node) else (_render(node), node)
        except RecursionError:  # e.g. a right-nested a + (b + (c + ...)) thousands deep
            found = None
        if found is not None and found[0] is not None:
            value, end = found
            out.append((value, getattr(node, "lineno", 0), getattr(end, "end_lineno", None) or getattr(node, "lineno", 0)))
    return out


def _chain_start(top: ast.AST) -> tuple[str, ast.AST] | None:
    """The longest start of a "a" + "b" + name chain that can be read ("ab"), and the node it
    ends at: the text before a variable part is still the prompt an edit may change."""
    spine = []
    node = top
    while _is_add(node):
        spine.append(node)
        node = node.left  # type: ignore[attr-defined]
    text = _render(node)
    if text is None:
        return None
    end = node
    for add in reversed(spine):  # from the innermost +
        right = _render(add.right)  # type: ignore[attr-defined]
        if right is None:
            break
        text, end = text + right, add
    return text, end


def _python_strings(text: str) -> list[str] | None:
    nodes = _python_nodes(text)
    return None if nodes is None else [value for value, _, _ in nodes]


def _data_strings(value: Any, out: list[str]) -> None:
    stack = [value]  # iterative: data files can nest deeper than Python's recursion limit
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            for k, v in reversed(list(item.items())):
                stack.extend((v, k))
        elif isinstance(item, list):
            stack.extend(reversed(item))


def prompt_candidates(text: str, path: str) -> list[str]:
    """Strings in a file that could be a prompt, in their runtime form where it can be known."""
    suffix = Path(path).suffix.lower()
    found: list[str] | None = None
    if suffix in (".py", ".pyi"):
        found = _python_strings(text)
    elif suffix in (".yaml", ".yml"):
        try:
            from . import yamlsafe

            found = []
            for doc in yamlsafe.load_all(text):
                _data_strings(doc, found)
        except Exception:
            found = None
    elif suffix == ".json":
        from .textutil import json_loads

        try:
            found = []
            _data_strings(json_loads(text), found)
        except (ValueError, RecursionError):
            found = None
    elif suffix in TEXT_SUFFIXES:
        found = [text]
    if found is None:
        found = [text, *_quoted_strings(text)]
    seen: dict[str, None] = {}
    for s in found:
        if len(s.strip()) >= MIN_PROMPT:
            seen.setdefault(s, None)
    return list(seen)


# --------------------------------------------------------------- matching


def norm(text: str) -> str:
    return " ".join((text or "").split())


def similarity(a: str, b: str, floor: float = 0.0) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    short, long_ = sorted((len(a), len(b)))
    if floor and 2 * short / (short + long_) < floor:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if floor and sm.quick_ratio() < floor:
        return 0.0
    return sm.ratio()


def best_match(candidates: list[str], target: str, floor: float) -> str | None:
    best, best_score = None, floor
    for c in candidates:
        score = similarity(c, target, floor)
        if score > best_score or (best is None and score >= floor and score > 0):
            best, best_score = c, score
    return best


def _spans(text: str, path: str) -> list[tuple[str, int, int]]:
    """Prompt candidates with the lines they occupy (1-based, inclusive)."""
    if Path(path).suffix.lower() in TEXT_SUFFIXES:
        return [(text, 1, text.count("\n") + 1)] if len(text.strip()) >= MIN_PROMPT else []
    nodes = _python_nodes(text) if Path(path).suffix.lower() in (".py", ".pyi") else None
    if nodes is None:  # other languages and data files: place each string by its first line
        lines = source_lines(text)
        nodes = []
        for value in prompt_candidates(text, path):
            probe = next((ln.strip() for ln in value.splitlines() if ln.strip()), "")[:40]
            if len(probe) < 8:
                continue
            at = next((i for i, line in enumerate(lines, 1) if probe in line), None)
            if at is not None:
                nodes.append((value, at, at + value.count("\n")))
    return [n for n in nodes if len(n[0].strip()) >= MIN_PROMPT]


def _python_slots(text: str) -> dict[str, list[str]] | None:
    """Strings by the place that holds them: `NAME = "..."` assignments (slot "=NAME")
    and keyword arguments (slot "call(keyword=)")."""
    from .textutil import parse_python

    tree = parse_python(text)
    if tree is None:
        return None
    slots: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = _render(node.value)
            for t in targets:
                if value is not None and isinstance(t, ast.Name):
                    slots.setdefault(f"={t.id}", []).append(value)
        elif isinstance(node, ast.Call):
            try:
                func = ast.unparse(node.func)  # type: ignore[attr-defined]
            except Exception:  # nosec B112  # a call it cannot render is simply not a slot
                continue
            for kw in node.keywords:
                value = _render(kw.value) if kw.arg else None
                if value is not None:
                    slots.setdefault(f"{func}({kw.arg}=)", []).append(value)
    return slots


def short_rewrite(base_text: str, base_path: str, new_text: str, new_path: str, base_prompt: str, exclude: set[str]) -> str | None:
    """A prompt rewritten to a few words ("Be brief."), which is too short to be found
    as a prompt anywhere: the new text of a whole prompt file, or the string that now
    fills the same assignment or keyword argument the prompt filled."""
    if Path(new_path).suffix.lower() in TEXT_SUFFIXES and Path(base_path).suffix.lower() in TEXT_SUFFIXES:
        return new_text if new_text.strip() and norm(new_text) != norm(base_prompt) else None
    if Path(base_path).suffix.lower() not in (".py", ".pyi") or Path(new_path).suffix.lower() not in (".py", ".pyi"):
        return None
    old, new = _python_slots(base_text), _python_slots(new_text)
    if old is None or new is None:
        return None
    found = set()
    for slot, values in old.items():
        if any(norm(v) == norm(base_prompt) for v in values) and len(new.get(slot, [])) == 1:
            value = new[slot][0]
            if value.strip() and norm(value) != norm(base_prompt) and norm(value) not in exclude:
                found.add(value)
    return found.pop() if len(found) == 1 else None


def positional_edit(base_text: str, base_path: str, new_text: str, new_path: str, base_prompt: str, exclude: set[str]) -> str | None:
    """The string that took a prompt's place, however different its wording: the prompt's
    lines in the base file are mapped through a line diff into the new file. None when
    nothing new sits there, or more than one string could be it."""
    where = next(((a, b) for value, a, b in _spans(base_text, base_path) if norm(value) == norm(base_prompt)), None)
    if where is None:
        return None
    first, last = where[0] - 1, where[1]  # 0-based, half-open
    lo = hi = None
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, source_lines(base_text), source_lines(new_text), autojunk=False).get_opcodes():
        if tag == "insert":
            overlaps = first <= i1 <= last
            start, end = j1, j2
        else:
            overlaps = i1 < last and i2 > first
            start, end = (j1 + max(first - i1, 0), j1 + min(last, i2) - i1) if tag == "equal" else (j1, j2)
        if overlaps and end > start:
            lo, hi = (start, end) if lo is None else (min(lo, start), max(hi, end))
    if lo is None:
        return None
    before = {norm(value) for value, _, _ in _spans(base_text, base_path)}
    scored = []
    for value, a, b in _spans(new_text, new_path):
        overlap = min(b, hi) - max(a - 1, lo)
        if overlap > 0 and norm(value) not in before and norm(value) not in exclude:
            scored.append((overlap, len(value), value))
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][:2] == scored[1][:2] and norm(scored[0][2]) != norm(scored[1][2]):
        return None  # two equally good candidates: do not guess
    return scored[0][2]


_SHAPES: list[Callable[[str], str]] = [
    lambda s: s,
    str.strip,
    textwrap.dedent,
    lambda s: textwrap.dedent(s).strip(),
    lambda s: textwrap.dedent(s).strip("\n"),
    norm,
]


def carry_edit(recorded: str, base_prompt: str, new_prompt: str) -> tuple[str, bool]:
    """The edited prompt as the model will receive it, shaped like the recorded one
    (same dedent/strip). Returns (prompt, approximate)."""
    for shape in _SHAPES:
        if shape(base_prompt) == recorded:
            return shape(new_prompt), False
    if base_prompt and base_prompt in recorded:  # the code prompt is part of a longer runtime prompt
        return recorded.replace(base_prompt, new_prompt, 1), False
    if norm(base_prompt) == norm(recorded):
        return textwrap.dedent(new_prompt).strip(), False
    return textwrap.dedent(new_prompt).strip(), True


@dataclass
class PromptChange:
    group: str
    old: str
    new: str | None
    files: list[str]
    status: str  # changed | removed | moved
    note: str = ""
    approximate: bool = False
    result: Any = None  # CompareResult

    @property
    def ran(self) -> bool:
        return self.result is not None

    @property
    def unchecked(self) -> int:
        """Cases whose check could not run (an invalid regex, an unreachable judge)."""
        return int(self.result.summary.get("unchecked") or 0) if self.result is not None else 0

    def passed(self, min_pass_rate: float | None) -> bool:
        if self.result is None:
            return True
        s = self.result.summary
        if not s.get("evaluated"):
            return self.result.inconclusive  # no case could be checked: no verdict, not a failure
        return min_pass_rate is None or s.get("pass_rate", 0.0) + 1e-9 >= min_pass_rate

    def to_dict(self, min_pass_rate: float | None = None) -> dict:
        d = {"group": self.group, "status": self.status, "files": self.files, "old_prompt": self.old, "new_prompt": self.new}
        if self.note:
            d["note"] = self.note
        if self.result is not None:
            d["summary"] = self.result.summary
            d["passed"] = self.passed(min_pass_rate)
        return d


def find_prompt_changes(
    prompts: dict[str, str],
    diffs: list[gh.FileDiff],
    read_base: Callable[[str], str | None] | None,
    read_current: Callable[[str], str | None],
) -> list[PromptChange]:
    """Which recorded prompts (group -> text) a set of file changes edits. Without
    `read_base` (no history), a file holds a prompt when it holds something close to it."""
    base_cache: dict[str, list[str]] = {}
    current_cache: dict[str, list[str]] = {}
    base_raw: dict[str, str | None] = {}
    current_raw: dict[str, str | None] = {}

    def base_text(path: str) -> str | None:
        if path not in base_raw:
            base_raw[path] = read_base(path) if read_base else None
        return base_raw[path]

    def current_text(path: str) -> str | None:
        if path not in current_raw:
            current_raw[path] = read_current(path)
        return current_raw[path]

    def base_candidates(path: str) -> list[str]:
        if path not in base_cache:
            text = base_text(path)
            base_cache[path] = prompt_candidates(text, path) if text is not None else []
        return base_cache[path]

    def current_candidates(path: str) -> list[str]:
        if path not in current_cache:
            text = current_text(path)
            current_cache[path] = prompt_candidates(text, path) if text is not None else []
        return current_cache[path]

    changes: list[PromptChange] = []
    for group, recorded in prompts.items():
        if len(recorded.strip()) < MIN_PROMPT:
            continue
        lived: list[tuple[gh.FileDiff, str]] = []
        for d in diffs:
            if read_base is None:
                if d.new_path and best_match(current_candidates(d.new_path), recorded, EDIT_RATIO) is not None:
                    lived.append((d, recorded))
                continue
            if d.old_path is None:
                continue
            exact = next((c for c in base_candidates(d.old_path) if norm(c) == norm(recorded)), None)
            found = exact or best_match(base_candidates(d.old_path), recorded, LIVES_RATIO)
            if found is not None:
                lived.append((d, found))
        if not lived:
            continue
        variants: dict[str, list[str]] = {}
        rewritten: set[str] = set()
        approximate = False
        removed: list[str] = []
        moved_to: list[str] = []
        unchanged = False
        homes = {d.old_path for d, _ in lived}
        others = {norm(p) for g, p in prompts.items() if g != group}

        def mine(cands: list[str]) -> list[str]:  # never mistake another group's prompt for this one
            return [c for c in cands if norm(c) not in others]

        for d, base_prompt in lived:
            now = current_candidates(d.new_path) if d.new_path else []
            if any(norm(c) == norm(base_prompt) for c in now):
                unchanged = True
                continue
            edited = best_match(mine(now), base_prompt, EDIT_RATIO)
            source = d.new_path
            positional = False
            if edited is None and d.new_path:  # rewritten so much that it reads like another prompt?
                old_text, new_text = base_text(d.old_path), current_text(d.new_path)
                if old_text is not None and new_text is not None:
                    edited = positional_edit(old_text, d.old_path, new_text, d.new_path, base_prompt, others)
                    if edited is None:
                        edited = short_rewrite(old_text, d.old_path, new_text, d.new_path, base_prompt, others)
                    positional = edited is not None
            if edited is None:  # gone from this file: moved elsewhere in the change?
                for other in diffs:
                    if not other.new_path or other.new_path == d.new_path or other.new_path in homes:
                        continue
                    cands = current_candidates(other.new_path)
                    if any(norm(c) == norm(base_prompt) for c in cands):
                        moved_to.append(other.new_path)
                        edited = None
                        break
                    edited = best_match(mine(cands), base_prompt, MOVE_RATIO)
                    if edited is not None:
                        source = other.new_path
                        break
                if edited is None:
                    if not moved_to:
                        removed.append(d.old_path or "")
                    continue
            new_prompt, approx = carry_edit(recorded, base_prompt, edited)
            approximate = approximate or approx
            if norm(new_prompt) == norm(recorded):
                unchanged = True
                continue
            if positional:
                rewritten.add(new_prompt)
            variants.setdefault(new_prompt, [])
            if source and source not in variants[new_prompt]:
                variants[new_prompt].append(source)
        for new_prompt, files in variants.items():
            note = ""
            if len(variants) > 1:
                note = "this prompt was edited differently in several places; each version is tested"
            elif any(f not in homes for f in files):
                note = f"moved from {', '.join(sorted(h for h in homes if h))}"
            elif new_prompt in rewritten:
                note = "rewritten: matched to the old prompt by its place in the file"
            changes.append(PromptChange(group, recorded, new_prompt, files, "changed", note, approximate))
        if not variants:
            if moved_to and not removed:
                changes.append(PromptChange(group, recorded, None, moved_to, "moved", "moved unchanged; nothing to test"))
            elif removed and not unchanged and not moved_to:
                changes.append(PromptChange(group, recorded, None, removed, "removed", "the prompt is no longer in this file; its tests were not run"))
    return changes


# ------------------------------------------------------------------- running


@dataclass
class CheckRun:
    model: str
    base: str | None
    changes: list[PromptChange] = field(default_factory=list)
    min_pass_rate: float | None = None
    files: list[str] = field(default_factory=list)
    ran_all: bool = False

    @property
    def tested(self) -> list[PromptChange]:
        return [c for c in self.changes if c.ran]

    @property
    def failing(self) -> list[PromptChange]:
        return [c for c in self.tested if not c.passed(self.min_pass_rate)]

    @property
    def passed(self) -> bool:
        return not self.failing

    @property
    def inconclusive(self) -> bool:
        """Nothing failed, but some cases could not be checked."""
        return self.passed and any(c.unchecked for c in self.tested)

    def to_dict(self) -> dict:
        return {
            "llmshift_version": __version__,
            "model": self.model,
            "base": self.base,
            "min_pass_rate": self.min_pass_rate,
            "files": self.files,
            "passed": self.passed,
            "inconclusive": self.inconclusive,  # passed, but some cases could not be checked
            "changes": [c.to_dict(self.min_pass_rate) for c in self.changes],
        }


def _is_source(path: str) -> bool:
    """Prompts live in code and prompt files, not in logs, test data or tool folders."""
    from .scanner import SKIP_DIRS

    p = Path(path)
    return bool(path) and not any(part in SKIP_DIRS for part in p.parts[:-1]) and p.suffix.lower() not in (".jsonl", ".ndjson", ".log")


def _decode(data: bytes | None) -> str | None:
    if data is None or len(data) > MAX_FILE_BYTES:
        return None
    from .scanner import decode_text

    return decode_text(data)


def resolve_base(root: Path, base: str | None, ctx: gh.Context) -> tuple[str, str]:
    """(commit, label) to compare against: the merge base with the PR's base branch."""
    if base:
        refs = [base]
    elif ctx.in_actions and (ctx.base_sha or ctx.base_ref):
        refs = [r for r in (ctx.base_sha, f"origin/{ctx.base_ref}" if ctx.base_ref else None, ctx.base_ref) if r]
    else:
        refs = ["origin/main", "origin/master", "main", "master"]
    shallow = "\n  In GitHub Actions, check out the full history: actions/checkout with `fetch-depth: 0`."
    for ref in refs:
        if not gh.ref_exists(ref, cwd=root):
            continue
        try:
            return gh.merge_base(ref, "HEAD", cwd=root), ref
        except gh.GitError:
            raise UsageError(f"cannot find where HEAD branched off {ref} (shallow clone?).{shallow}") from None
    raise UsageError(f"base {refs[0]!r} not found; pass --base REF.{shallow}")


def run_check(
    cases: list,
    model: Any,
    *,
    root: str | os.PathLike | None = None,
    base: str | None = None,
    files: list[str] | None = None,
    all_groups: bool = False,
    config: dict | None = None,
    judge: Any = None,
    concurrency: int = 4,
    cache: Any = None,
    allow_python: bool = False,
    severity: dict | None = None,
    min_pass_rate: float | None = None,
    ctx: gh.Context | None = None,
    show_progress: bool | None = None,
) -> CheckRun:
    """Find the prompts edited since `base` and test them on `model`."""
    from .compare import run_compare
    from .providers import resolve
    from .testset import group_key

    ctx = ctx or gh.context()
    cwd = Path(root or os.getcwd()).resolve()
    git_root = gh.repo_root(cwd)
    provider = resolve(model, config or {}) if isinstance(model, str) else model
    base_commit, base_label = None, None
    if git_root is not None:
        try:
            base_commit, base_label = resolve_base(git_root, base, ctx)
        except UsageError:
            if not files:
                raise
    elif not files:
        reason = gh.no_repo_reason(cwd) or "not a git repository"
        if "not a git repository" in reason.lower():  # rather than unsafe ownership or no git at all
            reason = "not a git repository"
        raise UsageError(f"{reason}: run llmshift check inside your checkout, or pass --files FILE...")
    top = git_root or cwd

    if files:
        diffs = []
        for f in files:
            p = Path(f)
            p = p if p.is_absolute() else cwd / p
            try:
                rel = p.resolve().relative_to(top).as_posix()
            except ValueError:
                raise UsageError(f"{f} is outside the repository") from None
            if p.is_dir():
                raise UsageError(f"--files takes files, and {f} is a directory")
            now = p.is_file()
            before = base_commit is None or gh.show(base_commit, rel, cwd=top) is not None
            if not now and not (base_commit and before):
                raise UsageError(f"{f} not found" + (f" (nor in {base_label})" if base_commit else ""))
            diffs.append(gh.FileDiff("M" if now and before else ("A" if now else "D"), rel if before else None, rel if now else None))
    else:
        diffs = gh.changed_files(base_commit, cwd=top) if base_commit else []
    diffs = [d for d in diffs if _is_source(d.new_path or d.old_path or "")]

    def read_base(path: str) -> str | None:
        return _decode(gh.show(base_commit, path, cwd=top)) if base_commit else None

    def read_current(path: str) -> str | None:
        p = top / path
        try:
            return _decode(p.read_bytes()) if p.is_file() else None
        except OSError:
            return None

    groups: dict[str, list] = {}
    for case in cases:
        groups.setdefault(group_key(case), []).append(case)
    # A hand-written group can hold cases with different system prompts: each prompt is
    # tracked on its own, so editing the less common one is noticed too.
    units: dict[str, tuple[str, list]] = {}  # name -> (group, its cases with one prompt)
    prompts: dict[str, str] = {}
    for group, members in groups.items():
        by_prompt = Counter(c.system_prompt for c in members).most_common()
        for i, (prompt, _) in enumerate(by_prompt):
            name = group if len(by_prompt) == 1 else f"{group} (prompt {i + 1} of {len(by_prompt)})"
            units[name] = (group, [c for c in members if c.system_prompt == prompt])
            prompts[name] = prompt
    changes = find_prompt_changes(prompts, diffs, read_base if base_commit else None, read_current)
    run = CheckRun(provider.spec, base_label, changes, min_pass_rate, sorted({d.new_path or d.old_path or "" for d in diffs}), all_groups)
    common = dict(config=config, judge=judge, concurrency=concurrency, cache=cache, allow_python=allow_python, severity=severity, show_progress=show_progress, min_pass_rate=min_pass_rate)
    for change in changes:
        if change.status != "changed":
            continue
        group, members = units[change.group]
        change.result = run_compare(members, provider, patches={group: {"system": change.new}}, label="check", **common)
    if all_groups:
        edited = {c.group for c in changes if c.status == "changed"}
        for name, (_group, members) in sorted(units.items()):
            if name in edited:
                continue
            change = PromptChange(name, prompts[name], None, [], "unchanged", "prompt unchanged; run because of --all")
            change.result = run_compare(members, provider, label="check", **common)
            changes.append(change)
    return run


# ------------------------------------------------------------------ reporting


def _prompt_diff(old: str, new: str, width: int = 100) -> str:
    def lines(text: str) -> list[str]:
        out = []
        for para in text.splitlines() or [""]:
            out.extend(textwrap.wrap(para, width) or [""])
        return out

    diff = difflib.unified_diff(lines(old), lines(new), "old prompt", "new prompt", lineterm="", n=1)
    return "\n".join(list(diff)[2:])


def _patterns(result: Any, n: int = 3) -> list[str]:
    out = []
    by_cat = (result.summary or {}).get("by_category") or {}
    for cat, info in sorted(by_cat.items(), key=lambda kv: -(kv[1].get("fail", 0) + kv[1].get("warn", 0))):
        for pattern, count in list((info.get("patterns") or {}).items())[:2]:
            out.append(f"{cat}: {pattern} ({count})")
    return out[:n]


def _rate(change: PromptChange) -> str:
    s = change.result.summary
    if s.get("total", 0) <= s.get("unchecked", 0):  # no case could be checked: no pass rate
        return "n/a"
    return f"{s.get('pass_rate', 0.0):.0%}"


def comment_marker(key: str | None = None) -> str:
    """The hidden first line that finds this check's comment again. Each key has its own
    comment, so two check jobs on one pull request (two test sets) do not overwrite each other."""
    safe = re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9._/-]+", "-", key or "")).strip("-")[:100]  # no "--" inside an HTML comment
    return f"<!-- llmshift:check key={safe} -->" if safe else gh.CHECK_MARKER


def check_markdown(run: CheckRun, marker: str = gh.CHECK_MARKER) -> str:
    threshold = f"{run.min_pass_rate:.0%}" if run.min_pass_rate is not None else "the threshold"
    tested = run.tested
    lines = [marker]
    if not run.changes:
        lines.append("## LLMShift prompt check: no prompt changes")
        lines.append("")
        lines.append("This change does not edit any prompt covered by the test set.")
        lines += ["", credit()]
        return show_bidi("\n".join(lines) + "\n")
    what = "prompt(s)" if run.ran_all else "prompt change(s)"  # --all also runs the prompts this change left alone
    if run.failing:
        lines.append(f"## LLMShift prompt check: {len(run.failing)} of {len(tested)} {what} below {threshold}")
    else:
        lines.append(f"## LLMShift prompt check: {len(tested)} {what} keep the recorded behaviour")
    lines.append("")
    lines.append(f"Tested on `{run.model}` against the recorded answers" + (f" (base `{run.base}`)." if run.base else "."))
    lines.append("")
    if tested:
        lines.append("| Prompt | Where | Cases | Pass rate | Changed | Result |")
        lines.append("|---|---|---:|---:|---:|---|")
        for c in tested:
            s = c.result.summary
            ok = c.passed(run.min_pass_rate)
            verdict = "**fail**" if not ok else f"**{c.unchecked} not checked**" if c.unchecked else "pass"
            lines.append(f"| `{c.group}` | {', '.join(f'`{f}`' for f in c.files) or '-'} | {s.get('total', 0)} | {_rate(c)} | {s.get('warned', 0)} | {verdict} |")
        lines.append("")
    moved = [c for c in run.changes if c.status in ("removed", "moved")]
    for c in moved:
        lines.append(f"- `{c.group}`: {c.status} ({', '.join(c.files)}). {c.note}")
    if moved:
        lines.append("")
    for c in tested:
        if c.new is None:
            continue
        lines.append(f"<details><summary><code>{c.group}</code>: the edit and what changed</summary>")
        lines.append("")
        if c.note:
            lines.append(f"_{c.note}_")
            lines.append("")
        if c.approximate:
            lines.append("_The code prompt differs from the recorded one (a template?); the edit was applied to the recorded prompt as closely as possible._")
            lines.append("")
        lines.append(md_fence(_prompt_diff(c.old, c.new)[:6000], "diff"))
        patterns = _patterns(c.result)
        if patterns:
            lines.append("")
            lines.append("What changed in the answers: " + _md("; ".join(patterns)))
        bad = [r for r in c.result.cases if r.status in ("fail", "error")][:3]
        for r in bad:
            why = r.error or next((x.message for x in r.checks if x.status in ("fail", "error")), "")
            lines.append(f"- `{r.id}`: {_md(why[:160])}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    lines += ["", credit()]
    return show_bidi("\n".join(lines).rstrip() + "\n")


def check_terminal(run: CheckRun) -> str:
    lines = [con.style("LLMShift check", "bold") + f"  model {run.model}" + (f"  base {run.base}" if run.base else "")]
    if not run.changes:
        lines.append("  No prompt changes found; nothing to check.")
        return "\n".join(lines)
    for c in run.changes:
        where = ", ".join(c.files)
        if c.result is None:
            lines.append(con.style(f"  {con.sym('info')} {c.group}  {c.status}: {where}  {c.note}".rstrip(), "grey"))
            continue
        s = c.result.summary
        ok = c.passed(run.min_pass_rate)
        mark = con.style(con.sym("fail"), "red") if not ok else con.style(con.sym("warn"), "yellow") if c.unchecked else con.style(con.sym("pass"), "green")
        extra = "" if ok or run.min_pass_rate is None else f" (below {run.min_pass_rate:.0%})"
        if c.unchecked:
            extra += f" ({c.unchecked} could not be checked: a check could not run)"
        changed = f", {s['warned']} changed" if s.get("warned") else ""
        lines.append(f"  {mark} {c.group}  {where or '-'}  {s.get('total', 0)} case(s)  pass rate {_rate(c)}{changed}{extra}")
        if c.note:
            lines.append(con.style(f"      {c.note}", "grey"))
        for p in _patterns(c.result):
            lines.append(con.style(f"      {p}", "grey"))
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI


def register(sub: Any) -> None:
    from . import cli

    p = sub.add_parser(
        "check",
        help="test only the prompts a change edits (for pull requests)",
        description="Find the system prompts this change edits (against --base) and run only their test cases, with the "
        "edited text, on the model in production. In a GitHub pull request, the result is posted as one comment.",
    )
    p.add_argument("--model", metavar="MODEL", type=cli.model_arg, help="model to test with (default: baseline, then candidate, from llmshift.yaml)")
    p.add_argument("--base", metavar="REF", help="git ref to compare with (default: the pull request's base in GitHub Actions, else origin/main)")
    p.add_argument("--files", nargs="+", metavar="FILE", help="only look at these files")
    p.add_argument("--all", action="store_true", help="also run the groups whose prompts did not change")
    p.add_argument("--min-pass-rate", type=cli.rate_arg, help="fail below this pass rate (default from llmshift.yaml: 0.95)")
    p.add_argument("--markdown", nargs="?", const="", metavar="FILE", help="also write the Markdown report (default: in the reports directory)")
    p.add_argument("--json", metavar="FILE", help="write the results as JSON")
    p.add_argument("--comment", action="store_true", help="post or update a comment on the pull request (automatic in GitHub Actions)")
    p.add_argument("--no-comment", action="store_true", help="never comment on the pull request")
    p.add_argument("--pr", type=int, metavar="NUMBER", help="pull request to comment on (implies --comment; default: from the GitHub event)")
    p.add_argument("--comment-key", metavar="KEY", help="keeps one comment per key, for several check jobs on one pull request (default: the test file, when it is not .llmshift/tests.jsonl)")
    cli._add_run_options(p)
    p.set_defaults(func=cmd_check)


def cmd_check(args: argparse.Namespace, cfg: dict) -> int:
    from . import cli

    cases, _path = cli._load_cases(cfg, args)
    model = args.model or cfg.get("baseline") or cfg.get("candidate")
    if not model:
        raise UsageError("which model should the prompts be tested on? Pass --model provider:model or set baseline in llmshift.yaml")
    min_rate = args.min_pass_rate if args.min_pass_rate is not None else cfg.get("min_pass_rate")
    ctx = gh.context()
    con.out(con.style(f"Looking for edited prompts in this change ({len(cases)} test case(s))...", "grey"), err=True)
    run = run_check(
        cases,
        model,
        base=args.base,
        files=args.files,
        all_groups=args.all,
        config=cfg,
        judge=args.judge or cfg.get("judge"),
        concurrency=args.concurrency or int(cfg.get("concurrency", 4)),
        cache=cli._cache(cfg, args),
        allow_python=args.allow_python or bool(cfg.get("allow_python_checks")),
        severity=cli._severity(cfg, args),
        min_pass_rate=float(min_rate) if min_rate is not None else None,
        ctx=ctx,
    )
    con.out(check_terminal(run))
    key = args.comment_key
    if key is None:
        root = gh.repo_root(os.getcwd())
        try:
            rel = Path(_path).resolve().relative_to(root).as_posix() if root else None
        except ValueError:
            rel = None
        key = None if rel in (None, ".llmshift/tests.jsonl") else rel
    marker = comment_marker(key)
    md = check_markdown(run, marker)
    if args.markdown is not None:
        target = args.markdown or Path(cfg.get("reports_dir") or ".llmshift/reports") / f"check-{cli._timestamp()}.md"
        cli._write(target, md)
        con.out(con.style(f"  report: {cli._rel(target)}", "grey"))
    if args.json:
        cli._write(args.json, report_json(run.to_dict()) + "\n")
    cli._step_summary(md)
    tested = run.tested
    if tested:
        rates = [c.result.summary.get("pass_rate", 0.0) for c in tested]
        cli._github_output("pass_rate", round(min(rates), 4))
    number = args.pr or ctx.pr_number
    if (args.comment or args.pr is not None or (ctx.in_actions and ctx.pr_number)) and not args.no_comment:
        if not ctx.repo:
            ctx.repo = gh.remote_repo(cwd=os.getcwd(), host=ctx.host)
        if not (ctx.token and ctx.repo and number):
            missing = [what for what, ok in (("GITHUB_TOKEN", ctx.token), ("the repository (GITHUB_REPOSITORY=owner/name)", ctx.repo), ("a PR number (--pr)", number)) if not ok]
            con.warn(f"not commenting on the pull request: needs {', '.join(missing)}")
        else:
            try:
                gh.upsert_comment(ctx, int(number), md, marker=marker)
                con.out(con.style(f"  commented on pull request #{number}", "grey"))
            except Exception as exc:  # e.g. read-only tokens on pull requests from forks
                con.warn(f"could not comment on pull request #{number}: {exc}")
    if not run.passed:
        return EXIT_FAIL
    return EXIT_USAGE if run.inconclusive else EXIT_OK
