"""`llmshift fix`: suggest prompt changes until the new model matches the old one.

For every prompt template with regressions:
1. derive targeted instructions from the failures (code fences, schema, labels,
   tone markers, length, structure, language, exact figures, tool use),
2. optionally add a worked example taken from cases *not* used for scoring,
3. optionally ask an optimizer model to rewrite the prompt,
then run each candidate prompt on the new model and keep the best one.
Nothing is changed in your code: you get the new prompt, a diff and where the
prompt lives in your repository.
"""

from __future__ import annotations

import datetime as _dt
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from . import tone as tonelib
from .compare import CaseResult, CompareResult, run_compare
from .jsonschema_lite import describe as describe_schema
from .providers import resolve
from .providers.base import ProviderError
from .stats import stored_rate
from .testset import TestCase, group_key, groups_of, unique_ids
from .textutil import credit, language_name, md_fence, normalize, plural, shorten, show_bidi, unified_diff, word_count

HEADER = "Output requirements:"


@dataclass
class GroupFix:
    group: str
    original_prompt: str
    suggested_prompt: str | None = None
    added: list[str] = field(default_factory=list)
    strategy: str = "none"
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    remaining: list[str] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    example_ids: list[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.suggested_prompt is not None and self.after.get("score", 0) > self.before.get("score", 0)

    @property
    def diff(self) -> str:
        if self.suggested_prompt is None:
            return ""
        return unified_diff(self.original_prompt, self.suggested_prompt, "current prompt", "suggested prompt")

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "original_prompt": self.original_prompt,
            "suggested_prompt": self.suggested_prompt,
            "added_instructions": self.added,
            "strategy": self.strategy,
            "before": self.before,
            "after": self.after,
            "resolved": self.resolved,
            "remaining": self.remaining,
            "prompt_locations": self.locations,
            "attempts": self.attempts,
            "notes": self.notes,
            "example_case_ids": self.example_ids,
            "diff": self.diff,
        }


@dataclass
class FixResult:
    candidate: str
    groups: list[GroupFix]
    before: dict
    after: dict | None = None
    created_at: str = ""
    excluded_examples: int = 0
    notes: list[str] = field(default_factory=list)  # what no prompt can fix (settings the model rejects)

    def to_dict(self) -> dict:
        return {
            "llmshift_version": __version__,
            "candidate": self.candidate,
            "created_at": self.created_at,
            "before": self.before,
            "after": self.after,
            "excluded_example_cases": self.excluded_examples,
            "notes": self.notes,
            "groups": [g.to_dict() for g in self.groups],
        }


def _score(cases: list[CaseResult]) -> float:
    """Case outcomes, plus partial credit from check scores (fewer tone differences, more facts...)."""
    if not cases:
        return 0.0
    total = 0.0
    for c in cases:
        if c.status == "error":
            continue  # a request that failed earns nothing (it has no checks to give it credit)
        total += {"pass": 1.0, "warn": 0.6}.get(c.status, 0.0)  # nosec B105  # a check status or setting name, not a password
        values = []
        for r in c.checks:
            if r.status == "pass":
                values.append(1.0)
            elif r.status == "fail":
                values.append(max(0.0, min(1.0, float(r.score))) if r.score is not None else 0.0)
        total += 0.3 * (sum(values) / len(values) if values else 1.0)
    return round(total / (1.3 * len(cases)), 4)


def _stats(cases: list[CaseResult]) -> dict:
    n = len(cases)
    counts = Counter(c.status for c in cases)
    return {
        "cases": n,
        "passed": counts["pass"],
        "changed": counts["warn"],
        "failed": counts["fail"],
        "errors": counts["error"],
        "pass_rate": stored_rate((counts["pass"] + counts["warn"]) / n) if n else 0.0,
        "score": _score(cases),
    }


def _problems(cases: list[CaseResult]) -> list[str]:
    seen: dict[str, int] = {}
    for c in cases:
        for r in c.checks:
            if r.status in ("fail", "error"):
                key = f"{r.category}: {r.message}"
                seen[key] = seen.get(key, 0) + 1
    return [k for k, _ in sorted(seen.items(), key=lambda kv: -kv[1])]


# ---------------------------------------------------------- rule synthesis


def _field_vocab(cases: list[TestCase], path: str) -> list[str]:
    from .formats import parse_json_lenient
    from .jsonschema_lite import get_path

    values = []
    for c in cases:
        value = parse_json_lenient(c.baseline_output)
        if value is None:
            continue
        found, v = get_path(value, path)
        if found and isinstance(v, (str, bool, int)):
            shown = ("true" if v else "false") if isinstance(v, bool) else str(v)
            if shown not in values:
                values.append(shown)
    return values


def _structure_rule(change: str) -> str | None:
    table = {
        "adds markdown headings": "Do not use markdown headings.",
        "no longer uses markdown headings": "Organise the answer under short markdown headings.",
        "adds a bulleted/numbered list": "Write in plain paragraphs, not bullet points or numbered lists.",
        "no longer uses a bulleted/numbered list": "Format the answer as a bulleted list.",
        "adds a numbered list": "Do not use numbered lists.",
        "no longer uses a numbered list": "Present the steps as a numbered list.",
        "adds a table": "Do not use tables.",
        "no longer uses a table": "Present the data in a markdown table.",
        "adds a fenced code block": "Do not use code blocks.",
        "no longer uses a fenced code block": "Put any code in a fenced code block.",
        "adds bold text": "Do not use bold text or other markdown emphasis.",
        "no longer uses bold text": "Use bold text to highlight the key points.",
    }
    return table.get(change)


def rule_instructions(failing: list[CaseResult], cases_by_id: dict[str, TestCase], group_cases: list[TestCase]) -> tuple[list[str], list[str]]:
    """Instructions that address the observed failures, plus notes for humans."""
    rules: Counter = Counter()
    notes: list[str] = []
    order: list[str] = []

    def add(text: str, weight: int = 1) -> None:
        if text not in rules:
            order.append(text)
        rules[text] += weight

    baseline_words = [word_count(c.baseline_output) for c in group_cases if c.baseline_output]
    median_words = int(statistics.median(baseline_words)) if baseline_words else None
    refusals = 0
    # Names that (almost) every baseline answer of this template mentions are part of
    # the expected style - e.g. the company name in a support reply.
    missing_entities: Counter = Counter()
    for res in failing:
        for r in res.checks:
            if r.type == "facts" and r.status == "fail":
                missing_entities.update(set((r.details or {}).get("missing_entities") or []))
    baselines = [c.baseline_output for c in group_cases if c.baseline_output]
    for entity, n in missing_entities.most_common(3):
        share = sum(1 for b in baselines if entity.lower() in b.lower()) / len(baselines) if baselines else 0
        if share >= 0.6 and n >= max(1, len(failing) // 3):
            example = next((s.strip() for b in baselines for s in re.split(r"(?<=[.!?])\s+", b) if entity.lower() in s.lower()), "")
            hint = f' (for example: "{shorten(example, 90)}")' if example else ""
            add(f'Mention "{entity}" as the existing answers do{hint}.', 2)
    for res in failing:
        case = cases_by_id.get(res.id)
        for r in res.checks:
            if r.status not in ("fail", "error"):
                continue
            d = r.details or {}
            if r.type == "json_valid":
                add("Return only the raw JSON object. Do not wrap it in markdown code fences (```) and do not add any text before or after it.", 3)
            elif r.type == "json_schema" and case is not None:
                schema = next((c.get("schema") for c in case.checks if c.get("type") == "json_schema"), None)
                if schema:
                    add(f"The JSON must have exactly this structure (keys and types): {describe_schema(schema)}", 2)
            elif r.type == "json_match":
                for m in d.get("mismatches") or []:
                    path = m.get("path", "")
                    vocab = _field_vocab(group_cases, path)
                    key = path.split(".")[-1].strip("[]'\"")
                    if 2 <= len(vocab) <= 12:
                        add(f'For "{key}", use exactly one of: {", ".join(vocab)}.')
                    if isinstance(m.get("expected"), str) and isinstance(m.get("actual"), str) and normalize(m["expected"]) == normalize(m["actual"]):
                        add(f'Write the value of "{key}" exactly as specified (same capitalisation and punctuation).')
            elif r.type in ("equals", "one_of") and case is not None:
                labels = next((c.get("values") for c in case.checks if c.get("type") == "one_of"), None)
                if labels:
                    add("Respond with exactly one of these labels and nothing else: " + ", ".join(map(str, labels)) + ".", 2)
                else:
                    add("Answer with the bare value only - no explanation, punctuation or extra words.", 2)
            elif r.type == "language" and case is not None:
                lang = next((c.get("expected") for c in case.checks if c.get("type") == "language"), None)
                if lang:
                    add(f"Always respond in {language_name(lang)}.", 2)
            elif r.type == "structure":
                for change in d.get("changes") or []:
                    rule = _structure_rule(change)
                    if rule:
                        add(rule)
            elif r.type == "length" and median_words:
                if "longer" in (r.message or ""):
                    add(f"Keep the answer concise: about {median_words} words, never more than {int(median_words * 1.5) + 5}.", 2)
                elif "shorter" in (r.message or ""):
                    add(f"Give a complete answer of about {median_words} words.", 2)
            elif r.type == "tone":
                for text in tonelib.instructions_for(d.get("differences") or []):
                    add(text)
            elif r.type == "facts" and case is not None:
                source = normalize(" ".join(str(m.get("content", "")) for m in case.messages if m.get("role") in ("user", "tool")))
                entities = set(d.get("missing_entities") or [])
                lost = [x for x in (d.get("missing") or []) if x not in entities] + [c.get("from", "") for c in d.get("changed") or []]
                if any(normalize(x) and normalize(x) in source for x in lost):
                    add("Copy figures, names, dates and IDs exactly as they appear in the input. Do not round, convert or paraphrase numbers.", 2)
                elif lost:
                    add("Be precise: include the specific figures, dates and IDs that answer the question.")
            elif r.type == "tool_calls":
                for exp in ((next((c for c in case.checks if c.get("type") == "tool_calls"), {}) or {}).get("expected") or []) if case else []:
                    if "no tool call" in (r.message or "") or "missing call" in (r.message or ""):
                        add(f"When the request needs it, call the `{exp.get('name')}` tool instead of answering in plain text.", 2)
                if "unexpected call" in (r.message or ""):
                    add("Only call a tool when it is required to answer.")
            elif r.type == "refusal":
                refusals += 1
    if refusals:
        notes.append(f"{refusals} case(s) are refused by the new model; review them manually - prompt changes cannot safely force an answer.")
    ranked = sorted(order, key=lambda t: (-rules[t], order.index(t)))
    return ranked[:10], notes


def compose(original: str, instructions: list[str], example: str | None = None) -> str:
    block = HEADER + "\n" + "\n".join(f"- {line}" for line in instructions) if instructions else ""
    parts = [p for p in (original.strip(), block, example or "") if p]
    return "\n\n".join(parts)


def _example_block(cases: list[TestCase]) -> str | None:
    shots = []
    for c in cases[:2]:
        user = c.user_text
        out = c.baseline_output
        if not user or not out:
            continue
        shots.append(f"Input:\n{shorten(user, 500)}\nOutput:\n{out[:900]}")
    if not shots:
        return None
    title = "Example of the expected output:" if len(shots) == 1 else "Examples of the expected output:"
    return title + "\n\n" + "\n\n".join(shots)


OPTIMIZER_SYSTEM = (
    "You are an expert prompt engineer. A team is moving an AI feature to a new model and needs the new model "
    "to behave exactly like the old one. You rewrite system prompts precisely and conservatively."
)


def optimize_prompt(optimizer: Any, original: str, failures: list[CaseResult], cases_by_id: dict[str, TestCase], current: str | None = None) -> str | None:
    examples = []
    for res in failures[:4]:
        case = cases_by_id.get(res.id)
        if case is None:
            continue
        problems = "; ".join(f"{r.category}: {r.message}" for r in res.checks if r.status in ("fail", "error"))[:600]
        examples.append(
            f"Input:\n{shorten(case.user_text, 700)}\n\nExpected output (old model):\n{res.baseline_output[:1200]}\n\n"
            f"Actual output (new model):\n{res.candidate_output[:1200]}\n\nProblems: {problems}"
        )
    if not examples:
        return None
    body = (
        f"Current system prompt:\n<prompt>\n{current or original or '(none)'}\n</prompt>\n\n"
        "With the new model some outputs no longer match the old model's. Examples:\n\n"
        + "\n\n---\n\n".join(f"### Example {i}\n{e}" for i, e in enumerate(examples, 1))
        + "\n\nRewrite the system prompt so the new model reproduces the old model's behaviour: the same format, facts, "
        "tone and length. Keep every original instruction and add explicit guidance only where needed. Do not mention "
        "'old model' or 'new model'. Reply with only the new system prompt inside <prompt></prompt> tags."
    )
    resp = optimizer.complete({"messages": [{"role": "system", "content": OPTIMIZER_SYSTEM}, {"role": "user", "content": body}], "params": {"temperature": 0}})
    text = resp.text or ""
    m = re.search(r"<prompt>\s*(.*?)\s*</prompt>", text, re.S)
    prompt = m.group(1).strip() if m else text.strip()
    if len(prompt) < 10 or len(prompt) > 20000:
        return None
    return prompt


def find_prompt_locations(prompt: str, paths: list[str], limit: int = 3) -> list[str]:
    """Where does this system prompt live in the codebase?"""
    from .scanner import _read_text, is_llmshift_output, iter_source_files

    lines = [ln.strip() for ln in prompt.splitlines() if len(ln.strip()) >= 20]
    if not lines:
        return []
    needle = lines[0][:60]
    needles = [needle, needle.replace('"', '\\"'), needle.replace("'", "\\'")]
    out = []
    try:
        files = list(iter_source_files(paths, include_docs=True))
    except FileNotFoundError:
        return []
    for _root, path in files[:20000]:
        if path.suffix.lower() in (".jsonl", ".ndjson"):  # recorded traffic and test files hold every prompt
            continue
        text, _skipped = _read_text(path)
        if not text or needle[:20] not in text and not any(n in text for n in needles):
            continue
        if is_llmshift_output(path, text):  # an earlier fix report quotes the prompt too
            continue
        if path.suffix.lower() == ".json" and any(k in text for k in ('"choices"', '"usage"', '"llmshift_record"', '"response"')):
            continue  # a saved request log: it holds the prompt, but the prompt does not live there
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(n in line for n in needles):
                try:
                    rel = Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
                except ValueError:
                    rel = Path(path).as_posix()
                out.append(f"{rel}:{lineno}")
                break
        if len(out) >= limit:
            break
    return out


def run_fix(
    cases: list[TestCase],
    candidate: Any,
    *,
    config: dict | None = None,
    judge: Any = None,
    optimizer: Any = None,
    concurrency: int = 4,
    cache: Any = None,
    allow_python: bool = False,
    severity: dict | None = None,
    rounds: int = 3,
    max_cases_per_group: int = 12,
    target: float = 1.0,
    source_paths: list[str] | None = None,
    previous: CompareResult | None = None,
    show_progress: bool | None = None,
    validate: bool = True,
) -> FixResult:
    config = config or {}
    cases = unique_ids(cases)
    cand = resolve(candidate, config) if isinstance(candidate, str) else candidate
    opt = resolve(optimizer, config) if isinstance(optimizer, str) else optimizer
    opt_error: str | None = None  # once the optimizer fails, the other groups do without it
    # A setting the model rejects fails a case whatever the prompt says: prompts are judged on
    # the answers, and the rejected settings are reported apart.
    severity = {**(severity or {}), "request": "info"}
    common = dict(judge=judge, config=config, concurrency=concurrency, cache=cache, allow_python=allow_python, severity=severity, show_progress=show_progress)
    initial = previous if previous is not None else run_compare(cases, cand, **common)
    refused = sorted({item.split("=", 1)[0] for c in initial.cases for r in c.checks if r.type == "request" for item in (r.details or {}).get("rejected") or []})
    by_result = {c.id: c for c in initial.cases}
    by_case = {c.id: c for c in cases}
    fixes: list[GroupFix] = []
    best_patches: dict[str, dict] = {}
    for group, members in sorted(groups_of(cases).items()):
        results = [by_result[c.id] for c in members if c.id in by_result]
        bad = [r for r in results if r.status in ("fail", "warn")]
        if not bad:
            continue
        original = members[0].system_prompt
        fix = GroupFix(group=group, original_prompt=original)
        # Scoring set: the failing cases plus a few passing ones as a regression guard.
        bad_ids = {r.id for r in bad}
        good = [c for c in members if c.id not in bad_ids and by_result.get(c.id) and by_result[c.id].status == "pass"]
        eval_cases = [by_case[r.id] for r in bad][:max_cases_per_group]
        eval_cases += good[: max(2, max_cases_per_group // 4)]
        eval_ids = {c.id for c in eval_cases}
        example_pool = [c for c in members if c.id not in eval_ids and c.has_baseline and by_result.get(c.id) and by_result[c.id].status != "error"]
        if not example_pool and len(eval_cases) >= 3:
            # Small template: hold one case out as the worked example and never score on it.
            holdout = next((c for c in eval_cases if by_result[c.id].status == "pass"), None) or eval_cases[-1]
            if holdout.has_baseline:
                example_pool = [holdout]
                eval_cases = [c for c in eval_cases if c.id != holdout.id]
        before = [by_result[c.id] for c in eval_cases]
        fix.before = _stats(before)
        best_score, best_patch, best_results = fix.before["score"], None, before
        instructions, notes = rule_instructions(bad, by_case, members)
        fix.notes.extend(notes)
        example = _example_block(example_pool)
        # (strategy, instructions, use_example, full system prompt)
        candidates: list[tuple[str, list[str], bool, str]] = []
        if instructions:
            candidates.append(("rules", instructions, False, compose(original, instructions)))
            if example:
                candidates.append(("rules+example", instructions, True, compose(original, instructions, example)))
        elif example:
            candidates.append(("example", [], True, compose(original, [], example)))
        used: list[str] = []
        tried: set[str] = {original}
        for round_no in range(max(1, rounds)):
            if opt is not None and opt_error is None:
                failing_now = [r for r in best_results if r.status in ("fail", "warn")]
                try:
                    rewritten = optimize_prompt(opt, original, failing_now or bad, by_case, best_patch["system"] if best_patch else None)
                except ProviderError as exc:  # a wrong key or model: the rule-based fixes still run
                    opt_error = str(exc)
                    fix.notes.append(f"The optimizer {opt.spec} failed, so only rule-based changes were tried: {exc}")
                    rewritten = None
                if rewritten:
                    candidates.append(("optimizer", [], False, rewritten))
            # A prompt already tried (the optimizer repeating itself) costs another run and teaches nothing.
            candidates = [c for c in candidates if c[3] not in tried]
            if not candidates:
                break
            tried.update(c[3] for c in candidates)
            round_best = None
            for strategy, added, with_example, prompt in candidates:
                patch = {"system": prompt}
                res = run_compare(eval_cases, cand, patches={group: patch}, **common)
                score = _score(res.cases)
                stats = _stats(res.cases)
                fix.attempts.append({"round": round_no + 1, "strategy": strategy, "score": score, "pass_rate": stats["pass_rate"], "errors": stats["errors"]})
                if stats["errors"] > fix.before["errors"]:
                    continue  # a prompt that makes requests fail is never an improvement
                if round_best is None or score > round_best[0]:
                    round_best = (score, strategy, added, with_example, res.cases)
                if score > best_score + 1e-9:
                    best_score, best_patch, best_results = score, patch, res.cases
                    fix.strategy, used = strategy, added
                    fix.example_ids = [c.id for c in example_pool[:2]] if with_example else []
            candidates = []
            if best_score >= target or round_best is None or round_best[1] == "optimizer":
                if best_score >= target:
                    break
                continue
            # Refine the round's best attempt with rules for whatever is still different.
            _, strategy, added, with_example, results = round_best
            still = [r for r in results if r.status in ("fail", "warn")]
            extra, _more = rule_instructions(still, by_case, members)
            extra = [x for x in extra if x not in added]
            if extra:
                combined = added + extra
                name = strategy.split("+round")[0] + f"+round{round_no + 2}"
                candidates.append((name, combined, with_example, compose(original, combined, example if with_example else None)))
        if best_patch is not None:
            fix.suggested_prompt = best_patch["system"]
            fix.added = used
            best_patches[group] = best_patch
        fix.after = _stats(best_results)
        before_problems = set(_problems(before))
        after_problems = _problems(best_results)
        fix.remaining = after_problems[:10]
        fix.resolved = [p for p in before_problems if p not in set(after_problems)][:10]
        if original:
            fix.locations = find_prompt_locations(original, source_paths or ["."])
        fixes.append(fix)

    # Cases quoted as worked examples in a suggested prompt would trivially match, so the
    # overall before/after numbers leave them out.
    examples = {cid for g in fixes for cid in g.example_ids}
    before_cases = [c for c in initial.cases if c.id not in examples]
    result = FixResult(candidate=cand.spec, groups=fixes, before=_stats(before_cases), created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    result.excluded_examples = len(examples)
    if refused:
        flags = " ".join(f"--drop-param {n}" for n in refused if n != "assistant prefill")
        result.notes.append(
            f"{cand.spec} rejects {', '.join(refused)} as the requests send them, so those calls fail with a 400 error whatever the "
            "prompt says. Remove them from the code (`llmshift upgrade` lists where)" + (f", then test with {flags}." if flags else ".")
        )
    if validate and best_patches:
        final = run_compare(cases, cand, patches=best_patches, **common)
        result.after = _stats([c for c in final.cases if c.id not in examples])
    return result


def patched_cases(cases: list[TestCase], fix: FixResult) -> list[TestCase]:
    """Copies of the test cases with the suggested prompts applied."""
    import copy

    from .messages import with_system_prompt

    by_group = {g.group: g.suggested_prompt for g in fix.groups if g.suggested_prompt}
    out = []
    for c in cases:
        new = copy.deepcopy(c)
        if group_key(c) in by_group:
            new.input["messages"] = with_system_prompt(c.messages, by_group[group_key(c)])
            new.metadata["prompt_fixed_for"] = fix.candidate
        out.append(new)
    return out


def fix_markdown(fix: FixResult) -> str:
    from .report import _md  # optimizer output and answers: no live links, images or mentions

    lines = [f"# LLMShift prompt fixes for {fix.candidate}", ""]
    b, a = fix.before, fix.after
    if a:
        lines.append(f"Overall pass rate: **{b['pass_rate']:.0%} -> {a['pass_rate']:.0%}** with the suggested prompts ({fix.created_at}).")
    else:
        lines.append(f"Overall pass rate before changes: **{b['pass_rate']:.0%}**.")
    lines.append("")
    for note in fix.notes:
        lines += [f"> **Note:** {_md(note)}", ""]
    for g in fix.groups:
        lines.append(f"## {_md(g.group)}")
        lines.append("")
        if g.locations:
            lines.append("Prompt found at: " + ", ".join(f"`{loc}`" for loc in g.locations))
            lines.append("")
        errs = f", {plural(g.before['errors'], 'error')}" if g.before.get("errors") else ""
        lines.append(f"- Before: {g.before.get('passed', 0)} of {g.before.get('cases', 0)} match the baseline ({g.before.get('failed', 0)} failing, {g.before.get('changed', 0)} changed{errs})")
        if g.suggested_prompt:
            errs = f", {plural(g.after['errors'], 'error')}" if g.after.get("errors") else ""
            lines.append(f"- After: {g.after.get('passed', 0)} of {g.after.get('cases', 0)} match the baseline ({g.after.get('failed', 0)} failing, {g.after.get('changed', 0)} changed{errs}) using **{g.strategy}**")
        else:
            lines.append("- No prompt change improved this template.")
        if g.added:
            lines.append("")
            lines.append("Instructions added:")
            lines += [f"- {_md(x)}" for x in g.added]
        if g.suggested_prompt:
            lines += ["", "<details><summary>Diff</summary>", "", md_fence(g.diff.rstrip(), "diff"), "", "</details>", ""]
            lines += ["Suggested system prompt:", "", md_fence(g.suggested_prompt, "text")]
        if g.remaining:
            lines += ["", "Still different after the fix:"] + [f"- {_md(x)}" for x in g.remaining[:8]]
        for note in g.notes:
            lines.append(f"> {_md(note)}")
        lines.append("")
    lines += ["", credit()]
    return show_bidi("\n".join(lines) + "\n")
