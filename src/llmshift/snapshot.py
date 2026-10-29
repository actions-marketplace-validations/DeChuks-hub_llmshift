"""`llmshift snapshot`: turn real usage logs into a test set automatically.

Pipeline: parse logs -> filter (model, time window, errors, truncated) ->
redact secrets/PII -> de-duplicate -> group by prompt template -> pick a
diverse sample per template -> (optionally) generate missing baselines ->
infer checks from the baseline outputs -> write JSONL.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import messages as msgs
from .agents import link_runs
from .infer import GroupStats, group_stats, infer_checks, merge_checks
from .logparse import DEFAULT_MAX_RECORDS, Interaction, LogStats, parse_logs, parse_time
from .pricing import USAGE_FIELDS
from .redact import Redactor
from .testset import TestCase, case_id_for
from .textutil import jaccard, stable_hash, tokenize


@dataclass
class SnapshotResult:
    cases: list[TestCase]
    stats: LogStats
    filtered: Counter = field(default_factory=Counter)
    unique: int = 0
    duplicates: int = 0
    groups: int = 0
    redactions: Counter = field(default_factory=Counter)
    generated: int = 0
    generation_errors: int = 0
    check_counts: Counter = field(default_factory=Counter)
    group_sizes: dict = field(default_factory=dict)


def _matches_model(model: str | None, patterns: list[str]) -> bool:
    if not patterns:
        return True
    if not model:
        return False
    m = model.lower()
    return any(fnmatch.fnmatch(m, p.lower()) or m == p.lower() for p in patterns)


def _diverse(items: list[Interaction], k: int, rng: random.Random) -> list[Interaction]:
    """Farthest-point sampling on token sets: k items covering the input space."""
    if len(items) <= k:
        return list(items)
    pool = list(items)
    rng.shuffle(pool)
    pool = pool[:2000]
    sets = [set(tokenize(msgs.last_user_text(it.messages))[:200]) for it in pool]
    chosen = [0]
    chosen_set = {0}
    min_dist = [1.0 - jaccard(sets[0], s) for s in sets]
    while len(chosen) < k:
        best, best_d = -1, -1.0
        for i, d in enumerate(min_dist):
            if i not in chosen_set and d > best_d:
                best, best_d = i, d
        if best < 0:
            break
        chosen.append(best)
        chosen_set.add(best)
        for i, s in enumerate(sets):
            d = 1.0 - jaccard(sets[best], s)
            if d < min_dist[i]:
                min_dist[i] = d
    return [pool[i] for i in chosen]


def _allocate(sizes: dict[str, int], max_cases: int, per_group: int) -> dict[str, int]:
    """How many cases each group gets: every group at least one, larger groups more."""
    quotas = {g: min(per_group, n) for g, n in sizes.items()}
    if sum(quotas.values()) <= max_cases:
        return quotas
    ordered = sorted(sizes, key=lambda g: (-sizes[g], g))
    if len(ordered) >= max_cases:
        return {g: (1 if i < max_cases else 0) for i, g in enumerate(ordered)}
    out = {g: 1 for g in ordered}
    remaining = max_cases - len(ordered)
    total_extra = sum(quotas[g] - 1 for g in ordered) or 1
    for g in ordered:
        out[g] += int(remaining * (quotas[g] - 1) / total_extra)
    leftover = max_cases - sum(out.values())
    for g in ordered:
        if leftover <= 0:
            break
        if out[g] < quotas[g]:
            out[g] += 1
            leftover -= 1
    return out


def build_snapshot(
    sources: Iterable[str],
    *,
    models: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    max_cases: int = 200,
    per_group: int = 25,
    seed: int = 7,
    redact_secrets: bool = True,
    redact_pii: bool = False,
    include_errors: bool = False,
    include_truncated: bool = False,
    baseline_provider: Any = None,
    regenerate: bool = False,
    judge: bool = False,
    concurrency: int = 4,
    cache: Any = None,
    max_records: int | None = DEFAULT_MAX_RECORDS,
) -> SnapshotResult:
    result = SnapshotResult(cases=[], stats=LogStats())
    since_dt, until_dt = parse_time(since), parse_time(until)
    if since and since_dt is None:
        raise ValueError(f"cannot parse --since {since!r} (use YYYY-MM-DD or ISO-8601)")
    if until and until_dt is None:
        raise ValueError(f"cannot parse --until {until!r} (use YYYY-MM-DD or ISO-8601)")
    until_inclusive = True
    if until_dt is not None and re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}\s*", str(until)):
        until_dt += _dt.timedelta(days=1)  # a date means "through the end of that day"
        until_inclusive = False

    def admit(inter: Interaction) -> bool:
        """Filters applied while the logs are read, so dropped calls never take memory."""
        reason = None
        if inter.error and not include_errors:
            reason = "API error recorded in the log"
        elif not (_matches_model(inter.model, models or []) or _matches_model(inter.metadata.get("served_model"), models or [])):
            reason = "different model" if inter.model else "model not recorded"
        elif since_dt or until_dt:
            ts = parse_time(inter.timestamp)
            if ts is None:
                reason = "no timestamp (time window given)"
            elif (since_dt and ts < since_dt) or (until_dt and (ts > until_dt if until_inclusive else ts >= until_dt)):
                reason = "outside the time window"
        if reason is None:
            if inter.finish_reason == "length" and not include_truncated:
                reason = "truncated output (finish_reason=length)"
            elif not inter.has_output and baseline_provider is None:
                reason = "no output recorded (pass --baseline-model to generate one)"
            elif not any(m.get("role") in ("user", "tool") for m in inter.messages) and not msgs.system_prompt(inter.messages):
                reason = "no prompt content"
        if reason is not None:
            result.filtered[reason] += 1
            return False
        return True

    kept, stats = parse_logs(sources, admit=admit, max_records=max_records)
    result.stats = stats

    redactor = Redactor(secrets=redact_secrets, pii=redact_pii)
    if redactor.patterns:
        for inter in kept:
            inter.messages = redactor.obj(inter.messages)
            inter.output = redactor.text(inter.output) if inter.output else inter.output
            inter.tool_calls = redactor.obj(inter.tool_calls)
            inter.vars = redactor.obj(inter.vars)
        result.redactions = Counter(redactor.counts)

    kept = link_runs(kept)
    unique: dict[str, Interaction] = {}
    for inter in kept:
        key = stable_hash({"m": inter.messages, "p": inter.params}, 24)
        if key in unique:
            result.duplicates += 1
            # Prefer the record that has an output.
            if not unique[key].has_output and inter.has_output:
                unique[key] = inter
            continue
        unique[key] = inter
    result.unique = len(unique)

    groups: dict[str, list[Interaction]] = {}
    for inter in unique.values():
        key = msgs.template_key(inter.messages, inter.params)
        groups.setdefault(key + "-agent" if inter.agent else key, []).append(inter)
    result.groups = len(groups)
    result.group_sizes = {g: len(v) for g, v in groups.items()}

    rng = random.Random(seed)  # nosec B311  # seeded on purpose: the same sample on every run; nothing secret
    quotas = _allocate({g: len(v) for g, v in groups.items()}, max(1, max_cases), max(1, per_group))
    selected: list[tuple[str, Interaction]] = []
    for g in sorted(groups):
        if quotas.get(g, 0) <= 0:
            continue
        with_output = [i for i in groups[g] if i.has_output]
        pool = with_output if len(with_output) >= quotas[g] or baseline_provider is None else groups[g]
        for inter in _diverse(sorted(pool, key=lambda i: (i.timestamp or "", i.source)), quotas[g], rng):
            selected.append((g, inter))

    if baseline_provider is not None:
        _generate_baselines(selected, baseline_provider, regenerate, concurrency, cache, result)

    stats_by_group: dict[str, GroupStats] = {}
    for g, members in groups.items():
        outputs = [i.output or "" for i in members if i.has_output]
        stats_by_group[g] = group_stats(outputs)

    for g, inter in selected:
        if not inter.has_output:
            continue
        inferred = infer_checks(inter.messages, inter.params, inter.output, inter.tool_calls, stats_by_group.get(g), judge=judge)
        if inter.agent:
            inferred = [{"type": "agent"}] + [c for c in inferred if c.get("type") != "tool_calls"]
        checks = merge_checks(inter.checks, inferred)
        inp = {"messages": inter.messages, "params": inter.params}
        baseline: dict = {"output": inter.output or ""}
        if inter.tool_calls:
            baseline["tool_calls"] = inter.tool_calls
        if inter.model:
            baseline["model"] = inter.model
        usage = {k: v for k, v in (inter.usage or {}).items() if k in USAGE_FIELDS and isinstance(v, (int, float)) and not isinstance(v, bool)}
        if usage:
            baseline["usage"] = usage  # what the answer cost, for `llmshift pick`
        metadata = dict(inter.metadata)
        metadata.update({k: v for k, v in (("source", inter.source), ("timestamp", inter.timestamp), ("format", inter.fmt)) if v})
        case = TestCase(
            id=inter.case_id or case_id_for(inp),
            input=inp,
            baseline=baseline,
            checks=checks,
            group=g,
            tags=list(inter.tags),
            vars=dict(inter.vars),
            metadata=metadata,
            agent=inter.agent,
        )
        result.cases.append(case)
        for c in checks:
            result.check_counts[c.get("type")] += 1
    return result


def _generate_baselines(selected: list[tuple[str, Interaction]], provider: Any, regenerate: bool, concurrency: int, cache: Any, result: SnapshotResult) -> None:
    from .runner import run_cases

    todo = [(g, i) for g, i in selected if regenerate or not i.has_output]
    if not todo:
        return
    temp_cases = [TestCase(id=f"gen{n}", input={"messages": i.messages, "params": i.params}, group=g) for n, (g, i) in enumerate(todo)]
    responses = run_cases(provider, temp_cases, concurrency=concurrency, cache=cache, label=f"baseline {provider.spec}")
    for case, (_group, inter) in zip(temp_cases, todo):
        resp = responses.get(case.id)
        if resp is None or not resp.ok:
            result.generation_errors += 1
            continue
        inter.output = resp.text
        inter.tool_calls = resp.tool_calls
        inter.model = resp.model or provider.model
        inter.finish_reason = resp.finish_reason
        inter.metadata["baseline_generated_by"] = provider.spec
        result.generated += 1
