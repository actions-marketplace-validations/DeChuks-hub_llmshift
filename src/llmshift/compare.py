"""`llmshift compare`: run the test set on a new model and show exactly what changed.

Each case's output is checked against what the old model did (format, facts,
tone, tool calls, refusals...). With a live baseline model, checks that also
fail for the old model are treated as noise rather than regressions.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import re
import statistics
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

from . import __version__
from .agents import trace as agent_trace
from .cache import ResponseCache
from .checks import CheckContext, CheckResult, Sample, run_checks
from .infer import wants_judge
from .messages import request_preview
from .providers import resolve
from .providers.anthropic import plan_thinking
from .providers.base import Provider, Response
from .runner import adaptations, run_cases
from .stats import cases_needed, mcnemar_p, pass_rate_interval, stored_rate
from .testset import TestCase, unique_ids
from .textutil import plural, word_count

JUDGE_CHECKS = ("llm_judge", "llm_label", "llm_score", "modelgraded")


def with_judge_check(case: TestCase) -> TestCase:
    """A judge given for a run (--judge, or judge: in llmshift.yaml) also judges free-text
    cases that have no judge check of their own, as `snapshot --with-judge-checks` would.
    Cases with imported model graders keep their own criteria."""
    if any(c.get("type") in JUDGE_CHECKS for c in case.checks) or not wants_judge(case.baseline_output, case.baseline_tool_calls, case.params):
        return case
    return replace(case, checks=[*case.checks, {"type": "llm_judge"}])


_OPTIONAL_FIELDS = (
    "candidate_tool_calls", "error", "latency_ms", "usage", "finish_reason", "model", "fingerprint", "unstable",
    "baseline_live_output", "baseline_latency_ms", "baseline_model", "baseline_live_status", "raw_status",
    "runs", "pass_fraction", "run_statuses", "flaky", "trace", "baseline_trace", "cost_usd", "baseline_cost_usd", "unchecked",
)


@dataclass
class CaseResult:
    id: str
    group: str
    status: str  # pass | warn | fail | error (no answer, or unchecked)
    input_preview: str = ""
    baseline_output: str = ""
    candidate_output: str = ""
    candidate_tool_calls: list = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None
    latency_ms: float | None = None
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    model: str | None = None
    fingerprint: str | None = None
    unstable: list[str] = field(default_factory=list)
    baseline_live_output: str | None = None
    baseline_latency_ms: float | None = None
    baseline_model: str | None = None
    baseline_live_status: str | None = None  # the old model, checked the same way
    raw_status: str | None = None  # before failures shared with the old model were set aside
    runs: int = 1
    pass_fraction: float | None = None  # share of repeated runs that passed
    run_statuses: list[str] = field(default_factory=list)
    flaky: bool = False
    trace: list[str] = field(default_factory=list)  # agent runs: one line per step
    baseline_trace: list[str] = field(default_factory=list)
    cost_usd: float | None = None
    baseline_cost_usd: float | None = None
    unchecked: bool = False  # answered, but a check could not run: no verdict, not the model's failure

    @property
    def failing(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status in ("fail", "error")]

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "group": self.group,
            "status": self.status,
            "input_preview": self.input_preview,
            "baseline_output": self.baseline_output,
            "candidate_output": self.candidate_output,
            "checks": [c.to_dict() for c in self.checks],
        }
        for key in _OPTIONAL_FIELDS:
            value = getattr(self, key)
            if value not in (None, [], {}, False) and not (key == "runs" and value == 1):
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CaseResult":
        r = cls(id=d["id"], group=d.get("group", ""), status=d.get("status", "error"))
        for key in ("input_preview", "baseline_output", "candidate_output", *_OPTIONAL_FIELDS):
            if key in d:
                setattr(r, key, d[key])
        r.checks = [CheckResult.from_dict(c) for c in d.get("checks") or []]
        return r


@dataclass
class CompareResult:
    candidate: str
    baseline: str
    cases: list[CaseResult]
    summary: dict = field(default_factory=dict)
    adaptations: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    tests_path: str | None = None
    judge: str | None = None
    created_at: str = ""
    min_pass_rate: float | None = None
    label: str = "compare"

    def to_dict(self) -> dict:
        return {
            "llmshift_version": __version__,
            "kind": self.label,
            "created_at": self.created_at,
            "candidate": self.candidate,
            "baseline": self.baseline,
            "judge": self.judge,
            "tests": self.tests_path,
            "min_pass_rate": self.min_pass_rate,
            "summary": self.summary,
            "insights": self.insights,
            "adaptations": self.adaptations,
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompareResult":
        return cls(
            candidate=d.get("candidate", "?"),
            baseline=d.get("baseline", "recorded"),
            cases=[CaseResult.from_dict(c) for c in d.get("cases") or []],
            summary=d.get("summary") or {},
            adaptations=d.get("adaptations") or [],
            insights=d.get("insights") or [],
            tests_path=d.get("tests"),
            judge=d.get("judge"),
            created_at=d.get("created_at", ""),
            min_pass_rate=d.get("min_pass_rate"),
            label=d.get("kind", "compare"),
        )

    @property
    def passed(self) -> bool:
        threshold = self.min_pass_rate if self.min_pass_rate is not None else 0.0
        s = self.summary
        if s.get("evaluated", 0) == 0:
            return False
        return s.get("pass_rate", 0.0) + 1e-9 >= threshold

    @property
    def inconclusive(self) -> bool:
        """No verdict (exit 2): some cases could not be checked (a check could not run) and the
        others pass, or no case could be checked at all."""
        s = self.summary
        unchecked = s.get("unchecked", 0)
        return bool(unchecked) and (unchecked >= s.get("total", 0) or self.passed)


def request_rejections(model: str, params: dict, resp: Response) -> list[str]:
    """What `model` rejects of a request as it is sent (temperature=0, assistant prefill, a
    thinking budget on Claude 4.7 and later, ...). Each was left out or changed to test the
    answer, so the real call fails with a 400 error. The provider reports what it changed
    (resp.rejected): Claude's thinking and effort only go out through the Anthropic API."""
    from .providers.anthropic import rejected_sampling

    found = rejected_sampling(model, params)
    names = {item.split("=", 1)[0] for item in found}
    for name in resp.rejected:
        if name not in names:
            names.add(name)
            found.append(f"{name}={json.dumps(params[name], default=str)}" if name in params else name)
    return found


def _said_by(note: str, rejected: list[str]) -> bool:
    """Whether an adaptation note only repeats a rejected setting the cases already report."""
    m = re.search(r"dropped (?:unsupported parameter )?'([\w.]+)'", note)
    if m:
        return {"stop_sequences": "stop"}.get(m.group(1), m.group(1)) in rejected
    return ("prefill" in note and "assistant prefill" in rejected) or ("forced tool choice" in note and "tool_choice" in rejected)


def rejection_check(model: str, rejected: list[str], ctx: CheckContext, advice: list[str] | None = None) -> CheckResult:
    """The case fails on what the model rejects; `advice` says what to send instead (thinking
    and effort are sent in a form the model takes, not left out)."""
    from .checks import severity_of

    them = "it" if len(rejected) == 1 else "them"
    plain = [item for item in rejected if item.split("=", 1)[0] not in ("thinking", "reasoning_effort")]
    message = f"{model} rejects {', '.join(rejected)} as this request sends {them}: the real call fails with a 400 error "
    if plain:
        those = "it" if len(plain) == 1 else "them"
        message += f"(llmshift left {those} out to test the answer). Remove {those} from your code"
        flags = " ".join(f"--drop-param {item.split('=', 1)[0]}" for item in plain if "=" in item)
        message += f", then test with {flags}." if flags else "."
    else:
        message += "(llmshift sent what the model takes instead to test the answer)."
    for line in dict.fromkeys(advice or []):
        message += " " + line
    return CheckResult("request", "request", severity_of({"type": "request"}, ctx), "fail", message, name="request settings", details={"rejected": rejected})


def case_status(results: list[CheckResult], error: str | None) -> str:
    """fail: an error-level check failed; error: no answer, or an error-level check could not
    run (an invalid regex, a judge that cannot be reached: the test failed, not the model);
    warn: a warning-level check failed; pass otherwise. Info-level checks never count."""
    if error:
        return "error"
    if any(r.status == "fail" and r.severity == "error" for r in results):
        return "fail"
    if any(r.status == "error" and r.severity == "error" for r in results):
        return "error"
    if any(r.status == "fail" and r.severity == "warn" for r in results):
        return "warn"
    return "pass"


def not_checked(results: list[CheckResult]) -> list[CheckResult]:
    """The error-level checks that could not run, which leave the case without a verdict."""
    return [r for r in results if r.status == "error" and r.severity == "error"]


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return round(values[idx], 1)


def failure_patterns(result: CheckResult) -> list[str]:
    """Short, countable descriptions of what went wrong in one check."""
    d = result.details or {}
    if result.type == "tone" and d.get("differences"):
        from .tone import EMOJI_RE

        return [EMOJI_RE.sub("", re.sub(r"\s*\(.*?\)", "", x["text"])).strip() for x in d["differences"]]
    if result.type == "structure" and d.get("changes"):
        return list(d["changes"])
    if result.type == "json_match" and (d.get("mismatches") or d.get("missing")):
        out = [f"{m['path']} changed" for m in d.get("mismatches") or []]
        return out + [f"{p} missing" for p in d.get("missing") or []]
    if result.type == "facts":
        out = []
        if d.get("changed"):
            out.append("numbers changed")
        if any(True for _ in d.get("missing") or []):
            out.append("facts missing")
        return out or ["facts differ"]
    if result.type == "json_schema" and d.get("errors"):
        return [re.sub(r"^\$[^:]*:\s*", "", e) for e in d["errors"][:3]]
    if result.type == "request" and d.get("rejected"):
        return [f"{item.split('=', 1)[0]} rejected" for item in d["rejected"]]
    if result.type in ("equals", "one_of"):
        return ["label/answer changed" if result.type == "equals" else "answer outside the label set"]
    if result.type == "similarity":
        return ["low similarity to the reference answer"]
    if result.type == "length":
        return ["much longer" if "longer" in (result.message or "") else "much shorter"]
    if result.type == "agent" and "baseline_steps" in d:
        looped = d.get("looped") or []
        out = [f"looped on {name}" for name in looped]
        if d.get("unfinished"):
            out.append("ran out of steps without a final answer")
        out += [f"missing tool call: {name}" for name in dict.fromkeys(d.get("missing") or [])]
        out += [f"extra tool call: {name}" for name in dict.fromkeys(d.get("extra") or []) if name not in looped]
        if "different order" in (result.message or ""):
            out.append("tools called in a different order")
        if d.get("unrecorded"):
            out.append("tool call with no recorded result")
        return out or ["tool arguments changed"]
    if result.type == "llm_judge" and d.get("differences"):
        return [_short(str(x)) for x in d["differences"][:3]]
    if result.type == "regex":  # the message quotes the pattern: its groups are not asides (/^(Yes|No)$/ is not /^$/)
        return [_short(result.message or "regex", 120)]
    msg = (result.message or result.type)[:300]  # a pattern is a short phrase; the regex below is quadratic in a long one
    for _ in range(10):  # asides in brackets, innermost first, so "(a (b) c)" goes whole; the "(s)" of "call(s)" stays
        shorter = re.sub(r"\s*\((?!s\))[^()]*\)", "", msg)
        if shorter == msg:
            break
        msg = shorter
    msg = re.sub(r'"[^"]*"', '"..."', msg)
    return [_short(msg) or result.type]


def _short(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return text


SKIP_REASONS = {
    "needs_judge": "needs a judge model (--judge)",
    "python_disabled": "python graders disabled (--allow-python)",
}


def summarize(cases: list[CaseResult], responses: dict[str, Response] | None = None) -> dict:
    counts = Counter(c.status for c in cases)
    unchecked = sum(1 for c in cases if c.unchecked)
    evaluated = len(cases) - counts["error"]
    by_category: dict[str, dict] = defaultdict(lambda: {"fail": 0, "warn": 0, "patterns": Counter(), "examples": {}})
    by_check: dict[str, Counter] = defaultdict(Counter)
    groups: dict[str, Counter] = defaultdict(Counter)
    skipped: Counter = Counter()
    unstable = 0
    for c in cases:
        groups[c.group]["total"] += 1
        groups[c.group]["unchecked" if c.unchecked else c.status] += 1
        unstable += len(c.unstable)
        cats_fail, cats_warn = set(), set()
        for r in c.checks:
            by_check[r.type][r.status] += 1
            if r.status == "skip":
                reason = (r.details or {}).get("skip_reason")
                if reason in SKIP_REASONS:
                    skipped[SKIP_REASONS[reason]] += 1
            if r.status == "fail" and r.severity in ("error", "warn"):
                (cats_fail if r.severity == "error" else cats_warn).add(r.category)
                for p in failure_patterns(r):
                    by_category[r.category]["patterns"][p] += 1
                    by_category[r.category]["examples"].setdefault(p, r.message)
        for r in not_checked(c.checks) if c.unchecked else ():
            reason = f"{r.name or r.type}: " + _short(r.message or "could not run", 110)
            by_category["unchecked"]["patterns"][reason] += 1
            by_category["unchecked"]["examples"].setdefault(reason, r.message)
        for cat in cats_fail:
            by_category[cat]["fail"] += 1
        for cat in cats_warn - cats_fail:
            by_category[cat]["warn"] += 1
    if unchecked:
        by_category["unchecked"]["fail"] = unchecked
    if counts["error"] > unchecked:
        by_category["errors"]["fail"] = counts["error"] - unchecked
        for c in cases:
            if c.status == "error" and not c.unchecked:
                by_category["errors"]["patterns"][re.sub(r"\s*\(.*?\)$", "", (c.error or "error"))[:90]] += 1
    latencies = [c.latency_ms for c in cases if c.latency_ms is not None and (c.status != "error" or c.unchecked)]  # answered
    base_lat = [c.baseline_latency_ms for c in cases if c.baseline_latency_ms is not None]
    tokens_in = sum(int((c.usage or {}).get("input_tokens") or 0) for c in cases)
    tokens_out = sum(int((c.usage or {}).get("output_tokens") or 0) for c in cases)
    # Average lengths over the same cases on both sides (answered, with a baseline).
    both = [c for c in cases if c.status != "error" and c.baseline_output and c.candidate_output]
    cand_words = [word_count(c.candidate_output) for c in both]
    base_words = [word_count(c.baseline_output) for c in both]
    total = len(cases)
    # Each case counts by the share of its runs that passed (0 or 1 without --repeat). A request
    # that got no answer counts against the pass rate; a case whose check could not run (the
    # test's fault, not the model's) is left out of it and reported apart.
    checked = [c for c in cases if not c.unchecked]
    fractions = [c.pass_fraction if c.pass_fraction is not None else float(c.status in ("pass", "warn")) for c in checked]
    rate, low, high = pass_rate_interval(fractions) if fractions else (0.0, 0.0, 0.0)
    flaky = [c.id for c in cases if c.flaky]
    return {
        "total": total,
        "evaluated": evaluated,
        "passed": counts["pass"],
        "warned": counts["warn"],
        "failed": counts["fail"],
        "errors": counts["error"] - unchecked,
        "unchecked": unchecked,
        "pass_rate": stored_rate(rate) if fractions else 0.0,
        "pass_rate_range": [round(low, 4), round(high, 4)] if fractions else None,
        "cases_needed": cases_needed(rate) if fractions else None,  # for a pass rate accurate to +/- 5 points
        "repeats": max((c.runs for c in cases), default=1),
        "flaky": len(flaky),
        "flaky_cases": flaky[:50],
        "unchanged_rate": round(counts["pass"] / len(checked), 4) if checked else 0.0,
        "by_category": {
            k: {
                "fail": v["fail"],
                "warn": v["warn"],
                "patterns": dict(v["patterns"].most_common(8)),
                "examples": {p: v["examples"].get(p, "") for p, _ in v["patterns"].most_common(8)},
            }
            for k, v in by_category.items()
        },
        "by_check": {k: dict(v) for k, v in sorted(by_check.items())},
        "groups": {k: dict(v) for k, v in sorted(groups.items())},
        "latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
        "baseline_latency_ms": {"p50": _percentile(base_lat, 0.5), "p95": _percentile(base_lat, 0.95)} if base_lat else None,
        "tokens": {"input": tokens_in, "output": tokens_out},
        "truncated": sum(1 for c in cases if c.finish_reason == "length"),
        "avg_words": {"baseline": round(statistics.mean(base_words), 1) if base_words else None, "candidate": round(statistics.mean(cand_words), 1) if cand_words else None},
        "skipped_checks": dict(skipped),
        "unstable_checks": unstable,
        "agent": _agent_summary(cases),
    }


def build_insights(summary: dict, cases: list[CaseResult], adapt: list[str]) -> list[str]:
    out = []
    evaluated = summary.get("evaluated", 0)
    if summary.get("unchecked"):
        n = summary["unchecked"]
        out.append(
            f"{n} case{'s' if n != 1 else ''} could not be checked, so the run is inconclusive: a check could not run "
            "(see \"Could not check\"). The pass rate leaves them out; fix the check or the judge and run again."
        )
    if summary.get("truncated"):
        out.append(
            f"{summary['truncated']} response(s) hit the output token limit (finish_reason=length). Reasoning models count "
            "thinking tokens toward max_tokens/max_completion_tokens - raise the limit for the new model."
        )
    fenced = sum(1 for c in cases for r in c.checks if r.type == "json_valid" and r.status == "fail" and (r.details or {}).get("fenced"))
    if fenced:
        out.append(f"JSON wrapped in markdown code fences in {fenced} case(s): parsers that call json.loads() on the raw text will break.")
    refusals = sum(1 for c in cases for r in c.checks if r.type == "refusal" and r.status == "fail")
    if refusals:
        out.append(f"The new model declined {refusals} request(s) that the old model answered.")
    aw = summary.get("avg_words") or {}
    if aw.get("baseline") and aw.get("candidate"):
        change = (aw["candidate"] - aw["baseline"]) / aw["baseline"]
        if abs(change) >= 0.3:
            out.append(f"Answers are {abs(change):.0%} {'longer' if change > 0 else 'shorter'} on average ({aw['baseline']:.0f} -> {aw['candidate']:.0f} words).")
    bl, cl = summary.get("baseline_latency_ms"), summary.get("latency_ms") or {}
    if bl and bl.get("p50") and cl.get("p50"):
        out.append(f"Median latency {bl['p50']:.0f} ms -> {cl['p50']:.0f} ms.")
    for note in adapt:
        out.append(f"Request adapted automatically: {note}.")
    for reason, n in (summary.get("skipped_checks") or {}).items():
        out.append(f"{n} check(s) skipped: {reason}.")
    if summary.get("unstable_checks"):
        out.append(f"{summary['unstable_checks']} check failure(s) ignored because the old model fails them too (noise).")
    runs = summary.get("repeats") or 1
    if runs > 1:
        if summary.get("flaky"):
            ids = ", ".join(summary.get("flaky_cases", [])[:5])
            out.append(f"Each case ran {runs} times: {summary['flaky']} flaky case(s) passed in some runs and failed in others ({ids}).")
        else:
            out.append(f"Each case ran {runs} times and gave the same result every time.")
    agent = summary.get("agent")
    if agent:
        steps = agent.get("avg_steps") or {}
        note = f"Agent tasks ({agent['tasks']}): {steps.get('baseline')} -> {steps.get('candidate')} steps on average"
        cost = agent.get("cost_per_task") or {}
        if cost.get("baseline") and cost.get("candidate") is not None:
            from .pricing import money

            change = (cost["candidate"] - cost["baseline"]) / cost["baseline"]
            note += f"; cost per task {money(cost['baseline'])} -> {money(cost['candidate'])} ({change:+.0%})"
        elif cost.get("candidate") is not None:
            from .pricing import money

            note += f"; cost per task {money(cost['candidate'])}"
        if agent.get("looped"):
            note += f"; {agent['looped']} task(s) looped"
        if agent.get("out_of_steps"):
            note += f"; {agent['out_of_steps']} ran out of steps"
        out.append(note + ".")
    vs = summary.get("vs_baseline")
    if vs and vs.get("compared"):
        b, c, p = vs["regressions"], vs["improvements"], vs["p_value"]
        if vs.get("significantly_worse"):
            out.append(f"Against the old model run live, {b} case(s) broke and {c} improved: significantly worse (McNemar p={p:.3f}).")
        elif b or c:
            out.append(f"Against the old model run live, {b} case(s) broke and {c} improved: not a significant difference (p={p:.2f}).")
    total, need = summary.get("total") or 0, summary.get("cases_needed") or 0
    checked = total - (summary.get("unchecked") or 0)  # the cases the pass rate is over
    if 0 < checked < need:
        low, high = summary.get("pass_rate_range") or (0, 1)
        out.append(
            f"With {plural(checked, 'checked case') if checked < total else plural(total, 'case')} the pass rate is only known to within {low:.0%}-{high:.0%}; "
            f"about {need} cases would pin it down to +/-5 points."
        )
    if summary.get("errors") and evaluated == 0:
        out.append("Every request failed - check the model name and API key.")
    return out


def run_compare(
    cases: list[TestCase],
    candidate: Any,
    *,
    baseline: Any = None,
    judge: Any = None,
    embedder: Any = None,
    config: dict | None = None,
    concurrency: int = 4,
    cache: ResponseCache | None = None,
    allow_python: bool = False,
    severity: dict | None = None,
    patches: Any = None,
    overrides: dict | None = None,
    tests_path: str | None = None,
    min_pass_rate: float | None = None,
    show_progress: bool | None = None,
    label: str = "compare",
    repeat: int = 1,
    judge_mode: str = "auto",
) -> CompareResult:
    config = config or {}
    cases = unique_ids(cases)
    cand: Provider = resolve(candidate, config) if isinstance(candidate, str) else candidate
    base: Provider | None = resolve(baseline, config) if isinstance(baseline, str) else baseline
    judge_p: Provider | None = resolve(judge, config) if isinstance(judge, str) else judge
    if judge_p is not None:
        cases = [with_judge_check(c) for c in cases]
    embed_p: Provider | None = resolve(embedder, config) if isinstance(embedder, str) else embedder
    runs = max(1, int(repeat or 1))
    if judge_mode not in ("auto", "always"):
        raise ValueError(f"judge mode must be auto or always, got {judge_mode!r}")

    # Run 1 shares cached answers with ordinary runs; the repeats are separate requests.
    all_responses = [
        run_cases(
            cand, cases, concurrency=concurrency, cache=cache, patches=patches, overrides=overrides,
            label=cand.spec if runs == 1 else f"{cand.spec} (run {i + 1}/{runs})", salt="" if i == 0 else f"repeat-{i}",
            show_progress=show_progress,
        )
        for i in range(runs)
    ]
    base_responses: dict[str, Response] = {}
    if base is not None:
        base_responses = run_cases(base, cases, concurrency=concurrency, cache=cache, overrides=overrides, label=f"baseline {base.spec}", show_progress=show_progress)

    grader_models: dict[str, Provider | None] = {}
    lock = threading.Lock()

    def resolve_model(spec: str) -> Provider | None:
        with lock:
            if spec not in grader_models:
                try:
                    grader_models[spec] = resolve(spec, config)
                except Exception:
                    grader_models[spec] = None
            return grader_models[spec]

    ctx = CheckContext(
        judge=judge_p,
        embedder=embed_p,
        allow_python=allow_python,
        severity=dict(severity or {}),
        resolve_model=resolve_model,
        judge_cache=cache,
        judge_mode=judge_mode,
    )

    def evaluate(case: TestCase, responses: dict[str, Response]) -> CaseResult:
        resp = responses.get(case.id) or Response(error="no response", error_code="missing")
        live = base_responses.get(case.id)
        eval_case = case
        if not case.has_baseline and live is not None and live.ok:
            eval_case = copy.copy(case)
            eval_case.baseline = {"output": live.text, "tool_calls": live.tool_calls, "model": live.model}
        result = CaseResult(
            id=case.id,
            group=case.group or "default",
            status="error",
            input_preview=request_preview(case.messages),
            baseline_output=eval_case.baseline_output,
            candidate_output=resp.text,
            candidate_tool_calls=resp.tool_calls,
            error=resp.error,
            latency_ms=resp.latency_ms,
            usage=resp.usage,
            finish_reason=resp.finish_reason,
            model=resp.model,
            fingerprint=resp.fingerprint,
        )
        if case.agent:
            result.baseline_trace = agent_trace(case.agent.get("steps") or [])
            result.trace = agent_trace(resp.steps)
        if live is not None:
            result.baseline_live_output = live.text if live.ok else None
            result.baseline_latency_ms = live.latency_ms if live.ok else None
            result.baseline_model = live.model
        if not resp.ok:
            result.status = "error"
            return result
        checks = run_checks(eval_case, Sample.from_response(resp, source=f"candidate {cand.spec}"), ctx)
        sent = {**case.params, **{k: v for k, v in (overrides or {}).items() if v is not None}}
        rejected = request_rejections(cand.model, sent, resp)
        if rejected:
            advice = plan_thinking(cand.model, sent)["advice"] if getattr(cand, "kind", None) == "anthropic" else []
            checks.append(rejection_check(cand.spec, rejected, ctx, advice))
        result.raw_status = case_status(checks, None)
        if live is not None and live.ok:
            base_checks = run_checks(eval_case, Sample.from_response(live, source=f"baseline {getattr(base, 'spec', '')}"), ctx)
            result.baseline_live_status = case_status(base_checks, None)
            for i, (c, b) in enumerate(zip(checks, base_checks)):
                if c.status == "fail" and b.status == "fail":
                    result.unstable.append(c.type)
                    checks[i] = CheckResult(c.type, c.category, c.severity, "skip", f"also fails on the old model (noise): {c.message}", c.score, c.name, c.details)
        result.checks = checks
        result.status = case_status(checks, None)
        if result.status == "error":  # answered, but a check that decides the case could not run
            result.unchecked = True
            reasons = dict.fromkeys(f"{r.name or r.type}: {r.message}" for r in not_checked(checks))
            result.error = "could not check: " + "; ".join(reasons)
        return result

    def evaluate_all(case: TestCase) -> CaseResult:
        return combine_runs([evaluate(case, responses) for responses in all_responses])

    from .runner import interruptible_map

    results = interruptible_map(evaluate_all, cases, concurrency)  # judge calls: Ctrl+C must stop them at once

    adapt = adaptations(cand) + (adaptations(base) if base else [])
    if any(c.agent for c in cases):
        _price_agent_tasks(results, cases, cand.spec, base.spec if base else None, config)
    summary = summarize(results)
    if base is not None:
        summary["vs_baseline"] = paired_comparison(results)
    recorded_models = sorted({c.baseline_model for c in cases if c.baseline_model})
    base_label = base.spec if base else ("recorded" + (f" ({', '.join(recorded_models[:3])})" if recorded_models else ""))
    res = CompareResult(
        candidate=cand.spec,
        baseline=base_label,
        cases=results,
        summary=summary,
        adaptations=adapt,
        tests_path=tests_path,
        judge=judge_p.spec if judge_p else None,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        min_pass_rate=min_pass_rate,
        label=label,
    )
    refused = [r for c in results for r in c.checks if r.type == "request" and r.status == "fail" and r.severity != "info"]
    names = sorted({item.split("=", 1)[0] for r in refused for item in (r.details or {}).get("rejected") or []})
    res.insights = build_insights(summary, results, [note for note in adapt if not _said_by(note, names)])
    if refused:
        flags = " ".join(f"--drop-param {n}" for n in names if n != "assistant prefill")
        res.insights.insert(0, (
            f"{plural(len(refused), 'request')} send settings {cand.spec} rejects ({', '.join(names)}): in production those calls fail "
            "with a 400 error, so the cases fail. Remove the settings from your code (`llmshift upgrade` lists where)"
            + (f", then test with {flags}." if flags else ".")
        ))
    if ctx.judge_failed:  # a dead judge is said first, not hidden behind "0 judge calls"
        res.insights.insert(0, f"The judge failed on {ctx.judge_failed} of {ctx.judge_calls + ctx.judge_failed} call(s), so those checks could not be decided: {ctx.judge_error}")
    if ctx.judge_calls or ctx.judge_skipped:
        note = f"{ctx.judge_calls} judge call(s) answered"
        if ctx.judge_skipped:
            note += f"; {ctx.judge_skipped} skipped because the rule-based checks already decided"
        res.insights.append(note + ".")
    return res


def _agent_summary(cases: list[CaseResult]) -> dict | None:
    tasks = [c for c in cases if c.baseline_trace]
    if not tasks:
        return None
    ran = [c for c in tasks if c.status != "error" and c.trace]
    looped = sum(1 for c in ran for r in c.checks if r.type == "agent" and (r.details or {}).get("looped"))
    stalled = sum(1 for c in ran if c.finish_reason == "step_limit")
    costs = [c.cost_usd for c in ran if c.cost_usd is not None]
    base_costs = [c.baseline_cost_usd for c in ran if c.baseline_cost_usd is not None]
    return {
        "tasks": len(tasks),
        "avg_steps": {
            "baseline": round(statistics.mean(len(c.baseline_trace) for c in ran), 2) if ran else None,
            "candidate": round(statistics.mean(len(c.trace) for c in ran), 2) if ran else None,
        },
        "looped": looped,
        "out_of_steps": stalled,
        "cost_per_task": {
            "baseline": round(statistics.mean(base_costs), 6) if base_costs else None,
            "candidate": round(statistics.mean(costs), 6) if costs else None,
        },
    }


def _price_agent_tasks(results: list[CaseResult], cases: list[TestCase], cand_spec: str, base_spec: str | None, config: dict) -> None:
    """Cost of each agent task, old and new, where prices are known."""
    from .pricing import overrides_from_config, price_for, usage_cost

    try:
        overrides = overrides_from_config(config)
    except ValueError:
        overrides = {}
    by_id = {c.id: c for c in cases}
    new_price = price_for(cand_spec, overrides)
    for r in results:
        case = by_id.get(r.id)
        if case is None or not case.agent:
            continue
        usage = r.usage or {}
        if new_price is not None and isinstance(usage.get("input_tokens"), int):
            r.cost_usd = round(usage_cost(new_price, usage)[0], 8)
        old_price = price_for(base_spec or case.baseline_model or (config or {}).get("baseline") or "", overrides)
        old_usage = (case.baseline or {}).get("usage") or {}
        if old_price is not None and isinstance(old_usage.get("input_tokens"), (int, float)):
            r.baseline_cost_usd = round(usage_cost(old_price, old_usage)[0], 8)


_ORDER = {"pass": 0, "warn": 1, "fail": 2, "error": 3}  # nosec B105  # a check status or setting name, not a password


def combine_runs(runs: list[CaseResult]) -> CaseResult:
    """One result per case from repeated runs: the most common outcome (the worse one
    on a tie), with the share of runs that passed and whether the case is flaky."""
    if len(runs) == 1:
        return runs[0]
    checked = [r for r in runs if not r.unchecked] or runs  # a run whose check could not run is no verdict
    statuses = [r.status for r in checked]
    counts = Counter(statuses)
    shown = max(counts, key=lambda st: (counts[st], _ORDER.get(st, 0)))
    result = next(r for r in checked if r.status == shown)
    passed = sum(1 for st in statuses if st in ("pass", "warn"))
    answered = sum(1 for st in statuses if st != "error")
    result.runs = len(runs)
    result.run_statuses = ["unchecked" if r.unchecked else r.status for r in runs]
    result.pass_fraction = round(passed / len(checked), 4)  # a run that got no answer has not passed
    # Flaky is the model answering differently, not a request that failed (a timeout, a 429).
    result.flaky = 0 < passed < answered
    return result


def paired_comparison(results: list[CaseResult]) -> dict:
    """The new model against the old one on the same cases (McNemar test): did more
    cases break than got fixed, beyond what chance explains?"""
    regressions = improvements = compared = 0
    for c in results:
        if c.baseline_live_status in (None, "error") or c.raw_status in (None, "error"):  # no answer, or not checked
            continue
        compared += 1
        old_ok, new_ok = c.baseline_live_status in ("pass", "warn"), c.raw_status in ("pass", "warn")
        regressions += old_ok and not new_ok
        improvements += new_ok and not old_ok
    p = mcnemar_p(regressions, improvements)
    return {
        "compared": compared,
        "regressions": regressions,
        "improvements": improvements,
        "p_value": round(p, 4),
        "significantly_worse": regressions > improvements and p < 0.05,
    }
