"""Renderers for compare results (terminal, Markdown, HTML, JUnit) and scan
results (terminal, Markdown, SARIF, GitHub annotations)."""

from __future__ import annotations

import html
import json
import re
from typing import Any

from . import __version__
from .checks import CATEGORY_TITLES
from .compare import CaseResult, CompareResult
from .console import bar, status_label, style, sym, truncate, width
from .formats import parse_json_lenient
from .textutil import PROJECT_URL, credit, diff_ratio, md_inert, plural, quoteattr, show_bidi, word_diff, xml_escape

STATUS_ORDER = {"fail": 0, "error": 1, "warn": 2, "pass": 3}  # nosec B105  # a check status or setting name, not a password
CATEGORY_ORDER = ["format", "facts", "tools", "safety", "semantics", "tone", "custom", "errors", "unchecked"]


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _shown_rate(s: dict) -> float | None:
    """The pass rate, or None ("n/a") when no case could be checked: it is over no case."""
    return s.get("pass_rate", 0.0) if s.get("total", 0) > s.get("unchecked", 0) else None


def range_text(s: dict) -> str:
    """"95% range 76%-88% over 50 cases x 3 runs; 2 flaky", or "over 48 of 50 cases" when 2 could
    not be checked (empty without a summary)."""
    rng = s.get("pass_rate_range")
    if not rng or not s.get("total"):
        return ""
    unchecked = s.get("unchecked") or 0  # left out of the pass rate, so of its range
    over = f"{s['total'] - unchecked} of {s['total']} cases" if unchecked else plural(s["total"], "case")
    text = f"95% range {_pct(rng[0])}-{_pct(rng[1])} over {over}"
    if (s.get("repeats") or 1) > 1:
        text += f" x {s['repeats']} runs"
    if s.get("flaky"):
        text += f"; {s['flaky']} flaky"
    return text


def _cat_title(cat: str) -> str:
    return CATEGORY_TITLES.get(cat, cat.title())


def _check_icon(status: str, severity: str) -> str:
    if status == "pass":
        return style(sym("pass"), "green")
    if status == "skip":
        return style(sym("skip"), "grey")
    if severity == "error" and status == "fail":
        return style(sym("fail"), "red", "bold")
    return style(sym("warn"), "yellow")


def terminal_diff(a: str, b: str, limit: int = 1200) -> str:
    segs = word_diff(a[:limit * 2], b[:limit * 2])
    out = []
    from .console import color_enabled

    color = color_enabled()
    for op, text in segs:
        if op == "=":
            out.append(text)
        elif op == "-":
            out.append(style(text, "red") if color else f"[-{text}-]")
        else:
            out.append(style(text, "green", "bold") if color else f"{{+{text}+}}")
    s = "".join(out)
    return s if len(s) <= limit * 3 else s[: limit * 3] + " ..."


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# ---------------------------------------------------------------- terminal


def compare_terminal(result: CompareResult, *, show: int = 5, verbose: bool = False) -> str:
    s = result.summary
    lines = []
    lines.append(style(f"LLMShift {result.label}", "bold") + f"  {result.baseline}  {sym('arrow')}  {style(result.candidate, 'bold')}")
    if result.tests_path:
        lines.append(style(f"Tests: {result.tests_path}  ({plural(s.get('total', 0), 'case')}, {plural(len(s.get('groups', {})), 'prompt template')})", "grey"))
    lines.append("")
    counts = (
        f"  {style(sym('pass') + ' ' + str(s.get('passed', 0)) + ' unchanged', 'green')}"
        f"   {style(sym('warn') + ' ' + str(s.get('warned', 0)) + ' changed', 'yellow')}"
        f"   {style(sym('fail') + ' ' + str(s.get('failed', 0)) + ' failed', 'red', 'bold')}"
    )
    if s.get("errors"):
        counts += f"   {style(sym('error') + ' ' + plural(s['errors'], 'error'), 'magenta', 'bold')}"
    if s.get("unchecked"):
        counts += f"   {style(sym('error') + ' ' + str(s['unchecked']) + ' not checked', 'magenta', 'bold')}"
    lines.append(counts)
    rate = s.get("pass_rate", 0.0)
    threshold = result.min_pass_rate
    verdict = ""
    if threshold is not None:
        ok = result.passed
        if result.inconclusive:  # the exit code is 2: what could be checked passed (or nothing could be), not everything could be
            verdict = "  " + style(f"{sym('warn')} inconclusive: {s['unchecked']} not checked", "yellow", "bold")
        else:
            verdict = "  " + (style(f"{sym('pass')} meets the {threshold:.0%} threshold", "green") if ok else style(f"{sym('fail')} below the {threshold:.0%} threshold", "red", "bold"))
    lines.append(f"  pass rate {bar(rate)} {_pct(_shown_rate(s))}{verdict}")
    if range_text(s):
        lines.append(style(f"            {range_text(s)}", "grey"))
    cats = s.get("by_category") or {}
    if cats:
        lines.append("")
        lines.append(style("What changed", "bold"))
        for cat in sorted(cats, key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99):
            info = cats[cat]
            n_fail, n_warn = info.get("fail", 0), info.get("warn", 0)
            if not n_fail and not n_warn:
                continue
            label = []
            if n_fail:
                label.append(style(f"{n_fail} not checked", "magenta") if cat == "unchecked" else style(f"{n_fail} failed", "red"))
            if n_warn:
                label.append(style(f"{n_warn} changed", "yellow"))
            lines.append(f"  {_cat_title(cat):<24} {', '.join(label)}")
            patterns = info.get("patterns") or {}
            examples = info.get("examples") or {}
            if cat == "tone":
                joined = " · ".join(f"{p} ({n})" for p, n in list(patterns.items())[:5]) if unicode_ok() else "; ".join(f"{p} ({n})" for p, n in list(patterns.items())[:5])
                lines.append(style(f"      {truncate(joined, width() - 8)}", "grey"))
                continue
            for p, n in list(patterns.items())[:3]:
                ex = examples.get(p) or ""
                text = f"{p} ({n})"
                if ex and not ex.strip().startswith(p.strip().removesuffix("...").rstrip()):
                    text += f", e.g. {ex}"
                lines.append(style(f"      {truncate(text, width() - 8)}", "grey"))
    groups = s.get("groups") or {}
    bad_groups = {g: v for g, v in groups.items() if v.get("fail") or v.get("warn") or v.get("error") or v.get("unchecked")}
    if len(groups) > 1 and bad_groups:
        lines.append("")
        lines.append(style("By prompt template", "bold"))
        for g, v in sorted(bad_groups.items(), key=lambda kv: (-kv[1].get("fail", 0), -kv[1].get("warn", 0)))[:10]:
            parts = []
            if v.get("fail"):
                parts.append(style(f"{v['fail']} failed", "red"))
            if v.get("warn"):
                parts.append(style(f"{v['warn']} changed", "yellow"))
            if v.get("error"):
                parts.append(style(plural(v["error"], "error"), "magenta"))
            if v.get("unchecked"):
                parts.append(style(f"{v['unchecked']} not checked", "magenta"))
            lines.append(f"  {truncate(g, 44):<44} {v.get('total', 0):>4} {'case ' if v.get('total', 0) == 1 else 'cases'}  " + ", ".join(parts))
    problem = _interleave_groups([c for c in result.cases if c.status != "pass"])
    if problem:
        limit = len(problem) if verbose or show < 0 else min(show, len(problem))
        lines.append("")
        title = f"Cases that changed ({len(problem)})"
        if limit < len(problem):
            title += f" - showing {limit}; use --show all or open the HTML report"
        lines.append(style(title, "bold"))
        for case in problem[:limit]:
            lines.append("")
            lines.extend(_case_block(case, verbose))
    if result.insights:
        lines.append("")
        lines.append(style("Notes", "bold"))
        for note in result.insights:
            lines.append(f"  {sym('info')} {note}")
    return "\n".join(lines)


def _interleave_groups(cases: list[CaseResult]) -> list[CaseResult]:
    """Round-robin over prompt templates (worst case of each first) so a short list covers them all."""
    buckets: dict[str, list[CaseResult]] = {}
    for c in sorted(cases, key=lambda c: (STATUS_ORDER.get(c.status, 9), c.id)):
        buckets.setdefault(c.group, []).append(c)
    out: list[CaseResult] = []
    while any(buckets.values()):
        heads = [b.pop(0) for b in buckets.values() if b]
        out.extend(sorted(heads, key=lambda c: (STATUS_ORDER.get(c.status, 9), c.group)))
    return out


def _case_block(case: CaseResult, verbose: bool) -> list[str]:
    out = [f"  {status_label(case.status)}  {style(case.id, 'bold')}  {style(case.group, 'grey')}"]
    if case.flaky:
        passed = sum(1 for st in case.run_statuses if st in ("pass", "warn"))
        out.append(style(f"      Flaky: passed {passed} of {case.runs} runs ({', '.join(case.run_statuses)})", "yellow"))
    if case.input_preview:
        out.append(f"      Input: {truncate(case.input_preview, width() - 14)}")
    if case.status == "error":
        out.append(style(f"      Error: {case.error}", "magenta"))
        return out
    if case.baseline_trace:
        arrow = f" {sym('arrow')} "
        out.append(style("      Steps (old): ", "grey") + truncate(arrow.join(case.baseline_trace), width() - 20))
        out.append(style("      Steps (new): ", "grey") + truncate(arrow.join(case.trace) or "(none)", width() - 20))
    for r in case.checks:
        if r.status == "pass" or (r.status == "skip" and not verbose):
            continue
        cat = r.category
        out.append(f"      {_check_icon(r.status, r.severity)} {cat:<9} {truncate(r.message or r.type, width() - 24)}")
    if case.baseline_output or case.candidate_output:
        short = len(case.baseline_output) + len(case.candidate_output) < 1500
        structured = parse_json_lenient(case.baseline_output) is not None
        if not (verbose or short):
            out.append(style("      (long outputs - see the HTML report for the full diff)", "grey"))
        elif structured or diff_ratio(case.baseline_output, case.candidate_output) > 0.5:
            out.append(style("      Old:", "grey") + " " + _indent(case.baseline_output.strip(), "           ").lstrip())
            out.append(style("      New:", "grey") + " " + _indent(case.candidate_output.strip(), "           ").lstrip())
        else:
            out.append(style("      Diff (old -> new):", "grey"))
            out.append(_indent(terminal_diff(case.baseline_output, case.candidate_output), "        "))
    return out


def unicode_ok() -> bool:
    from .console import unicode_enabled

    return unicode_enabled()


# ---------------------------------------------------------------- markdown


_MENTION_RE = re.compile(r"(?<![\w`])@(?=[A-Za-z0-9])")


def _md(text: str) -> str:
    """Safe inside a Markdown table cell: one line, escaped pipes and tags, no links,
    images or autolinks, and no live @mentions (a model's answer quoted in a PR comment
    must not notify anyone or load a tracking pixel)."""
    text = md_inert(show_bidi(" ".join(str(text or "").split()))).replace("|", "\\|").replace("<", "&lt;")
    return _MENTION_RE.sub("@\u200b", text)


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")
_STATUSES = {"pass", "warn", "fail", "error", "skip"}


def _xml_text(text: str) -> str:
    """XML 1.0 forbids most control characters (ANSI escapes, form feeds) even when escaped;
    bidi controls, which would reorder what the page shows, are shown as escapes."""
    return show_bidi(_CONTROL_RE.sub("", text or ""))


def _status(value: str) -> str:
    return value if value in _STATUSES else "error"


def compare_markdown(result: CompareResult, *, max_cases: int = 15, title: str = "LLMShift report") -> str:
    s = result.summary
    lines = [f"## {title}", ""]
    lines.append(f"**{result.baseline}** → **{result.candidate}**" + (f" · tests: `{result.tests_path}`" if result.tests_path else ""))
    lines.append("")
    verdict = ""
    if result.min_pass_rate is not None:
        verdict = " ✅ meets threshold" if result.passed else f" ❌ below the {result.min_pass_rate:.0%} threshold"
        if result.inconclusive:
            verdict = f" ⚠️ inconclusive: {s['unchecked']} not checked"
    lines.append("| Pass rate | Unchanged | Changed | Failed | Errors |")
    lines.append("|---|---|---|---|---|")
    not_checked = f" (+{s['unchecked']} not checked)" if s.get("unchecked") else ""
    lines.append(f"| **{_pct(_shown_rate(s))}**{verdict} | {s.get('passed', 0)} | {s.get('warned', 0)} | {s.get('failed', 0)} | {s.get('errors', 0)}{not_checked} |")
    if range_text(s):
        lines += ["", f"_{range_text(s)}_"]
    cats = s.get("by_category") or {}
    rows = [(c, cats[c]) for c in sorted(cats, key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99) if cats[c].get("fail") or cats[c].get("warn")]
    if rows:
        lines += ["", "### What changed", "", "| Category | Failed | Changed | Most common |", "|---|---|---|---|"]
        for cat, info in rows:
            top = "; ".join(f"{p} ({n})" for p, n in list((info.get("patterns") or {}).items())[:3])
            lines.append(f"| {_cat_title(cat)} | {info.get('fail', 0)} | {info.get('warn', 0)} | {_md(top)} |")
    if result.insights:
        lines += ["", "### Notes", ""]
        lines += [f"- {_md(n)}" for n in result.insights]
    problem = sorted([c for c in result.cases if c.status != "pass"], key=lambda c: (STATUS_ORDER.get(c.status, 9), c.id))
    if problem:
        lines += ["", f"### Cases that changed ({len(problem)})", ""]
        for case in problem[:max_cases]:
            icon = {"fail": "❌", "warn": "⚠️", "error": "💥"}.get(case.status, "")
            cats_failing = sorted({r.category for r in case.checks if r.status in ("fail", "error")})
            lines.append(f"<details><summary>{icon} <code>{_xml_text(html.escape(case.id))}</code> · {_xml_text(html.escape(case.group))} · {html.escape(', '.join(cats_failing) or case.status)}</summary>")
            lines.append("")
            if case.input_preview:
                lines.append(f"**Input:** {_md(case.input_preview)}")
                lines.append("")
            if case.status == "error":
                lines.append(f"**Error:** {_md(case.error or '')}")
            else:
                lines.append("| | Check | Result |")
                lines.append("|---|---|---|")
                for r in case.checks:
                    if r.status in ("fail", "error"):
                        mark = "❌" if (r.severity == "error" and r.status == "fail") else "⚠️"
                        lines.append(f"| {mark} | {r.category} / `{r.type}` | {_md(r.message)} |")
                lines.append("")
                lines.append("```diff")
                for line in (case.baseline_output or "").splitlines()[:30]:
                    lines.append("- " + show_bidi(line))
                for line in (case.candidate_output or "").splitlines()[:30]:
                    lines.append("+ " + show_bidi(line))
                lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")
        if len(problem) > max_cases:
            lines.append(f"_{len(problem) - max_cases} more - see the HTML report._")
    lines.append("")
    lines.append(credit())
    return show_bidi("\n".join(lines) + "\n")


# -------------------------------------------------------------------- junit


def compare_junit(result: CompareResult) -> str:
    s = result.summary
    suite = f"llmshift {result.label}: {result.candidate}"
    total_time = sum((c.latency_ms or 0) for c in result.cases) / 1000.0
    errors = s.get("errors", 0) + s.get("unchecked", 0)  # a case whose check could not run is an error, not a failure
    out = ['<?xml version="1.0" encoding="UTF-8"?>']
    out.append(f"<testsuites name=\"llmshift\" tests=\"{s.get('total', 0)}\" failures=\"{s.get('failed', 0)}\" errors=\"{errors}\" time=\"{total_time:.3f}\">")
    out.append(f"  <testsuite name={quoteattr(_xml_text(suite))} tests=\"{s.get('total', 0)}\" failures=\"{s.get('failed', 0)}\" errors=\"{errors}\" skipped=\"0\" time=\"{total_time:.3f}\">")
    for case in result.cases:
        t = (case.latency_ms or 0) / 1000.0
        out.append(f"    <testcase classname={quoteattr(_xml_text(case.group))} name={quoteattr(_xml_text(case.id))} time=\"{t:.3f}\">")
        if case.status == "error":
            kind = "not_checked" if case.unchecked else "api_error"
            out.append(f"      <error message={quoteattr(_xml_text(case.error or 'error')[:300])} type=\"{kind}\"/>")
        elif case.status == "fail":
            failing = [r for r in case.checks if r.status == "fail" and r.severity == "error"]
            msg = "; ".join(f"{r.category}: {r.message}" for r in failing)[:500]
            body = "\n".join(f"[{r.category}/{r.type}] {r.message}" for r in case.checks if r.status in ("fail", "error"))
            body += "\n\n--- old\n" + (case.baseline_output or "")[:2000] + "\n--- new\n" + (case.candidate_output or "")[:2000]
            out.append(f"      <failure message={quoteattr(_xml_text(msg))} type=\"regression\">{xml_escape(_xml_text(body))}</failure>")
        elif case.status == "warn":
            notes = "\n".join(f"[{r.category}/{r.type}] {r.message}" for r in case.checks if r.status in ("fail", "error"))
            out.append(f"      <system-out>{xml_escape(_xml_text('changed (not failing): ' + notes))}</system-out>")
        out.append("    </testcase>")
    out.append("  </testsuite>")
    out.append("</testsuites>")
    return show_bidi("\n".join(out) + "\n")


# --------------------------------------------------------------------- html

_HTML_CSS = """
:root{--bg:#f7f7f8;--panel:#ffffff;--text:#1c1d22;--muted:#5f6370;--border:#e3e4e8;--fail:#c62f3a;--fail-bg:#fdecee;
--warn:#a45c00;--warn-bg:#fff4e0;--pass:#1f7a4a;--pass-bg:#e7f6ee;--err:#7a3db8;--err-bg:#f3eafd;--del:#fbe2e4;--ins:#dff3e6;
--code:#f2f3f5;--accent:#3557d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121317;--panel:#1b1c21;--text:#e8e9ed;--muted:#a0a3ad;
--border:#2c2e35;--fail:#ff6b75;--fail-bg:#3a1d21;--warn:#ffb648;--warn-bg:#382a14;--pass:#4ccf8a;--pass-bg:#17301f;--err:#c39bff;
--err-bg:#2a1f3a;--del:#4a2227;--ins:#1e3a28;--code:#24262d;--accent:#8aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.sub{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:18px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}.card .n{font-size:26px;font-weight:650}
.card .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--border);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border);vertical-align:top}th{font-size:12px;color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:0}.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.pill.fail{color:var(--fail);background:var(--fail-bg)}.pill.warn{color:var(--warn);background:var(--warn-bg)}
.pill.pass{color:var(--pass);background:var(--pass-bg)}.pill.error{color:var(--err);background:var(--err-bg)}.pill.skip{color:var(--muted);background:var(--code)}
.bar{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.bar button{border:1px solid var(--border);background:var(--panel);color:var(--text);
padding:5px 12px;border-radius:999px;cursor:pointer;font:inherit}.bar button.on{border-color:var(--accent);color:var(--accent)}
.bar input,.bar select{flex:1;min-width:180px;border:1px solid var(--border);background:var(--panel);color:var(--text);border-radius:8px;padding:6px 10px;font:inherit}
details.case{background:var(--panel);border:1px solid var(--border);border-radius:10px;margin:8px 0}
details.case>summary{cursor:pointer;padding:10px 12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;list-style:none}
details.case>summary::-webkit-details-marker{display:none}.cid{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
.grp{color:var(--muted);font-size:12px}.prev{color:var(--muted);flex-basis:100%;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.body{padding:0 12px 14px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media (max-width:760px){.cols{grid-template-columns:1fr}}
pre{background:var(--code);border-radius:8px;padding:10px;margin:6px 0;white-space:pre-wrap;word-break:break-word;font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;max-height:420px;overflow:auto}
del{background:var(--del);text-decoration:line-through}ins{background:var(--ins);text-decoration:none}.h{font-size:12px;color:var(--muted);margin-top:10px;font-weight:600}
ul.notes{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px 10px 10px 30px;margin:0}
footer{color:var(--muted);font-size:12px;margin-top:30px}.hidden{display:none}
"""

_HTML_JS = """
(function(){var f='all',q='',g='';var cases=[].slice.call(document.querySelectorAll('details.case'));
function apply(){cases.forEach(function(c){var ok=(f==='all'||c.dataset.status===f)&&(!g||c.dataset.group===g)&&(!q||c.dataset.text.indexOf(q)>=0);c.classList.toggle('hidden',!ok);});}
[].forEach.call(document.querySelectorAll('.bar button'),function(b){b.addEventListener('click',function(){[].forEach.call(document.querySelectorAll('.bar button'),function(x){x.classList.remove('on')});b.classList.add('on');f=b.dataset.f;apply();});});
var s=document.getElementById('q');if(s){s.addEventListener('input',function(){q=s.value.toLowerCase();apply();});}
var gs=document.getElementById('g');if(gs){gs.addEventListener('change',function(){g=gs.value;apply();});}apply();})();
"""


def _h(text: Any) -> str:
    """Text for an HTML page: escaped, and without the control characters (NUL, ESC...)
    that model output can carry and HTML cannot show."""
    return html.escape(_xml_text(str(text)))


def _html_diff(a: str, b: str) -> str:
    out = []
    for op, text in word_diff(a or "", b or ""):
        t = _h(text)
        out.append(t if op == "=" else (f"<del>{t}</del>" if op == "-" else f"<ins>{t}</ins>"))
    return "".join(out)


def compare_html(result: CompareResult) -> str:
    s = result.summary
    e = _h
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">']
    parts.append(f"<title>LLMShift report</title><style>{_HTML_CSS}</style></head><body><main>")
    parts.append(f"<h1>LLMShift {e(result.label)}: {e(result.baseline)} &rarr; {e(result.candidate)}</h1>")
    meta = [f"generated {e(result.created_at)}"]
    if result.tests_path:
        meta.append(f"tests <code>{e(result.tests_path)}</code>")
    if result.judge:
        meta.append(f"judge {e(result.judge)}")
    parts.append(f"<div class=\"sub\">{' &middot; '.join(meta)}</div>")
    verdict = ""
    if result.min_pass_rate is not None:
        verdict = f"<div class=\"l\">{'meets' if result.passed else 'below'} the {result.min_pass_rate:.0%} threshold</div>"
        if result.inconclusive:
            verdict = f"<div class=\"l\">inconclusive: {s['unchecked']} not checked</div>"
    parts.append('<div class="cards">')
    rng = f"<div class=\"l\">{e(range_text(s))}</div>" if range_text(s) else ""
    parts.append(f"<div class=\"card\"><div class=\"l\">Pass rate</div><div class=\"n\">{_pct(_shown_rate(s))}</div>{verdict}{rng}</div>")
    tiles = [("Unchanged", "passed", "pass"), ("Changed", "warned", "warn"), ("Failed", "failed", "fail"), ("Errors", "errors", "error")]
    if s.get("unchecked"):
        tiles.append(("Not checked", "unchecked", "error"))
    for label, key, cls in tiles:
        parts.append(f"<div class=\"card\"><div class=\"l\">{label}</div><div class=\"n\"><span class=\"pill {cls}\">{s.get(key, 0)}</span></div></div>")
    parts.append("</div>")
    cats = s.get("by_category") or {}
    rows = [c for c in sorted(cats, key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99) if cats[c].get("fail") or cats[c].get("warn")]
    if rows:
        parts.append("<h2>What changed</h2><table><tr><th>Category</th><th>Failed</th><th>Changed</th><th>Most common</th></tr>")
        for cat in rows:
            info = cats[cat]
            top = "; ".join(f"{p} ({n})" for p, n in list((info.get("patterns") or {}).items())[:4])
            parts.append(f"<tr><td>{e(_cat_title(cat))}</td><td>{info.get('fail', 0)}</td><td>{info.get('warn', 0)}</td><td>{e(top)}</td></tr>")
        parts.append("</table>")
    if result.insights:
        parts.append("<h2>Notes</h2><ul class=\"notes\">" + "".join(f"<li>{e(n)}</li>" for n in result.insights) + "</ul>")
    groups = sorted({c.group for c in result.cases})
    parts.append(f"<h2>Cases ({len(result.cases)})</h2>")
    parts.append('<div class="bar">')
    for f, label in (("all", "All"), ("fail", "Failed"), ("warn", "Changed"), ("error", "Errors"), ("pass", "Unchanged")):
        on = ' class="on"' if f == "all" else ""
        parts.append(f'<button data-f="{f}"{on}>{label}</button>')
    parts.append('<input id="q" type="search" placeholder="Search input, output or check messages">')
    if len(groups) > 1:
        parts.append('<select id="g"><option value="">All prompt templates</option>' + "".join(f"<option>{e(g)}</option>" for g in groups) + "</select>")
    parts.append("</div>")
    for case in sorted(result.cases, key=lambda c: (STATUS_ORDER.get(c.status, 9), c.group, c.id)):
        failing = [r for r in case.checks if r.status in ("fail", "error")]
        text_index = " ".join([case.id, case.group, case.input_preview, case.candidate_output[:2000], case.baseline_output[:2000]] + [r.message for r in case.checks]).lower()
        parts.append(f"<details class=\"case\" data-status=\"{_status(case.status)}\" data-group=\"{e(case.group)}\" data-text=\"{e(text_index)}\">")
        cats_failing = ", ".join(sorted({r.category for r in failing}))
        label = {"warn": "CHANGED"}.get(case.status, case.status.upper())
        if case.flaky:
            label += f" (flaky {sum(1 for st in case.run_statuses if st in ('pass', 'warn'))}/{case.runs})"
        parts.append(f"<summary><span class=\"pill {_status(case.status)}\">{e(label)}</span><span class=\"cid\">{e(case.id)}</span><span class=\"grp\">{e(case.group)}</span><span class=\"grp\">{e(cats_failing)}</span><span class=\"prev\">{e(case.input_preview)}</span></summary><div class=\"body\">")
        if case.status == "error":
            parts.append(f"<div class=\"h\">Error</div><pre>{e(case.error or '')}</pre>")
        else:
            if case.baseline_trace:
                parts.append("<div class=\"h\">Steps</div><table><tr><th>#</th><th>Old model</th><th>New model</th></tr>")
                for i in range(max(len(case.baseline_trace), len(case.trace))):
                    old = case.baseline_trace[i] if i < len(case.baseline_trace) else ""
                    new = case.trace[i] if i < len(case.trace) else ""
                    parts.append(f"<tr><td>{i + 1}</td><td>{e(old)}</td><td>{e(new)}</td></tr>")
                parts.append("</table>")
            parts.append("<table><tr><th></th><th>Category</th><th>Check</th><th>Result</th></tr>")
            for r in sorted(case.checks, key=lambda r: {"fail": 0, "error": 1, "skip": 3, "pass": 4}.get(r.status, 2)):  # nosec B105  # a check status or setting name, not a password
                cls = "fail" if (r.status == "fail" and r.severity == "error") else ("warn" if r.status in ("fail", "error") else r.status)
                text = "changed" if cls == "warn" and r.status == "fail" else r.status
                parts.append(f"<tr><td><span class=\"pill {_status(cls)}\">{e(text)}</span></td><td>{e(r.category)}</td><td><code>{e(r.type)}</code></td><td>{e(r.message)}</td></tr>")
            parts.append("</table>")
            parts.append('<div class="cols"><div><div class="h">Old model (baseline)</div>')
            parts.append(f"<pre>{e(case.baseline_output)}</pre></div><div><div class=\"h\">New model ({e(case.model or result.candidate)})</div><pre>{e(case.candidate_output)}</pre>")
            if case.candidate_tool_calls:
                parts.append(f"<div class=\"h\">Tool calls</div><pre>{e(json.dumps(case.candidate_tool_calls, indent=2, ensure_ascii=False))}</pre>")
            parts.append("</div></div>")
            small_edit = diff_ratio(case.baseline_output, case.candidate_output) <= 0.5
            if case.baseline_output and case.candidate_output and small_edit and parse_json_lenient(case.baseline_output) is None:
                parts.append(f"<div class=\"h\">Word diff</div><pre>{_html_diff(case.baseline_output, case.candidate_output)}</pre>")
        parts.append("</div></details>")
    parts.append(f"<footer>Generated by <a href=\"{PROJECT_URL}\">llmshift</a> {e(__version__)}</footer></main><script>{_HTML_JS}</script></body></html>")
    return show_bidi("".join(parts))


# --------------------------------------------------------------------- scan


def _scan_label(lk: Any, warn_days: int) -> tuple[str, tuple]:
    if lk is None:
        return "?", ("grey",)
    if lk.status == "retired":
        return f"{sym('fail')} RETIRED", ("red", "bold")
    if lk.status == "retiring":
        d = lk.days_left
        text = f"{sym('warn')} {d} DAY{'S' if d != 1 else ''} LEFT"
        return (text, ("yellow", "bold")) if d is not None and d <= warn_days else (f"{sym('info')} RETIRES IN {d}d", ("cyan",))
    if lk.status == "deprecated":
        return f"{sym('warn')} DEPRECATED", ("yellow",)
    if lk.status == "unknown":
        return "? UNKNOWN", ("grey",)
    if lk.severity == "notice":
        return f"{sym('info')} NOTE", ("cyan",)
    if lk.alias:
        return f"{sym('info')} ALIAS", ("cyan",)
    return f"{sym('pass')} OK", ("green",)


def scan_terminal(scan: Any, *, verbose: bool = False, max_locations: int = 4) -> str:
    models = scan.models()
    counted = f"{plural(scan.files_scanned, 'file')}, {plural(len(scan.findings), 'model reference')}, {plural(len(models), 'distinct model')}"
    lines = [style("LLMShift scan", "bold") + f"  {counted}" + style(f"   (lifecycle data: {scan.registry_updated})", "grey")]
    if not models:
        lines.append("")
        lines.append("  No model IDs found. Try --include-docs, or check the paths you passed.")
        lines += [style(s, "grey") for s in _skipped(scan)]
        return "\n".join(lines)
    lines.append("")
    for m in models:
        lk = m["lookup"]
        if lk is not None and lk.status == "active" and lk.severity == "ok" and not verbose:
            continue
        label, st = _scan_label(lk, scan.warn_days)
        provider = lk.provider if lk else "?"
        lines.append(f"  {style(f'{label:<18}', *st)} {style(m['model'], 'bold')}  {style(provider, 'grey')}")
        if lk is not None:
            lines.append(f"  {'':<18} {lk.message}")
        for f in m["findings"][:max_locations]:
            where = f.location
            extra = f"  (via {f.via})" if f.via else ""
            ctx = truncate(f.context, max(20, width() - len(where) - 26)) if f.context else (f.call or f.kind)
            lines.append(style(f"  {'':<18}   {where}  {ctx}{extra}", "grey"))
        if len(m["findings"]) > max_locations:
            lines.append(style(f"  {'':<18}   ... {len(m['findings']) - max_locations} more", "grey"))
    ok_models = [m for m in models if m["lookup"] and m["lookup"].status == "active" and m["lookup"].severity == "ok"]
    if ok_models and not verbose:
        ok_label = sym("pass") + " OK"
        ok_cell = style(f"{ok_label:<18}", "green")
        names = ", ".join(m["model"] for m in ok_models[:12]) + (" ..." if len(ok_models) > 12 else "")
        lines.append("  " + ok_cell + " " + names)
    c = scan.counts()
    lines.append("")
    summary = []
    if c["retired"]:
        summary.append(style(f"{c['retired']} retired", "red", "bold"))
    if c["retiring_soon"]:
        summary.append(style(f"{c['retiring_soon']} retiring within {scan.warn_days} days", "yellow"))
    if c["retiring"] - c["retiring_soon"]:
        summary.append(f"{c['retiring'] - c['retiring_soon']} retiring later")
    if c["deprecated"]:
        summary.append(style(f"{c['deprecated']} deprecated", "yellow"))
    if c["alias"]:
        summary.append(f"{c['alias']} unpinned alias")
    if c["unknown"]:
        summary.append(f"{c['unknown']} unknown")
    if c["active"]:
        summary.append(style(f"{c['active']} ok", "green"))
    lines.append("Summary: " + ", ".join(summary))
    lines += [style(s, "grey") for s in _skipped(scan)]
    return "\n".join(lines)


def _skipped(scan: Any, shown: int = 5) -> list[str]:
    """Files the scan could not read: always listed, so "nothing found" never hides them."""
    out = [f"  skipped {err}" for err in scan.errors[:shown]]
    if len(scan.errors) > shown:
        out.append(f"  ... and {len(scan.errors) - shown} more file(s) skipped")
    return out


def scan_markdown(scan: Any) -> str:
    lines = ["## LLMShift scan", "", f"{plural(scan.files_scanned, 'file')} scanned, {plural(len(scan.findings), 'model reference')}, lifecycle data from {scan.registry_updated}.", ""]
    models = scan.models()
    skipped = ["- " + s.strip() for s in _skipped(scan)]
    if not models:
        return show_bidi("\n".join(lines + ["No model IDs found."] + ([""] + skipped if skipped else [])) + "\n")
    lines += ["| Status | Model | Provider | Retires | Replacement | Where |", "|---|---|---|---|---|---|"]
    for m in models:
        lk = m["lookup"]
        label = _scan_label(lk, scan.warn_days)[0]
        icon = {"critical": "🔴", "warning": "🟠", "notice": "🔵", "info": "⚪", "ok": "🟢"}.get(lk.severity if lk else "info", "")
        where = ", ".join(f"`{f.location}`" for f in m["findings"][:3]) + (f" +{len(m['findings']) - 3}" if len(m["findings"]) > 3 else "")
        retires = lk.retires.isoformat() if lk and lk.retires else ""
        lines.append(f"| {icon} {label} | `{m['model']}` | {lk.provider if lk else ''} | {retires} | {('`' + lk.replacement + '`') if lk and lk.replacement else ''} | {where} |")
    if skipped:
        lines += ["", *skipped]
    return show_bidi("\n".join(lines) + "\n")


_SARIF_RULES = {
    "retired-model": ("error", "Model is retired", "The provider has shut this model down; calls fail or will fail."),
    "retiring-model": ("warning", "Model retires soon", "The provider has announced a shutdown date for this model."),
    "deprecated-model": ("warning", "Model is deprecated", "The provider has deprecated this model."),
    "model-alias": ("note", "Unpinned model alias", "Aliases can be re-pointed to a new model without notice."),
    "unknown-model": ("note", "Model not in the lifecycle registry", "LLMShift has no lifecycle data for this model."),
    "platform-retirement": ("note", "Retired on another platform", "The same model ID is retired on another platform."),
}


def _rule_for(lk: Any, warn_days: int) -> str | None:
    if lk is None:
        return None
    if lk.status == "retired":
        return "retired-model"
    if lk.status == "retiring":
        return "retiring-model" if (lk.days_left or 0) <= warn_days else "deprecated-model"
    if lk.status == "deprecated":
        return "deprecated-model"
    if lk.status == "unknown":
        return "unknown-model"
    if lk.severity == "notice":
        return "platform-retirement"
    if lk.alias:
        return "model-alias"
    return None


def _in_workspace(path: str) -> str:
    """A finding's path as GitHub places annotations and code-scanning results: relative to
    the checkout (GITHUB_WORKSPACE), whatever folder the command ran in."""
    import os
    from pathlib import Path

    workspace = os.environ.get("GITHUB_WORKSPACE")
    if not workspace:
        return path
    try:
        rel = os.path.relpath(os.path.abspath(path), workspace)
    except ValueError:  # another drive on Windows
        return path
    return path if rel.startswith("..") else Path(rel).as_posix()


def _sarif_uri(path: str) -> str:
    """Relative paths stay relative (percent-encoded); absolute ones become file:/// URIs."""
    import urllib.parse
    from pathlib import Path, PureWindowsPath

    if re.match(r"^[A-Za-z]:[\\/]", path) or path.startswith("/"):
        try:
            return Path(path).resolve().as_uri()
        except ValueError:
            return "file:///" + urllib.parse.quote(PureWindowsPath(path).as_posix())
    return urllib.parse.quote(path.replace("\\", "/"))


def scan_sarif(scan: Any) -> dict:
    results = []
    for f in scan.findings:
        rule = _rule_for(f.lookup, scan.warn_days)
        if not rule:
            continue
        line = getattr(f, "file_line", None) or f.line  # notebooks: the line in the .ipynb file
        results.append(
            {
                "ruleId": rule,
                "level": _SARIF_RULES[rule][0],
                "message": {"text": f"{f.model}: {f.lookup.message}" if f.lookup else f.model},
                "locations": [{"physicalLocation": {"artifactLocation": {"uri": _sarif_uri(_in_workspace(f.file))}, "region": {"startLine": max(1, line), "startColumn": max(1, f.col)}}}],
            }
        )
    rules = [
        {"id": rid, "name": rid, "shortDescription": {"text": short}, "fullDescription": {"text": full}, "defaultConfiguration": {"level": level}}
        for rid, (level, short, full) in _SARIF_RULES.items()
    ]
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{"tool": {"driver": {"name": "llmshift", "version": __version__, "rules": rules}}, "results": results}],
    }


def _gh_escape(value: str, prop: bool = False) -> str:
    value = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if prop:
        value = value.replace(":", "%3A").replace(",", "%2C")
    return value


def github_annotations(scan: Any) -> list[str]:
    out = []
    seen = set()
    for f in scan.findings:
        lk = f.lookup
        if lk is None or lk.severity not in ("critical", "warning"):
            continue
        key = (f.file, f.line, f.model)
        if key in seen:
            continue
        seen.add(key)
        level = "error" if lk.severity == "critical" else "warning"
        title = f"{'Retired' if lk.status == 'retired' else 'Retiring' if lk.status == 'retiring' else 'Deprecated'} model {f.model}"
        line = getattr(f, "file_line", None) or f.line
        out.append(f"::{level} file={_gh_escape(_in_workspace(f.file), True)},line={line},col={f.col},title={_gh_escape(title, True)}::{_gh_escape(lk.message)}")
    return out
