"""compare (incl. differential baseline), report renderers, and fix."""

from __future__ import annotations

import json
import xml.dom.minidom

from llmshift import report
from llmshift.cache import ResponseCache
from llmshift.compare import CompareResult, case_status, run_compare
from llmshift.demo import MESSAGES, REPLY_PROMPT, SUMMARY_PROMPT, TICKETS, TRIAGE_PROMPT, UPDATES
from llmshift.fix import compose, fix_markdown, patched_cases, run_fix
from llmshift.infer import infer_checks
from llmshift.providers.mock import MockProvider, legacy
from llmshift.testset import TestCase


def demo_cases():
    cases = []
    templates = ((TRIAGE_PROMPT, TICKETS, {"temperature": 0, "response_format": {"type": "json_object"}}), (REPLY_PROMPT, MESSAGES, {}), (SUMMARY_PROMPT, UPDATES, {}))
    for g, (prompt, inputs, params) in enumerate(templates):
        for i, text in enumerate(inputs):
            msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
            out = legacy({"messages": msgs, "params": params}).text
            cases.append(TestCase(id=f"t{g}-{i}", input={"messages": msgs, "params": params}, baseline={"output": out}, checks=infer_checks(msgs, params, out), group=prompt[:20]))
    return cases


def test_duplicate_ids_are_handled():
    case = demo_cases()[0]
    res = run_compare([case, case], "mock:legacy", show_progress=False)
    assert [c.id for c in res.cases] == [case.id, case.id + "-2"] and res.summary["passed"] == 2


def test_compare_legacy_is_clean_and_nextgen_regresses():
    cases = demo_cases()
    same = run_compare(cases, "mock:legacy", show_progress=False)
    assert same.summary["passed"] == len(cases) and same.summary["pass_rate"] == 1.0
    res = run_compare(cases, "mock:nextgen", show_progress=False, min_pass_rate=0.95)
    s = res.summary
    assert s["failed"] > 0 and not res.passed
    cats = s["by_category"]
    assert cats["format"]["fail"] == len(TICKETS)
    assert "JSON is wrapped in a markdown code fence" in cats["format"]["patterns"]
    assert cats["facts"]["fail"] >= len(UPDATES)
    assert cats["tone"]["warn"] >= 1
    assert any("code fences" in i for i in res.insights)
    roundtrip = CompareResult.from_dict(json.loads(json.dumps(res.to_dict())))
    assert roundtrip.summary == res.summary and len(roundtrip.cases) == len(res.cases)


def test_differential_baseline_marks_noise():
    case = TestCase(id="c1", input={"messages": [{"role": "user", "content": "q"}], "params": {}}, baseline={"output": "A"}, checks=[{"type": "equals", "value": "A"}, {"type": "not_empty"}])
    res = run_compare([case], MockProvider("new", fn=lambda r: "B"), baseline=MockProvider("old", fn=lambda r: "C"), show_progress=False)
    c = res.cases[0]
    assert c.status == "pass" and c.unstable == ["equals"]
    assert c.checks[0].status == "skip" and "noise" in c.checks[0].message
    assert c.baseline_live_output == "C"


def test_live_baseline_fills_missing_reference():
    case = TestCase(id="c1", input={"messages": [{"role": "user", "content": "q"}], "params": {}}, checks=[{"type": "similarity", "threshold": 0.9}])
    res = run_compare([case], MockProvider("new", fn=lambda r: "the same answer"), baseline=MockProvider("old", fn=lambda r: "the same answer"), show_progress=False)
    assert res.cases[0].status == "pass" and res.cases[0].baseline_output == "the same answer"


def test_errors_and_cache(tmp_path):
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if "boom" in req["messages"][-1]["content"]:
            raise RuntimeError("provider exploded")
        return "fine"

    cases = [TestCase(id=f"c{i}", input={"messages": [{"role": "user", "content": t}], "params": {}}, baseline={"output": "fine"}, checks=[{"type": "equals", "value": "fine"}]) for i, t in enumerate(["ok", "boom"])]
    cache = ResponseCache(tmp_path / "cache")
    provider = MockProvider("p", fn=flaky)
    res = run_compare(cases, provider, cache=cache, show_progress=False)
    assert res.summary["errors"] == 1 and res.summary["passed"] == 1
    assert next(c for c in res.cases if c.status == "error").error.endswith("provider exploded")
    run_compare(cases, provider, cache=cache, show_progress=False)
    assert calls["n"] == 3  # the successful response came from the cache; the error was retried


def test_case_status_rules():
    from llmshift.checks import CheckResult

    def r(status, severity):
        return CheckResult("x", "facts", severity, status)

    assert case_status([r("pass", "error")], None) == "pass"
    assert case_status([r("fail", "warn")], None) == "warn"
    assert case_status([r("fail", "error"), r("fail", "warn")], None) == "fail"
    assert case_status([r("error", "error")], None) == "error"  # could not be checked: neither a pass nor the model's failure
    assert case_status([r("error", "error"), r("fail", "error")], None) == "fail"  # a failure that could be checked still counts
    assert case_status([r("error", "warn")], None) == "pass"  # a warning-level check that could not run changes nothing
    assert case_status([r("fail", "info")], None) == "pass"
    assert case_status([], "timeout") == "error"


def test_reports_render():
    res = run_compare(demo_cases(), "mock:nextgen", show_progress=False, min_pass_rate=0.9, tests_path="t.jsonl")
    term = report.compare_terminal(res, show=3)
    assert "What changed" in term and "Broken format" in term and "below the 90% threshold" in term
    md = report.compare_markdown(res)
    assert md.startswith("## LLMShift report") and "| Broken format |" in md and "```diff" in md
    page = report.compare_html(res)
    assert page.startswith("<!doctype html>") and "<del>" in page and 'data-status="fail"' in page and "<script>" in page
    assert "<script>alert" not in page
    junit = report.compare_junit(res)
    dom = xml.dom.minidom.parseString(junit.encode("utf-8"))  # nosec B318  # llmshift's own report, not untrusted XML
    assert dom.getElementsByTagName("testcase").length == len(res.cases)
    assert dom.getElementsByTagName("failure").length == res.summary["failed"]


def test_html_escapes_untrusted_content():
    case = TestCase(id="x<y>", input={"messages": [{"role": "user", "content": "<img src=x onerror=alert(1)>"}], "params": {}}, baseline={"output": "<b>old</b>"}, checks=[{"type": "equals", "value": "<b>old</b>"}])
    res = run_compare([case], MockProvider("p", fn=lambda r: "<script>alert(1)</script>"), show_progress=False)
    page = report.compare_html(res)
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<img src=x" not in page


def test_fix_improves_demo_prompts(tmp_path):
    cases = demo_cases()
    result = run_fix(cases, "mock:nextgen", show_progress=False, source_paths=[str(tmp_path)])
    assert result.after["pass_rate"] > result.before["pass_rate"]
    assert result.after["passed"] >= result.before["passed"] + 10
    by_prefix = {g.group[:6]: g for g in result.groups}
    triage = by_prefix[TRIAGE_PROMPT[:6]]
    assert any("code fences" in line for line in triage.added)
    assert triage.suggested_prompt.startswith(TRIAGE_PROMPT)
    summary = by_prefix[SUMMARY_PROMPT[:6]]
    assert any("Do not round" in line for line in summary.added) and summary.after["passed"] == summary.after["cases"]
    assert any('"medium" -> "high"' in item for item in triage.remaining)  # honest about what prompts cannot fix
    md = fix_markdown(result)
    assert "Suggested system prompt" in md and "```diff" in md
    fixed = patched_cases(cases, result)
    assert fixed[0].system_prompt != cases[0].system_prompt and cases[0].system_prompt == TRIAGE_PROMPT
    check = run_compare(fixed, "mock:nextgen", show_progress=False)
    assert abs(check.summary["pass_rate"] - result.after["pass_rate"]) < 1e-9


def test_fix_nothing_to_do():
    cases = demo_cases()[:3]
    result = run_fix(cases, "mock:legacy", show_progress=False)
    assert result.groups == [] and result.after is None


def test_compose():
    assert compose("Base.", ["A", "B"]) == "Base.\n\nOutput requirements:\n- A\n- B"
    assert compose("", [], "Example") == "Example"
