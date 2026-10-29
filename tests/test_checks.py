"""Check implementations and graders."""

from __future__ import annotations

import json

import pytest

from llmshift.checks import CheckContext, Sample, evaluate, run_checks
from llmshift.graders import _get_choice, chat_to_text, safe_eval
from llmshift.providers.base import Response
from llmshift.providers.mock import MockProvider


def ev(check, text, case, ctx=None, **sample_kw):
    return evaluate(check, Sample(text=text, **sample_kw), case, ctx or CheckContext())


def test_format_checks(make_case):
    case = make_case('{"a": 1}')
    assert ev({"type": "json_valid"}, '{"a": 1}', case).status == "pass"
    r = ev({"type": "json_valid"}, '```json\n{"a": 1}\n```', case)
    assert r.status == "fail" and r.category == "format" and r.severity == "error" and "code fence" in r.message
    assert ev({"type": "json_valid", "allow_fences": True}, '```json\n{"a": 1}\n```', case).status == "pass"
    assert ev({"type": "json_valid"}, '"just a string"', case).status == "fail"
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    assert ev({"type": "json_schema", "schema": schema}, '```json\n{"a": 2}\n```', case).status == "pass"
    r = ev({"type": "json_schema", "schema": schema}, '{"b": 2}', case)
    assert r.status == "fail" and "missing required key 'a'" in r.message
    r = ev({"type": "json_schema", "schema": schema}, '{"a": 2, "c": 1}', case)
    assert r.status == "pass" and r.details["extra_keys"] == ["c"]
    assert ev({"type": "not_empty"}, "", case).status == "fail"
    assert ev({"type": "not_empty"}, "", case, tool_calls=[{"name": "f", "arguments": {}}]).status == "pass"


def test_structure_language_regex(make_case):
    case = make_case("- a\n- b")
    r = ev({"type": "structure", "expect": {"list": True, "headings": False}}, "# Title\nplain text", case)
    assert r.status == "fail" and "adds markdown headings" in r.message and "no longer uses a bulleted/numbered list" in r.message
    r = ev({"type": "language", "expected": "en"}, "Gracias por tu mensaje. El pedido llegará el lunes y no tienes que hacer nada más.", case)
    assert r.status == "fail" and "Spanish" in r.message
    assert ev({"type": "regex", "pattern": r"^\d{3}$"}, "123", case).status == "pass"
    assert ev({"type": "regex", "pattern": "secret", "negate": True}, "a secret", case).status == "fail"
    assert ev({"type": "regex", "pattern": "("}, "x", case).status == "error"


def test_label_checks(make_case):
    case = make_case("positive")
    assert ev({"type": "equals", "value": "positive"}, " positive\n", case).status == "pass"
    r = ev({"type": "equals", "value": "positive"}, "Positive.", case)
    assert r.status == "fail" and "differs only in case" in r.message
    assert ev({"type": "equals", "value": "positive", "case_sensitive": False}, "POSITIVE", case).status == "pass"
    assert ev({"type": "equals", "value": "1299", "numeric": True}, "$1,299.00", case).status == "pass"
    assert ev({"type": "one_of", "values": ["a", "b"]}, "c", case).status == "fail"
    assert ev({"type": "not_equals", "value": "x"}, "x", case).status == "fail"


def test_contains_family(make_case):
    case = make_case("x")
    assert ev({"type": "contains", "value": ["30 days", "refund"]}, "Refund within 30 days", case).status == "fail"  # case-sensitive
    assert ev({"type": "contains", "value": ["30 days", "refund"], "case_sensitive": False}, "Refund within 30 days", case).status == "pass"
    assert ev({"type": "contains_any", "value": ["Paris", "Lyon"]}, "It is Paris.", case).status == "pass"
    assert ev({"type": "contains_any", "value": ["paris"], "ignore_case": True}, "It is PARIS.", case).status == "pass"
    assert ev({"type": "not_contains", "value": ["As an AI"]}, "as an ai model", case).status == "fail"
    assert ev({"type": "starts_with", "value": ["time"]}, " time flies", case).status == "pass"
    assert ev({"type": "starts_with", "value": ["Time"]}, "time flies", case).status == "fail"


def test_oss_semantics(make_case):
    case = make_case("x")
    assert ev({"type": "fuzzy_match", "value": ["The Season 11"]}, "season 11!", case).status == "pass"
    assert ev({"type": "fuzzy_match", "value": ["Paris"]}, "London", case).status == "fail"
    assert ev({"type": "json_equals", "values": ['{"a": 1, "b": [1, 2]}']}, '{"b": [1, 2], "a": 1.0}', case).status == "pass"
    assert ev({"type": "json_equals", "values": ['{"a": null}']}, '{"a": null}', case).status == "fail"  # openai/evals: null never matches
    assert ev({"type": "json_equals", "values": ['{"a": 1}']}, '```json\n{"a": 1}\n```', case).status == "fail"  # fences not stripped


def test_templated_values_and_targets(make_case):
    case = make_case("x", vars={"label": "Hardware", "ref": {"city": "Paris"}})
    assert ev({"type": "equals", "value": "{{item.label}}"}, "Hardware", case).status == "pass"
    check = {"type": "equals", "value": "{{item.ref.city}}", "target": "{{sample.output_json.city}}"}
    assert ev(check, '{"city": "Paris"}', case).status == "pass"
    assert ev(check, '{"city": "Rome"}', case).status == "fail"


def test_json_match(make_case):
    case = make_case('{"category": "billing", "urgency": "low", "refund": true}')
    check = {"type": "json_match", "fields": {"$.category": "billing", "$.urgency": "low", "$.refund": True, "$.tags": ["a", "b"]}}
    assert ev(check, '{"category": "billing", "urgency": "low", "refund": true, "tags": ["b", "a"]}', case).status == "pass"
    r = ev(check, '{"category": "billing", "urgency": "high", "refund": 1}', case)
    assert r.status == "fail" and r.category == "facts"
    assert '$.urgency: "low" -> "high"' in r.message and "$.refund" in r.message and "$.tags: missing" in r.message
    assert ev(check, "not json", case).status == "skip"


def test_facts_check(make_case):
    case = make_case("Order #4521 arrives on March 14 with Acme Express.")
    check = {"type": "facts", "facts": [{"kind": "id", "text": "#4521"}, {"kind": "date", "text": "March 14", "value": "--03-14"}, {"kind": "entity", "text": "Acme Express"}]}
    assert ev(check, "Your order #4521 will be delivered by Acme Express on March 14.", case).status == "pass"
    r = ev(check, "Your order #4521 will be delivered on March 16 by Acme Express.", case)
    assert r.status == "fail" and r.severity == "error" and '"March 14"' in r.message
    soft = ev(check, "Your order #4521 arrives on March 14.", case)
    assert soft.status == "fail" and soft.severity == "warn"  # only a name went missing


def test_tone_length_similarity_refusal(make_case):
    base = "Thank you for contacting us. We have reviewed your order and it will ship on Monday. Please let us know if we can help further."
    case = make_case(base)
    r = ev({"type": "tone"}, "Hey!! 🎉 Awesome news - your order ships Monday!!! Super excited for you, totally amazing stuff!", case)
    assert r.status == "fail" and r.severity == "warn" and "emoji" in r.message
    assert ev({"type": "tone"}, "ok", case).status == "skip"
    r = ev({"type": "length", "min_words": 10, "max_words": 30, "baseline_words": 20}, "word " * 60, case)
    assert r.status == "fail" and "much longer" in r.message
    assert ev({"type": "similarity", "threshold": 0.5}, base, case).status == "pass"
    assert ev({"type": "similarity", "threshold": 0.5}, "completely unrelated words here", case).status == "fail"
    assert ev({"type": "refusal"}, "I'm sorry, but I can't help with that.", case).status == "fail"
    assert ev({"type": "refusal"}, "", case, refusal="policy").status == "fail"


def test_tool_calls(make_case):
    case = make_case("")
    check = {"type": "tool_calls", "expected": [{"name": "get_weather", "arguments": {"city": "Paris", "days": 3, "note": "a long free text argument that may vary a lot"}}]}
    ok = ev(check, "", case, tool_calls=[{"name": "get_weather", "arguments": {"city": "paris", "days": 3, "note": "other words"}}])
    assert ok.status == "pass"
    r = ev(check, "It is sunny in Paris.", case)
    assert r.status == "fail" and "no tool call" in r.message
    r = ev(check, "", case, tool_calls=[{"name": "get_weather", "arguments": {"city": "Rome", "days": 3, "note": "x"}}])
    assert r.status == "fail" and 'get_weather.city: "Paris" -> "Rome"' in r.message
    r = ev(check, "", case, tool_calls=[{"name": "search", "arguments": {}}])
    assert "missing call(s) to get_weather" in r.message and "unexpected call(s) to search" in r.message
    r = ev(check, "", case, tool_calls=[{"name": "get_weather", "arguments": "{not json"}])
    assert "not valid JSON" in r.message


def test_unknown_and_crashing_checks(make_case):
    case = make_case("x")
    assert ev({"type": "does_not_exist"}, "x", case).status == "error"
    assert ev({"type": "json_schema", "schema": "not a dict"}, '{"a": 1}', case).status == "error"


def test_severity_overrides(make_case):
    case = make_case("x")
    r = evaluate({"type": "tone"}, Sample(text="x"), case, CheckContext(severity={"tone": "error"}))
    assert r.severity == "error"
    r = evaluate({"type": "equals", "value": "a", "severity": "warn"}, Sample(text="b"), case, CheckContext())
    assert r.severity == "warn"


# ------------------------------------------------------------------ graders


def scripted_judge(replies):
    calls = []

    def fn(request):
        calls.append(request)
        return replies.pop(0) if replies else ""

    provider = MockProvider("judge", fn=fn)
    provider.calls = calls
    return provider


def test_llm_judge(make_case):
    case = make_case("The refund window is 30 days.", user="How long is the refund window?")
    judge = scripted_judge([json.dumps({"verdict": "fail", "reason": "different window", "differences": ["30 days -> 14 days"]}), '{"verdict": "pass", "reason": "same"}'])
    ctx = CheckContext(judge=judge)
    r = ev({"type": "llm_judge"}, "You have 14 days.", case, ctx)
    assert r.status == "fail" and "30 days -> 14 days" in r.message
    assert ev({"type": "llm_judge"}, "30 days.", case, ctx).status == "pass"
    prompt = judge.calls[0]["messages"][1]["content"]
    assert "How long is the refund window?" in prompt and "The refund window is 30 days." in prompt
    assert ev({"type": "llm_judge"}, "x", case).status == "skip"  # no judge configured


def test_llm_label_and_score(make_case):
    case = make_case("x", vars={"response": "I love it"})
    judge = scripted_judge(['{"label": "positive"}', "I think it is negative", '{"steps": [], "result": 0.8}', "score: 0.2"])
    ctx = CheckContext(judge=judge)
    check = {
        "type": "llm_label",
        "input": [{"type": "message", "role": "developer", "content": {"type": "input_text", "text": "Classify"}}, {"role": "user", "content": "Statement: {{item.response}} / {{sample.output_text}}"}],
        "labels": ["positive", "neutral", "negative"],
        "passing_labels": ["positive"],
    }
    assert ev(check, "great", case, ctx).status == "pass"
    sent = judge.calls[0]["messages"]
    assert sent[0]["role"] == "developer" and "Statement: I love it / great" in sent[1]["content"]
    assert ev(check, "bad", case, ctx).status == "fail"
    score = {"type": "llm_score", "input": [{"role": "user", "content": "Rate {{sample.output_text}}"}], "range": [0, 1], "pass_threshold": 0.5}
    assert ev(score, "a", case, ctx).status == "pass"
    assert ev(score, "b", case, ctx).status == "fail"


def test_judge_from_grader_model(make_case):
    case = make_case("x")
    judge = scripted_judge(['{"label": "yes"}'])
    ctx = CheckContext(resolve_model=lambda spec: judge if spec == "gpt-4o-2024-08-06" else None)
    check = {"type": "llm_label", "model": "gpt-4o-2024-08-06", "input": [{"role": "user", "content": "?"}], "labels": ["yes", "no"], "passing_labels": ["yes"]}
    assert ev(check, "x", case, ctx).status == "pass"


def test_modelgraded_fact_and_closedqa(make_case):
    case = make_case("x", vars={"input": [{"role": "user", "content": "Capital of France?"}], "ideal": "Paris"})
    judge = scripted_judge(["Reasoning...\nD", "Reasoning...\nC", "Some reasoning\nY\nY"])
    ctx = CheckContext(judge=judge)
    fact = {
        "type": "modelgraded",
        "prompt": "Q: {input}\nExpert: {ideal}\nSubmission: {completion}",
        "choice_strings": list("ABCDE"),
        "eval_type": "cot_classify",
        "passing_choices": ["A", "B", "C", "E"],
    }
    assert ev(fact, "Lyon", case, ctx).status == "fail"
    assert ev(fact, "Paris", case, ctx).status == "pass"
    sent = judge.calls[0]["messages"][0]["content"]
    assert "Q: Capital of France?" in sent and "Submission: Lyon" in sent and 'single choice from "A" or "B"' in sent
    closedqa = {"type": "modelgraded", "prompt": "{input} {completion} {criteria}", "choice_strings": ["Y", "N"], "choice_scores": {"Y": 1.0, "N": 0.0}, "spec_eval_type": "cot_classify", "args": {"criteria": "correct?"}}
    assert ev(closedqa, "Paris", case, ctx).status == "pass"


def test_get_choice_and_chat_to_text():
    assert _get_choice("blah\nA\nC", "cot_classify", None, list("ABCDE")) == "C"
    assert _get_choice("B. because", "classify", None, list("ABCDE")) == "B"
    assert _get_choice("nothing here", "classify", None, ["Y", "N"]) == "__invalid__"
    assert chat_to_text([{"role": "user", "content": "hi"}]) == "hi"
    assert chat_to_text([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]) == "s\nUser: u\n"


def test_python_grader(make_case):
    case = make_case("x", vars={"label": "yes"})
    src = "def grade(sample, item):\n    return 1.0 if sample['output_text'] == item['label'] else 0.0\n"
    check = {"type": "python", "source": src, "pass_threshold": 0.5}
    assert ev(check, "yes", case).status == "skip"
    ctx = CheckContext(allow_python=True)
    assert ev(check, "yes", case, ctx).status == "pass"
    assert ev(check, "no", case, ctx).status == "fail"
    broken = ev({"type": "python", "source": "def grade(s, i):\n    raise ValueError('boom')\n"}, "yes", case, ctx)
    assert broken.status == "fail" and "boom" in broken.message
    missing = ev({"type": "python", "source": "x = 1"}, "yes", case, ctx)
    assert missing.status == "fail" and "grade" in missing.message
    typo = ev({"type": "python", "source": "def grade(s, i)\n    return 1.0\n"}, "yes", case, ctx)
    assert typo.status == "fail" and "SyntaxError" in typo.message
    helper = "import math\n\ndef _half(x):\n    return x / 2\n\ndef grade(s, i):\n    return _half(2.0) if math.pi else 0.0\n"
    assert ev({"type": "python", "source": helper}, "yes", case, ctx).status == "pass"


def test_multi_and_safe_eval(make_case):
    assert safe_eval("(a + b) / 2", {"a": 1, "b": 0}) == 0.5
    assert safe_eval("max(a, b) ^ 2", {"a": 0.5, "b": 0.2}) == 0.25
    for bad in ("__import__('os')", "a.b", "[1][0]", "open('x')"):
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval(bad, {"a": 1})
    case = make_case("x", vars={"name": "Ada", "email": "ada@example.com"})
    check = {
        "type": "multi",
        "graders": {
            "name": {"type": "similarity", "metric": "fuzzy_match", "threshold": 0.9, "target": "{{sample.output_json.name}}", "reference": "{{item.name}}"},
            "email": {"type": "equals", "value": "{{item.email}}", "target": "{{sample.output_json.email}}"},
        },
        "calculate_output": "(name + email) / 2",
        "pass_threshold": 0.9,
    }
    assert ev(check, '{"name": "Ada", "email": "ada@example.com"}', case).status == "pass"
    assert ev(check, '{"name": "Ada", "email": "x@example.com"}', case).status == "fail"


def test_run_checks_order(make_case):
    case = make_case("hello", checks=[{"type": "not_empty"}, {"type": "equals", "value": "hello"}])
    results = run_checks(case, Sample.from_response(Response(text="hello")), CheckContext())
    assert [r.type for r in results] == ["not_empty", "equals"] and all(r.status == "pass" for r in results)


def test_a_regex_that_backtracks_without_end_is_stopped(make_case, monkeypatch):
    """^(\\w+\\s?)*$ took more than a minute on a 141-character answer it does not match."""
    import time

    from llmshift import userregex

    monkeypatch.setattr(userregex, "TIMEOUT", 0.5)
    case = make_case("x")
    start = time.perf_counter()
    r = ev({"type": "regex", "pattern": r"^(\w+\s?)*$"}, "a" * 141 + "!", case)
    assert r.status == "error" and "backtracks" in r.message and r"\w+(?:\s\w+)*" in r.message
    assert ev({"type": "regex", "pattern": r"^\w+(?:\s\w+)*$"}, "a" * 141 + "!", case).status == "fail"  # the rewrite it suggests
    assert ev({"type": "regex", "pattern": r"\d+ items"}, "1" * 40000 + " items", case).status == "pass"
    schema = {"type": "object", "properties": {"name": {"type": "string", "pattern": r"^(\w+\s?)*$"}}, "patternProperties": {r"^(\w+\s?)*$": {"type": "string"}}}
    r = ev({"type": "json_schema", "schema": schema}, json.dumps({"name": "a" * 141 + "!", "b" * 60 + "!": "x"}), case)
    assert r.status == "fail" and "backtracks" in r.message
    assert time.perf_counter() - start < 30
    start = time.perf_counter()
    for i in range(300):  # the worker stays up between searches
        assert ev({"type": "regex", "pattern": r"order #\d+"}, f"your order #{i} ships", case).status == "pass"
    assert time.perf_counter() - start < 10


@pytest.mark.parametrize("reply, status", [
    ("The new answer does not pass: the window changed.", "fail"),
    ("It doesn't fail; the facts are the same.", "pass"),
    ("I do not think it will pass.", "fail"),
    ("This fails.", "fail"),
    ("There are no differences and it passes.", "pass"),
    ("Not only does it pass, it is clearer.", "pass"),
    ('{"verdict": "maybe"} No idea.', "error"),
])
def test_a_judge_answering_in_prose_is_read_with_its_negations(make_case, reply, status):
    case = make_case("The refund window is 30 days.", user="How long is the refund window?")
    ctx = CheckContext(judge=scripted_judge([reply]))
    assert ev({"type": "llm_judge"}, "You have 30 days.", case, ctx).status == status


def test_judge_replies_are_read_in_linear_time(make_case):
    import time

    from llmshift.graders import score_from_text

    start = time.perf_counter()
    for reply in ("score" + " " * 40000, "1" * 40000, "not " * 20000 + "fail"):  # 20 to 40 s each before
        score_from_text(reply, 0, 10)
    ctx = CheckContext(judge=scripted_judge(["not " * 20000 + "a pass"]))
    assert ev({"type": "llm_judge"}, "x", make_case("y"), ctx).status == "fail"
    assert time.perf_counter() - start < 5
