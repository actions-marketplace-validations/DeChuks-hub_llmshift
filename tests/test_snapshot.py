"""Log parsing (all formats) and snapshot (logs -> test set)."""

from __future__ import annotations

import gzip
import json

import pytest

from conftest import chat_response, write_jsonl
from llmshift.infer import group_stats, infer_checks, select_fields
from llmshift.logparse import parse_logs, parse_time
from llmshift.providers.mock import MockProvider
from llmshift.snapshot import build_snapshot

SYS = "You are a helpful assistant for Acme."


def chat_pair(user, out, model="gpt-4o-2024-05-13", ts="2026-10-01T10:00:00Z", **params):
    return {"timestamp": ts, "request": {"model": model, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}], **params}, "response": chat_response(out, model=model)}


def test_formats_are_detected(tmp_path):
    records = [
        chat_pair("q1", "a1"),
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "flat"}], "choices": [{"message": {"role": "assistant", "content": "flat answer"}, "finish_reason": "stop"}]},
        {"request": {"model": "claude-x", "system": "be nice", "max_tokens": 10, "messages": [{"role": "user", "content": "hey"}]}, "response": {"type": "message", "model": "claude-x", "content": [{"type": "text", "text": "hello!"}], "stop_reason": "end_turn"}},
        {"request": {"model": "gpt-5.6", "instructions": "sys", "input": "resp q", "text": {"format": {"type": "json_object"}}, "max_output_tokens": 50}, "response": {"object": "response", "status": "completed", "model": "gpt-5.6", "output": [{"type": "message", "content": [{"type": "output_text", "text": "{\"a\": 1}"}]}]}},
        {"request": {"contents": [{"role": "user", "parts": [{"text": "gem q"}]}], "systemInstruction": {"parts": [{"text": "gsys"}]}, "generationConfig": {"temperature": 0.2}}, "response": {"candidates": [{"content": {"parts": [{"text": "gem a"}]}, "finishReason": "STOP"}], "modelVersion": "gemini-x"}},
        {"request": {"body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "helicone"}]}}, "response": {"body": chat_response("h answer")}},
        {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "dataset q"}, {"role": "assistant", "content": "dataset a"}]},
        {"conversations": [{"from": "human", "value": "sharegpt q"}, {"from": "gpt", "value": "sharegpt a"}]},
        {"inputs": {"messages": [[{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "HumanMessage"], "kwargs": {"content": "lc q"}}]]}, "outputs": {"generations": [[{"text": "lc a"}]]}, "extra": {"invocation_params": {"model_name": "gpt-4o"}}},
        {"input": "langfuse q", "output": {"role": "assistant", "content": "langfuse a"}, "model": "gpt-4o", "startTime": "2026-10-02T00:00:00Z"},
        {"prompt": "generic q", "completion": "generic a", "system_prompt": "gs"},
        {"input": [{"role": "user", "content": "q"}], "ideal": "x"},
        {"what": "ever"},
    ]
    write_jsonl(tmp_path / "mixed.jsonl", records)
    (tmp_path / "bad.jsonl").write_text('{"prompt": "ok", "response": "fine"}\nnot json\n', encoding="utf-8")
    interactions, stats = parse_logs([str(tmp_path)])
    outputs = {i.output for i in interactions}
    expected = {"a1", "flat answer", "hello!", '{"a": 1}', "gem a", "h answer", "dataset a", "sharegpt a", "lc a", "langfuse a", "generic a", "fine"}
    assert expected <= outputs
    assert stats.skipped["invalid JSON line"] == 1
    assert stats.skipped["unrecognised record"] == 1
    assert any("openai/evals" in reason for reason in stats.skipped)
    by_out = {i.output: i for i in interactions}
    assert by_out["hello!"].messages[0] == {"role": "system", "content": "be nice"}
    assert by_out['{"a": 1}'].params == {"max_completion_tokens": 50, "response_format": {"type": "json_object"}}
    assert by_out["gem a"].params == {"temperature": 0.2} and by_out["gem a"].model == "gemini-x"
    assert by_out["generic a"].messages[0]["content"] == "gs"
    assert by_out["lc a"].model == "gpt-4o"


def test_batch_join_csv_and_gzip(tmp_path):
    write_jsonl(tmp_path / "batch_in.jsonl", [{"custom_id": "r1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "batch q"}]}}])
    write_jsonl(tmp_path / "batch_out.jsonl", [{"id": "b1", "custom_id": "r1", "response": {"status_code": 200, "body": chat_response("batch a")}}])
    (tmp_path / "rows.csv").write_text("question,answer,model\nWhat is 2+2?,4,gpt-4o\n", encoding="utf-8")
    with gzip.open(tmp_path / "zipped.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"prompt": "gz q", "response": "gz a"}) + "\n")
    interactions, stats = parse_logs([str(tmp_path)])
    by_out = {i.output: i for i in interactions}
    assert by_out["batch a"].fmt == "openai_batch" and by_out["batch a"].metadata["custom_id"] == "r1"
    assert by_out["4"].messages == [{"role": "user", "content": "What is 2+2?"}]
    assert "gz a" in by_out


def test_parse_time():
    assert parse_time("2026-10-21T10:00:00Z").year == 2026
    assert parse_time("2026-10-21 10:00:00.123456789").microsecond == 123456
    assert parse_time("1760000000").year == 2025
    assert parse_time("2026-10-21").day == 21
    assert parse_time("garbage") is None


def test_snapshot_end_to_end(tmp_path):
    records = []
    for i in range(30):
        records.append(chat_pair(f"Classify ticket {i}: my order {i} is late", json.dumps({"label": ["shipping", "billing"][i % 2], "score": 0.9, "summary": f"order {i} late"}), ts=f"2026-10-{(i % 28) + 1:02d}T10:00:00Z", temperature=0, response_format={"type": "json_object"}))
    records.append(chat_pair("Classify ticket 0: my order 0 is late", "dup", ts="2026-10-02T00:00:00Z"))  # different output, same request? (differs: no params)
    records.append(chat_pair("Old one", "x", model="gpt-3.5-turbo"))
    records.append({**chat_pair("err", "e"), "status_code": 500})
    truncated = chat_pair("cut", "partial")
    truncated["response"]["choices"][0]["finish_reason"] = "length"
    records.append(truncated)
    records.append(chat_pair("My key is sk-" + "proj-abcdefghijklmnopqrstuvwxyz123456 and mail bob@example.com", "Noted bob@example.com"))
    write_jsonl(tmp_path / "logs.jsonl", records)
    res = build_snapshot([str(tmp_path / "logs.jsonl")], models=["gpt-4o*"], max_cases=10, per_group=8)
    assert res.filtered["different model"] == 1
    assert res.filtered["API error recorded in the log"] == 1
    assert res.filtered["truncated output (finish_reason=length)"] == 1
    assert len(res.cases) <= 10 and res.groups >= 2
    json_cases = [c for c in res.cases if c.params.get("response_format")]
    assert json_cases
    checks = {c["type"]: c for c in json_cases[0].checks}
    assert {"json_valid", "json_schema", "json_match"} <= set(checks)
    assert set(checks["json_schema"]["schema"]["required"]) == {"label", "score", "summary"}
    fields = checks["json_match"]["fields"]
    assert "$.label" in fields and "$.score" not in fields and "$.summary" not in fields  # free text excluded via group stats
    secret_case = next(c for c in res.cases if "key is" in c.user_text)
    assert "sk-proj" not in secret_case.user_text and "<SECRET_1>" in secret_case.user_text
    assert "bob@example.com" in secret_case.user_text  # PII redaction is opt-in
    pii = build_snapshot([str(tmp_path / "logs.jsonl")], redact_pii=True)
    case = next(c for c in pii.cases if "key is" in c.user_text)
    assert "<EMAIL_1>" in case.user_text and case.baseline_output == "Noted <EMAIL_1>"


def test_snapshot_time_window_and_generation(tmp_path):
    write_jsonl(tmp_path / "l.jsonl", [
        {"timestamp": "2026-01-05T00:00:00Z", "prompt": "early", "response": "a"},
        {"timestamp": "2026-06-05T00:00:00Z", "prompt": "late", "response": "b"},
        {"timestamp": "2026-06-06T00:00:00Z", "prompt": "no output"},
    ])
    res = build_snapshot([str(tmp_path / "l.jsonl")], since="2026-03-01")
    assert [c.user_text for c in res.cases] == ["late"]
    assert res.filtered["outside the time window"] == 1
    gen = MockProvider("gen", fn=lambda r: "generated: " + r["messages"][-1]["content"])
    res = build_snapshot([str(tmp_path / "l.jsonl")], baseline_provider=gen)
    outputs = {c.user_text: c.baseline_output for c in res.cases}
    assert outputs["no output"] == "generated: no output" and outputs["early"] == "a"
    assert res.generated == 1
    with pytest.raises(ValueError):
        build_snapshot([str(tmp_path / "l.jsonl")], since="not a date")


def test_infer_checks_kinds():
    msgs = [{"role": "user", "content": "q"}]
    label = infer_checks(msgs, {}, "positive", group=group_stats(["positive", "negative", "neutral", "positive", "negative"]))
    assert [c["type"] for c in label] == ["not_empty", "refusal", "one_of", "equals"]
    assert label[2]["values"] == ["negative", "neutral", "positive"]
    prose = infer_checks(msgs, {"temperature": 0}, "The capital of France is Paris. It has about 2.1 million residents and hosts the Louvre museum.")
    types = [c["type"] for c in prose]
    assert {"language", "structure", "length", "facts", "tone", "similarity"} <= set(types)
    tools = infer_checks(msgs, {}, "", [{"name": "lookup", "arguments": {"id": 7}}])
    assert tools == [{"type": "tool_calls", "expected": [{"name": "lookup", "arguments": {"id": 7}}]}]
    refused = infer_checks(msgs, {}, "I'm sorry, but I can't help with that.")
    assert [c["type"] for c in refused] == ["not_empty"]
    fenced = infer_checks(msgs, {}, '```json\n{"a": "x"}\n```')
    assert fenced[2] == {"type": "json_valid", "allow_fences": True}
    creative = infer_checks(msgs, {"temperature": 1.0}, "Once upon a time Alice met Bob in Paris and they founded Acme with 3 friends.")
    assert next(c for c in creative if c["type"] == "facts")["severity"] == "warn"


def test_select_fields_rules():
    value = {"id": "ORD-1", "ok": True, "count": 3, "confidence": 0.93, "ratio": 0.5, "price": 12.5, "note": "a longer free text note here", "tags": ["a", "b"], "created_at": "2026-01-01"}
    fields = select_fields(value)
    assert fields == {"$.id": "ORD-1", "$.ok": True, "$.count": 3, "$.price": 12.5, "$.tags": ["a", "b"]}


def test_test_files_keep_line_separator_characters(tmp_path):
    """U+2028, U+2029 and U+0085 stay unescaped in JSON strings; reading the test file
    must not take them for line ends (str.splitlines() does)."""
    from llmshift.testset import load_tests, save_tests

    weird = "first second third\x85fourth"
    log = tmp_path / "log.jsonl"
    write_jsonl(log, [{"request": {"model": "gpt-4o", "messages": [{"role": "user", "content": weird}]}, "response": chat_response(weird)}])
    cases = build_snapshot([str(log)]).cases
    save_tests(cases, tmp_path / "tests.jsonl")
    loaded = load_tests(tmp_path / "tests.jsonl")
    assert [c.baseline_output for c in loaded] == [weird] and loaded[0].messages[-1]["content"] == weird


def test_utf16_logs_corrupt_gzip_and_odd_tools(tmp_path):
    """PowerShell 5.1's `>` writes UTF-16; a .gz can be cut short; tools come in odd
    shapes. Each file or record that cannot be read is counted, never fatal."""
    def rec(q, a, **kw):
        return {"request": {"model": "gpt-4o", "messages": [{"role": "user", "content": q}], **kw}, "response": chat_response(a)}

    (tmp_path / "ps.jsonl").write_bytes(("\n".join(json.dumps(rec(f"q{i}", f"a{i}")) for i in range(3)) + "\n").encode("utf-16"))
    (tmp_path / "doc.json").write_bytes(json.dumps([rec("doc q", "doc a")]).encode("utf-16"))
    (tmp_path / "excel.csv").write_bytes("input,output\nhello there,hi back\n".encode("utf-16"))
    (tmp_path / "cut.jsonl.gz").write_bytes(gzip.compress(b'{"x": 1}\n')[:-12] + b"garbage!!!!!")
    (tmp_path / "tools.jsonl").write_text(json.dumps(rec("t q", "t a", tools=[{"function": "x"}, {"name": 5}, 7])) + "\n" + json.dumps(rec("u q", "u a", tools=5)) + "\n", encoding="utf-8")
    res = build_snapshot([str(tmp_path)])
    assert sorted(c.baseline_output for c in res.cases) == ["a0", "a1", "a2", "doc a", "hi back", "t a", "u a"]
    assert dict(res.stats.skipped) == {"unreadable file": 1}


def test_null_items_in_arrays_do_not_fail_their_own_baseline():
    from llmshift.checks import CheckContext, Sample, evaluate
    from llmshift.testset import TestCase

    outputs = ['{"rows": [["Alice", 30], ["Bob", null]]}', '{"tags": ["a"]}', '{"tags": [null]}']
    stats = group_stats(outputs)
    for out in outputs:
        case = TestCase(id="c", input={"messages": [{"role": "user", "content": "q"}], "params": {"response_format": {"type": "json_object"}}}, baseline={"output": out})
        for check in infer_checks(case.messages, case.params, out, [], stats):
            result = evaluate(check, Sample(text=out), case, CheckContext())
            assert result.status in ("pass", "skip"), (check["type"], result.message)
