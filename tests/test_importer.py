"""OpenAI Evals importer: hosted API (emulated), exported JSON and the OSS registry."""

from __future__ import annotations

import json

import pytest

from conftest import write_jsonl
from llmshift import cli
from llmshift.checks import CheckContext, Sample, run_checks
from llmshift.importers import openai_evals as oe

EVAL = {
    "object": "eval",
    "id": "eval_abc",
    "name": "Ticket categories",
    "data_source_config": {"type": "custom", "schema": {"type": "object", "properties": {"item": {"type": "object"}}}},
    "testing_criteria": [
        {"name": "Match label", "id": "Match label-1111", "type": "string_check", "input": "{{ sample.output_text }}", "reference": "{{ item.correct_label }}", "operation": "eq"},
        {"name": "Mentions", "id": "Mentions-2222", "type": "string_check", "input": "{{sample.output_text}}", "reference": "{{item.correct_label}}", "operation": "ilike"},
        {"name": "Similar", "id": "Similar-3333", "type": "text_similarity", "input": "{{sample.output_text}}", "reference": "{{item.correct_label}}", "evaluation_metric": "fuzzy_match", "pass_threshold": 0.8},
        {
            "name": "Judge",
            "id": "Judge-4444",
            "type": "label_model",
            "model": "o3-mini",
            "input": [{"type": "message", "role": "developer", "content": {"type": "input_text", "text": "Is it right?"}}, {"type": "message", "role": "user", "content": {"type": "input_text", "text": "{{item.ticket_text}} => {{sample.output_text}}"}}],
            "labels": ["correct", "incorrect"],
            "passing_labels": ["correct"],
            "sampling_params": None,
        },
        {"name": "Score", "id": "Score-5555", "type": "score_model", "model": "gpt-4.1-2025-04-14", "input": [{"role": "user", "content": "Rate {{sample.output_text}}"}], "range": [1, 7], "pass_threshold": 5.5},
        {"name": "Py", "id": "Py-6666", "type": "python", "source": "def grade(sample, item):\n    return 1.0 if sample['output_text'] == item['correct_label'] else 0.0\n", "pass_threshold": 0.5},
        {"name": "Mystery", "type": "telepathy"},
    ],
}
TEMPLATE_RUN = {
    "object": "eval.run",
    "id": "evalrun_1",
    "eval_id": "eval_abc",
    "status": "completed",
    "model": "gpt-4.1",
    "created_at": 100,
    "data_source": {
        "type": "completions",
        "source": {"type": "file_id", "id": "file-123"},
        "input_messages": {"type": "template", "template": [{"type": "message", "role": "developer", "content": {"type": "input_text", "text": "Categorize: Hardware, Software or Other."}}, {"type": "message", "role": "user", "content": {"type": "input_text", "text": "{{item.ticket_text}}"}}]},
        "model": "gpt-4.1",
        "sampling_params": {"max_completions_tokens": 2048, "temperature": 0},
    },
}
ITEMS = [
    {"ticket_text": "My monitor won't turn on!", "correct_label": "Hardware"},
    {"ticket_text": "I'm in vim and I can't quit!", "correct_label": "Software"},
    {"ticket_text": "Best restaurants in Cleveland?", "correct_label": "Other"},
]


def output_item(i, item, output, passed):
    return {
        "object": "eval.run.output_item",
        "id": f"outputitem_{i}",
        "run_id": "evalrun_1",
        "eval_id": "eval_abc",
        "status": "pass" if passed else "fail",
        "datasource_item_id": i,
        "datasource_item": item,
        "results": [
            {"name": "Match label-1111", "passed": passed, "score": 1.0 if passed else 0.0},
            {"name": "Mentions-2222", "passed": item["correct_label"].lower() in output.lower(), "score": 1.0},
        ],
        "sample": {
            "input": [{"role": "developer", "content": "Categorize: Hardware, Software or Other."}, {"role": "user", "content": item["ticket_text"], "tool_calls": None}],
            "output": [{"role": "assistant", "content": output, "tool_calls": None}],
            "finish_reason": "stop",
            "model": "gpt-4.1-2025-04-14",
            "temperature": 1.0,
            "max_completion_tokens": 2048,
            "seed": 42,
        },
    }


def test_convert_with_output_items_and_fidelity():
    outs = [output_item(0, ITEMS[0], "Hardware", True), output_item(1, ITEMS[1], "software", False), output_item(2, ITEMS[2], "Other", True)]
    cases, info = oe.convert_hosted(EVAL, run=TEMPLATE_RUN, output_items=outs)
    assert info.cases == 3 and info.with_baseline == 3 and info.run_id == "evalrun_1"
    assert any("telepathy" in w for w in info.warnings)
    case = cases[0]
    assert case.messages == [{"role": "developer", "content": "Categorize: Hardware, Software or Other."}, {"role": "user", "content": "My monitor won't turn on!"}]
    assert case.params == {"temperature": 0, "max_completion_tokens": 2048, "seed": 42}
    assert case.baseline == {"output": "Hardware", "model": "gpt-4.1-2025-04-14"}
    assert case.vars == ITEMS[0] and case.group.startswith("ticket-categories-")
    checks = {c["name"]: c for c in case.checks}
    assert checks["Match label"] == {"type": "equals", "value": "Hardware", "strip": False, "category": "custom", "name": "Match label", "grader_id": "Match label-1111", "origin": "openai_evals:string_check"}  # exact, as OpenAI's eq
    assert checks["Mentions"]["type"] == "contains" and checks["Mentions"]["case_sensitive"] is False
    assert checks["Similar"]["type"] == "similarity" and checks["Similar"]["threshold"] == 0.8
    assert checks["Judge"]["type"] == "llm_label" and checks["Judge"]["model"] == "o3-mini" and "sampling_params" not in checks["Judge"]
    assert checks["Score"]["type"] == "llm_score" and checks["Score"]["range"] == [1, 7]
    assert checks["Py"]["type"] == "python"
    agreement = oe.verify_against_recorded(cases)
    assert agreement["graded"] == 6 and agreement["agree"] == 6 and agreement["rate"] == 1.0
    res = run_checks(cases[1], Sample(text="Software"), CheckContext(allow_python=True))
    statuses = {r.name: r.status for r in res}
    assert statuses["Match label"] == "pass" and statuses["Py"] == "pass" and statuses["Judge"] == "skip"


def test_template_rendering_from_items_and_item_reference():
    items = [{"item": it} for it in ITEMS]
    cases, info = oe.convert_hosted(EVAL, run=TEMPLATE_RUN, items=items)
    assert info.with_baseline == 0 and cases[2].messages[1]["content"] == "Best restaurants in Cleveland?"
    run = {"id": "r", "status": "completed", "data_source": {"type": "completions", "input_messages": {"type": "item_reference", "item_reference": "item.chat"}}}
    ev = {"id": "e2", "name": "Ref", "testing_criteria": []}
    cases, _ = oe.convert_hosted(ev, run=run, items=[{"item": {"chat": [{"role": "user", "content": "from item"}]}}])
    assert cases[0].messages == [{"role": "user", "content": "from item"}]
    logs_eval = {"id": "e3", "name": "Logs", "data_source_config": {"type": "logs"}, "testing_criteria": []}
    cases, info = oe.convert_hosted(logs_eval)
    assert cases == [] and "llmshift snapshot" in info.warnings[0]


def test_multi_and_responses_sampling():
    g = {"type": "multi", "name": "m", "graders": {"a": {"type": "string_check", "input": "{{sample.output_text}}", "reference": "x", "operation": "eq"}}, "calculate_output": "a"}
    check = oe.convert_grader(g, {}, [])
    assert check["type"] == "multi" and check["graders"]["a"]["type"] == "equals"
    params = oe._sampling_to_params({"max_completion_tokens": 9, "text": {"format": {"type": "json_schema", "name": "n", "schema": {"type": "object"}}}, "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}]})
    assert params["response_format"]["type"] == "json_schema" and params["tools"][0]["function"]["name"] == "f"


def serve_evals_api(server):
    evals = [dict(EVAL, id="eval_abc"), {**EVAL, "id": "eval_def", "name": "Second"}]
    server.route("GET", "/v1/evals?", lambda r: {"object": "list", "data": evals[:1], "has_more": True, "last_id": "eval_abc"} if "after=" not in r["path"] else {"object": "list", "data": evals[1:], "has_more": False})
    server.route("GET", "/v1/evals/eval_abc/runs/evalrun_1/output_items", lambda r: {"object": "list", "data": [output_item(i, it, it["correct_label"], True) for i, it in enumerate(ITEMS)], "has_more": False})
    server.route("GET", "/v1/evals/eval_abc/runs", lambda r: {"object": "list", "data": [TEMPLATE_RUN], "has_more": False})
    server.route("GET", "/v1/evals/eval_def/runs", lambda r: {"object": "list", "data": [{**TEMPLATE_RUN, "id": "evalrun_2", "eval_id": "eval_def"}], "has_more": False})
    server.route("GET", "/v1/evals/eval_def/runs/evalrun_2/output_items", lambda r: {"object": "list", "data": [], "has_more": False})
    server.route("GET", "/v1/files/file-123/content", lambda r: "\n".join(json.dumps({"item": it}) for it in ITEMS).encode())


def test_import_from_api(server, monkeypatch):
    serve_evals_api(server)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = oe.EvalsClient(base_url=server.url + "/v1")
    result = oe.import_from_api(all_evals=True, client=client)
    assert [e.name for e in result.evals] == ["Ticket categories", "Second"]
    assert result.evals[0].with_baseline == 3 and result.evals[1].cases == 3 and result.evals[1].with_baseline == 0
    assert all(r["headers"]["authorization"] == "Bearer sk-test" for r in server.requests)
    warnings = oe.grader_model_warnings(result.cases)
    assert any("o3-mini" in w and "retires" in w for w in warnings)
    with pytest.raises(ValueError):
        oe.import_from_api(client=client)


def test_save_raw_keeps_every_api_response(server, monkeypatch, tmp_path, capsys):
    # The tests are a translation; the grader definitions and data sources as OpenAI held them are the record
    # to compare against later, and after 30 November the API is gone. So --save-raw keeps what was read.
    serve_evals_api(server)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", server.url + "/v1")
    monkeypatch.chdir(tmp_path)
    argv = ["import", "openai-evals", "--api", "--all", "--split", "evals", "--no-verify", "--save-raw"]
    assert cli.main(argv) == 0
    [saved] = (tmp_path / "evals").glob("openai-evals-raw-*.json")
    assert f"saved {len(server.requests)} API responses" in capsys.readouterr().out
    text = saved.read_text(encoding="utf-8")
    assert "sk-test-secret" not in text  # response bodies only, never the request headers
    responses = json.loads(text)["responses"]
    assert len(responses) == len(server.requests)  # every response the import read
    assert responses[0]["body"]["data"][0]["testing_criteria"] == EVAL["testing_criteria"]  # the graders, as OpenAI held them
    data_file = next(r for r in responses if r["path"] == "/files/file-123/content")
    assert [json.loads(line) for line in data_file["text"].splitlines()] == [{"item": it} for it in ITEMS]

    asked = len(server.requests)
    assert cli.main(argv) == cli.EXIT_USAGE  # an archive is never replaced without --force...
    assert len(server.requests) == asked  # ...and that is known before the API is called
    assert "--save-raw FILE" in capsys.readouterr().err
    assert cli.main(["import", "openai-evals", "--api", "--all", "--no-verify", "-o", "one.jsonl", "--save-raw", "keep/raw.json"]) == 0
    assert len(json.loads((tmp_path / "keep" / "raw.json").read_text(encoding="utf-8"))["responses"]) == asked
    assert cli.main(["import", "openai-evals", "eval.json", "--save-raw"]) == cli.EXIT_USAGE  # an exported file is the raw copy already


def test_import_from_export_file(tmp_path):
    (tmp_path / "eval.json").write_text(json.dumps({"eval": EVAL, "runs": [TEMPLATE_RUN], "output_items": [output_item(0, ITEMS[0], "Hardware", True)]}), encoding="utf-8")
    result = oe.import_from_export(str(tmp_path / "eval.json"))
    assert len(result.cases) == 1 and result.cases[0].baseline_output == "Hardware"
    (tmp_path / "bare.json").write_text(json.dumps(EVAL), encoding="utf-8")
    write_jsonl(tmp_path / "items.jsonl", [{"item": {"input": "hello", "correct_label": "Other"}}])
    result = oe.import_from_export(str(tmp_path / "bare.json"), data=str(tmp_path / "items.jsonl"))
    assert result.cases[0].messages == [{"role": "user", "content": "hello"}]
    (tmp_path / "nothing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        oe.import_from_export(str(tmp_path / "nothing.json"))


@pytest.fixture
def registry(tmp_path):
    root = tmp_path / "registry"
    (root / "evals").mkdir(parents=True)
    (root / "data" / "basic").mkdir(parents=True)
    (root / "modelgraded").mkdir()
    (root / "evals" / "basic.yaml").write_text(
        """
capitals:
  id: capitals.dev.v0
  metrics: [accuracy]
capitals.dev.v0:
  class: evals.elsuite.basic.match:Match
  args:
    samples_jsonl: basic/capitals.jsonl
    few_shot_jsonl: basic/fewshot.jsonl
    num_few_shot: 1
mentions:
  id: mentions.dev.v0
mentions.dev.v0:
  class: evals.elsuite.basic.includes:Includes
  args:
    samples_jsonl: basic/capitals.jsonl
    ignore_case: true
fuzzy: fuzzy.dev.v0
fuzzy.dev.v0:
  class: evals.elsuite.basic.fuzzy_match:FuzzyMatch
  args: {samples_jsonl: basic/capitals.jsonl}
jsonish:
  id: jsonish.dev.v0
jsonish.dev.v0:
  class: evals.elsuite.basic.json_match:JsonMatch
  args: {samples_jsonl: basic/json.jsonl}
facts:
  id: facts.dev.v0
facts.dev.v0:
  class: evals.elsuite.modelgraded.classify:ModelBasedClassify
  args:
    samples_jsonl: basic/capitals.jsonl
    eval_type: cot_classify
    modelgraded_spec: fact
custom:
  id: custom.dev.v0
custom.dev.v0:
  class: my.module:CustomEval
  args: {samples_jsonl: basic/capitals.jsonl}
""",
        encoding="utf-8",
    )
    write_jsonl(
        root / "data" / "basic" / "capitals.jsonl",
        [
            {"input": [{"role": "system", "content": "Answer concisely."}, {"role": "user", "content": "Capital of France?"}], "ideal": "Paris"},
            {"input": "Capital of Japan?", "ideal": ["Tokyo", "Tōkyō"]},
        ],
    )
    write_jsonl(root / "data" / "basic" / "fewshot.jsonl", [{"sample": [{"role": "system", "content": "Capital of Italy?", "name": "example_user"}, {"role": "system", "content": "Rome", "name": "example_assistant"}]}])
    write_jsonl(root / "data" / "basic" / "json.jsonl", [{"input": "Give JSON", "ideal": '{"a": 1}'}])
    return root


def test_oss_registry(registry):
    result = oe.import_oss_registry(str(registry))
    by_eval = {e.name: e for e in result.evals}
    assert by_eval["capitals"].cases == 2 and by_eval["capitals"].checks["starts_with"] == 2
    assert by_eval["mentions"].checks["contains_any"] == 2 and by_eval["fuzzy"].checks["fuzzy_match"] == 2
    assert by_eval["jsonish"].checks["json_equals"] == 1 and by_eval["facts"].checks["modelgraded"] == 2
    assert by_eval["custom"].cases == 0 and any("not supported" in w for w in by_eval["custom"].warnings)
    cap = next(c for c in result.cases if c.metadata["eval"] == "capitals" and "France" in c.user_text)
    assert [m["role"] for m in cap.messages] == ["system", "user", "assistant", "user"]  # few-shot inserted, roles converted
    assert cap.params == {"temperature": 0}
    fact = next(c for c in result.cases if c.metadata["eval"] == "facts")
    check = fact.checks[0]
    assert check["passing_choices"] == ["A", "B", "C", "E"] and check["eval_type"] == "cot_classify" and "[Expert]: {ideal}" in check["prompt"]
    ctx = CheckContext()
    tokyo = next(c for c in result.cases if c.metadata["eval"] == "mentions" and "Japan" in c.user_text)
    assert run_checks(tokyo, Sample(text="It is tokyo."), ctx)[0].status == "pass"
    only = oe.import_oss_registry(str(registry), eval_names=["capit*"])
    assert [e.name for e in only.evals] == ["capitals"]
    with pytest.raises(ValueError):
        oe.import_oss_registry(str(registry), eval_names=["nope"])


def test_canon_content():
    assert oe.canon_content({"type": "input_text", "text": "hi"}) == "hi"
    assert oe.canon_content([{"type": "input_text", "text": "a"}, {"type": "input_image", "image_url": "http://x/y.png"}]) == [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "http://x/y.png"}}]
