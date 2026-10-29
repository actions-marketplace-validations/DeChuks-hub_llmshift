"""End-to-end CLI runs (offline, simulated models)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmshift import __version__
from llmshift.cli import main
from llmshift.demo import create_project


@pytest.fixture
def demo(tmp_path, monkeypatch):
    root = create_project(str(tmp_path / "demo"))
    monkeypatch.chdir(root)
    return root


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_version_and_help(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert f"llmshift {__version__}" in capsys.readouterr().out
    assert main([]) == 2


def test_full_workflow(demo, capsys):
    code, out, _ = run(capsys, "scan")
    assert code == 1 and "RETIRED" in out and "claude-3-5-haiku-20241022" in out
    code, out, _ = run(capsys, "scan", "--fail-on", "never", "--format", "json")
    assert code == 0 and json.loads(out)["summary"]["retired"] == 1
    code, out, _ = run(capsys, "snapshot", "logs")
    assert code == 0 and "wrote 18 test case(s)" in out and Path(".llmshift/tests.jsonl").exists()
    code, out, _ = run(capsys, "compare", "--to", "mock:legacy")
    assert code == 0 and "18 unchanged" in out
    code, out, _ = run(capsys, "compare", "--to", "mock:nextgen", "--markdown", "summary.md", "--junit", "junit.xml", "--show", "all")
    assert code == 1 and "below the 95% threshold" in out and "Broken format" in out
    assert Path("summary.md").read_text(encoding="utf-8").startswith("## LLMShift report")
    reports = sorted(Path(".llmshift/reports").glob("compare-*.json"))
    assert reports and list(Path(".llmshift/reports").glob("compare-*.html"))
    code, out, _ = run(capsys, "report", str(reports[-1]), "--html", "again.html")
    assert code == 0 and Path("again.html").exists()
    code, out, _ = run(capsys, "fix", "--to", "mock:nextgen", "--write-tests", "fixed.jsonl", "--from-results", str(reports[-1]))
    assert code == 0 and "Add to the system prompt" in out and "prompt at app/triage.py" in out
    code, out, _ = run(capsys, "compare", "--tests", "fixed.jsonl", "--to", "mock:nextgen", "--min-pass-rate", "0.8")
    assert code == 0
    code, out, _ = run(capsys, "watch", "--model", "mock:nextgen", "--alert", "file:alert.json")
    assert "first run for this model" in out
    code, out, _ = run(capsys, "watch", "--model", "mock:nextgen", "--fail-on", "never")
    assert code == 0 and "previous run" in out


def test_watch_threshold_and_schedule_validation(demo, capsys):
    run(capsys, "snapshot", "logs")
    code, out, _ = run(capsys, "watch", "--model", "mock:nextgen", "--dry-run")
    assert code == 1 and "below the 95% minimum" in out  # falls back to the project-wide min_pass_rate
    code, out, _ = run(capsys, "watch", "--model", "mock:legacy", "--dry-run")
    assert code == 0 and "no drift detected" in out
    code, _, err = run(capsys, "watch", "--model", "mock:legacy", "--every", "soon")
    assert code == 2 and "cannot parse interval" in err
    code, _, err = run(capsys, "watch", "--model", "mock:legacy", "--at", "25:99")
    assert code == 2 and "HH:MM" in err
    code, _, err = run(capsys, "watch", "--model", "mock:legacy", "--every", "1h", "--at", "02:00")
    assert code == 2 and "not both" in err


def test_filters_and_limits(demo, capsys):
    run(capsys, "snapshot", "logs")
    code, out, _ = run(capsys, "compare", "--to", "mock:legacy", "--limit", "2", "--no-report")
    assert code == 0 and "2 unchanged" in out and not Path(".llmshift/reports").exists()
    code, out, _ = run(capsys, "compare", "--to", "mock:legacy", "--group", "summarise", "--no-report")
    assert code == 0 and "4 unchanged" in out
    code, _, err = run(capsys, "compare", "--to", "mock:legacy", "--group", "nothing-matches")
    assert code == 2 and "no test cases selected" in err
    code, _, err = run(capsys, "compare", "--to", "mock:legacy", "--severity", "tone=loud")
    assert code == 2 and "--severity level" in err


def test_helpful_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code, _, err = run(capsys, "compare", "--to", "mock:legacy")
    assert code == 2 and "llmshift snapshot" in err
    Path("t.jsonl").write_text('{"id": "a", "input": {"messages": [{"role": "user", "content": "x"}]}}\n', encoding="utf-8")
    code, _, err = run(capsys, "compare", "--tests", "t.jsonl")
    assert code == 2 and "--to provider:model" in err
    code, _, err = run(capsys, "compare", "--tests", "t.jsonl", "--to", "openai:gpt-6-astra")
    assert code == 2 and "OPENAI_API_KEY" in err  # every request failed -> clear message
    Path("bad.jsonl").write_text("{broken\n", encoding="utf-8")
    code, _, err = run(capsys, "compare", "--tests", "bad.jsonl", "--to", "mock:echo")
    assert code == 2 and "bad.jsonl:1" in err
    code, _, err = run(capsys, "snapshot", "missing-dir")
    assert code == 2 and "no such file" in err
    Path("empty.jsonl").write_text('{"nothing": "useful"}\n', encoding="utf-8")
    code, out, err = run(capsys, "snapshot", "empty.jsonl")
    assert code == 1 and "unrecognised record" in out


def test_init_models_doctor_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code, out, _ = run(capsys, "init", "--github", "--candidate", "anthropic:claude-sonnet-5-5")
    assert code == 0 and Path("llmshift.yaml").exists() and Path(".github/workflows/llmshift-watch.yml").exists()
    assert "anthropic:claude-sonnet-5-5" in Path("llmshift.yaml").read_text(encoding="utf-8")
    assert ".llmshift/cache/" in Path(".gitignore").read_text(encoding="utf-8")
    from llmshift._vendor import yaml

    for wf in Path(".github/workflows").glob("*.yml"):
        assert "jobs" in yaml.safe_load(wf.read_text(encoding="utf-8"))
    code, out, _ = run(capsys, "init")
    assert "already exists" in out
    code, out, _ = run(capsys, "models", "gpt-4o-2024", "--provider", "openai")
    assert code == 0 and "gpt-4o-2024-05-13" in out and "2026-10-23" in out
    code, out, _ = run(capsys, "models", "--retiring-within", "3", "--format", "json")
    assert code == 0 and all(m["days_left"] <= 3 for m in json.loads(out))
    code, out, _ = run(capsys, "doctor")
    assert code == 0 and "lifecycle registry" in out and "none (only mock:" in out
    code, out, _ = run(capsys, "cache", "clear")
    assert code == 0 and "removed 0" in out


def test_import_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ev = {"object": "eval", "id": "eval_1", "name": "Labels", "testing_criteria": [{"type": "string_check", "name": "eq", "id": "eq-1", "input": "{{sample.output_text}}", "reference": "{{item.label}}", "operation": "eq"}]}
    Path("eval.json").write_text(json.dumps(ev), encoding="utf-8")
    Path("items.jsonl").write_text("\n".join(json.dumps({"item": {"input": f"Say {w}", "label": w}}) for w in ("yes", "no")), encoding="utf-8")
    code, out, _ = run(capsys, "import", "openai-evals", "eval.json", "--data", "items.jsonl", "-o", "imported.jsonl")
    assert code == 0 and "wrote 2 test case(s)" in out
    (tmp_path / "bot.py").write_text("def answer(req):\n    return req['messages'][-1]['content'].split()[-1]\n", encoding="utf-8")
    code, out, _ = run(capsys, "compare", "--tests", "imported.jsonl", "--to", f"python:{(tmp_path / 'bot.py').as_posix()}:answer", "--no-report")
    assert code == 0 and "2 unchanged" in out
    code, _, err = run(capsys, "import", "openai-evals")
    assert code == 2 and "pass a source" in err


def test_demo_command(tmp_path, capsys):
    code = main(["demo", "--dir", str(tmp_path / "d")])
    captured = capsys.readouterr()
    out = captured.out
    assert code == 0 and "Demo complete" in out and "Overall pass rate" in out
    assert (tmp_path / "d" / ".llmshift" / "tests.fixed.jsonl").exists()
    # The demo carries on after each step, so CI exit-code notes would only confuse.
    assert "Exit code" not in out + captured.err


def test_demo_in_a_new_relative_folder_where_resolve_keeps_it_relative(tmp_path, monkeypatch, capsys):
    # On Windows, Python 3.9 before 3.9.23 (3.9.13 is its last installer) returns
    # Path("new").resolve() unchanged while nothing along the path exists; the demo then
    # looked for new/llmshift.yaml from inside new/.
    import pathlib

    real = pathlib.Path.resolve

    def resolve_like_old_windows(self, strict=False):
        if not self.is_absolute() and not any(p.exists() for p in [self, *self.parents] if str(p) != "."):
            return self
        return real(self, strict=strict)

    monkeypatch.setattr(pathlib.Path, "resolve", resolve_like_old_windows)
    monkeypatch.chdir(tmp_path)
    code = main(["demo", "--dir", "proj"])
    out = capsys.readouterr().out
    assert code == 0 and "Demo complete" in out
    assert f"Created a demo project in {tmp_path / 'proj'}" in out
    assert (tmp_path / "proj" / ".llmshift" / "tests.fixed.jsonl").exists()


def test_github_actions_outputs(demo, monkeypatch, capsys):
    summary = demo / "summary.md"
    outputs = demo / "outputs.txt"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    code, out, _ = run(capsys, "scan")
    assert code == 1 and "::error file=app/summaries.py,line=10" in out and "::warning file=app/triage.py" in out
    assert "## LLMShift scan" in summary.read_text(encoding="utf-8")
    assert "retired=1" in outputs.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "argv, words",
    [
        (["compare", "--to", "mock:nextgen", "--temperature", "99"], "must be from 0 to 2, got 99"),
        (["compare", "--to", "mock:nextgen", "--temperature", "nan"], "must be from 0 to 2"),
        (["compare", "--to", "mock:nextgen", "--max-tokens", "0"], "must be 1 or more, got 0"),
        # An unset variable in CI must not quietly test the configured model instead.
        (["compare", "--to", ""], "argument --to/--model/--candidate: is empty"),
        (["compare", "--to", "mock:nextgen", "--judge", " "], "argument --judge: is empty"),
        (["upgrade", "--from", "gpt-4o-2024-05-13", "--to", ""], "argument --to: is empty"),
        (["safety", "--baseline-model", ""], "argument --baseline-model: is empty"),
        (["watch", "--model", ""], "argument --model: is empty"),
        (["pick", "--models", " , "], "argument --models/--model: is empty"),
        (["embeddings", "--from", "", "--to", "mock:embed-v2", "--corpus", "docs"], "argument --from: is empty"),
        (["init", "--candidate", ""], "argument --candidate: is empty"),
    ],
    ids=lambda v: "-".join(v) if isinstance(v, list) else None,
)
def test_bad_run_options_stop_before_any_request(argv, words, capsys):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2 and words in capsys.readouterr().err


def test_embeddings_checks_the_corpus_before_announcing_work(in_tmp, capsys):
    code, _, err = run(capsys, "embeddings", "--from", "mock:embed-v1", "--to", "mock:embed-v2", "--corpus", "missing-folder")
    assert code == 2 and "corpus path not found: missing-folder" in err and "Embedding a sample" not in err
    (in_tmp / "docs").mkdir()
    for i in range(3):
        (in_tmp / "docs" / f"doc{i}.txt").write_text(f"Refunds take {i + 3} days. Orders ship from warehouse {i}.", encoding="utf-8")
    code, _, err = run(capsys, "embeddings", "--from", "mock:embed-v1", "--to", "mock:embed-v2", "--corpus", "docs")
    assert code in (0, 1) and "Embedding a sample of your corpus with mock:embed-v1 and mock:embed-v2" in err


def test_scan_counts_in_words_and_names_the_replacement(in_tmp, capsys):
    (in_tmp / "app.py").write_text('client.messages.create(model="claude-sonnet-4-5", max_tokens=10)\n', encoding="utf-8")
    code, out, _ = run(capsys, "scan", "--fail-on", "never")
    assert code == 0
    assert "1 file, 1 model reference, 1 distinct model" in out and "1 files" not in out
    # The next step names the replacement of the most urgent model, and how to keep the check in CI.
    assert "llmshift compare --to claude-sonnet-5-5" in out and "llmshift init --github" in out
    code, out, _ = run(capsys, "scan", "--fail-on", "never", "--format", "markdown")
    assert "1 file scanned, 1 model reference," in out
