"""Torture tests: hostile, huge, broken and unlucky inputs must never crash or hang
llmshift, and every failure must come with a clear message."""

from __future__ import annotations

import gzip
import io
import json
import os
import random
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import pytest

from llmshift import cli, console, facts, formats, logparse, textutil, tone
from llmshift.checks import CHECKS, CheckContext, Sample, evaluate
from llmshift.config import ConfigError, load_config
from llmshift.demo import create_project
from llmshift.providers import http
from llmshift.providers.base import ProviderError
from llmshift.providers.openai import OpenAIProvider
from llmshift.redact import Redactor
from llmshift.testset import TestCase, TestFileError, load_tests

SEED = 20261022
WEIRD = ["", " ", "\u200b", "\u202e", "\ufeff", "\x00", "\u2028", "\u0301", "\U0001f600", "\U0001f469\u200d\U0001f469\u200d\U0001f467", "\u65e5\u672c\u8a9e", "\u0639\u0631\u0628\u0649", "\ud83d", "\t", "\r\n", "\\", '"', "'", "{", "]"]


def _text(rng: random.Random, n: int = 40) -> str:
    return "".join(rng.choice(WEIRD + list("abc xyz 0123456789.,$%#@")) for _ in range(rng.randint(0, n)))


def _value(rng: random.Random, depth: int = 0):
    pick = rng.randint(0, 9 if depth < 3 else 5)
    if pick == 0:
        return None
    if pick == 1:
        return rng.choice([True, False])
    if pick == 2:
        return rng.choice([0, -1, 1.5, 10**30, -(10**30), 1e308])
    if pick in (3, 4, 5):
        return _text(rng)
    if pick in (6, 7):
        return [_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {rng.choice(["role", "content", "messages", "text", "model", "tool_calls", "name", _text(rng, 5)]): _value(rng, depth + 1) for _ in range(rng.randint(0, 4))}


def _timed(limit: float):
    start = time.perf_counter()

    def done() -> float:
        took = time.perf_counter() - start
        assert took < limit, f"took {took:.1f}s (limit {limit}s)"
        return took

    return done


# ------------------------------------------------------------------- YAML


def _alias_bomb(levels: int = 9) -> str:
    lines = ["a0: &a0 [" + ",".join(['"lol"'] * 9) + "]"]
    lines += [f"a{i}: &a{i} [" + ",".join([f"*a{i - 1}"] * 9) + "]" for i in range(1, levels)]
    return "\n".join(lines) + "\ncandidate: mock:legacy\n"


def test_yaml_alias_bomb_in_config_is_refused_at_once(in_tmp, capsys):
    (in_tmp / "llmshift.yaml").write_text(_alias_bomb(), encoding="utf-8")
    done = _timed(5)
    assert cli.main(["doctor"]) == 2
    done()
    assert "expands to more than" in capsys.readouterr().err


def test_yaml_bombs_and_recursion_in_test_files(tmp_path):
    bomb = tmp_path / "bomb.yaml"
    bomb.write_text(_alias_bomb() + "cases: *a8\n", encoding="utf-8")
    done = _timed(5)
    with pytest.raises(TestFileError, match="expands to more than"):
        load_tests(bomb)
    done()
    for text in ("a: &a [*a]\n", "x: " + "[" * 3000 + "]" * 3000 + "\n"):
        f = tmp_path / "t.yaml"
        f.write_text(text, encoding="utf-8")
        with pytest.raises(TestFileError):
            load_tests(f)


def test_recursive_and_deep_config(in_tmp):
    for text in ("a: &a [*a]\ncandidate: x\n", "x: " + "[" * 3000 + "]" * 3000 + "\n", "x: " + "{a: " * 2000 + "1" + "}" * 2000 + "\n"):
        (in_tmp / "llmshift.yaml").write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config()


# ------------------------------------------------------------------- logs


def _hostile_records(rng: random.Random, n: int) -> list[str]:
    lines = []
    for _ in range(n):
        kind = rng.randint(0, 7)
        if kind == 0:
            lines.append("".join(chr(rng.randint(1, 0x2FFF)) for _ in range(rng.randint(1, 60))).replace("\n", " "))
        elif kind == 1:
            lines.append(json.dumps(_value(rng)))
        elif kind == 2:
            lines.append(json.dumps({"request": _value(rng), "response": _value(rng)}))
        elif kind == 3:
            lines.append(json.dumps({"model": _value(rng), "messages": _value(rng), "choices": _value(rng)}))
        elif kind == 4:
            lines.append(json.dumps({"custom_id": _value(rng), "body": _value(rng), "response": _value(rng)}))
        elif kind == 5:
            lines.append(json.dumps({"input": _value(rng), "output": _value(rng), "timestamp": _value(rng)}))
        elif kind == 6:
            lines.append("[" * rng.randint(1000, 5000) + "]" * 10)
        else:
            lines.append(json.dumps({"llmshift_record": 1, "request": {"model": "m", "messages": [{"role": "user", "content": _text(rng)}]}, "response": {"choices": [{"message": {"content": _text(rng)}}]}}))
    return lines


def test_hostile_log_records_never_crash(tmp_path):
    rng = random.Random(SEED)
    lines = _hostile_records(rng, 1500)
    (tmp_path / "hostile.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", errors="surrogatepass")
    done = _timed(30)
    inters, stats = logparse.parse_logs([str(tmp_path)])
    done()
    assert stats.records + sum(stats.skipped.values()) >= len([x for x in lines if x.strip()]) - sum(stats.skipped.values())
    assert all(isinstance(i.messages, list) for i in inters)
    for inter in inters:  # whatever was accepted can be stored
        json.dumps({"m": inter.messages, "o": inter.output}, ensure_ascii=False).encode("utf-8")


def test_a_line_bomb_in_a_gzip_log_is_skipped_in_bounded_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(logparse, "MAX_LINE_BYTES", 1024 * 1024)
    good = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "choices": [{"message": {"content": "hello"}}]})
    with gzip.open(tmp_path / "bomb.jsonl.gz", "wb") as fh:
        for _ in range(30):
            fh.write(b"a" * (1024 * 1024))
        fh.write(b"\n" + good.encode() + b"\n")
    tracemalloc.start()
    inters, stats = logparse.parse_logs([str(tmp_path)])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(inters) == 1 and any("too large" in r for r in stats.skipped)
    assert peak < 16 * 1024 * 1024, f"peak memory {peak / 1e6:.0f} MB"


def test_oversized_json_documents_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(logparse, "MAX_DOCUMENT_BYTES", 1000)
    (tmp_path / "big.json").write_text(json.dumps([{"prompt": "x" * 50, "output": "y"}] * 40), encoding="utf-8")
    inters, stats = logparse.parse_logs([str(tmp_path)])
    assert not inters and any("too large" in r for r in stats.skipped)


def test_huge_logs_keep_a_fair_sample_per_template(tmp_path):
    rows = []
    for template, n in (("billing", 2000), ("shipping", 900), ("legal", 30)):
        for i in range(n):
            rows.append(json.dumps({"model": "gpt-4o", "messages": [{"role": "system", "content": f"You answer {template} questions."}, {"role": "user", "content": f"q{i}"}],
                                    "choices": [{"message": {"content": f"a{i}"}}]}))
    (tmp_path / "log.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    inters, stats = logparse.parse_logs([str(tmp_path)], max_records=300)
    assert len(inters) == 300 and stats.parsed == 2930 and stats.sampled_out == 2630
    by = {}
    for inter in inters:
        by.setdefault(inter.messages[0]["content"], 0)
        by[inter.messages[0]["content"]] += 1
    assert len(by) == 3 and by["You answer legal questions."] >= 10  # rare templates keep a place


def test_line_separators_bom_and_crlf_in_logs(tmp_path):
    rec = {"model": "gpt-4o", "messages": [{"role": "user", "content": "line\u2028sep\u2029para\x85next"}], "choices": [{"message": {"content": "ok"}}]}
    data = "\ufeff" + json.dumps(rec, ensure_ascii=False) + "\r\n" + json.dumps(rec, ensure_ascii=False) + "\r\n"
    (tmp_path / "x.jsonl").write_bytes(data.encode("utf-8"))
    inters, stats = logparse.parse_logs([str(tmp_path)])
    assert len(inters) == 2 and not stats.skipped
    assert inters[0].messages[0]["content"] == "line\u2028sep\u2029para\x85next"


def test_binary_and_corrupt_files(tmp_path):
    rng = random.Random(SEED)
    (tmp_path / "noise.jsonl").write_bytes(bytes(rng.randint(0, 255) for _ in range(200_000)))
    noisy = "".join(json.dumps({"prompt": os.urandom(30).hex(), "output": os.urandom(30).hex()}) + "\n" for _ in range(2000))
    whole = gzip.compress(noisy.encode())
    (tmp_path / "trunc.jsonl.gz").write_bytes(whole[: len(whole) // 2])  # cut mid-stream
    (tmp_path / "empty.jsonl").write_bytes(b"")
    (tmp_path / "bad.csv").write_bytes(b'prompt,output\n"unterminated,\n' + b"\x00" * 100)
    inters, stats = logparse.parse_logs([str(tmp_path)])
    assert stats.files == 4 and any("unreadable file" in r for r in stats.skipped)
    # The complete records before the cut in the .gz are kept; the rest is reported unreadable.
    salvaged = [i for i in inters if i.source.endswith(".gz") or ".gz:" in i.source]
    assert 0 < len(salvaged) < 2000 and len(inters) - len(salvaged) <= 1


# ----------------------------------------------------------------- checks


def _random_check(rng: random.Random, ctype: str) -> dict:
    check = {"type": ctype}
    for key in rng.sample(["value", "values", "pattern", "schema", "expected", "path", "facts", "threshold", "min_words", "max_words", "language", "metric", "reference", "negate", "case_sensitive", "severity", "fields", "labels"], 4):
        check[key] = _value(rng)
    if rng.random() < 0.3:
        check["severity"] = rng.choice(["error", "warn", "info", "nonsense", None])
    if ctype == "regex":
        check["pattern"] = rng.choice(["a+", "[", "(", "\\d{3}", "(?i)hello", ""])
    return check


def test_every_check_survives_hostile_parameters_and_outputs():
    rng = random.Random(SEED)
    ctx = CheckContext()
    case = TestCase(id="t", input={"messages": [{"role": "user", "content": "q"}], "params": {}}, baseline={"output": '{"a": 1, "b": [1, 2]}', "tool_calls": [{"name": "f", "arguments": {"x": 1}}]})
    evaluate({"type": "not_empty"}, Sample(text="x"), case, ctx)  # registers the model graders
    done = _timed(60)
    for _ in range(25):
        for ctype in list(CHECKS):
            sample = Sample(text=_text(rng, 200) if rng.random() < 0.7 else json.dumps(_value(rng)), tool_calls=_value(rng) if isinstance(_value(rng), list) else [])
            result = evaluate(_random_check(rng, ctype), sample, case, ctx)
            assert result.status in ("pass", "fail", "skip", "error"), (ctype, result)
            assert isinstance(result.message, str)
    done()


def test_text_algorithms_stay_fast_on_adversarial_input():
    big_digits = "1" * 400_000
    cases = [
        lambda: facts.extract(big_digits),
        lambda: facts.extract("$1,000,000.00 on 2024-01-01 at 10:30 for #A-1234; " * 8000),
        lambda: facts.match(facts.extract("Revenue rose 12.5% to $4.2 million on March 3."), "x" * 300_000),
        lambda: Redactor(pii=True).text("4" * 300_000 + " a@b.co " * 20_000),
        lambda: formats.find_embedded_json("{" * 200_000),
        lambda: formats.find_embedded_json('{"a":' * 60_000),
        lambda: formats.is_refusal("I can't " * 100_000),
        lambda: tone.profile("😀 Great!!! " * 50_000),
        lambda: textutil.word_diff("alpha beta " * 15_000, "alpha gamma " * 15_000),
        lambda: textutil.rouge_l("w " * 60_000, "w x " * 30_000),
        lambda: textutil.detect_language("le " * 200_000),
        # Runaway model output that used to take a minute in one regex each:
        lambda: facts.extract("Answer:" + "\n" * 32_000),
        lambda: formats.strip_fences("```json\n{" + "\t" * 64_000 + "x"),
        lambda: facts.extract("Code: " + "a-" * 16_000),
        lambda: Redactor(pii=True).text("Contact: " + "a-" * 16_000),
    ]
    for fn in cases:
        done = _timed(15)
        fn()
        done()
    from llmshift.scanner import scan_generic

    done = _timed(15)
    scan_generic("\n".join('model = "' + "a1" * 995 + '"' for _ in range(900)), code=True)  # 697 s before
    done()


def test_scanner_on_minified_and_binary_sources(in_tmp):
    from llmshift.scanner import scan_paths

    (in_tmp / "app.min.js").write_text("var a=1;" * 180_000 + 'm="gpt-4o-2024-05-13";', encoding="utf-8")
    (in_tmp / "vendor.bundle.js").write_text("var b=2;" * 400_000, encoding="utf-8")  # over the 2 MB per-file cap
    (in_tmp / "blob.py").write_bytes(os.urandom(500_000))
    (in_tmp / "deep.py").write_text("x = " + "(" * 5000 + "1" + ")" * 5000 + "\nmodel = 'gpt-4o'\n", encoding="utf-8")
    done = _timed(30)
    result = scan_paths(["."])
    done()
    assert "gpt-4o-2024-05-13" in {f.model for f in result.findings}
    assert any("vendor.bundle.js" in e and "larger than" in e for e in result.errors)


# --------------------------------------------------------------- providers


def _provider(server, **options) -> OpenAIProvider:
    return OpenAIProvider("gpt-test", options={"base_url": server.url + "/v1", "api_key": "k", "max_retries": 1, "timeout": 2, **options})


REQUEST = {"messages": [{"role": "user", "content": "hi"}], "params": {}}


def _ok(text="fine"):
    return {"id": "x", "object": "chat.completion", "model": "gpt-test", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


def test_provider_chaos(server, monkeypatch):
    delays: list[float] = []
    monkeypatch.setattr(http, "_sleep", lambda seconds, sleep: delays.append(seconds))
    behaviours = iter([
        (200, "<html>gateway</html>"),  # garbage 200
        (502, "<html>bad gateway</html>"), (200, _ok("after a 502")),  # retried
        (429, {"error": {"message": "slow down"}}, {"Retry-After": "99999"}),  # more than a day: not waited for
        (429, {"error": {"message": "slow down"}}, {"Retry-After": "-5"}), (200, _ok()),
        (429, {"error": {"message": "slow down"}}, {"Retry-After": "soon"}), (200, _ok()),
        (200, "[" * 50_000 + "]" * 50_000),  # nested too deeply
        (200, json.dumps(_ok("broken X emoji")).replace("X", "\\ud83d")),
        (200, {"error": {"message": "model overloaded", "type": "server_error"}}),
    ])
    server.route("POST", "/v1/chat/completions", lambda req: next(behaviours))
    p = _provider(server)
    with pytest.raises(ProviderError):
        p.complete(REQUEST)
    assert p.complete(REQUEST).text == "after a 502"
    waited = len(delays)
    with pytest.raises(ProviderError, match=r"asks to wait 1 day before trying again, longer than llmshift waits \(60 s\)"):
        p.complete(REQUEST)
    assert len(delays) == waited  # failed at once instead of sleeping a minute per retry
    assert max(delays) <= http.MAX_DELAY
    p.complete(REQUEST)
    p.complete(REQUEST)
    assert all(0 <= d <= http.MAX_DELAY for d in delays)
    with pytest.raises(ProviderError, match="nested too deeply"):
        p.complete(REQUEST)
    assert p.complete(REQUEST).text == "broken � emoji"
    with pytest.raises(ProviderError):
        p.complete(REQUEST)


def test_a_slow_server_is_not_called_unreachable_and_a_day_long_wait_is_refused(server, monkeypatch):
    import email.utils

    from llmshift.runner import _repeats

    monkeypatch.setattr(http, "_sleep", lambda seconds, sleep: None)
    server.route("POST", "/v1/chat/completions", lambda req: (time.sleep(2), _ok())[1])
    with pytest.raises(ProviderError) as info:
        _provider(server, timeout=0.5).complete(REQUEST)
    message = str(info.value)
    assert info.value.code == "timeout" and "no answer from 127.0.0.1" in message and "within 0.5s" in message
    assert "was reached but is slow" in message and "cannot reach" not in message
    assert _repeats(info.value)  # every case would wait as long: the run stops early, as before
    in_two_days = email.utils.formatdate(time.time() + 2 * 86400, usegmt=True)
    server.route("POST", "/v1/chat/completions", lambda req: (429, {"error": {"message": "quota"}}, {"Retry-After": in_two_days}))
    before = len(server.requests)
    with pytest.raises(ProviderError, match="asks to wait 2 days") as info:
        _provider(server).complete(REQUEST)
    assert len(server.requests) - before == 1 and not info.value.retryable
    assert [http._human_wait(s) for s in (90, 900, 7200, 86400, 3 * 86400)] == ["90 s", "15 min", "2 h", "1 day", "3 days"]


def test_resets_timeouts_and_oversized_bodies(server, monkeypatch):
    monkeypatch.setattr(http, "_sleep", lambda seconds, sleep: None)

    def reset(req):
        raise ConnectionResetError("boom")

    server.route("POST", "/v1/chat/completions", reset)
    with pytest.raises(ProviderError) as exc:
        _provider(server).complete(REQUEST)
    assert exc.value.code == "connection_error"
    server.route("POST", "/v1/chat/completions", lambda req: (time.sleep(3), _ok())[1])
    done = _timed(12)
    with pytest.raises(ProviderError):
        _provider(server, timeout=0.5).complete(REQUEST)
    done()
    monkeypatch.setattr(http, "MAX_RESPONSE_BYTES", 10_000)
    server.route("POST", "/v1/chat/completions", lambda req: _ok("x" * 50_000))
    with pytest.raises(ProviderError, match="larger than"):
        _provider(server).complete(REQUEST)


def test_compare_survives_a_provider_that_fails_at_random(server, monkeypatch, make_case):
    from llmshift.compare import run_compare

    monkeypatch.setattr(http, "_sleep", lambda seconds, sleep: None)
    rng = random.Random(SEED)
    lock = __import__("threading").Lock()

    def flaky(req):
        with lock:
            roll = rng.random()
        if roll < 0.2:
            return 500, {"error": {"message": "internal"}}
        if roll < 0.3:
            return 200, "not json"
        if roll < 0.35:
            raise ConnectionResetError("reset")
        return _ok("hello")

    server.route("POST", "/v1/chat/completions", flaky)
    cases = [make_case("hello", user=f"q{i}", id=f"c{i}") for i in range(40)]
    res = run_compare(cases, _provider(server, max_retries=2), concurrency=8, show_progress=False)
    s = res.summary
    assert s["total"] == 40 and s["evaluated"] + s["errors"] == 40 and s["evaluated"] > 20


# ------------------------------------------------------------ environment


@pytest.fixture
def demo(tmp_path, monkeypatch):
    root = create_project(str(tmp_path / "demo"))
    monkeypatch.chdir(root)
    assert cli.main(["snapshot", "logs"]) == 0
    return root


def test_ctrl_c_during_a_run_exits_130_and_the_next_run_works(demo, monkeypatch):
    from llmshift import runner

    real_wait = runner.wait

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "wait", interrupted)
    assert cli.main(["compare", "--to", "mock:legacy", "--no-cache", "--no-report"]) == 130
    assert http.CANCEL.is_set()
    monkeypatch.setattr(runner, "wait", real_wait)
    assert cli.main(["compare", "--to", "mock:legacy", "--no-cache", "--no-report"]) == 0
    assert not http.CANCEL.is_set()


def test_ctrl_c_in_a_watch_loop(demo, monkeypatch):
    from llmshift import watch

    def stop(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(watch, "sleep_until_next", stop)
    assert cli.main(["watch", "--model", "mock:legacy", "--every", "1m"]) == 130


def test_unwritable_outputs_exit_2_with_the_reason(demo, capsys):
    Path("afile.txt").write_text("x", encoding="utf-8")
    assert cli.main(["compare", "--to", "mock:legacy", "--html", "afile.txt/report.html", "--no-report"]) == 2
    assert "cannot write afile.txt" in capsys.readouterr().err.replace("\\", "/")


def test_a_full_disk_never_breaks_a_run(demo, monkeypatch):
    import errno

    from llmshift import cache

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(cache.fileio, "temp_file", full)
    assert cli.main(["compare", "--to", "mock:legacy", "--no-report"]) == 0


def test_parallel_processes_share_one_cache_and_one_log(tmp_path):
    script = (
        "import sys, json\n"
        "from llmshift.record import log\n"
        "from llmshift.cache import ResponseCache\n"
        "from llmshift.compare import run_compare\n"
        "from llmshift.testset import TestCase\n"
        "cases = [TestCase(id=f'c{i}', input={'messages': [{'role': 'user', 'content': f'q{i}'}], 'params': {}}, baseline={'output': f'q{i}'}) for i in range(30)]\n"
        "run_compare(cases, 'mock:echo', cache=ResponseCache(sys.argv[1]), show_progress=False)\n"
        "for i in range(150):\n"
        "    log({'model': 'm', 'messages': [{'role': 'user', 'content': 'x' * 3000 + str(i)}]}, {'text': 'y'}, path=sys.argv[2])\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    procs = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path / "cache"), str(tmp_path / "log.jsonl")], env=env) for _ in range(4)]
    assert all(p.wait(timeout=180) == 0 for p in procs)
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 600 and all(json.loads(x)["llmshift_record"] == 1 for x in lines)
    for f in (tmp_path / "cache").rglob("*.json"):
        assert isinstance(json.loads(f.read_text(encoding="utf-8")), dict)


def test_unicode_and_spaces_in_paths(tmp_path, monkeypatch):
    root = create_project(str(tmp_path / "проект с пробелами 😀"))
    monkeypatch.chdir(root)
    assert cli.main(["snapshot", "logs"]) == 0
    assert cli.main(["compare", "--to", "mock:legacy", "--html", "отчёт 😀.html", "--no-report"]) == 0
    assert (root / "отчёт 😀.html").exists()


def test_legacy_console_encodings_never_crash(monkeypatch):
    for encoding in ("cp1252", "ascii", "latin-1"):
        stream = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")
        monkeypatch.setattr(sys, "stdout", stream)
        console.out("emoji 😀, check ✓, arrow →, cjk 日本, rtl عربى, broken \ud83d")
        stream.flush()
        assert stream.buffer.getvalue()


# -------------------------------------------------------------- CLI fuzz


COMMANDS = [
    ["scan"], ["models"], ["snapshot", "logs"], ["compare"], ["fix"], ["watch"], ["import", "openai-evals"], ["report", "x.json"],
    ["doctor"], ["cache", "clear"], ["pick"], ["check"], ["upgrade"], ["safety"], ["shadow", "report"], ["judge", "calibrate"],
    ["embeddings"], ["digest"], ["calendar"], ["init"],
]


def test_every_command_rejects_garbage_arguments(in_tmp, capsys):
    for argv in COMMANDS:
        with pytest.raises(SystemExit) as exc:
            cli.main(argv + ["--definitely-not-an-option"])
        assert exc.value.code == 2, argv
        with pytest.raises(SystemExit) as exc:
            cli.main(argv[:1] + ["--help"] if argv[0] not in ("import", "shadow", "judge", "cache") else argv[:2] + ["--help"])
        assert exc.value.code == 0, argv
    capsys.readouterr()


def test_every_command_in_an_empty_folder_fails_cleanly(in_tmp, capsys, monkeypatch):
    monkeypatch.setenv("LLMSHIFT_REGISTRY_URL", "http://127.0.0.1:9/models.json")
    for argv in COMMANDS + [["update", "--check"], ["judge", "label"]]:
        try:
            code = cli.main(argv)
        except SystemExit as exc:  # argparse: missing required arguments
            code = exc.code
        out = capsys.readouterr()
        assert code in (0, 1, 2), (argv, code)
        assert "Traceback" not in out.err and "unexpected" not in out.err, (argv, out.err)


def test_config_fuzz_never_crashes(in_tmp):
    rng = random.Random(SEED)
    keys = ["concurrency", "timeout", "max_retries", "min_pass_rate", "repeat", "cache", "judge_mode", "tests", "baseline", "scan", "snapshot", "fix", "watch", "providers", "pricing", "severity"]
    from llmshift._vendor import yaml

    for _ in range(200):
        doc = {rng.choice(keys): _value(rng) for _ in range(rng.randint(1, 5))}
        (in_tmp / "llmshift.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8", errors="surrogatepass")
        try:
            load_config()
        except ConfigError:
            pass


def test_test_file_fuzz_never_crashes(tmp_path):
    rng = random.Random(SEED)
    for i in range(300):
        rec = {"id": _value(rng), "input": _value(rng), "baseline": _value(rng), "checks": _value(rng), "agent": _value(rng) if rng.random() < 0.3 else None}
        f = tmp_path / f"t{i}.jsonl"
        f.write_text(json.dumps({k: v for k, v in rec.items() if v is not None}) + "\n", encoding="utf-8")
        try:
            load_tests(f)
        except TestFileError:
            pass


def test_json_nesting_limit_is_the_same_on_every_platform(tmp_path):
    # The C parser's own limit depends on the OS and Python version: 5000 levels parse on
    # Linux and macOS with Python 3.12+, not on Windows. Verdicts must not differ.
    limit = textutil.MAX_JSON_DEPTH
    ok, deep = "[" * limit + "]" * limit, "[" * (limit + 1) + "]" * (limit + 1)
    assert textutil.json_loads(ok) and not textutil.json_too_deep(ok)
    with pytest.raises(RecursionError):
        textutil.json_loads(deep)
    assert formats.find_embedded_json("x" + "[" * 5000 + "]" * 5000) is None
    assert formats.analyze_json("[" * 5000 + "]" * 5000)["valid"] is False
    assert formats.find_embedded_json('Answer: {"a": [1, {"b": "]]]"}]} done')[0] == {"a": [1, {"b": "]]]"}]}
    # Brackets inside strings never count, escaped quotes don't end a string, escaped backslashes do.
    assert not textutil.json_too_deep('{"s": "' + "[" * 500 + '"}')
    assert not textutil.json_too_deep('{"s": "a\\"' + "[" * 300 + '"}')
    assert textutil.json_too_deep('{"s": "a\\\\", "t": ' + "[" * 250 + "]" * 250 + "}")
    # The same limit applies to log lines and test files.
    deep_record = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "choices": [{"message": {"content": "ok"}}]})
    (tmp_path / "logs.jsonl").write_text(deep_record + "\n" + '{"x": ' + deep + "}\n", encoding="utf-8")
    inters, stats = logparse.parse_logs([str(tmp_path / "logs.jsonl")])
    assert len(inters) == 1 and stats.skipped["JSON nested too deeply"] == 1
    (tmp_path / "t.jsonl").write_text('{"id": "a", "input": ' + deep + "}\n", encoding="utf-8")
    with pytest.raises(TestFileError, match="nested too deeply"):
        load_tests(tmp_path / "t.jsonl")


def test_adversarial_json_nesting_scan_stays_linear():
    started = time.perf_counter()
    for text in ('["' + '\\"' * 400_000, ('"' + "\\" * 1000 + '"[') * 2000, '["' + "[" * 800_000 + '"]', "[" * 800_000, "[" + ",".join(["[1]"] * 200_000) + "]"):
        textutil.json_too_deep(text)
    assert time.perf_counter() - started < 5


def test_deep_python_never_crashes_scan_or_check(in_tmp):
    """A 3,000-term `+` chain or a 600-branch elif ladder: the model call next to it is
    still found, prompts in the file are still read, and nothing raises RecursionError."""
    from llmshift.prcheck import prompt_candidates
    from llmshift.scanner import scan_paths

    chain = "SQL = " + " + ".join(f'"part {i} "' for i in range(3000)) + "\n"
    ladder = "def route(x):\n    if x == 0:\n        return 0\n" + "".join(f"    elif x == {i}:\n        return {i}\n" for i in range(1, 600))
    call = 'client.chat.completions.create(model="gpt-4o-2024-05-13", messages=[])\n'
    for name, body in (("chain.py", chain), ("ladder.py", ladder)):
        (in_tmp / name).write_text("from openai import OpenAI\nclient = OpenAI()\n" + body + call, encoding="utf-8")
        result = scan_paths([name])
        assert [f.model for f in result.findings] == ["gpt-4o-2024-05-13"] and not result.errors, name
    start = time.perf_counter()
    cands = prompt_candidates(chain, "chain.py")
    assert time.perf_counter() - start < 5 and "".join(f"part {i} " for i in range(3000)) in cands
    deep_json = "[" * 5000 + '"You are a careful assistant who answers briefly."' + "]" * 5000
    assert prompt_candidates(deep_json, "fixture.json")  # read as text, not a RecursionError


def test_warnings_about_the_users_code_are_not_printed(in_tmp, capsys):
    """Python 3.12+ warns about invalid escape sequences while parsing; that is about
    the user's code, so llmshift does not print it, and other warnings are untouched."""
    import warnings

    from llmshift.textutil import parse_python

    (in_tmp / "app.py").write_text('import re\nPAT = re.compile("\\d+")\nMODEL = "gpt-4o-2024-05-13"\n', encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # as under PYTHONWARNINGS=error: a parse must not fail
        assert parse_python('x = "\\d"') is not None
        assert cli.main(["scan", "app.py", "--fail-on", "never"]) == 0
    assert "SyntaxWarning" not in capsys.readouterr().err
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        compile('x = "\\d"', "host_app.py", "exec")  # the host app's own code still warns as usual
    assert any(issubclass(w.category, SyntaxWarning) for w in seen) or sys.version_info < (3, 12)


def test_a_folder_that_refuses_new_files_fails_at_once(tmp_path, monkeypatch):
    """On Windows, tempfile.mkstemp before Python 3.13 retries such a folder about two
    billion times (a hang); llmshift's own temp files give up on the first refusal."""
    from llmshift import fileio
    from llmshift.testset import atomic_write

    attempts = []
    real_open = os.open

    def refuse(path, flags, *args, **kwargs):
        if flags & os.O_EXCL:
            attempts.append(path)
            raise PermissionError(13, "Access is denied", path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", refuse)
    with pytest.raises(PermissionError):
        fileio.temp_file(tmp_path)
    with pytest.raises(PermissionError):
        atomic_write(tmp_path / "tests.jsonl", "{}\n")
    assert len(attempts) == 2


def test_redaction_knows_every_provider_key_and_real_world_numbers(monkeypatch):
    monkeypatch.setenv("MY_SERVICE_API_KEY", "0123456789abcdef0123456789abcdef")  # no recognisable shape
    keys = [
        "xai-" + "ABCDEFGHIJKLMNOPQRSTUV", "gsk_" + "ABCDEFGHIJKLMNOPQRSTUV", "pplx-" + "ABCDEFGHIJKLMNOPQRSTUV", "fw_" + "ABCDEFGHIJKLMNOPQRSTUV",
        "hf_" + "ABCDEFGHIJKLMNOPQRSTUV", "r8_" + "ABCDEFGHIJKLMNOPQRSTUV", "sk_live_" + "ABCDEFGHIJKLMNOP", "ASIA" + "ABCDEFGHIJKLMNOP",
        "AIza" + "SyA1234567890abcdefghijklmnopqrstu-", "0123456789abcdef0123456789abcdef",
    ]
    out = Redactor(secrets=True).text(" | ".join(keys))
    assert not any(k in out for k in keys), out
    pii = Redactor(secrets=True, pii=True)
    text = "IBAN gb82west12345698765432, phones +44 20 7946 0958, 01 23 45 67 89, card 4111.1111.1111.1111 expires"
    out = pii.text(text)
    assert "gb82" not in out and "7946" not in out and "45 67" not in out and "4111" not in out and "> expires" in out
    # ...and leaves look-alikes alone: no check digits, too few digits.
    keep = "SKU AB12 CDEF GHIJ KLMN, score +10 20 30, version 1.2.3"
    assert pii.text(keep) == keep


def test_wrong_typed_fields_in_test_files_name_the_place(tmp_path):
    base = {"input": {"messages": [{"role": "user", "content": "hi"}]}, "baseline": {"output": "ok"}}
    for rec, words in (
        ({**base, "input": {"messages": [{"role": "user", "content": "hi"}], "params": 5}}, "'params' must be an object"),
        ({**base, "tags": 5}, "'tags' must be a list"),
        ({**base, "vars": "x"}, "'vars' must be an object"),
        ({**base, "metadata": [1]}, "'metadata' must be an object"),
    ):
        (tmp_path / "t.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
        with pytest.raises(TestFileError, match=words) as info:
            load_tests(tmp_path / "t.jsonl")
        assert "t.jsonl:1" in str(info.value)
    (tmp_path / "t.yaml").write_text("- input: hi\n  baseline: 2024-02-30\n", encoding="utf-8")
    with pytest.raises(TestFileError, match="t.yaml: invalid YAML value"):
        load_tests(tmp_path / "t.yaml")
    (tmp_path / "u.yaml").write_text("- input: hi\n  vars: {1: a, b: c}\n  baseline: ok\n", encoding="utf-8")
    assert len(load_tests(tmp_path / "u.yaml")) == 1  # keys of mixed types are fine


def test_ctrl_c_stops_judge_and_embedding_calls_at_once():
    """The judge phase of compare, judge calibrate and embeddings wait on worker threads:
    Ctrl+C must stop them at once, not after every queued call has finished."""
    import _thread
    import threading

    from llmshift.runner import interruptible_map

    def slow(x):
        end = time.monotonic() + 30
        while time.monotonic() < end and not http.CANCEL.is_set():
            time.sleep(0.01)
        return x

    threading.Timer(0.3, _thread.interrupt_main).start()
    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        interruptible_map(slow, range(8), 2)
    assert time.monotonic() - start < 5 and http.CANCEL.is_set()
    http.CANCEL.clear()
    assert interruptible_map(lambda x: x * 2, range(5), 3) == [0, 2, 4, 6, 8]  # order kept


def test_a_watch_loop_survives_an_outage(demo, monkeypatch, tmp_path):
    """A scheduled watch reports an outage (alert, critical) and checks again next time,
    instead of exiting for good on the first network failure."""
    from llmshift import watch
    from llmshift.runner import RunAborted

    real, calls, sleeps = watch.run_watch, [], []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RunAborted("cannot reach api.example.com: connection refused")
        return real(*args, **kwargs)

    def tick(*args):
        sleeps.append(1)
        if len(sleeps) >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch, "run_watch", flaky)
    monkeypatch.setattr(watch, "sleep_until_next", tick)
    alert = tmp_path / "outage.json"
    assert cli.main(["watch", "--model", "mock:legacy", "--every", "1m", "--alert", f"file:{alert}"]) == 130
    assert len(calls) == 2 and json.loads(alert.read_text(encoding="utf-8"))["title"] == "mock:legacy: every request failed"


def test_wrong_numbers_are_refused_before_anything_runs(demo, capsys):
    for argv, words in (
        (["compare", "--to", "mock:legacy", "--min-pass-rate", "95"], "for 95% write 0.95"),
        (["compare", "--to", "mock:legacy", "--limit", "-17"], "must be 1 or more"),
        (["compare", "--to", "mock:legacy", "--show", "x"], "a number or 'all'"),
        (["compare", "--to", "mock:legacy", "--concurrency", "0"], "must be 1 or more"),
        (["fix", "--to", "mock:nextgen", "--rounds", "0"], "must be 1 or more"),
        (["watch", "--model", "mock:legacy", "--max-drop", "-1"], "from 0 to 1"),
        (["snapshot", "logs", "--max-cases", "-3"], "must be 1 or more"),
    ):
        with pytest.raises(SystemExit) as info:
            cli.main(argv)
        assert info.value.code == 2 and words in capsys.readouterr().err, argv
    assert not Path(".llmshift/reports").exists() or not list(Path(".llmshift/reports").glob("compare-*"))  # nothing ran
