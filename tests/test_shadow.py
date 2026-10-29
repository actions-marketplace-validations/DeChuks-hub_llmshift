"""Shadow mode: background copies of live calls, and the shadow report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import warnings

import pytest

import llmshift.shadow as shadow_mod
from llmshift import cli
from llmshift.providers import resolve
from llmshift.providers.mock import MockProvider
from llmshift.record import wrap
from llmshift.shadow import Shadow, build_cases, load_shadow_logs

SYSTEM = "You are a support ticket triage assistant. Classify the ticket and reply with JSON."
TICKETS = ["I was charged twice for order #4521", "The app crashes when I log in", "Where is my parcel? It is late", "I need a refund for invoice 88"]


class _Completions:
    def __init__(self, spec: str = "mock:legacy"):
        self.provider = resolve(spec)
        self.stream_chunks = None

    def create(self, **kw):
        resp = self.provider.complete({"messages": kw["messages"], "params": {}})
        if kw.get("stream"):
            text = resp.text
            return iter([{"model": "gpt-4o", "choices": [{"delta": {"content": text[:10]}}]}, {"choices": [{"delta": {"content": text[10:]}, "finish_reason": "stop"}]}])
        return {"model": "gpt-4o-2024-08-06", "choices": [{"index": 0, "message": {"role": "assistant", "content": resp.text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 12}}


class FakeClient:
    def __init__(self, spec: str = "mock:legacy"):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(spec)


def _ask(client, text, **kw):
    return client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}], **kw)


def _lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_shadow_copies_calls_without_changing_the_answer(tmp_path):
    shadow = Shadow("mock:nextgen", rate=1.0, path=tmp_path / "shadow.jsonl")
    client = wrap(FakeClient(), path=tmp_path / "rec.jsonl", shadow=shadow)
    answers = [_ask(client, t)["choices"][0]["message"]["content"] for t in TICKETS]
    assert all(a.startswith("{") for a in answers)  # the app still gets the real (legacy) answers
    assert shadow.flush(10)
    rows = _lines(tmp_path / "shadow.jsonl")
    assert len(rows) == 4 and shadow.stats["written"] == 4 and shadow.stats["errors"] == 0
    row = rows[0]
    assert row["llmshift_shadow"] == 1 and row["api"] == "chat"
    assert row["baseline"]["output"].startswith("{") and row["shadow"]["output"].startswith("```json")
    assert row["shadow"]["spec"] == "mock:nextgen" and row["request"]["messages"][0]["role"] == "system"
    assert len(_lines(tmp_path / "rec.jsonl")) == 4  # recording still happens


def test_slow_or_broken_shadow_models_never_affect_the_app(tmp_path):
    slow = MockProvider("slow", fn=lambda r: (time.sleep(0.6), "late")[1])
    shadow = Shadow(slow, rate=1.0, path=tmp_path / "s.jsonl", max_workers=1, max_pending=2)
    client = wrap(FakeClient(), record=False, shadow=shadow)
    start = time.perf_counter()
    for t in TICKETS:
        _ask(client, t)
    assert time.perf_counter() - start < 0.5  # never waits for the shadow model
    assert shadow.stats["dropped"] >= 1  # beyond max_pending, copies are dropped, not queued
    assert shadow.flush(10)
    assert not (tmp_path / "rec.jsonl").exists()

    def broken(request):
        raise RuntimeError("shadow model down")

    bad = Shadow(MockProvider("bad", fn=broken), rate=1.0, path=tmp_path / "bad.jsonl")
    client = wrap(FakeClient(), record=False, shadow=bad)
    assert _ask(client, TICKETS[0])["choices"][0]["message"]["content"].startswith("{")
    assert bad.flush(10)
    row = _lines(tmp_path / "bad.jsonl")[0]
    assert "shadow model down" in row["shadow"]["error"] and bad.stats["errors"] == 1


def test_unknown_shadow_model_is_logged_not_raised(tmp_path):
    shadow = Shadow("nosuchprovider:model-x", rate=1.0, path=tmp_path / "s.jsonl")
    client = wrap(FakeClient(), record=False, shadow=shadow)
    _ask(client, TICKETS[0])
    assert shadow.flush(10)
    assert "cannot tell which provider serves" in _lines(tmp_path / "s.jsonl")[0]["shadow"]["error"]


def test_sampling_rate_and_validation(tmp_path):
    none = Shadow("mock:nextgen", rate=0.0, path=tmp_path / "n.jsonl")
    client = wrap(FakeClient(), record=False, shadow=none)
    for t in TICKETS * 5:
        _ask(client, t)
    assert none.stats["seen"] == 20 and none.stats["written"] == 0
    some = Shadow("mock:nextgen", rate=0.5, path=tmp_path / "s.jsonl", seed=3)
    client = wrap(FakeClient(), record=False, shadow=some)
    for t in TICKETS * 10:
        _ask(client, t)
    assert some.flush(10)
    assert 8 <= some.stats["written"] <= 32
    with pytest.raises(ValueError):
        Shadow("mock:nextgen", rate=1.5)


def test_streams_and_shadow_given_as_a_spec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = wrap(FakeClient(), path=tmp_path / "rec.jsonl", shadow="mock:frugal", shadow_rate=1.0)
    chunks = list(_ask(client, TICKETS[1], stream=True))
    assert len(chunks) == 2
    shadow = client._shadow
    assert shadow.flush(10)
    row = _lines(tmp_path / "logs" / "llmshift-shadow.jsonl")[0]
    assert row["baseline"]["output"] == row["shadow"]["output"]  # frugal answers like legacy
    assert "stream" not in row["request"]["params"]


def test_secrets_are_redacted_in_the_shadow_log(tmp_path):
    shadow = Shadow("mock:legacy", rate=1.0, path=tmp_path / "s.jsonl")
    client = wrap(FakeClient(), record=False, shadow=shadow)
    _ask(client, "my key is sk-" + "proj-abcdefghijklmnopqrstuvwxyz123456 please help")
    assert shadow.flush(10)
    text = (tmp_path / "s.jsonl").read_text(encoding="utf-8")
    assert "sk-proj-abcdefghij" not in text and "<SECRET_1>" in text


def test_concurrent_shadowing_keeps_lines_whole(tmp_path):
    shadow = Shadow("mock:nextgen", rate=1.0, path=tmp_path / "s.jsonl", max_workers=4, max_pending=1000)
    client = wrap(FakeClient(), record=False, shadow=shadow)
    threads = [threading.Thread(target=lambda i=i: [_ask(client, f"{TICKETS[i % 4]} ({i}-{j})") for j in range(10)]) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert shadow.flush(20)
    assert len(_lines(tmp_path / "s.jsonl")) == 60


def _shadow_log(tmp_path, spec):
    shadow = Shadow(spec, rate=1.0, path=tmp_path / f"{spec.replace(':', '-')}.jsonl")
    client = wrap(FakeClient(), record=False, shadow=shadow)
    for t in TICKETS:
        _ask(client, t)
    assert shadow.flush(10)
    return shadow.path


def test_shadow_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    bad = _shadow_log(tmp_path, "mock:nextgen")
    good = _shadow_log(tmp_path, "mock:frugal")
    records = load_shadow_logs([str(bad)])
    cases, answers, spec = build_cases(records)
    assert len(cases) == 4 and spec == "mock:nextgen" and sorted(answers) == sorted(c.id for c in cases)
    assert cli.main(["shadow", "report", str(good)]) == 0
    out = capsys.readouterr().out
    assert "LLMShift shadow" in out and "4 unchanged" in out and "4 shadowed request(s)" in out
    assert cli.main(["shadow", "report", str(bad), "--format", "json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "shadow" and data["baseline"] == "live traffic" and data["summary"]["failed"] == 4
    assert cli.main(["shadow", "report", str(tmp_path / "missing.jsonl")]) == 2
    assert cli.main(["shadow", "report", str(bad), "--html"]) == 1
    assert list((tmp_path / ".llmshift" / "reports").glob("shadow-*.html"))


HANGING_APP = """
import sys, time
import llmshift.shadow as shadow_mod
from llmshift.providers.mock import MockProvider
shadow_mod.EXIT_GRACE = 0.5
shadow = shadow_mod.Shadow(MockProvider("hang", fn=lambda r: (time.sleep(2), "late")[1]), rate=1.0, path=sys.argv[1], max_workers=1)
answer = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}
for i in range(10):
    shadow.observe("chat", {"model": "gpt-4o", "messages": [{"role": "user", "content": f"q{i}"}]}, answer)
"""


def test_a_hanging_shadow_model_does_not_hold_up_the_app_exit(tmp_path):
    # 10 queued copies of 2 s each: the exit waits for the grace period, not for the copies.
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", HANGING_APP, str(tmp_path / "s.jsonl")], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert time.perf_counter() - start < 12


def test_idle_workers_stop_and_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow_mod, "IDLE_SECONDS", 0.1)
    shadow = Shadow("mock:nextgen", rate=1.0, path=tmp_path / "s.jsonl", max_workers=2)
    client = wrap(FakeClient(), record=False, shadow=shadow)
    _ask(client, TICKETS[0])
    assert shadow.flush(10)
    deadline = time.monotonic() + 10
    while shadow._threads and time.monotonic() < deadline:
        time.sleep(0.05)
    assert shadow._threads == []  # nothing to do: no threads left behind
    _ask(client, TICKETS[1])
    assert shadow.flush(10) and shadow.stats["written"] == 2


def test_thread_limits_and_exits_inside_the_shadow_model(tmp_path, monkeypatch):
    shadow = Shadow("mock:nextgen", rate=1.0, path=tmp_path / "s.jsonl")
    with monkeypatch.context() as m:
        m.setattr(threading.Thread, "start", lambda self: (_ for _ in ()).throw(RuntimeError("can't start new thread")))
        with pytest.warns(RuntimeWarning, match="can't start new thread"):
            shadow.observe("chat", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}, {"choices": [{"message": {"content": "x"}}]})
    assert shadow.stats["errors"] == 1 and shadow._pending == 0 and shadow.flush(0.1)

    def exits(request):
        raise SystemExit(3)

    quits = Shadow(MockProvider("quits", fn=exits), rate=1.0, path=tmp_path / "q.jsonl", max_workers=1)
    client = wrap(FakeClient(), record=False, shadow=quits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in TICKETS[:2]:
            _ask(client, t)
        assert quits.flush(10)
    assert quits.stats["errors"] == 2 and len(quits._threads) == 1  # the worker survived both


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_shadowing_and_recording_work_in_a_forked_child(tmp_path):
    from llmshift.record import _path_lock

    shadow = Shadow(MockProvider("slow", fn=lambda r: (time.sleep(0.5), "late")[1]), rate=1.0, path=tmp_path / "s.jsonl", max_workers=1)
    client = wrap(FakeClient(), path=tmp_path / "rec.jsonl", shadow=shadow)
    _ask(client, TICKETS[0])  # a copy is in flight in the parent
    held = threading.Event()
    release = threading.Event()

    def hold_lock():  # another thread is writing to the log at the moment of the fork
        with _path_lock(tmp_path / "rec.jsonl"):
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    held.wait(10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # "use of fork() may lead to deadlocks": that is the point
        pid = os.fork()
    if pid == 0:  # child
        code = 1
        try:
            _ask(client, TICKETS[1])
            code = 0 if shadow.flush(10) and shadow.stats["written"] >= 1 else 2
        finally:
            os._exit(code)
    release.set()
    holder.join()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        pytest.fail("the forked child hung")
    assert os.waitstatus_to_exitcode(status) == 0
    assert shadow.flush(10)


def test_repeated_requests_keep_their_own_answers(tmp_path, monkeypatch, capsys):
    """The same question asked 200 times has 200 real and 200 shadow answers: each shadow
    answer is checked against its own real one, whatever order the threads run in."""
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "dup.jsonl"
    request = {"messages": [{"role": "user", "content": "Name a random fruit."}], "params": {"temperature": 1}}
    with open(log, "w", encoding="utf-8") as fh:
        for i in range(200):
            answer = f"fruit number {i}"
            fh.write(json.dumps({"llmshift_shadow": 1, "ts": "2026-10-22T10:00:00Z", "request": request, "baseline": {"output": answer}, "shadow": {"spec": "mock:same", "output": answer}}) + "\n")
    assert cli.main(["shadow", "report", str(log), "--concurrency", "16", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["passed"] == 200 and data["summary"]["failed"] == 0
