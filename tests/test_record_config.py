"""llmshift.record (client wrapper) and configuration loading."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import warnings

import pytest

from llmshift.config import expand_env, load_config
from llmshift.record import Recorder, log, wrap
from llmshift.snapshot import build_snapshot


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self):
        def conv(v):
            if isinstance(v, Obj):
                return v.model_dump()
            if isinstance(v, list):
                return [conv(x) for x in v]
            return v

        return {k: conv(v) for k, v in self.__dict__.items()}


def completion(text):
    return Obj(id="c", object="chat.completion", model="gpt-4o-2024-08-06", choices=[Obj(index=0, message=Obj(role="assistant", content=text), finish_reason="stop")])


class FakeCompletions:
    def create(self, **kwargs):
        if kwargs.get("stream"):
            return iter([Obj(model="gpt-4o", choices=[Obj(delta=Obj(content="Hel"), finish_reason=None)]), Obj(model="gpt-4o", choices=[Obj(delta=Obj(content="lo"), finish_reason="stop")])])
        if kwargs.get("model") == "boom":
            raise RuntimeError("API down")
        return completion("echo " + kwargs["messages"][-1]["content"])


class FakeAsyncCompletions:
    async def create(self, **kwargs):
        return completion("async " + kwargs["messages"][-1]["content"])


class FakeClient:
    def __init__(self, completions):
        self.chat = Obj(completions=completions)
        self.api_key = "sk-" + "should-not-leak-abcdefghijklmnopqrstuvwxyz"


def test_wrap_records_calls_and_snapshot_reads_them(tmp_path):
    path = tmp_path / "llm.jsonl"
    client = wrap(FakeClient(FakeCompletions()), path=path)
    out = client.chat.completions.create(model="gpt-4o", temperature=0, messages=[{"role": "user", "content": "hi"}])
    assert out.choices[0].message.content == "echo hi"
    assert client.api_key.startswith("sk-")  # attributes pass through
    chunks = list(client.chat.completions.create(model="gpt-4o", stream=True, messages=[{"role": "user", "content": "stream me"}]))
    assert len(chunks) == 2
    try:
        client.chat.completions.create(model="boom", messages=[{"role": "user", "content": "x"}])
    except RuntimeError:
        pass
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and all(rec["llmshift_record"] == 1 for rec in lines)
    assert lines[0]["request"]["temperature"] == 0 and lines[0]["response"]["choices"][0]["message"]["content"] == "echo hi"
    assert lines[1]["response"]["choices"][0]["message"]["content"] == "Hello"
    assert lines[2]["error"].startswith("RuntimeError") and lines[2]["response"] is None
    res = build_snapshot([str(path)])
    outputs = sorted(c.baseline_output for c in res.cases)
    assert outputs == ["Hello", "echo hi"]  # the failed call is filtered out


def test_async_wrap(tmp_path):
    path = tmp_path / "a.jsonl"
    client = wrap(FakeClient(FakeAsyncCompletions()), path=path)
    result = asyncio.run(client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "q"}]))
    assert result.choices[0].message.content == "async q"
    assert json.loads(path.read_text(encoding="utf-8"))["response"]["choices"][0]["message"]["content"] == "async q"


def test_recorder_redacts_and_never_raises(tmp_path):
    rec = Recorder(tmp_path / "r.jsonl")
    rec.log({"model": "m", "messages": [{"role": "user", "content": "key sk-" + "proj-abcdefghijklmnopqrstuvwxyz0123"}]}, {"choices": []})
    assert "sk-proj" not in (tmp_path / "r.jsonl").read_text(encoding="utf-8")
    blocked = tmp_path / "dir_not_file"
    blocked.mkdir()
    bad = Recorder(blocked)  # writing to a directory fails
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bad.log({"model": "m"}, None)
        bad.log({"model": "m"}, None)
    assert len(caught) == 1  # warns once, never raises
    log({"model": "m", "messages": []}, {"choices": []}, path=tmp_path / "manual.jsonl")
    assert (tmp_path / "manual.jsonl").exists()


def test_sampling(tmp_path):
    rec = Recorder(tmp_path / "s.jsonl", sample_rate=0.0)
    client = wrap(FakeClient(FakeCompletions()), recorder=rec)
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert not (tmp_path / "s.jsonl").exists()


def test_config_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg["tests"] == ".llmshift/tests.jsonl" and cfg.path is None and cfg["scan"]["fail_on"] == "retired"
    monkeypatch.setenv("CAND", "openai:gpt-6-astra")
    (tmp_path / "llmshift.yaml").write_text("candidate: ${CAND}\njudge: ${JUDGE:-openai:gpt-5.6-terra}\nscan:\n  warn_days: 30\n", encoding="utf-8")
    sub = tmp_path / "pkg" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    cfg = load_config()
    assert cfg["candidate"] == "openai:gpt-6-astra" and cfg["judge"] == "openai:gpt-5.6-terra"
    assert cfg["scan"]["warn_days"] == 30 and cfg["scan"]["fail_on"] == "retired"  # deep-merged with defaults
    assert expand_env(["${CAND}", {"x": "$NOT_EXPANDED"}]) == ["openai:gpt-6-astra", {"x": "$NOT_EXPANDED"}]


def test_pyproject_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[tool.llmshift]\ncandidate = "anthropic:claude-sonnet-5-5"\nmin_pass_rate = 0.9\n', encoding="utf-8")
    cfg = load_config()
    try:
        import tomllib  # noqa: F401
    except ImportError:
        return
    assert cfg["candidate"] == "anthropic:claude-sonnet-5-5" and cfg["min_pass_rate"] == 0.9


def test_bad_config(tmp_path, monkeypatch):
    import pytest

    from llmshift.config import ConfigError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "llmshift.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config()
    with pytest.raises(ConfigError):
        load_config("missing.yaml")


class FakeMessageStream:
    """Shaped like anthropic's MessageStream: iterate events, then read the snapshot."""

    def __init__(self, text):
        self.text = text
        self.closed = False

    def __iter__(self):
        yield Obj(type="text", text=self.text)

    @property
    def current_message_snapshot(self):
        return Obj(id="m", type="message", role="assistant", model="claude-sonnet-5-5", content=[Obj(type="text", text=self.text)],
                   stop_reason="end_turn", usage=Obj(input_tokens=5, output_tokens=2))

    def close(self):
        self.closed = True


class FakeStreamManager:
    def __init__(self, stream=None, error=None):
        self.stream, self.error, self.exited = stream, error, False

    def __enter__(self):
        if self.error:
            raise self.error
        return self.stream

    def __exit__(self, *exc):
        self.exited = True
        self.stream.close()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *exc):
        self.__exit__(*exc)


class FakeMessages:
    def __init__(self):
        self.managers = []

    def stream(self, **kwargs):
        if kwargs["model"] == "boom":
            manager = FakeStreamManager(error=RuntimeError("overloaded"))
        else:
            manager = FakeStreamManager(FakeMessageStream("streamed " + kwargs["messages"][-1]["content"]))
        self.managers.append(manager)
        return manager

    def parse(self, **kwargs):
        return Obj(id="m", type="message", role="assistant", model="claude-sonnet-5-5", content=[Obj(type="text", text='{"category": "billing"}')], stop_reason="end_turn")


class SchemaModel:
    """Duck-typed like a pydantic model class used as a response format."""

    @classmethod
    def model_json_schema(cls):
        return {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}


def test_sdk_stream_helpers_are_recorded_when_the_block_ends(tmp_path):
    path = tmp_path / "helpers.jsonl"
    messages = FakeMessages()
    client = wrap(Obj(messages=messages), path=path)
    with client.messages.stream(model="claude-sonnet-5-5", max_tokens=50, messages=[{"role": "user", "content": "hi"}]) as stream:
        assert [e.text for e in stream] == ["streamed hi"]  # the app sees the SDK's own stream
    assert messages.managers[0].exited and messages.managers[0].stream.closed
    try:
        with client.messages.stream(model="boom", max_tokens=50, messages=[{"role": "user", "content": "x"}]):
            pass
    except RuntimeError:
        pass

    async def use_async():
        async with client.messages.stream(model="claude-sonnet-5-5", max_tokens=50, messages=[{"role": "user", "content": "async"}]) as stream:
            return [e.text for e in stream]

    assert asyncio.run(use_async()) == ["streamed async"]
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and [rec["api"] for rec in lines] == ["messages"] * 3
    assert lines[0]["response"]["content"][0]["text"] == "streamed hi" and lines[0]["latency_ms"] >= 0
    assert lines[1]["error"] == "RuntimeError: overloaded" and lines[1]["response"] is None
    assert lines[2]["response"]["content"][0]["text"] == "streamed async"
    res = build_snapshot([str(path)])
    assert sorted(c.baseline_output for c in res.cases) == ["streamed async", "streamed hi"]


def test_pydantic_response_formats_are_recorded_as_json_schema(tmp_path):
    path = tmp_path / "formats.jsonl"
    messages = FakeMessages()
    completions = FakeCompletions()
    completions.parse = lambda **kw: completion('{"category": "billing"}')
    client = wrap(Obj(chat=Obj(completions=completions), messages=messages), path=path)
    client.chat.completions.parse(model="gpt-4o", messages=[{"role": "user", "content": "classify"}], response_format=SchemaModel)
    client.messages.parse(model="claude-sonnet-5-5", max_tokens=50, messages=[{"role": "user", "content": "classify"}], output_format=SchemaModel)
    chat, claude = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rf = chat["request"]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"]["required"] == ["category"] and rf["json_schema"]["name"] == "SchemaModel"
    assert claude["request"]["output_config"]["format"]["schema"]["required"] == ["category"] and "output_format" not in claude["request"]
    res = build_snapshot([str(path)])
    formats = [c.input["params"].get("response_format") for c in res.cases]
    assert len(formats) == 2 and all(f and f["json_schema"]["schema"]["required"] == ["category"] for f in formats)  # replayable


class NotGiven:  # the class of openai.NOT_GIVEN (and of anthropic's, and of other generated SDKs)
    def __bool__(self):
        return False


def _chunk(text=None, finish=None, index=0, refusal=None):
    return Obj(model="gpt-4o", choices=[Obj(index=index, delta=Obj(content=text, refusal=refusal), finish_reason=finish)])


class StreamingCompletions:
    def __init__(self, chunks, error=None):
        self.chunks, self.error = chunks, error

    def create(self, **kwargs):
        def gen():
            yield from self.chunks
            if self.error is not None:
                raise self.error

        return gen()


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_mocked_client_and_sdk_sentinels(tmp_path):
    """Apps test with unittest.mock: recording a MagicMock answer must finish at once,
    and openai.NOT_GIVEN-style arguments are no value at all, never {}."""
    import time
    from unittest import mock

    path = tmp_path / "m.jsonl"
    client = wrap(mock.MagicMock(), path=path)
    start = time.perf_counter()
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], temperature=NotGiven(), top_p=NotGiven())
    assert time.perf_counter() - start < 5
    rec = _records(path)[0]
    assert rec["request"] == {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}


def test_streams_that_did_not_finish_are_not_expected_answers(tmp_path):
    path = tmp_path / "s.jsonl"
    whole = [_chunk("Hel"), _chunk("lo", finish="stop")]
    for chunks, error, stop_after in ((whole, None, None), (whole[:1], ConnectionError("reset by peer"), None), (whole[:1], None, None), (whole, None, 1)):
        client = wrap(FakeClient(StreamingCompletions(chunks, error)), path=path)
        stream = client.chat.completions.create(model="gpt-4o", stream=True, messages=[{"role": "user", "content": "q"}])
        got = []
        try:
            for chunk in stream:
                got.append(chunk)
                if stop_after and len(got) >= stop_after:
                    stream.close()  # the user pressed stop
                    break
        except ConnectionError:
            pass
    complete, dropped, cut, stopped = _records(path)
    assert "error" not in complete and complete["response"]["choices"][0]["message"]["content"] == "Hello"
    assert dropped["error"].startswith("ConnectionError") and "finish reason" in cut["error"] and "closed the stream" in stopped["error"]
    assert [c.baseline_output for c in build_snapshot([str(path)]).cases] == ["Hello"]  # only the whole answer becomes a test


def test_streamed_choices_refusals_and_later_edits(tmp_path):
    path = tmp_path / "n.jsonl"
    chunks = [_chunk("A", index=0), _chunk("B", index=1), _chunk("a", index=0, finish="stop"), _chunk(None, index=1, refusal="I can't."), _chunk("b", index=1, finish="stop")]
    client = wrap(FakeClient(StreamingCompletions(chunks)), path=path)
    messages = [{"role": "user", "content": "q"}]
    stream = client.chat.completions.create(model="gpt-4o", stream=True, n=2, messages=messages)
    messages.append({"role": "assistant", "content": "(typing...)"})  # the app edits its list while the stream runs
    list(stream)
    rec = _records(path)[0]
    assert [c["message"]["content"] for c in rec["response"]["choices"]] == ["Aa", "Bb"]
    assert rec["response"]["choices"][1]["message"]["refusal"] == "I can't."
    assert rec["request"]["messages"] == [{"role": "user", "content": "q"}]


def test_recording_failures_never_raise_even_as_errors(tmp_path):
    blocked = tmp_path / "a_directory"
    blocked.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # pytest's filterwarnings = error, or python -W error
        client = wrap(FakeClient(FakeCompletions()), path=blocked)
        assert client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}]).choices[0].message.content == "echo hi"


def test_a_broken_last_line_and_server_side_context(tmp_path):
    from llmshift import fileio

    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"llmshift_record": 1, "request": {"model": "gpt-4o", "messages": [{"role"')  # a writer died mid-line
    rec = {"llmshift_record": 1, "api": "chat", "request": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}, "response": completion("hello").model_dump()}
    fileio.append(path, (json.dumps(rec) + "\n").encode())
    follow_up = {"llmshift_record": 1, "api": "responses", "request": {"model": "gpt-5", "input": "And my name?", "previous_response_id": "resp_1"}, "response": {"object": "response", "output": []}}
    fileio.append(path, (json.dumps(follow_up) + "\n").encode())
    res = build_snapshot([str(path)])
    assert [c.baseline_output for c in res.cases] == ["hello"]  # the record after the broken line is whole
    assert any("context stored by OpenAI" in reason for reason in res.stats.skipped)


def test_the_wrapped_client_still_looks_like_the_client(tmp_path):
    import copy

    import pytest

    path = tmp_path / "c.jsonl"
    raw = FakeClient(FakeCompletions())
    client = wrap(wrap(raw, path=path), path=path)  # wrapping twice records once
    assert isinstance(client, FakeClient) and "chat" in dir(client)
    assert isinstance(copy.copy(client), FakeClient)
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert len(_records(path)) == 1
    with pytest.raises(ValueError):
        Recorder(tmp_path / "x.jsonl", sample_rate=float("nan"))


class FakeRawResponse:
    """What the SDKs' with_raw_response.create(...) returns: headers now, the answer on parse()."""

    def __init__(self, answer):
        self.headers = {"x-ratelimit-remaining-requests": "99"}
        self._answer = answer
        self.parses = 0

    def parse(self):
        self.parses += 1
        return self._answer


class FakeStreamingResponse:
    """What with_streaming_response.create(...) returns: a context manager around the response."""

    def __init__(self, answer, fail=False):
        self._answer, self._fail = answer, fail

    def __enter__(self):
        if self._fail:
            raise RuntimeError("400 bad request")
        return FakeRawResponse(self._answer)

    def __exit__(self, *exc):
        return False


class FakeRawCompletions:
    def create(self, **kwargs):
        return FakeRawResponse(completion("raw " + kwargs["messages"][-1]["content"]))


class FakeStreamingCompletions:
    def create(self, **kwargs):
        return FakeStreamingResponse(completion("streamed " + kwargs["messages"][-1]["content"]), fail=kwargs.get("model") == "boom")


def test_raw_and_streaming_response_calls_are_recorded(tmp_path):
    path = tmp_path / "raw.jsonl"
    completions = FakeCompletions()
    completions.with_raw_response = FakeRawCompletions()
    completions.with_streaming_response = FakeStreamingCompletions()
    client = wrap(FakeClient(completions), path=path)
    raw = client.chat.completions.with_raw_response.create(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
    assert raw.headers["x-ratelimit-remaining-requests"] == "99"  # the app still gets the HTTP response
    assert raw.parse().choices[0].message.content == "raw a" and raw.parse() is raw.parse()
    with client.chat.completions.with_streaming_response.create(model="gpt-4o", messages=[{"role": "user", "content": "b"}]) as resp:
        assert resp.parse().choices[0].message.content == "streamed b"
    try:
        with client.chat.completions.with_streaming_response.create(model="boom", messages=[{"role": "user", "content": "c"}]):
            pass
    except RuntimeError:
        pass
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [rec["response"]["choices"][0]["message"]["content"] if rec.get("response") else rec["error"] for rec in lines] == [
        "raw a", "streamed b", "RuntimeError: 400 bad request",
    ]  # parsed three times, recorded once


def _sdk_checked(fn):
    """What the OpenAI and Anthropic SDKs put around an async client's create (required_args):
    a plain function that returns its coroutine, made with functools.wraps."""

    @functools.wraps(fn)
    def checked(*args, **kwargs):
        return fn(*args, **kwargs)

    return checked


class FakeAsyncResponseManager:
    """What an async client's with_streaming_response.create(...) returns: the request is
    made on entering."""

    def __init__(self, request):
        self._request = request

    async def __aenter__(self):
        return await self._request

    async def __aexit__(self, *exc):
        return False


class SdkAsyncCompletions:
    """Shaped like openai's AsyncCompletions: create as above, and with_raw_response and
    with_streaming_response built from whatever create is when first asked for (a tracer's,
    if one patched it), asking it for the HTTP response."""

    @_sdk_checked
    async def create(self, *, model: str, messages: list, stream: bool = False, extra_headers: dict | None = None):
        answer = completion("async " + messages[-1]["content"])
        return FakeRawResponse(answer) if "X-Stainless-Raw-Response" in (extra_headers or {}) else answer

    @functools.cached_property
    def with_raw_response(self):
        create = self.create

        @functools.wraps(create)
        async def raw_create(**kwargs):
            return await create(**kwargs, extra_headers={"X-Stainless-Raw-Response": "raw"})

        return Obj(create=raw_create)

    @functools.cached_property
    def with_streaming_response(self):
        create = self.create

        @functools.wraps(create)
        def streaming_create(**kwargs):
            return FakeAsyncResponseManager(create(**kwargs, extra_headers={"X-Stainless-Raw-Response": "stream"}))

        return Obj(create=streaming_create)


def _instructor_is_async(func):
    """instructor.utils.is_async: instructor patches an async client's create when it is true."""
    is_coroutine = inspect.iscoroutinefunction(func)
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
        is_coroutine = is_coroutine or inspect.iscoroutinefunction(func)
    return is_coroutine


def test_wrapped_methods_look_like_the_sdk_methods(tmp_path):
    path = tmp_path / "a.jsonl"
    raw = FakeClient(SdkAsyncCompletions()).chat.completions.create
    client = wrap(FakeClient(SdkAsyncCompletions()), path=path)
    create = client.chat.completions.create
    assert _instructor_is_async(create) and inspect.iscoroutinefunction(create)  # LangSmith checks the latter
    assert create.__name__ == "create" and inspect.signature(create) == inspect.signature(raw)
    assert asyncio.run(create(model="gpt-4o", messages=[{"role": "user", "content": "q"}])).choices[0].message.content == "async q"
    streaming = client.chat.completions.with_streaming_response.create
    assert not inspect.iscoroutinefunction(streaming)  # it returns a context manager at once, as the SDK's does

    async def use():
        async with streaming(model="gpt-4o", messages=[{"role": "user", "content": "b"}]) as resp:
            return resp.parse()

    assert asyncio.run(use()).choices[0].message.content == "async b"
    assert [r["response"]["choices"][0]["message"]["content"] for r in _records(path)] == ["async q", "async b"]
    stream = wrap(Obj(messages=FakeMessages()), path=tmp_path / "b.jsonl").messages.stream
    assert stream.__name__ == "stream" and not inspect.iscoroutinefunction(stream)


class Observed:
    """A shadow that counts what it is shown."""

    def __init__(self):
        self.seen = []

    def observe(self, api, request, response):
        self.seen.append(api)


def _trace(resource, seen, keep_trail=True):
    """Patch resource.create as langsmith.wrappers.wrap_openai does: its own function around
    the create it found (async for an async one), which takes out its own argument."""
    original = resource.create

    def create(*args, **kwargs):
        seen.append(kwargs.pop("langsmith_extra", None))
        return original(*args, **kwargs)

    async def acreate(*args, **kwargs):
        seen.append(kwargs.pop("langsmith_extra", None))
        return await original(*args, **kwargs)

    is_async = inspect.iscoroutinefunction(original) or inspect.iscoroutinefunction(getattr(original, "__wrapped__", None))
    traced = acreate if is_async else create
    resource.create = functools.wraps(original)(traced) if keep_trail else traced


@pytest.mark.parametrize("keep_trail", [True, False])
def test_a_tracer_that_patches_create_after_wrap_records_each_call_once(tmp_path, keep_trail):
    """LangSmith's wrap_openai(client) sets client.chat.completions.create to a function around
    wrap()'s, and every call was recorded and shadowed twice."""
    path, shadow, seen = tmp_path / "sync.jsonl", Observed(), []
    client = wrap(FakeClient(FakeCompletions()), path=path, shadow=shadow)
    _trace(client.chat.completions, seen, keep_trail)
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], langsmith_extra={"run_id": "r1"})
    list(client.chat.completions.create(model="gpt-4o", stream=True, messages=[{"role": "user", "content": "s"}]))
    with pytest.raises(RuntimeError):
        client.chat.completions.create(model="boom", messages=[{"role": "user", "content": "x"}])
    recs = _records(path)
    assert len(recs) == 3 and len(seen) == 3 and len(shadow.seen) == 2  # a failed call is not shadowed
    if keep_trail:  # the recording create inside the tracer's records what was sent
        assert "langsmith_extra" not in recs[0]["request"]

    path, shadow, seen = tmp_path / "async.jsonl", Observed(), []
    client = wrap(FakeClient(SdkAsyncCompletions()), path=path, shadow=shadow)
    _trace(client.chat.completions, seen, keep_trail)

    async def calls():
        await asyncio.gather(*(client.chat.completions.create(model=f"gpt-4o-{i}", messages=[{"role": "user", "content": "q"}]) for i in range(3)))
        raw = await client.chat.completions.with_raw_response.create(model="gpt-4o", messages=[{"role": "user", "content": "raw"}])
        async with client.chat.completions.with_streaming_response.create(model="gpt-4o", messages=[{"role": "user", "content": "streamed"}]) as resp:
            return raw.parse(), resp.parse()

    parsed, streamed = asyncio.run(calls())
    assert parsed.choices[0].message.content == "async raw" and streamed.choices[0].message.content == "async streamed"
    recs = _records(path)
    assert len(recs) == 5 and len(seen) == 5 and len(shadow.seen) == 5
    assert sorted(r["request"]["model"] for r in recs) == ["gpt-4o", "gpt-4o", "gpt-4o-0", "gpt-4o-1", "gpt-4o-2"]
    assert [r["response"]["choices"][0]["message"]["content"] for r in recs][3:] == ["async raw", "async streamed"]  # the answers, not HTTP responses


def test_a_call_not_sampled_is_not_sampled_again_inside_a_tracer(tmp_path):
    rec = Recorder(tmp_path / "s.jsonl")
    answers = iter([False, True])
    rec.sampled = lambda: next(answers)
    client = wrap(FakeClient(FakeCompletions()), recorder=rec)
    _trace(client.chat.completions, [], keep_trail=False)
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert not (tmp_path / "s.jsonl").exists() and next(answers) is True  # asked once


def test_an_odd_callable_never_breaks_the_proxy(tmp_path):
    class Odd:
        def __call__(self, **kwargs):
            return completion("odd")

        @property
        def __wrapped__(self):
            raise RuntimeError("not today")

    path = tmp_path / "o.jsonl"
    client = wrap(FakeClient(Obj(create=Odd())), path=path)
    assert client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}]).choices[0].message.content == "odd"
    assert len(_records(path)) == 1
