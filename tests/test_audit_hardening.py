"""Regression tests for the second audit: what outside text (provider errors, model
answers, scanned files) can do to a terminal or a PR comment, and URLs with secrets."""

from __future__ import annotations

import re

import pytest

from llmshift import console
from llmshift.github import fit_markdown
from llmshift.providers import http
from llmshift.providers.base import ProviderError
from llmshift.report import _md
from llmshift.textutil import md_fence


def test_terminal_output_obeys_only_its_own_colours(monkeypatch, capsys):
    hostile = "\x1b]0;pwned\x07 \x1b[2J\x1b[8mhidden\x1b[0m ok\rover \u202eevil\u202c\ttab \x9b31m"
    console.set_color(True)
    try:
        coloured = console.safe(console.style("fine", "green") + " " + hostile)
        assert coloured.startswith("\x1b[32mfine\x1b[0m ")  # what style() adds stays
        for raw in ("\x1b]", "\x07", "\x1b[2J", "\x1b[8m", "\r", "\u202e", "\x9b"):
            assert raw not in coloured
        assert "\\u001b]0;pwned\\u0007" in coloured and "\\u202e" in coloured and "\ttab" in coloured
        assert "ok\nover" in coloured  # a carriage return cannot overwrite the line
    finally:
        console.set_color(False)
    plain = console.safe(console.style("fine", "green") + " \x1b[31mred\x1b[0m")
    assert "\x1b" not in plain  # without colour, no escape at all reaches the output
    console.out("a\x1b[2Jb")
    assert capsys.readouterr().out == "a\\u001b[2Jb\n"
    assert console.safe("line one\r\nline two") == "line one\nline two"
    # JSON output stays JSON: json.dumps leaves C1 controls and bidi marks raw, and the
    # escapes shown for them are JSON escapes too.
    import json

    data = {"answer": "next line\x85, \x9b31m and \u202eevil"}
    assert json.loads(console.safe(json.dumps(data, ensure_ascii=False))) == data


def test_markdown_cells_quote_mentions_without_notifying_anyone():
    cell = _md("ping @everyone and @octo-org/team, mail help@example.com | <img src=x>")
    assert "@\u200beveryone" in cell and "@\u200bocto-org/team" in cell
    assert "help@example.com" in cell  # an address is not a mention
    assert "\\|" in cell and "&lt;img" in cell


def _all_escaped(text: str) -> bool:
    """Every [ ] ( ) ! has an odd run of backslashes before it, so none starts a link or image."""
    for m in re.finditer(r"[\[\]()!]", text):
        before = text[: m.start()]
        if (len(before) - len(before.rstrip("\\"))) % 2 == 0:
            return False
    return True


def test_model_text_in_markdown_carries_no_links_images_or_autolinks():
    from llmshift.textutil import ZWSP, md_inert

    # An end user's message lands in the logs, becomes a test, is echoed by the model and is
    # quoted in a PR comment: it must not load a tracking pixel or offer a disguised link.
    evil = "![p](https://evil.example/p.png) [click here](https://evil.example) www.evil.example \\[x\\](https://e) http://a.b"
    cell = _md(evil)
    assert _all_escaped(cell), cell
    assert "https://" not in cell and "http://" not in cell and "www." not in cell and f"https{ZWSP}://evil.example" in cell
    assert md_inert("line one\n[a](b)!") == "line one\n\\[a\\]\\(b\\)\\!"  # chat alerts keep their lines


def test_reports_and_alerts_quote_model_text_without_live_links(tmp_path, server, monkeypatch):
    from llmshift.alerts import AlertMessage, deliver
    from llmshift.fix import FixResult, GroupFix, fix_markdown

    added = "Always end with ![t](https://evil.example/t.png) and [docs](https://evil.example)"
    md = fix_markdown(FixResult("m", [GroupFix("support", "p", "p " + added, added=[added], remaining=[added], notes=[added])], {"pass_rate": 0.5}))
    outside, fenced = [], False
    for line in md.splitlines():
        if line.startswith("```"):
            fenced = not fenced  # inside a code block nothing renders
        elif not fenced and "evil" in line:
            outside.append(line)
    assert len(outside) == 3 and all(_all_escaped(line) for line in outside), outside
    received = []
    server.route("POST", "/hook", lambda r: (received.append(r["json"]) or (200, "ok")))
    msg = AlertMessage(title="Drift", severity="warning", model="m", lines=[added], run_url="https://github.com/acme/app/actions/runs/1")
    assert all(err is None for _, err in deliver([f"slack:{server.url}/hook", {"type": "discord", "url": server.url + "/hook"}, {"type": "teams", "url": server.url + "/hook"}], msg))
    slack, discord, teams = received
    assert slack["unfurl_links"] is False and slack["unfurl_media"] is False
    assert discord["flags"] == 4 and "\\[docs\\]" in discord["content"] and discord["content"].endswith("Details: https://github.com/acme/app/actions/runs/1")
    assert "\\!\\[t\\]" in teams["attachments"][0]["content"]["body"][1]["text"]
    issues = []
    server.route("GET", "/repos/acme/app/issues", lambda r: [])
    server.route("POST", "/repos/acme/app/issues", lambda r: (issues.append(r["json"]) or (201, {"number": 1})))
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/app")
    monkeypatch.setenv("GITHUB_API_URL", server.url)
    assert deliver(["github"], msg) == [("github", None)]
    line = next(x for x in issues[0]["body"].splitlines() if "evil" in x)
    assert _all_escaped(line[2:]) and "https://evil" not in line


def test_reports_show_trojan_source_controls_as_escapes():
    import json

    from llmshift import report
    from llmshift.compare import run_compare
    from llmshift.providers.mock import MockProvider
    from llmshift.testset import TestCase
    from llmshift.textutil import report_json

    rlo, lri = chr(0x202E), chr(0x2066)  # right-to-left override, left-to-right isolate
    case = TestCase(id="c" + rlo + "1", input={"messages": [{"role": "user", "content": "Pay invoice " + rlo + "fdp.exe"}], "params": {}},
                    baseline={"output": "Approved " + lri + "x"}, checks=[{"type": "equals", "value": "Approved"}])
    result = run_compare([case], MockProvider("mock", fn=lambda r: "Paid " + rlo + "gnp.exe"), show_progress=False)
    outputs = {
        "markdown": report.compare_markdown(result), "html": report.compare_html(result), "junit": report.compare_junit(result),
        "json": report_json(result.to_dict()), "md_cell": _md("x" + rlo + "y"), "fence": md_fence("a" + rlo + "b"),
    }
    for name, text in outputs.items():
        assert rlo not in text and lri not in text, name
        assert "\\u202e" in text, name
    assert json.loads(outputs["json"])["cases"][0]["candidate_output"] == "Paid " + rlo + "gnp.exe"  # JSON reads back unchanged
    assert chr(0x200F) in _md("rtl mark " + chr(0x200F))  # directional marks reorder nothing and stay


def test_code_blocks_cannot_be_closed_by_their_content():
    block = md_fence(" ```\n@everyone\n````", "diff")
    fence = block.splitlines()[0][: -len("diff")]
    assert fence == "`````" and block.endswith("\n`````")
    # Cut short, the comment closes the long fence it left open, and the <details> too.
    body = "<details>\n" + md_fence(" ```\n" + "x\n" * 5000, "diff") + "\n</details>\n"
    cut = fit_markdown(body, limit=3000)
    assert cut.count("````\n") == 1 and "</details>" in cut
    tail = cut.split("````\n")[1]
    assert "x\n" not in tail


def test_urls_with_a_password_are_refused_without_showing_it():
    with pytest.raises(ProviderError) as info:
        http.request("GET", "http://user:s3cret@127.0.0.1:9/v1/models", max_retries=0, timeout=2)
    assert info.value.code == "invalid_url" and "s3cret" not in str(info.value) and "127.0.0.1:9" in str(info.value)
    assert http.display_url("https://bob:hunter2@gw.example.com:8443/v1?key=abc") == "https://gw.example.com:8443/v1"


@pytest.mark.parametrize("text, kept", [
    ("glpat-AbCdEfGhIjKlMnOpQrSt12", ""),
    ("npm_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8", ""),
    ("postgres://admin:hunter2pw@db:5432/app", "postgres://admin:"),
    ("redis://:s3cr3tpass@cache:6379/0", "redis://:"),
    ('{"api-key": "0123456789abcdef0123456789abcdef"}', '{"api-key": "'),
    ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "aws_secret_access_key = "),
    ("aws secret access key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "aws secret access key "),
    ("--aws-secret-access-key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "--aws-secret-access-key "),
    ("aws_secret_access_key => wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "aws_secret_access_key => "),
    ("password=hunter2", "password="),
    ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "Authorization: Basic "),
])
def test_more_secret_shapes_are_redacted(text, kept):
    from llmshift.redact import Redactor

    out = Redactor().text(text)
    assert out.startswith(kept + "<SECRET_1>"), out


@pytest.mark.parametrize("text", [
    "max_tokens=1000, prompt_tokens: 512, token_count=7",
    "token = tokenizer.encode(text)",
    "password = input('Enter password')",
    "Enter your password: carefully, then press OK",
    "api_key = os.environ['OPENAI_API_KEY']",
    "secret=True, apikey=12345678",
    'url = f"https://{user}:{password}@{host}"',
    "Use https://example.com:8080/path, a basic understanding of keys",
])
def test_code_and_prose_about_secrets_is_left_alone(text):
    from llmshift.redact import Redactor

    assert Redactor().text(text) == text


def test_aws_secret_keys_are_known_by_what_surrounds_them():
    from llmshift.redact import Redactor

    key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"  # no prefix: just 40 characters
    assert Redactor().text("my keys are " + "AKIA" + f"IOSFODNN7EXAMPLE and {key}") == "my keys are <SECRET_2> and <SECRET_1>"
    assert key not in Redactor().text(f"the secret access key is {key}.")
    hash_line = "secret access key rotated in commit 3f786850e387550fdab836ed7e6dc881de23001b"
    assert Redactor().text(hash_line) == hash_line


def test_secret_settings_in_json_are_redacted_by_their_name():
    from llmshift.redact import Redactor

    data = {"headers": {"api-key": "0123456789abcdef0123456789abcdef"}, "args": {"password": "hunter2", "token": "none", "max_tokens": "abc123x"}}
    out = Redactor().obj(data)
    assert out["headers"]["api-key"].startswith("<SECRET_") and out["args"]["password"].startswith("<SECRET_")
    assert out["args"]["token"] == "none" and out["args"]["max_tokens"] == "abc123x"
    assert Redactor(secrets=False).obj(data) == data


def test_compressed_answers_are_unpacked_and_cut_answers_retried(server, monkeypatch):
    import gzip
    import json
    import zlib

    body = json.dumps({"ok": "café"}).encode("utf-8")
    raw_deflate = zlib.compressobj(wbits=-15)
    server.route("GET", "/gzip", lambda req: (200, gzip.compress(body), {"Content-Encoding": "gzip"}))
    server.route("GET", "/deflate", lambda req: (200, raw_deflate.compress(body) + raw_deflate.flush(), {"Content-Encoding": "deflate"}))
    server.route("GET", "/br", lambda req: (200, b"\x0b\x02\x80", {"Content-Encoding": "br"}))
    assert http.request("GET", server.url + "/gzip", max_retries=0) == {"ok": "café"}
    assert server.requests[-1]["headers"]["accept-encoding"] == "identity"
    assert http.request("GET", server.url + "/deflate", max_retries=0) == {"ok": "café"}
    with pytest.raises(ProviderError, match="compressed with br"):
        http.request("GET", server.url + "/br", max_retries=0)
    # A body cut short is fetched again; a web page where the API should be is not.
    answers = iter([b'{"choices": [{"mess', body])
    server.route("GET", "/cut", lambda req: (200, next(answers)))
    assert http.request("GET", server.url + "/cut", sleep=lambda s: None) == {"ok": "café"}
    before = len(server.requests)
    server.route("GET", "/html", lambda req: (200, b"<!doctype html><title>Login</title>"))
    with pytest.raises(ProviderError, match="invalid JSON"):
        http.request("GET", server.url + "/html", sleep=lambda s: None)
    assert len(server.requests) == before + 1
    # A compressed bomb is unpacked no further than the size limit.
    monkeypatch.setattr(http, "MAX_RESPONSE_BYTES", 1024 * 1024)
    server.route("GET", "/bomb", lambda req: (200, gzip.compress(b"0" * (20 * 1024 * 1024)), {"Content-Encoding": "gzip"}))
    with pytest.raises(ProviderError, match="larger than"):
        http.request("GET", server.url + "/bomb", max_retries=0)


def test_answer_text_follows_the_declared_charset_only_when_it_is_not_utf8():
    from email.message import Message

    def headers(ctype):
        m = Message()
        m["Content-Type"] = ctype
        return m

    latin = headers("application/json; charset=ISO-8859-1")
    assert http._text('{"a": "café"}'.encode("latin-1"), latin) == '{"a": "café"}'
    assert http._text('{"a": "café"}'.encode("utf-8"), latin) == '{"a": "café"}'  # the usual mislabel
    assert http._text('{"a": "é"}'.encode("utf-16"), headers("application/json")) == '{"a": "é"}'
    assert http._text('{"a": 1}'.encode("utf-8-sig"), None) == '{"a": 1}'
    assert http._text(b'{"a": "\xff"}', headers("application/json; charset=nonsense")) == '{"a": "\ufffd"}'


def test_provider_errors_say_what_failed_before_the_message():
    assert str(ProviderError("bad key", status=401, code="invalid_api_key")) == "[HTTP 401, invalid_api_key] bad key"
    assert str(ProviderError("slow down", status=403, code="403")) == "[HTTP 403] slow down"
    assert str(ProviderError("no answer", code="timeout")) == "[timeout] no answer"
    assert str(ProviderError("plain")) == "plain"


def test_source_files_hold_no_invisible_characters():
    """Zero-width and bidi characters in source are invisible in review (and GitHub warns
    about them): they are written as escapes."""
    import pathlib
    import unicodedata

    root = pathlib.Path(__file__).resolve().parents[1]
    found = []
    for path in [*root.joinpath("src", "llmshift").rglob("*.py"), *root.joinpath("tests").rglob("*.py")]:
        if "_vendor" in path.parts:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(unicodedata.category(c) in ("Cf", "Zl", "Zp") or c == "\N{NO-BREAK SPACE}" for c in line):
                found.append(f"{path.name}:{n}")
    assert not found


@pytest.mark.parametrize("spec, says", [
    ("opneai:gpt-4o", "did you mean openai:gpt-4o?"),
    ("antropic:claude-x", "did you mean anthropic:claude-x?"),
    ("command:x", "type: command"),
    ("llama3.1:8b", "e.g. openai:llama3.1:8b or ollama:llama3.1:8b"),
])
def test_unknown_providers_get_a_useful_hint(spec, says):
    from llmshift.providers import resolve

    with pytest.raises(ProviderError) as info:
        resolve(spec)
    assert info.value.code == "unknown_provider" and says in str(info.value)


def test_embedding_and_cohere_ids_go_to_the_right_provider():
    from llmshift.providers import infer_provider

    assert infer_provider("text-embedding-004") == infer_provider("text-embedding-005") == "gemini"
    assert infer_provider("text-embedding-3-small") == infer_provider("text-embedding-ada-002") == "openai"
    assert infer_provider("command-r-plus") == infer_provider("command") == "cohere"
    assert infer_provider("commander") is None


@pytest.mark.parametrize("settings, says", [
    ({"timeout": "abc"}, "timeout must be a number"),
    ({"timeout": 0}, "timeout must be a number of at least 0.1"),
    ({"max_retries": -1}, "max_retries must be a whole number"),
    ({"max_retries": 2.5}, "max_retries must be a whole number"),
    ({"headers": ["X-A: 1"]}, "headers must be a mapping"),
    ({"headers": {"X-A": {"b": 1}}}, "headers.X-A must be a single value"),
    ({"extra_body": "x"}, "extra_body must be a mapping"),
    ({"base_url": "localhost:8000/v1"}, "must start with https:// or http://"),
])
def test_bad_provider_settings_are_reported_when_read(settings, says):
    from llmshift.providers import resolve

    with pytest.raises(ProviderError) as info:
        resolve("openai:gpt-4o", {"providers": {"openai": settings}})
    assert info.value.fatal and says in str(info.value)


def test_azure_needs_an_endpoint_and_accepts_its_host_name(monkeypatch):
    from llmshift.providers import resolve

    with pytest.raises(ProviderError) as info:
        resolve("azure:dep")
    assert info.value.code == "missing_endpoint" and info.value.fatal
    p = resolve("azure:dep", {"providers": {"azure": {"endpoint": "my-res.openai.azure.com", "api_key": "k"}}})
    assert p.base_url == "https://my-res.openai.azure.com/openai/deployments/dep"
    assert resolve("openai:gpt-4o", {"providers": {"openai": {"headers": {"X-A": 1, "X-B": True}}}}).extra_headers == {"X-A": "1", "X-B": "true"}


def test_python_providers_time_out_report_their_own_errors_and_notice_edited_helpers(tmp_path, monkeypatch):
    import time

    from llmshift.providers import resolve

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "prompts.py").write_text("PROMPT = 'one'\n", encoding="utf-8")
    (tmp_path / "pkg" / "llm.py").write_text(
        "import time\nfrom pkg.prompts import PROMPT\n\n"
        "def answer(req):\n    return PROMPT\n\n"
        "def slow(req):\n    time.sleep(20)\n\n"
        "def broken(req):\n    return {}['x']\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        p = resolve("python:pkg.llm:answer")
        assert p.complete({"messages": []}).text == "one"
        before = p.identity()
        (tmp_path / "pkg" / "prompts.py").write_text("PROMPT = 'two'\n", encoding="utf-8")
        assert resolve("python:pkg.llm:answer").identity() != before  # a cached answer is not reused
        start = time.monotonic()
        with pytest.raises(ProviderError) as info:
            resolve("python:pkg.llm:slow", {"timeout": 0.5}).complete({"messages": []})
        assert info.value.code == "timeout" and time.monotonic() - start < 10
        with pytest.raises(KeyError):  # your bug, reported as yours
            resolve("python:pkg.llm:broken").complete({"messages": []})
    finally:
        for name in ("pkg", "pkg.llm", "pkg.prompts"):
            __import__("sys").modules.pop(name, None)


def test_wrapping_a_client_per_request_shares_one_shadow(tmp_path):
    from llmshift.record import wrap

    client = type("OpenAI", (), {"__module__": "openai"})()
    path = tmp_path / "shadow.jsonl"
    first = wrap(client, record=False, shadow="mock:echo", shadow_rate=1.0, shadow_path=path)._shadow
    again = [wrap(client, record=False, shadow="mock:echo", shadow_rate=1.0, shadow_path=path)._shadow for _ in range(50)]
    assert all(s is first for s in again)
    assert wrap(client, record=False, shadow="mock:echo", shadow_rate=0.5, shadow_path=path)._shadow is not first
    home = wrap(client, record=False, shadow="mock:echo", shadow_path="~/llmshift-shadow-test.jsonl")._shadow
    assert "~" not in str(home.path)
    first.close(0)
    assert wrap(client, record=False, shadow="mock:echo", shadow_rate=1.0, shadow_path=path)._shadow is not first  # a closed one is replaced


HOSTILE_EXPORTS = [
    {"object": "eval", "name": {"a": 1}, "data_source_config": "x", "testing_criteria": 5},
    {"eval": {"object": "eval", "id": "e1", "name": "x", "testing_criteria": []}, "runs": "x", "output_items": 5, "items": {"a": 1}},
    {"eval": {"object": "eval", "id": "e1", "name": "x", "testing_criteria": [{"type": "string_check", "operation": 5, "reference": "a"}]},
     "runs": [{"object": "eval.run", "id": 5, "status": [], "data_source": "x", "created_at": "soon"}, 1, None],
     "output_items": [{"run_id": "r1", "sample": "x", "datasource_item": 5, "results": "y", "datasource_item_id": "b"},
                      {"run_id": "r1", "sample": {"input": 5}, "datasource_item": {"q": 1}, "results": [1], "datasource_item_id": 3}]},
    {"object": "eval", "name": "x", "testing_criteria": [{"type": "text_similarity", "pass_threshold": "high"}, {"type": "multi", "graders": [1]}]},
]


@pytest.mark.parametrize("doc", HOSTILE_EXPORTS, ids=["types", "wrapper", "runs-and-items", "graders"])
def test_hostile_eval_exports_import_or_explain(tmp_path, doc):
    import json

    from llmshift.importers import openai_evals as oe

    path = tmp_path / "eval.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    oe.import_from_export(str(path), max_items=5)  # no crash; odd fields count as missing


@pytest.mark.parametrize("raw, says", [
    (b"[" * 100000 + b"]" * 100000, "nested too deeply"),
    (b'{"object": "eval", "name": "\xff"}', "not UTF-8 text"),
    (b'{"object": "eval", "name": "x"}\n{not json\n', "line 2: not valid JSON"),
], ids=["deep", "not-utf8", "bad-line"])
def test_unreadable_eval_exports_name_the_file(tmp_path, raw, says):
    from llmshift.importers import openai_evals as oe

    path = tmp_path / "eval.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError) as info:
        oe.import_from_export(str(path))
    assert str(path) in str(info.value) and says in str(info.value)


def test_utf16_exports_and_samples_are_read(tmp_path):
    import json

    from llmshift.importers import openai_evals as oe

    ev = {"object": "eval", "id": "e1", "name": "caps", "testing_criteria": [{"type": "string_check", "name": "x", "input": "{{sample.output_text}}", "reference": "{{item.a}}", "operation": "eq"}]}
    path = tmp_path / "eval.json"
    path.write_bytes(json.dumps({"eval": ev, "items": [{"item": {"input": "q", "a": "b"}}]}).encode("utf-16"))
    assert len(oe.import_from_export(str(path)).cases) == 1


def test_hostile_registries_import_or_explain(tmp_path):
    import gzip
    import json

    from llmshift.importers import openai_evals as oe

    (tmp_path / "evals").mkdir()
    (tmp_path / "data").mkdir()
    sample = {"input": [{"role": "user", "content": "Capital of France?"}], "ideal": "Paris"}
    weird = [sample, {"input": 5, "ideal": None}, {"input": [1, None, {"role": 5, "content": [1]}], "ideal": ["Paris", 5]}, "x", 5]
    (tmp_path / "data" / "s.jsonl").write_text("\n".join(json.dumps(s) for s in weird) + "\n" + "[" * 5000 + "]" * 5000, encoding="utf-8")
    (tmp_path / "data" / "bad.jsonl.gz").write_bytes(gzip.compress(b'{"input": "x"}')[:12])
    spec = "  class: evals.elsuite.basic.match:Match\n"
    (tmp_path / "evals" / "x.yaml").write_text(
        "good:\n  id: good.v0\ngood.v0:\n" + spec + "  args:\n    samples_jsonl: s.jsonl\n    few_shot_jsonl: s.jsonl\n    num_few_shot: lots\n"
        "strargs:\n  id: strargs.v0\nstrargs.v0:\n" + spec + "  args: hello\n"
        "intpath:\n  id: intpath.v0\nintpath.v0:\n" + spec + "  args:\n    samples_jsonl: 5\n"
        "gz:\n  id: gz.v0\ngz.v0:\n" + spec + "  args:\n    samples_jsonl: bad.jsonl.gz\n"
        "outside:\n  id: outside.v0\noutside.v0:\n" + spec + "  args:\n    samples_jsonl: " + str(tmp_path.parent / "elsewhere.jsonl") + "\n",
        encoding="utf-8",
    )
    (tmp_path.parent / "elsewhere.jsonl").write_text(json.dumps(sample), encoding="utf-8")
    result = oe.import_oss_registry(str(tmp_path))
    assert {c.metadata["eval"] for c in result.cases} == {"good"} and len(result.cases) == 2
    said = "\n".join(result.warnings)
    for words in ("num_few_shot must be a number", "strargs: no samples_jsonl", "samples file name must be text", "gz: ", "nested too deeply", "outside the registry"):
        assert words in said, words


@pytest.mark.parametrize("text, expected", [
    ("2026-03-01T12:00:00+00", "2026-03-01T12:00:00+00:00"),
    ("2026-03-01T12:00:00,123Z", "2026-03-01T12:00:00.123000+00:00"),
    ("2026-03-01T12:00:00.5-08", "2026-03-01T12:00:00.500000-08:00"),
    ("20260301T120000Z", "2026-03-01T12:00:00+00:00"),
    ("20260301T120000+0100", "2026-03-01T12:00:00+01:00"),
    ("2026-W09-7", "2026-03-01T00:00:00+00:00"),
    ("Sun, 01 Mar 2026 12:00:00 GMT", "2026-03-01T12:00:00+00:00"),
    ("2026-02-30T00:00:00Z", None),
    ("garbage Text", None),
])
def test_log_times_read_the_same_on_every_python(text, expected):
    from llmshift.logparse import parse_time

    got = parse_time(text)
    assert (got.isoformat() if got else None) == expected


def test_registry_dates_read_the_same_on_every_python(monkeypatch):
    import datetime as dt

    from llmshift.registry import _date, today

    assert _date("2026-03-01") == _date("20260301") == _date("2026-03-01T12:00:00Z") == dt.date(2026, 3, 1)
    assert _date("2026-02-30") is None and _date("202603011") is None and _date(None) is None
    monkeypatch.setenv("LLMSHIFT_TODAY", "20261022")
    assert today() == dt.date(2026, 10, 22)
    monkeypatch.setenv("LLMSHIFT_TODAY", "soon")
    assert today() == dt.date.today()


def test_log_sampling_keeps_a_fair_share_of_late_templates():
    import random
    from collections import Counter

    from llmshift.logparse import Interaction, _Sample

    def call(template, i):
        it = Interaction(messages=[{"role": "system", "content": f"template number {template} " * 3}, {"role": "user", "content": str(i)}])
        it.case_id = f"{template}|{i}"
        return it

    sample, n = _Sample(1000), 0
    for _ in range(25):  # 40 templates fill the sample before a new one appears
        for t in range(40):
            sample.add(call(f"T{t}", n))
            n += 1
    for i in range(500):
        sample.add(call("late", n + i))
    kept = Counter(it.case_id.split("|")[0] for it in sample.items())
    assert len(sample.items()) == 1000 and kept["late"] >= 20 and min(kept.values()) >= 20
    rng = random.Random(5)
    for _ in range(200):  # never more than the limit, in order, every template kept while they fit
        limit, total, templates = rng.choice([5, 50, 120]), rng.randrange(0, 1500), rng.randrange(1, 40)
        weights = [rng.random() ** 3 for _ in range(templates)]
        sample, seq = _Sample(limit), []
        for i in range(total):
            t = rng.choices(range(templates), weights)[0]
            seq.append(t)
            sample.add(call(t, i))
        ids = [int(it.case_id.split("|")[1]) for it in sample.items()]
        assert len(ids) <= limit and ids == sorted(ids) and sample.dropped == total - len(ids)
        if len(set(seq)) <= limit:
            assert {seq[i] for i in ids} == set(seq)


def test_mcnemar_is_exact_and_fast_for_large_runs():
    import math
    import time

    from llmshift.stats import mcnemar_p

    def exact(b, c):
        n = b + c
        return 1.0 if n == 0 or b == 0 else sum(math.comb(n, k) for k in range(b, n + 1)) / 2**n

    for b, c in [(501, 500), (600, 450), (520, 600), (900, 300), (2, 1500), (1500, 2), (700, 700)]:
        assert math.isclose(mcnemar_p(b, c), exact(b, c), rel_tol=1e-9, abs_tol=1e-300)
    start = time.perf_counter()
    assert 0 < mcnemar_p(5200, 4800) < 1e-4 and mcnemar_p(10, 40000) == 1.0
    assert time.perf_counter() - start < 1.0


def test_schema_refs_to_true_false_and_junk():
    from llmshift.jsonschema_lite import describe, validate

    assert validate({"a": 1}, {"$defs": {"x": False}, "properties": {"a": {"$ref": "#/$defs/x"}}}) == ["$.a: no value is allowed here"]
    assert validate({"a": 1}, {"$defs": {"x": True}, "properties": {"a": {"$ref": "#/$defs/x"}}}) == []
    assert validate({"a": 1}, {"properties": {"a": {"$ref": 5}}}) == [] and validate(1, "junk") == [] and describe({"$ref": 5}) == "5"
    assert validate(float("nan"), {"type": "number", "minimum": 0}) == ["$: nan is not a JSON number"]
    assert validate(5, {"type": "number", "minimum": "10", "maximum": float("nan"), "multipleOf": "x"}) == []
    assert validate("abc", {"type": "string", "minLength": "5", "pattern": 5}) == []


def test_json_checks_reject_nan_and_infinity():
    from llmshift.formats import analyze_json

    for text in ('{"a": NaN}', '{"a": Infinity}', "[-Infinity]"):
        info = analyze_json(text)
        assert not info["valid"] and "not allowed in JSON" in info["error"]
    assert analyze_json('{"a": 1.5}')["valid"]


def test_inferred_checks_pass_on_their_own_baseline():
    from llmshift import infer
    from llmshift.checks import CheckContext, Sample, run_checks
    from llmshift.testset import TestCase

    for output in ('{"status\\n": "open"}', "Order at Domino's Pizza tonight.", "McDonald's Corporation reported results.", "|---|---|\n|---|---|"):
        messages = [{"role": "user", "content": "q"}]
        checks = infer.infer_checks(messages, {}, output)
        case = TestCase(id="c", input={"messages": messages, "params": {}}, baseline={"output": output}, checks=checks)
        failing = [(r.type, r.message) for r in run_checks(case, Sample(text=output), CheckContext()) if r.status not in ("pass", "skip")]
        assert failing == [], (output, failing)


def test_global_options_work_after_the_subcommand(tmp_path, monkeypatch, capsys):
    from llmshift import cli

    monkeypatch.chdir(tmp_path)
    assert cli.main(["models", "gpt-4o", "--no-color"]) == 0
    assert cli.main(["shadow", "report", "--config", "missing.yaml"]) == 2
    assert "config file not found: missing.yaml" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--help"], ["scan", "--help"], ["--version"]])
def test_help_into_a_closed_pipe_exits_quietly(args):
    import os
    import subprocess
    import sys

    read_end, write_end = os.pipe()
    os.close(read_end)  # the reader is gone before anything is written
    try:
        proc = subprocess.run([sys.executable, "-m", "llmshift", *args], stdout=write_end, stderr=subprocess.PIPE, timeout=60)
    finally:
        os.close(write_end)
    assert b"Exception ignored" not in proc.stderr and b"Traceback" not in proc.stderr, proc.stderr
    assert proc.returncode in (0, 1)


def test_files_saved_as_utf16_are_read_and_bad_bytes_name_the_file(tmp_path):
    import json

    from llmshift.calibrate import load_labels
    from llmshift.config import load_config
    from llmshift.testset import TestFileError, load_tests

    case = {"id": "c1", "input": {"messages": [{"role": "user", "content": "hi"}]}, "baseline": {"output": "hello"}}
    tests = tmp_path / "tests.jsonl"
    tests.write_bytes(json.dumps(case).encode("utf-16"))  # what PowerShell 5.1's > writes
    assert [c.id for c in load_tests(tests)] == ["c1"]
    config = tmp_path / "llmshift.yaml"
    config.write_bytes("candidate: mock:nextgen\n".encode("utf-16"))
    assert load_config(str(config))["candidate"] == "mock:nextgen"
    labels = tmp_path / "labels.jsonl"
    labels.write_bytes(json.dumps({"candidate": "x", "label": "pass"}).encode("utf-16"))
    assert load_labels(labels)[0]["label"] == "pass"
    tests.write_bytes(b'{"id": "c1", "input": "\xff"}')
    with pytest.raises(TestFileError) as info:
        load_tests(tests)
    assert str(tests) in str(info.value) and "not UTF-8 text" in str(info.value)


def test_corrupted_watch_history_and_calibration_files_are_survived(tmp_path):
    import json

    from llmshift.calibrate import calibration_note
    from llmshift.watch import _load_run, detect_drift, history_paths, load_history

    hist, runs = history_paths(tmp_path, "mock:legacy")
    hist.parent.mkdir(parents=True, exist_ok=True)
    rows = [5, "x", [1], {"run_id": "../../etc/passwd", "summary": "bad"}, {"run_id": "20261001T000000Z", "tests": "t", "summary": {"evaluated": "many", "pass_rate": "high", "similarity": float("nan"), "models": [1], "failing_categories": "x"}},
            {"run_id": "20261002T000000Z", "tests": "t", "summary": {"evaluated": 10, "pass_rate": 0.9, "similarity": 0.8}}]
    hist.write_bytes(("\n".join(json.dumps(r) for r in rows) + "\n").encode() + b"\xff\xfe{broken\n")
    history = load_history(tmp_path, "mock:legacy")
    assert [h.get("run_id") for h in history] == ["../../etc/passwd", "20261001T000000Z", "20261002T000000Z"]
    assert _load_run(runs, "../../etc/passwd") == {} and _load_run(runs, None) == {}
    summary = {"total": 10, "evaluated": 10, "passed": 5, "pass_rate": 0.5, "error_rate": 0.0, "similarity": 0.5}
    detect_drift(summary, {}, history, {}, model="mock:legacy", min_pass_rate=0.8, max_pass_rate_drop=0.05, max_similarity_drop=0.1, retirement_warn_days=30, fingerprint="t")
    cal = tmp_path / "judge-calibration.json"
    cal.write_text(json.dumps({"judges": {"mock:judge": {"rate": "90%", "n": "lots"}}}), encoding="utf-8")
    note = calibration_note("mock:judge", {"cache_dir": str(tmp_path / "cache")})
    assert "cannot be read" in note and "llmshift judge calibrate --judge mock:judge" in note


def test_misspelt_settings_are_reported_and_severity_off_works(tmp_path, monkeypatch, capsys):
    from llmshift import cli
    from llmshift.checks import normalize_severity
    from llmshift.config import load_config

    monkeypatch.chdir(tmp_path)
    (tmp_path / "llmshift.yaml").write_text(
        "min_pass_rat: 0.9\nscan:\n  warn_day: 30\nwatch:\n  modle: x\nseverity:\n  tone: warnn\n  tonee: warn\n  similarity: off\n", encoding="utf-8"
    )
    said = "\n".join(load_config().warnings)
    for words in ("'min_pass_rat' (did you mean 'min_pass_rate'?)", "'scan.warn_day' (did you mean 'warn_days'?)", "'watch.modle' (did you mean 'model'?)",
                  "severity.tone must be error, warn or info, got 'warnn'", "unknown check type 'severity.tonee' (did you mean 'tone'?)"):
        assert words in said, words
    assert "similarity" not in said  # YAML reads `off` as false: that is "info"
    assert normalize_severity(False) == "info" and normalize_severity(True) is None and normalize_severity("WARNING") == "warn"
    assert cli.main(["models", "gpt-4o"]) == 0
    assert "did you mean 'min_pass_rate'" in capsys.readouterr().err


def test_registry_files_and_settings_are_checked(tmp_path, monkeypatch):
    import json

    from llmshift.config import ConfigError, load_config
    from llmshift.registry import Registry

    bad = tmp_path / "reg.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="entry 1 must be an object with a model"):
        Registry.load([str(bad)])
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"models": [{"id": "my-model", "provider": "openai", "status": "retired", "retires": "2026-01-01"}]}), encoding="utf-8")
    assert Registry.load([str(good)]).lookup("my-model").status == "retired"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "llmshift.yaml").write_text("scan:\n  registry: [1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="scan.registry must be a registry file name"):
        load_config()


def test_shadow_report_finds_the_project_log_from_a_subfolder(tmp_path, monkeypatch, capsys):
    import json

    from llmshift import cli

    (tmp_path / "llmshift.yaml").write_text("min_pass_rate: 0.0\n", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    record = {"llmshift_shadow": 1, "ts": "2026-10-01T00:00:00Z", "api": "chat",
              "request": {"messages": [{"role": "user", "content": "hi"}], "params": {}},
              "baseline": {"output": "hello there"}, "shadow": {"spec": "mock:echo", "output": "hello there"}}
    (tmp_path / "logs" / "llmshift-shadow.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    monkeypatch.chdir(tmp_path / "src")
    assert cli.main(["shadow", "report"]) == 0
    assert "1 shadowed request" in capsys.readouterr().out


def test_scan_reads_paths_past_260_characters_and_lists_skipped_files(tmp_path):
    import os

    from llmshift.fileio import long_path
    from llmshift.scanner import scan_paths

    deep = tmp_path
    while len(str(deep)) < 300:
        deep = deep / "a_fairly_long_directory_name_x"
    os.makedirs(long_path(deep, always=True), exist_ok=True)
    with open(long_path(deep / "app.py", always=True), "w", encoding="utf-8") as fh:
        fh.write('model = "gpt-4o"\n')
    (tmp_path / "big.py").write_text('model = "gpt-4o-mini"\n' + "# x\n" * 1_500_000, encoding="utf-8")
    result = scan_paths([str(tmp_path)])
    assert any(f.model == "gpt-4o" for f in result.findings)
    assert any("big.py: larger than" in e for e in result.errors)


def test_case_ids_stay_unique_and_stable_however_cases_are_selected(tmp_path):
    import json

    from llmshift.testset import TestCase, load_tests, unique_ids

    recs = [{"id": i, "input": {"messages": [{"role": "user", "content": "q"}]}, "group": g} for i, g in [("a", "x"), ("a", "y"), ("a-2", "x"), ("b", "y")]]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n{broken", encoding="utf-8")
    assert [c.id for c in load_tests(path, limit=4)] == ["a", "a-2", "a-2-2", "b"]  # stops before the broken line
    assert [c.id for c in load_tests(path, limit=2, keep=lambda c: c.group == "y")] == ["a-2", "b"]  # same ids as in the full set
    same = [TestCase(id=i, input={"messages": []}) for i in ("a", "a", "a-2")]
    assert [c.id for c in unique_ids(same)] == ["a", "a-2", "a-2-2"] and [c.id for c in same] == ["a", "a", "a-2"]


def _dead_judge_module(tmp_path):
    (tmp_path / "deadjudge.py").write_text(
        "from llmshift.providers.base import ProviderError\n\n"
        "def answer(req):\n    raise ProviderError('Incorrect API key provided', status=401, code='invalid_api_key')\n",
        encoding="utf-8",
    )
    return f"python:{tmp_path / 'deadjudge.py'}:answer"


def test_a_dead_judge_is_reported_not_counted_as_zero_calls(tmp_path):
    from llmshift.compare import run_compare
    from llmshift.testset import TestCase

    cases = [TestCase(id=f"c{i}", input={"messages": [{"role": "user", "content": f"q{i}"}]}, baseline={"output": f"answer {i}"}, checks=[{"type": "llm_judge"}]) for i in range(3)]
    res = run_compare(cases, "mock:echo", judge=_dead_judge_module(tmp_path), cache=None, judge_mode="always", show_progress=False)
    assert res.insights[0].startswith("The judge failed on 3 of 3 call(s)") and "Incorrect API key" in res.insights[0]
    messages = [r.message for c in res.cases for r in c.checks if r.type == "llm_judge"]
    assert messages and all(m.startswith("the grading model failed") for m in messages)


def test_calibrating_a_dead_judge_keeps_the_last_good_check(tmp_path, monkeypatch, capsys):
    import json

    from llmshift import cli

    monkeypatch.chdir(tmp_path)
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps({"id": f"l{i}", "candidate": f"answer {i}", "reference": "answer", "label": "pass"}) for i in range(6)), encoding="utf-8")
    saved = tmp_path / ".llmshift" / "judge-calibration.json"
    saved.parent.mkdir()
    before = json.dumps({"judges": {"x": {"judge": "x", "n": 20, "agree": 19, "rate": 0.95}}})
    saved.write_text(before, encoding="utf-8")
    assert cli.main(["judge", "calibrate", "--labels", str(labels), "--judge", _dead_judge_module(tmp_path), "--no-cache"]) == 1
    assert "gave no usable verdict on 6 pair(s): [HTTP 401, invalid_api_key] Incorrect API key provided" in capsys.readouterr().err
    assert saved.read_text(encoding="utf-8") == before


def test_a_failed_request_in_repeated_runs_is_not_flakiness():
    from llmshift.compare import CaseResult, combine_runs

    def runs(*statuses):
        return combine_runs([CaseResult(id="c", group="g", status=s) for s in statuses])

    one_error = runs("pass", "pass", "error")
    assert not one_error.flaky and one_error.pass_fraction == 0.6667  # counted against, not called flaky
    assert runs("pass", "fail", "pass").flaky and not runs("fail", "error", "fail").flaky


def test_fix_survives_a_dead_optimizer_and_stops_when_it_repeats_itself(tmp_path):
    from test_compare_fix import demo_cases

    from llmshift.fix import run_fix
    from llmshift.providers.mock import MockProvider

    calls = {"n": 0}

    def same_prompt(request):
        calls["n"] += 1
        return "<prompt>You are a helpful assistant. Answer the question.</prompt>"

    dead = _dead_judge_module(tmp_path)
    result = run_fix(demo_cases(), "mock:nextgen", optimizer=dead, rounds=3, show_progress=False, source_paths=[str(tmp_path)])
    notes = [n for g in result.groups for n in g.notes]
    assert any("The optimizer" in n and "only rule-based changes were tried" in n for n in notes)
    assert result.after["pass_rate"] > result.before["pass_rate"]  # the rules still fixed what they could
    result = run_fix(demo_cases(), "mock:nextgen", optimizer=MockProvider("same", fn=same_prompt), rounds=4, show_progress=False, source_paths=[str(tmp_path)])
    for group in result.groups:
        optimizer_runs = [a for a in group.attempts if a["strategy"] == "optimizer"]
        assert len(optimizer_runs) <= 1  # the same rewrite is tried once, not every round


def test_snapshot_reads_shadow_logs_as_recorded_traffic(tmp_path):
    import json

    from llmshift.snapshot import build_snapshot

    recs = [{"llmshift_shadow": 1, "ts": "2026-10-01T00:00:00Z", "api": "chat",
             "request": {"messages": [{"role": "system", "content": "You are support."}, {"role": "user", "content": f"question {i}"}], "params": {"temperature": 0}},
             "baseline": {"output": f"answer {i}", "model": "gpt-4o-2024-08-06"}, "shadow": {"spec": "mock:echo", "output": "x"}} for i in range(3)]
    recs.append({"llmshift_shadow": 1, "request": "junk", "baseline": None})
    path = tmp_path / "llmshift-shadow.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    res = build_snapshot([str(path)])
    assert sorted(c.baseline_output for c in res.cases) == ["answer 0", "answer 1", "answer 2"]


def test_init_takes_its_models_from_the_code(tmp_path, monkeypatch):
    from llmshift import cli
    from llmshift.config import load_config

    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text('client.chat.completions.create(model="gpt-4o-2024-08-06", messages=[])\n', encoding="utf-8")
    assert cli.main(["init"]) == 0
    cfg = load_config()
    assert cfg["baseline"] == "openai:gpt-4o-2024-08-06" and cfg["candidate"] and cfg["candidate"] != cfg["baseline"]
    (tmp_path / "app.py").unlink()
    (tmp_path / "llmshift.yaml").unlink()  # a re-run would keep the models it names
    assert cli.main(["init"]) == 0
    cfg = load_config()
    assert cfg["baseline"] is None and cfg["candidate"] is None  # nothing to go on: placeholders, never a retiring model
    assert "# baseline: provider:model" in (tmp_path / "llmshift.yaml").read_text(encoding="utf-8")


def test_scanner_resolves_names_in_their_own_scope():
    from llmshift.scanner import scan_python

    code = (
        'MODEL = "gpt-4o"\n'
        "def setup():\n"
        '    model = "claude-3-5-sonnet-20240620"\n'
        "def ask(client, model):\n"
        "    return client.chat.completions.create(model=model, messages=[])\n"
        "def ask_local(client):\n"
        "    model = pick_model()\n"
        "    return client.chat.completions.create(model=model, messages=[])\n"
        "class Bot:\n"
        '    model = "gpt-3.5-turbo"\n'
        "    def run(self, client):\n"
        "        return client.chat.completions.create(model=model, messages=[])\n"
        "def outer():\n"
        '    m = "o1-mini"\n'
        "    def inner(client):\n"
        "        return client.chat.completions.create(model=m, messages=[])\n"
        "def with_default(client, model=MODEL):\n"
        "    return client.chat.completions.create(model=model, messages=[])\n"
    )
    calls = {(line, value) for line, _col, value, kind, _call, _via in scan_python(code, llm_hint=True) if kind == "call"}
    assert calls == {(16, "o1-mini"), (18, "gpt-4o")}  # never a parameter, a shadowed local or a class attribute


def test_an_html_error_page_is_summed_up_not_dumped():
    err = http._parse_error(404, b"<!DOCTYPE html><html><head><title>404 Not Found</title></head><body>" + b"x" * 5000 + b"</body></html>", "https://example.com/x")
    assert str(err) == "[HTTP 404, not_found] the server sent a web page (404 Not Found), not an API answer (example.com)"
    assert "x" * 50 not in str(http._parse_error(502, b"<html><body>" + b"x" * 500 + b"</body></html>", "https://example.com/x"))


def test_env_var_defaults_resolve_at_the_call_and_json_logs_are_no_prompt_home(tmp_path):
    import json

    from llmshift.fix import find_prompt_locations
    from llmshift.scanner import scan_python

    code = 'import os\nMODEL = os.getenv("LLM_MODEL", "gpt-4o-2024-08-06")\nclient.chat.completions.create(model=MODEL, messages=[])\n'
    assert ("gpt-4o-2024-08-06", "MODEL, line 2") in {(v, via) for _l, _c, v, kind, _call, via in scan_python(code, llm_hint=True) if kind == "call"}
    prompt = "You are Acme support. Always answer politely and briefly."
    (tmp_path / "app.py").write_text(f'SYSTEM = "{prompt}"\n', encoding="utf-8")
    (tmp_path / "calls.json").write_text(json.dumps([{"request": {"messages": [{"role": "system", "content": prompt}]}, "response": {"choices": []}}], indent=1), encoding="utf-8")
    (tmp_path / "prompts.json").write_text(json.dumps({"support": prompt}), encoding="utf-8")
    found = find_prompt_locations(prompt, [str(tmp_path)], limit=10)
    assert {f.split("/")[-1].split(":")[0] for f in found} == {"app.py", "prompts.json"}


def test_watch_writes_the_pass_rate_output_and_outputs_cannot_be_injected(tmp_path, monkeypatch):
    from llmshift import cli

    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    cli._github_output("recommended", "evil\npass_rate=1.0")

    def outputs(text):  # as GitHub reads the file: name=value, or name<<DELIMITER ... DELIMITER
        found, lines = {}, iter(text.splitlines())
        for line in lines:
            if "<<" in line:
                name, delimiter = line.split("<<", 1)
                value = []
                for inner in lines:
                    if inner == delimiter:
                        break
                    value.append(inner)
                found[name] = "\n".join(value)
            elif "=" in line:
                name, value = line.split("=", 1)
                found[name] = value
        return found

    assert outputs(out.read_text(encoding="utf-8")) == {"recommended": "evil\npass_rate=1.0"}  # one output, not two
    monkeypatch.chdir(tmp_path)
    assert cli.main(["demo", "--dir", "proj"]) == 0
    monkeypatch.chdir(tmp_path / "proj")
    out.write_text("", encoding="utf-8")
    cli.main(["watch", "--model", "mock:nextgen", "--dry-run"])
    assert out.read_text(encoding="utf-8").startswith("pass_rate=")


def test_annotations_name_files_from_the_checkout_root(tmp_path, monkeypatch):
    from llmshift.report import github_annotations, scan_sarif
    from llmshift.scanner import scan_paths

    app = tmp_path / "services" / "app"
    app.mkdir(parents=True)
    (app / "main.py").write_text('client.chat.completions.create(model="gpt-3.5-turbo-0301", messages=[])\n', encoding="utf-8")
    monkeypatch.chdir(app)  # the action's working-directory
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    scan = scan_paths(["."])
    assert scan.findings and all(f.file == "main.py" for f in scan.findings)
    assert all("file=services/app/main.py," in a for a in github_annotations(scan))
    uris = {loc["physicalLocation"]["artifactLocation"]["uri"] for r in scan_sarif(scan)["runs"][0]["results"] for loc in r["locations"]}
    assert uris == {"services/app/main.py"}


def test_redaction_and_failure_patterns_stay_linear_on_hostile_text():
    import time

    from llmshift.checks import CheckResult
    from llmshift.compare import failure_patterns
    from llmshift.redact import Redactor

    start = time.perf_counter()
    Redactor().text("x-api-key=." * 20000)  # every match attempt used to rescan the rest of the run
    Redactor().text("secret access key: " * 20000 + ("AKIA" + "IOSFODNN7EXAMPLE ") * 5000)
    failure_patterns(CheckResult("contains", "facts", "error", "fail", "a" + " " * 200000 + "x"))
    assert time.perf_counter() - start < 5  # quadratic, this took minutes


def test_global_options_after_the_subcommand_keep_abbreviations_unique():
    from llmshift import cli

    parser = cli.build_parser()
    ns = parser.parse_args(cli._global_options_first(["compare", "--no-c", "--con", "3", "--config", "x.yaml", "--no-color"]))
    assert ns.no_cache and ns.concurrency == 3 and ns.config == "x.yaml" and ns.no_color  # --no-c is --no-cache, as in 0.1.0
    ns = parser.parse_args(cli._global_options_first(["check", "--files", "a.py", "b.py", "--no-color"]))
    assert ns.files == ["a.py", "b.py"] and ns.no_color
    assert parser.parse_args(cli._global_options_first(["scan", "--", "--config"])).paths == ["--config"]


def test_source_compiles_without_warnings():
    """An invalid escape sequence in a string is a SyntaxWarning, and an error under -W error."""
    import pathlib
    import warnings

    root = pathlib.Path(__file__).resolve().parents[1]
    found = []
    for path in [*root.joinpath("src", "llmshift").rglob("*.py"), *root.joinpath("tests").rglob("*.py"), *root.joinpath("scripts").glob("*.py")]:
        if "_vendor" in path.parts:
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        found += [f"{path.name}:{w.lineno}: {w.message}" for w in caught]
    assert not found


# ---------------------------------------------------------------- independent review


def test_scan_comments_do_not_swallow_code(tmp_path):
    from llmshift.scanner import scan_paths

    slash = chr(92)
    (tmp_path / "client.js").write_text(
        "const base = url.replace(/" + slash + "/*$/, '');\n"  # a regex, not a comment
        'client.chat.completions.create({ model: "gpt-3.5-turbo-0613" });\n', encoding="utf-8")
    (tmp_path / "main.tf").write_text(
        'user_data = <<EOF\nrm -rf /var/lib/apt/lists/*\nEOF\nfoundation_model = "anthropic.claude-v2:1"\n', encoding="utf-8")
    (tmp_path / "Chat.vue").write_text('<template>\n<!-- the old bot used model: "gpt-3.5-turbo-0301" -->\n</template>\n', encoding="utf-8")
    (tmp_path / "q.sql").write_text('# model: "gpt-3.5-turbo-0301" in a MySQL comment\nSELECT 1;\n', encoding="utf-8")
    found = {(f.file.split("/")[-1], f.model) for f in scan_paths([str(tmp_path)]).findings}
    assert ("client.js", "gpt-3.5-turbo-0613") in found and ("main.tf", "anthropic.claude-v2:1") in found
    assert not any(name in ("Chat.vue", "q.sql") for name, _ in found)


def test_scanner_reads_annotations_and_class_scope_defaults():
    from llmshift.scanner import scan_python

    code = (
        "from typing import Literal\n"
        'MODEL = "gpt-4o"\n'
        'def ask(model: Literal["gpt-3.5-turbo-0613"] = "gpt-4-0613") -> Literal["text-davinci-003"]:\n'
        "    pass\n"
        "class Bot:\n"
        '    MODEL = "gpt-3.5-turbo-0613"\n'
        "    def run(self, client, model=MODEL):\n"
        "        client.chat.completions.create(model=model, messages=[])\n"
    )
    found = {(line, value, kind) for line, _c, value, kind, _call, _via in scan_python(code, llm_hint=True)}
    assert (3, "gpt-3.5-turbo-0613", "string") in found and (3, "text-davinci-003", "string") in found
    assert (8, "gpt-3.5-turbo-0613", "call") in found and (8, "gpt-4o", "call") not in found


def test_prompt_check_reads_the_text_before_a_variable():
    from llmshift.prcheck import _python_strings

    code = 'PROMPT = "You are a support assistant for ACME Corp. " + "Answer in two sentences at most. " + "Customer: " + name\n'
    assert "You are a support assistant for ACME Corp. Answer in two sentences at most. Customer: " in _python_strings(code)


def test_upgrade_swaps_engine_outside_azure_only(tmp_path, monkeypatch):
    import subprocess

    from llmshift.upgrade import plan_upgrade

    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "old_sdk.py").write_text('import openai\nr = openai.Completion.create(engine="text-davinci-003", prompt="hi")\n', encoding="utf-8")
    (tmp_path / "azure_app.py").write_text('from openai import AzureOpenAI\nclient = AzureOpenAI(azure_endpoint="https://x.openai.azure.com")\nr = client.completions.create(engine="text-davinci-003", prompt="hi")\n', encoding="utf-8")
    plan = plan_upgrade(["."], old="text-davinci-003", new="gpt-3.5-turbo-instruct", root=tmp_path)
    assert 'engine="gpt-3.5-turbo-instruct"' in plan.diff and "old_sdk.py" in plan.diff  # the pre-1.0 SDK named the model engine=
    assert "azure_app.py" not in plan.diff  # in Azure code it names a deployment
    assert any("azure_app.py" in n and "Azure deployment" in n for n in plan.notes)


def test_upgrade_commits_are_signed_where_the_repository_signs(tmp_path):
    import subprocess

    from llmshift import github as gh

    ident = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", *ident, "-C", str(tmp_path), "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    assert gh.commit_files(tmp_path, {"a.txt": b"x"}, {"a.txt": "100644"}, "plain", config=ident)
    subprocess.run(["git", "-C", str(tmp_path), "config", "commit.gpgSign", "true"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "gpg.program", "false"], check=True)  # a signer that always fails
    with pytest.raises(gh.GitError):  # signing was attempted, as `git commit` would
        gh.commit_files(tmp_path, {"a.txt": b"y"}, {"a.txt": "100644"}, "signed", config=ident)


def test_a_stream_left_at_its_finish_is_recorded_whole(tmp_path):
    import json

    from llmshift.record import _StreamProxy, _StreamAccumulator

    chunks = [
        {"model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Paris"}, "finish_reason": None}]},
        {"model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"model": "gpt-4o", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
    ]
    got = []
    stream = _StreamProxy(iter(chunks), lambda acc, error: got.append((acc.response("chat"), error)))
    for chunk in stream:  # the app stops at the finish reason, before the usage chunk
        if chunk["choices"] and chunk["choices"][0]["finish_reason"]:
            break
    stream.close()
    assert got and got[0][1] is None and json.dumps(got[0][0]).count("Paris") == 1
    assert _StreamAccumulator  # imported: the name the proxy builds on


def test_large_sdk_responses_stay_data_and_cycles_end():
    from llmshift.record import to_dict

    tokens = [{"token": "a", "logprob": -0.1, "bytes": [97], "top_logprobs": [{"token": "b", "logprob": -1.0, "bytes": [98]}] * 20}] * 3000
    big = to_dict({"choices": [{"message": {"content": "x"}, "logprobs": {"content": tokens}}], "usage": {"prompt_tokens": 1}})
    assert isinstance(big["choices"][0]["message"], dict) and isinstance(big["usage"], dict)
    loop: dict = {"a": [1]}
    loop["self"] = loop
    assert isinstance(to_dict(loop)["self"], str)


def test_python_functions_have_no_time_limit_unless_one_is_set(tmp_path):
    import time

    from llmshift.providers import resolve

    (tmp_path / "slowish.py").write_text("import time\ndef f(r):\n    time.sleep(1.2)\n    return 'done'\n", encoding="utf-8")
    spec = f"python:{tmp_path / 'slowish.py'}:f"
    assert resolve(spec, {"providers": {"python": {"timeout": 0.5}}}).options.get("timeout_set")
    start = time.monotonic()
    assert resolve(spec).complete({"messages": []}).text == "done" and time.monotonic() - start >= 1.1


def test_redaction_skips_only_what_is_all_encoded_data():
    import base64

    from llmshift.redact import Redactor

    key = "AKIA" + "IOSFODNN7EXAMPLE"
    prose = ("word " * 900) + key + " password = Hunter2Secret99"
    assert key not in Redactor().text(prose) and "Hunter2Secret99" not in Redactor().text(prose)
    tail = "data:image/png;base64,iVBORw0KGgo= and the key sk-ant-api03-" + "A" * 40
    assert "sk-ant-api03" not in Redactor().text(tail)
    blob = "data:image/png;base64," + base64.b64encode(bytes(range(256)) * 50).decode()
    assert Redactor().text(blob) == blob


def test_server_waits_are_read_only_when_readable():
    from email.message import Message

    from llmshift.providers import http

    h = Message()
    h["Retry-After"] = "soon"
    assert http._header_delay(h) is None and http._retry_delay(h, 3) >= 4.0  # backs off per attempt
    h2 = Message()
    h2["Retry-After"] = "7"
    assert http._header_delay(h2) == 7.0
    assert http.display_url("admin:hunter2pw@gpu-box:8000/v1?key=x") == "gpu-box:8000/v1"


def test_maps_clear_a_stale_cancel_and_stop_after_a_failure():
    from llmshift.providers import http
    from llmshift.runner import interruptible_map

    http.CANCEL.set()  # left over from an earlier Ctrl+C in this process
    assert interruptible_map(lambda x: x * 2, [1, 2, 3], 2) == [2, 4, 6]
    sent = []

    def call(x):
        sent.append(x)
        if x == 0:
            raise ValueError("401")
        return x

    with pytest.raises(ValueError, match="401"):
        interruptible_map(call, range(10), 1)
    assert sent == [0]


def test_logging_self_referencing_metadata_returns(tmp_path):
    import threading

    from llmshift.record import Recorder

    meta: dict = {"run": "x"}
    meta["self"] = meta
    rec = Recorder(tmp_path / "log.jsonl")
    done = threading.Event()
    threading.Thread(target=lambda: (rec.log({"messages": [{"role": "user", "content": "hi"}]}, None, metadata=meta), done.set()), daemon=True).start()
    assert done.wait(10)


def test_append_tolerates_targets_that_cannot_seek(tmp_path, monkeypatch):
    import errno
    import os

    from llmshift import fileio

    target = tmp_path / "out.jsonl"
    target.write_bytes(b"partial")

    def no_seek(*args):
        raise OSError(errno.ESPIPE, "Illegal seek")

    monkeypatch.setattr(os, "lseek", no_seek)
    monkeypatch.setattr(fileio, "_lock", lambda fd: None)
    monkeypatch.setattr(fileio, "_unlock", lambda fd: None)
    fileio.append(target, b'{"a": 1}' + bytes([10]))
    assert target.read_bytes().endswith(b'{"a": 1}' + bytes([10]))


def test_agent_tool_calls_pair_and_order_like_the_model_ran_them():
    from llmshift.agents import compare_steps
    from llmshift.checks import _pair_calls

    many = [{"name": "search", "arguments": {"q": str(i)}} for i in range(8)]
    assert isinstance(_pair_calls("search", many, many[:7], "auto"), list)  # more recorded than replayed: no IndexError
    base = [{"tool_calls": [{"name": "search", "arguments": {}}, {"name": "fetch", "arguments": {}}]}, {"text": "done"}]
    split = [{"tool_calls": [{"name": "search", "arguments": {}}]}, {"tool_calls": [{"name": "fetch", "arguments": {}}]}, {"text": "done"}]
    swapped = [{"tool_calls": [{"name": "fetch", "arguments": {}}]}, {"tool_calls": [{"name": "search", "arguments": {}}]}, {"text": "done"}]
    sequential = [{"tool_calls": [{"name": "search", "arguments": {}}]}, {"tool_calls": [{"name": "fetch", "arguments": {}}]}, {"text": "done"}]
    assert not any("different order" in c for c in compare_steps(base, split)[1])  # parallel became sequential
    assert any("different order" in c for c in compare_steps(sequential, swapped)[1])


def test_json_checks_from_0_1_0_pass_on_a_nan_baseline():
    from llmshift.checks import CheckContext, Sample, evaluate
    from llmshift.testset import TestCase

    out = '{"sensor": "t1", "reading": NaN}'
    checks = [{"type": "json_valid"}, {"type": "json_schema", "schema": {"type": "object", "properties": {"reading": {"type": "number"}}}},
              {"type": "json_match", "fields": {"$.reading": float("nan")}}]
    case = TestCase(id="c", input={"messages": [{"role": "user", "content": "q"}], "params": {}}, baseline={"output": out}, checks=checks)
    assert [evaluate(c, Sample(text=out), case, CheckContext()).status for c in checks] == ["pass", "pass", "pass"]


def test_digest_needs_no_url_when_there_is_nothing_to_send(tmp_path, monkeypatch, capsys):
    from llmshift import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    (tmp_path / "app.py").write_text('MODEL = "gpt-5.6-sol"\n', encoding="utf-8")
    (tmp_path / "llmshift.yaml").write_text("digest:\n  alerts:\n    - type: slack\n      url: ${SLACK_WEBHOOK_URL}\n", encoding="utf-8")
    assert cli.main(["digest", "."]) == 0


def test_imports_logs_and_test_files_in_their_legitimate_shapes(tmp_path):
    import json

    from llmshift.logparse import parse_logs
    from llmshift.testset import load_tests

    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in [
        {"input": {"messages": [{"role": "user", "content": "Hi"}], "conversation": [{"role": "user", "content": "earlier"}]}, "output": "Hello", "model": "gpt-4o"},
        {"request": {"messages": [{"role": "user", "content": "Summarise"}], "model": "gpt-4o", "prompt": {"name": "summary-v2", "version": 3}},
         "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "model": "gpt-4o"}},
    ]), encoding="utf-8")
    assert len(parse_logs([str(log)])[0]) == 2
    tests = tmp_path / "t.jsonl"
    tests.write_text(json.dumps({"id": "c1", "input": {"messages": [{"role": "user", "content": "q"}], "params": []}, "vars": [], "metadata": []}), encoding="utf-8")
    assert load_tests(tests)[0].params == {}


def test_config_knows_every_check_type_and_code_spans_keep_model_ids():
    from llmshift.config import config_warnings
    from llmshift.pick import _code

    assert config_warnings({"severity": {"includes": "warn"}}, None) == []
    assert _code("workers:@cf/meta/llama|x") == "workers:@cf/meta/llama" + chr(92) + "|x"


def test_terminal_keeps_ordinary_rtl_marks():
    from llmshift import console

    text = "shalom " + chr(0x200F) + " and " + chr(0x061C)
    assert console.safe(text) == text and chr(0x202E) not in console.safe(chr(0x202E))


def test_prices_skip_unknown_entries_for_another_name(monkeypatch):
    from llmshift import pricing

    monkeypatch.setattr(pricing, "_table", lambda: ({"gemini/gemini-x": [0.0, 0.0], "gemini-x": [0.3, 2.5]}, "2026-10-21"))
    price = pricing.price_for("gemini:gemini-x")
    assert price is not None and price.input == 0.3
