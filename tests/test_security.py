"""Security rules for llmshift's own code, checked on every test run.

Automated scanners read every file of every PyPI release. One of them filed an advisory
over a single exec call (GCVE-110-VULN-PYPI-2026-000499), so these tests fail on the
patterns such scanners report, before a release can carry one. Where llmshift runs code on
purpose (python graders, `python:` and `command` providers), SECURITY.md says so and the
tests below pin down how.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import sys

import pytest

from llmshift.providers import resolve

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "llmshift"
OWN_CODE = [p for p in PACKAGE.rglob("*.py") if "_vendor" not in p.parts]
# Everything the source archive ships, which is what scanners of PyPI releases read.
SHIPPED = OWN_CODE + [p for d in ("tests", "scripts", "examples") for p in (ROOT / d).rglob("*.py")]

BANNED_BUILTINS = {"exec", "eval", "compile", "__import__", "execfile"}
BANNED_ATTRIBUTES = {
    ("os", "system"), ("os", "popen"), ("os", "startfile"), ("subprocess", "getoutput"), ("subprocess", "getstatusoutput"),
    ("os", "execl"), ("os", "execlp"), ("os", "execv"), ("os", "execvp"), ("os", "spawnl"), ("os", "spawnv"),
    ("pickle", "load"), ("pickle", "loads"), ("marshal", "load"), ("marshal", "loads"), ("shelve", "open"),
    ("yaml", "load"), ("yaml", "load_all"), ("yaml", "unsafe_load"), ("yaml", "full_load"),
    ("tempfile", "mktemp"), ("ssl", "_create_unverified_context"),
    ("importlib", "import_module"), ("runpy", "run_path"), ("runpy", "run_module"),
}
# The one place that imports code by name on purpose: `python:module:function` providers
# run the user's own module, as pytest imports their tests (see SECURITY.md).
DYNAMIC_IMPORT_ALLOWED = {"providers/custom.py"}


def _calls(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _where(path: pathlib.Path, node: ast.AST) -> str:
    return f"{path.relative_to(PACKAGE).as_posix()}:{node.lineno}"


def test_no_dynamic_code_execution():
    hits = []
    for path in OWN_CODE:
        rel = path.relative_to(PACKAGE).as_posix()
        for call in _calls(path):
            func = call.func
            if isinstance(func, ast.Name) and func.id in BANNED_BUILTINS:
                hits.append(f"{_where(path, call)} {func.id}")
            elif isinstance(func, ast.Attribute):
                base = ast.unparse(func.value).split(".")[-1]
                if (base, func.attr) in BANNED_ATTRIBUTES and not (func.attr == "import_module" and rel in DYNAMIC_IMPORT_ALLOWED):
                    hits.append(f"{_where(path, call)} {base}.{func.attr}")
                if func.attr in ("exec_module", "spec_from_file_location") and rel not in DYNAMIC_IMPORT_ALLOWED:
                    hits.append(f"{_where(path, call)} {func.attr}")
    assert hits == []


def test_no_shell_and_no_unverified_connections():
    hits = []
    for path in OWN_CODE:
        for call in _calls(path):
            for kw in call.keywords:
                if kw.arg == "shell" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                    hits.append(f"{_where(path, call)} shell={ast.unparse(kw.value)}")
                if kw.arg == "verify" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                    hits.append(f"{_where(path, call)} verify=False")
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr in ("md5", "sha1") and not any(k.arg == "usedforsecurity" for k in call.keywords):
                hits.append(f"{_where(path, call)} {func.attr} without usedforsecurity=False")
    assert hits == []


def test_no_bind_all_interfaces_literal():
    hits = [
        _where(path, node)
        for path in OWN_CODE
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant) and node.value == "0.0.0.0"  # nosec B104  # the rule itself; nothing binds
    ]
    assert hits == []


def test_no_exec_or_eval_text_in_anything_shipped():
    # Some scanners match text, not syntax: a call written inside a string or a help message
    # counts. This covers the tests, scripts and examples too, since the source archive ships them.
    pattern = re.compile(r"\b(?:exec|eval)\s*\(")
    hits = [
        f"{path.relative_to(ROOT).as_posix()}:{n}"
        for path in SHIPPED
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert hits == []


def test_no_key_shaped_text_in_anything_shipped():
    # Secret scanners (GitHub's among them) report key-shaped text even when it is a test's
    # fake. Tests build their fake keys from pieces, as "AKIA" + "...", so nothing matches.
    pattern = re.compile(
        r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|xox[abprs]-[A-Za-z0-9-]{10,}"
        r"|AIza[\w-]{30,}|sk-(?:proj-|ant-)?[\w-]{20,}|sk_live_\w{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----\s*[A-Za-z0-9+/=]{20,}"
        r"|(?:xai-|gsk_|pplx-|fw_|hf_|r8_)[A-Za-z0-9]{16,}"
    )
    shipped = SHIPPED + [ROOT / name for name in ("README.md", "CHANGELOG.md", "SECURITY.md", "action.yml")]
    hits = [
        f"{path.relative_to(ROOT).as_posix()}:{n}"
        for path in shipped
        if path.exists()
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert hits == []


def test_no_hidden_bidirectional_text():
    # "Trojan Source": direction overrides make code read differently from how it runs.
    bidi = {chr(c) for c in [*range(0x202A, 0x202F), *range(0x2066, 0x206A)]}
    hits = [
        f"{path.relative_to(ROOT).as_posix()}:{n}"
        for path in SHIPPED
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if bidi & set(line)
    ]
    assert hits == []


def test_releases_wait_for_the_security_scan():
    workflows = ROOT / ".github" / "workflows"
    if not workflows.is_dir():
        pytest.skip("the source archive has no workflows")
    release = (workflows / "release.yml").read_text(encoding="utf-8")
    assert "uses: ./.github/workflows/security.yml" in release
    assert re.search(r"publish:\s*\n\s*needs: \[[^\]]*\bsecurity\b", release)


def test_yaml_is_only_read_with_the_safe_loader():
    for path in OWN_CODE:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\b(?:Unsafe|Full)?Loader\s*=\s*yaml\.(?:Loader|UnsafeLoader|FullLoader)\b", text), path
        assert "yaml.Loader" not in text and "yaml.UnsafeLoader" not in text and "yaml.FullLoader" not in text, path


def test_command_providers_never_use_a_shell(tmp_path):
    # Shell syntax in a command reaches the program as plain words: nothing is chained or redirected.
    marker = tmp_path / "injected.txt"
    script = tmp_path / "argv.py"
    script.write_text("import json, sys\nsys.stdin.read()\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
    command = f'"{sys.executable}" "{script}" "two words" && echo injected > "{marker}"'
    prov = resolve("cmd:m", {"providers": {"cmd": {"type": "command", "command": command}}})
    args = json.loads(prov.complete({"messages": [{"role": "user", "content": "hi"}], "params": {}}).text)
    assert args[0] == "two words" and "&&" in args
    assert not marker.exists()


def test_programs_are_never_found_in_the_current_directory(tmp_path, monkeypatch):
    # Windows starts "git" from the current directory before PATH, so a git.exe in a cloned
    # repository would run (GitPython's CVE-2023-40590). llmshift runs git by full path.
    from llmshift.fileio import find_program

    planted = tmp_path / ("git.exe" if sys.platform == "win32" else "git")
    planted.write_bytes(b"")
    planted.chmod(0o700)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", os.pathsep.join([".", ""]))
    assert find_program("git") is None
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_program("git") == str(planted)
    source = "\n".join(p.read_text(encoding="utf-8") for p in OWN_CODE)
    assert not re.search(r"""\[\s*["'](?:git|taskkill)["']""", source), "start git and taskkill by full path"


def test_the_config_search_stays_inside_your_project_and_home(tmp_path, monkeypatch):
    # A config can name programs to run, so one planted in a shared parent folder must not
    # be read (as git's CVE-2022-24765, where any user could create C:\.git).
    from llmshift import config

    home = tmp_path / "home"
    sub = home / "proj" / "sub"
    sub.mkdir(parents=True)
    (tmp_path / "llmshift.yaml").write_text("candidate: mock:above-home\n", encoding="utf-8")
    monkeypatch.setattr(config, "_home", lambda: home.resolve())
    assert config.find_config(sub) is None  # stops at the home directory
    (home / "proj" / "llmshift.yaml").write_text("candidate: mock:mine\n", encoding="utf-8")
    assert config.find_config(sub) == home / "proj" / "llmshift.yaml"
    assert config.find_config(sub, stop=sub) is None  # an MCP server's project bounds it too
    (home / "proj" / "sub" / ".git").mkdir()
    assert config.find_config(sub) is None  # and the git root does


def test_on_windows_a_shared_folder_is_searched_only_up_to_its_git_root(tmp_path, monkeypatch):
    from llmshift import config

    monkeypatch.setattr(config, "_windows_rules", lambda: True)
    monkeypatch.setattr(config, "_home", lambda: (tmp_path / "somebody").resolve())  # outside home
    sub = tmp_path / "shared" / "proj" / "sub"
    sub.mkdir(parents=True)
    (tmp_path / "shared" / "llmshift.yaml").write_text("candidate: mock:planted\n", encoding="utf-8")
    assert config.find_config(sub) is None  # no git: only the folder itself
    (tmp_path / "shared" / "proj" / ".git").mkdir()
    assert config.find_config(sub) is None  # the git root is proj, below the planted file
    (tmp_path / "shared" / "proj" / "llmshift.yaml").write_text("candidate: mock:mine\n", encoding="utf-8")
    assert config.find_config(sub) == tmp_path / "shared" / "proj" / "llmshift.yaml"


def test_the_windows_rule_applies_on_windows_and_only_there(tmp_path, monkeypatch):
    # The test above swaps the switch in, and test folders sit inside the home folder, where
    # the rule never applies. So this one keeps the real switch: a switch that is always off
    # would quietly drop the protection on Windows, and every other test would still pass.
    from llmshift import config

    monkeypatch.setattr(config, "_home", lambda: (tmp_path / "somebody").resolve())  # outside home
    sub = tmp_path / "shared" / "proj" / "sub"
    sub.mkdir(parents=True)
    planted = tmp_path / "shared" / "llmshift.yaml"
    planted.write_text("candidate: mock:planted\n", encoding="utf-8")
    if sys.platform == "win32":
        assert config.find_config(sub) is None  # no git: only the folder itself
    else:
        assert config.find_config(sub) == planted  # yours, so it is read


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="POSIX file owners, as a normal user")
def test_a_config_owned_by_another_user_is_ignored_and_reported(tmp_path, monkeypatch):
    from llmshift import config

    (tmp_path / "llmshift.yaml").write_text("candidate: mock:planted\n", encoding="utf-8")
    real = os.geteuid()
    monkeypatch.setattr(config, "_home", lambda: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "geteuid", lambda: real + 4242 if real else 4242)  # as if someone else ran it
    cfg = config.load_config()
    assert cfg.path is None and any("another user owns it" in w for w in cfg.warnings)


def test_the_mcp_server_never_reads_a_config_from_above_its_project(tmp_path, monkeypatch):
    from llmshift import config, mcp

    seen = {}
    real_load = config.load_config

    def spy(path=None, *, stop=None):
        seen["stop"] = stop
        return real_load(path, stop=stop)

    monkeypatch.setattr(config, "load_config", spy)
    (tmp_path / "proj").mkdir()
    server = mcp.Server(root=tmp_path / "proj")
    server.call_tool(mcp.TOOLS["llmshift_doctor"], {})
    assert seen["stop"] == server.root


def test_the_official_openai_api_is_matched_by_host_name():
    from llmshift.providers.openai import OpenAIProvider

    def official(url: str) -> bool:
        return OpenAIProvider("m", options={"base_url": url, "api_key": "x"}).official

    assert official("https://api.openai.com/v1") and official("https://API.OpenAI.com/v1")
    assert not official("https://api.openai.com.example.net/v1") and not official("https://example.net/api.openai.com/v1")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_appended_files_are_readable_by_their_owner_only(tmp_path):
    from llmshift import fileio

    path = tmp_path / "recording.jsonl"
    fileio.append(path, b'{"x": 1}\n')
    assert path.stat().st_mode & 0o077 == 0


def test_third_party_actions_are_pinned_to_a_commit():
    # A tag such as @release/v1 can be moved to other code; a commit cannot. GitHub's own
    # actions (actions/*, github/*) are exempt, as CodeQL's rule exempts them.
    workflows = ROOT / ".github" / "workflows"
    if not workflows.is_dir():
        pytest.skip("the source archive has no workflows")
    loose = [
        f"{path.name}: {ref}"
        for path in sorted(workflows.glob("*.yml"))
        for ref in re.findall(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", path.read_text(encoding="utf-8"), re.M)
        if not ref.startswith(("actions/", "github/", "./")) and not re.search(r"@[0-9a-f]{40}$", ref)
    ]
    assert loose == []


@pytest.mark.skipif(sys.platform == "win32", reason="Windows splits the command line itself")
def test_command_with_an_unclosed_quote_is_a_config_error():
    from llmshift.providers.base import ProviderError

    with pytest.raises(ProviderError) as info:
        resolve("cmd:m", {"providers": {"cmd": {"type": "command", "command": 'python "unclosed'}}})
    assert info.value.code == "invalid_spec"
