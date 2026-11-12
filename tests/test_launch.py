"""Launch checks: scanner false alarms, pyproject settings on every Python version,
report flags, and package metadata."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import llmshift
from llmshift import cli
from llmshift.config import ConfigError, load_config
from llmshift.demo import create_project
from llmshift.scanner import scan_paths

ROOT = Path(__file__).resolve().parents[1]


def _toml(path: Path) -> dict:
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- scan


def test_model_values_in_files_without_ai_usage_must_be_model_ids(in_tmp):
    (in_tmp / "tests").mkdir()
    (in_tmp / "tests" / "test_size_finder.py").write_text(
        "from size_finder import recommend\n\n\ndef test_full_length():\n"
        '    assert recommend({"height": 180, "model": "full"}) == "L"\n'
        '    assert recommend({"model": "slim-fit-v2"}) == "M"\n',
        encoding="utf-8",
    )
    (in_tmp / "fixtures.yaml").write_text("garment:\n  model: full\n  model_name: classic-cut\n", encoding="utf-8")
    (in_tmp / "data.json").write_text(json.dumps({"model": "full", "size_model": "v3-tall"}), encoding="utf-8")
    assert scan_paths(["."]).findings == []
    # Known model IDs still count anywhere.
    (in_tmp / "notes.yaml").write_text("model: gpt-4o-2024-05-13\n", encoding="utf-8")
    assert [f.model for f in scan_paths(["."]).findings] == ["gpt-4o-2024-05-13"]


def test_custom_model_names_count_in_files_that_use_an_llm(in_tmp):
    (in_tmp / "app.py").write_text(
        'from openai import OpenAI\n\nclient = OpenAI()\n'
        'client.chat.completions.create(model="acme-support-ft-v3", messages=[])\n'
        'client.chat.completions.create(model="full", messages=[])\n',
        encoding="utf-8",
    )
    (in_tmp / "llm.yaml").write_text("model: acme-router-v2\nmax_tokens: 400\ntemperature: 0.2\n", encoding="utf-8")
    models = {f.model for f in scan_paths(["."]).findings}
    assert models == {"acme-support-ft-v3", "acme-router-v2"}  # "full" is never a model ID


# -------------------------------------------------------- pyproject config


def test_tomli_is_a_dependency_before_python_311():
    deps = _toml(ROOT / "pyproject.toml")["project"]["dependencies"]
    tomli = [d for d in deps if d.replace(" ", "").startswith("tomli")]
    assert tomli and re.search(r"python_version\s*<\s*['\"]3\.11['\"]", tomli[0])


def test_pyproject_settings_are_read(in_tmp):
    (in_tmp / "pyproject.toml").write_text('[project]\nname = "app"\n\n[tool.llmshift]\ncandidate = "mock:nextgen"\n', encoding="utf-8")
    cfg = load_config()
    assert cfg["candidate"] == "mock:nextgen" and cfg.path.name == "pyproject.toml"


def test_unreadable_pyproject_settings_are_an_error_not_ignored(in_tmp, monkeypatch):
    (in_tmp / "pyproject.toml").write_text('[tool.llmshift]\ncandidate = "mock:nextgen"\n', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "tomllib", None)  # as on Python < 3.11 without tomli
    monkeypatch.setitem(sys.modules, "tomli", None)
    with pytest.raises(ConfigError, match="tomli"):
        load_config()


def test_broken_toml_with_settings_is_reported(in_tmp):
    (in_tmp / "pyproject.toml").write_text('[tool.llmshift]\ncandidate = "unterminated\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse TOML"):
        load_config()


def test_other_projects_pyproject_files_are_left_alone(in_tmp):
    (in_tmp / "pyproject.toml").write_text('[project]\nname = "app"\ndependencies = ["llmshift"\n', encoding="utf-8")  # broken, not ours
    cfg = load_config()
    assert cfg.path is None and cfg["tests"] == ".llmshift/tests.jsonl"


# ------------------------------------------------------------ report flags


@pytest.fixture
def demo_tests(tmp_path, monkeypatch):
    root = create_project(str(tmp_path / "demo"))
    monkeypatch.chdir(root)
    assert cli.main(["snapshot", "logs"]) == 0
    return root


def test_report_flags_take_an_optional_file_name(demo_tests, capsys):
    reports = Path(".llmshift/reports")
    cli.main(["compare", "--to", "mock:legacy", "--no-report", "--markdown", "--junit"])
    assert sorted(p.suffix for p in reports.iterdir()) == [".md", ".xml"]  # only what was asked for
    md = next(reports.glob("compare-*.md"))
    assert md.read_text(encoding="utf-8").startswith("## LLMShift report")
    assert next(reports.glob("compare-*.junit.xml")).read_text(encoding="utf-8").startswith("<?xml")
    cli.main(["compare", "--to", "mock:legacy", "--no-report", "--json", "results.json", "--markdown", "pr.md"])
    assert Path("results.json").exists() and Path("pr.md").exists()
    capsys.readouterr()
    assert cli.main(["report", "results.json", "--html", "--junit"]) == 0
    assert Path("results.html").exists() and Path("results.junit.xml").exists()
    assert "wrote results.html" in capsys.readouterr().out


def test_watch_html_flag_without_a_file_name(demo_tests):
    cli.main(["watch", "--model", "mock:legacy", "--html", "--dry-run", "--limit", "3"])
    assert list(Path(".llmshift/reports").glob("watch-*.html"))


def test_help_shows_report_flags_take_files(capsys):
    with pytest.raises(SystemExit):
        cli.main(["compare", "--help"])
    assert "--markdown [FILE]" in capsys.readouterr().out


# ---------------------------------------------------------------- metadata


def test_package_metadata_is_launch_ready():
    meta = _toml(ROOT / "pyproject.toml")["project"]
    assert meta["name"] == "llmshift" and meta["version"] == llmshift.__version__
    assert meta["scripts"] == {"llmshift": "llmshift.cli:entry"}
    assert meta["authors"] == [{"name": "Chukwu Nzubechi"}]  # PyPI shows metadata to everyone: a name, no e-mail address
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "Chukwu Nzubechi" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert re.search(r'^author:\s*"Chukwu Nzubechi"', (ROOT / "action.yml").read_text(encoding="utf-8"), re.M)
    assert (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").count(f"## {llmshift.__version__} ") == 1


def test_third_party_notices_carry_the_full_licenses_into_both_packages():
    text = (ROOT / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
    assert "Copyright (c) 2023 Berri AI" in text and "Copyright (c) 2006-2016 Kirill Simonov" in text  # price list, PyYAML
    assert text.count("Permission is hereby granted, free of charge") == 2 and text.count('THE SOFTWARE IS PROVIDED "AS IS"') == 2
    targets = _toml(ROOT / "pyproject.toml")["tool"]["hatch"]["build"]["targets"]
    assert "THIRD_PARTY_NOTICES" in targets["sdist"]["include"]
    assert targets["wheel"]["force-include"] == {"THIRD_PARTY_NOTICES": "llmshift/THIRD_PARTY_NOTICES"}


def test_action_examples_use_this_version():
    """The `uses:` lines name the moving major tag (v0, which the release workflow moves) or
    this exact version: a bump must move an exact pin too, or whoever copies it keeps running
    the old action."""
    files = [ROOT / "README.md", *sorted((ROOT / "examples" / "workflows").glob("*.yml"))]
    refs = {f.name: re.findall(r"DeChuks-hub/llmshift@([\w.-]+)", f.read_text(encoding="utf-8")) for f in files}
    major = "v" + llmshift.__version__.split(".")[0]
    assert sum(map(len, refs.values())) >= 5 and {r for found in refs.values() for r in found} <= {major, f"v{llmshift.__version__}"}, refs


def test_the_mcp_registry_entry_matches_this_release():
    """The release publishes server.json to the MCP Registry, which takes a version once and
    checks that the README on PyPI names the server (mcp-name:)."""
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    assert server["version"] == llmshift.__version__ and len(server["description"]) <= 100
    package = server["packages"][0]
    assert (package["registryType"], package["identifier"], package["version"]) == ("pypi", "llmshift", llmshift.__version__)
    assert [a["value"] for a in package["packageArguments"]] == ["mcp"]  # uvx llmshift mcp
    assert f"<!-- mcp-name: {server['name']} -->" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert server["name"] == "io.github.DeChuks-hub/llmshift"  # the namespace GitHub sign-in grants this repository's owner


def test_the_mcp_registry_entry_lists_the_servers_tools():
    """Directories that read the MCP Registry showed "tools not listed": server.json has no field
    for tools. The registry keeps up to 4 KB of publisher-provided metadata, so the entry lists
    each tool's name and the first sentence of its description, taken from the server itself."""
    from llmshift import mcp

    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    provided = server["_meta"]["io.modelcontextprotocol.registry/publisher-provided"]
    expected = [{"name": t.name, "description": re.split(r"(?<=\.)\s+(?=[A-Z`])", t.description.strip(), maxsplit=1)[0]} for t in mcp.TOOLS.values()]
    assert provided["tools"] == expected  # a tool added, renamed or reworded must be listed the same way
    assert len(json.dumps(provided, ensure_ascii=False).encode("utf-8")) <= 4096  # larger and the registry refuses the release


# What a workflow can pass to and read from the action: `@v0` moves to every 0.x release,
# so a name removed or renamed here would break workflows that never chose to upgrade.
V0_INPUTS = {"command", "args", "python-version", "version", "working-directory"}
V0_OUTPUTS = {"pass-rate", "retired", "retiring", "recommended"}


def test_the_action_keeps_its_inputs_and_outputs_within_v0():
    from llmshift._vendor import yaml

    action = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))
    assert V0_INPUTS <= set(action["inputs"]) and V0_OUTPUTS <= set(action["outputs"])
    assert all(not spec.get("required") or name in V0_INPUTS for name, spec in action["inputs"].items())  # no new required input


def test_the_action_meets_the_marketplace_rules():
    """GitHub Marketplace lists the action from action.yml at the release tag: it refuses a
    description of 125 characters or more, and shows the name, icon and colour."""
    from llmshift._vendor import yaml

    action = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))
    assert action["name"] and len(action["description"]) < 125
    assert action["branding"]["icon"] and action["branding"]["color"]


def test_python_39_is_tested_on_a_linux_that_still_ships_it():
    """setup-python has no Python 3.9 for Ubuntu 26.04, which ubuntu-latest became in October 2026
    (the job fails before any test runs). The 3.9 jobs run on ubuntu-24.04, so every push and every
    release still tests the oldest supported Python on Linux."""
    from llmshift._vendor import yaml

    for name in ("ci.yml", "release.yml"):
        path = ROOT / ".github" / "workflows" / name
        if not path.exists():
            pytest.skip("the workflows are not in the source archive")
        matrix = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["test"]["strategy"]["matrix"]
        jobs = {(os_name, py) for os_name in matrix["os"] for py in matrix["python-version"]}
        jobs -= {(e["os"], e["python-version"]) for e in matrix.get("exclude", [])}
        jobs |= {(i["os"], i["python-version"]) for i in matrix.get("include", [])}
        assert ("ubuntu-latest", "3.9") not in jobs and ("ubuntu-24.04", "3.9") in jobs, name


# The first major release of each of GitHub's own actions that runs on Node.js 24 (`runs.using`
# in its action.yml). An older one makes every run warn that Node.js 20 is deprecated.
NODE24 = {
    "actions/checkout": 5, "actions/setup-python": 6, "actions/cache": 5, "actions/upload-artifact": 6,
    "actions/download-artifact": 7, "actions/configure-pages": 6, "actions/upload-pages-artifact": 5, "actions/deploy-pages": 5,
}


def test_github_actions_run_on_node24(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "--github"])
    files = [ROOT / "action.yml", *sorted((ROOT / "examples" / "workflows").glob("*.yml")), *sorted((tmp_path / ".github" / "workflows").glob("*.yml"))]
    files += sorted((ROOT / ".github" / "workflows").glob("*.yml"))  # not in the source archive
    used = [(f.name, action, int(major)) for f in files for action, major in re.findall(r"uses:\s*(actions/[\w-]+)@v(\d+)", f.read_text(encoding="utf-8"))]
    assert len(used) >= 10
    for name, action, major in used:
        assert action in NODE24, f"{name}: add {action} to NODE24 with its first Node 24 major"
        assert major >= NODE24[action], f"{name}: {action}@v{major} runs on Node 20; use v{NODE24[action]} or later"


def test_no_leftovers_of_the_old_name_or_placeholders():
    shipped = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "action.yml", ROOT / "pyproject.toml", ROOT / "LICENSE"]
    shipped += [p for p in (ROOT / "src").rglob("*") if p.suffix in (".py", ".json")]
    shipped += [p for p in (ROOT / "examples").rglob("*") if p.is_file()]
    for path in shipped:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"model ?shift", text, re.I), path
        assert "your-org" not in text and "contributors" not in text.lower(), path


def test_readme_links_work_on_pypi():
    """The README is also the PyPI page, where relative links lead nowhere, so links
    are absolute and the ones into this repository name files that exist."""
    links = re.findall(r"\]\(([^)\s]+)\)", (ROOT / "README.md").read_text(encoding="utf-8"))
    assert links
    for link in links:
        assert re.match(r"(https?://|mailto:|#)", link), f"relative link {link!r} breaks on PyPI"
        m = re.match(r"https://github\.com/DeChuks-hub/llmshift/(?:blob|tree)/main/(.+)", link)
        if m:
            assert (ROOT / m.group(1)).exists(), link


def _subparsers(parser):
    import argparse

    return next((a.choices for a in parser._actions if isinstance(a, argparse._SubParsersAction)), {})


def test_every_command_and_flag_in_the_docs_exists():
    """Each `llmshift <command> --flag` written in the README, CHANGELOG or example
    workflows must be accepted by that command's parser."""
    docs = [ROOT / "README.md", ROOT / "CHANGELOG.md", *sorted((ROOT / "examples" / "workflows").glob("*.yml"))]
    commands = _subparsers(cli.build_parser())
    checked = 0
    for path in docs:
        for line in path.read_text(encoding="utf-8").splitlines():
            for m in re.finditer(r"(?<![\w/.-])llmshift ([a-z][\w-]*)((?: [^`|#;&]*)?)", line):
                name, rest = m.group(1), m.group(2)
                assert name in commands, f"{path.name}: no command `llmshift {name}` (in {line.strip()!r})"
                parser = commands[name]
                words = rest.split()
                sub = _subparsers(parser)
                if sub and words and words[0] in sub:
                    parser = sub[words.pop(0)]
                known = {opt for a in parser._actions for opt in a.option_strings}
                for word in words:
                    flag = word.split("=", 1)[0].rstrip(",.)")
                    if flag.startswith("--"):
                        assert flag in known, f"{path.name}: `llmshift {name}` has no {flag} (in {line.strip()!r})"
                        checked += 1
    assert checked > 40  # the docs really were checked


@pytest.mark.parametrize(
    "argv",
    [
        ["scan", "--warn-days", "-5"],
        ["models", "gpt-4o", "--warn-days", "-1"],
        ["models", "--retiring-within", "soon"],
        ["upgrade", "--auto", "--warn-days", "-5"],
        ["digest", "--within", "-3"],
        ["calendar", "--out", "site", "--days-back", "-1"],
    ],
)
def test_day_counts_must_not_be_negative(argv, in_tmp, capsys):
    try:
        code = cli.main(argv)
    except SystemExit as exc:  # argparse rejects the value itself
        code = exc.code
    assert code == 2
    err = capsys.readouterr().err
    assert "zero or more" in err or "whole number of days" in err


NO_PYYAML = """
import sys
sys.modules["yaml"] = None  # any `import yaml` now fails: llmshift must use its bundled copy
from pathlib import Path
from llmshift import cli
from llmshift.config import load_config
from llmshift.testset import load_tests, save_tests
root = Path(sys.argv[1])
(root / "llmshift.yaml").write_text("candidate: mock:nextgen\\nmin_pass_rate: 0.5\\n", encoding="utf-8")
assert load_config(str(root / "llmshift.yaml"))["candidate"] == "mock:nextgen"
(root / "t.yaml").write_text("cases:\\n  - prompt: hi\\n    baseline: hello\\n", encoding="utf-8")
cases = load_tests(root / "t.yaml")
save_tests(cases, root / "copy.yaml")
assert len(load_tests(root / "copy.yaml")) == 1
assert cli.main(["init", "--github"]) == 0
print("ok")
"""


def test_llmshift_never_needs_an_installed_pyyaml(tmp_path):
    proc = subprocess.run([sys.executable, "-c", NO_PYYAML, str(tmp_path)], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")))
    assert proc.returncode == 0 and "ok" in proc.stdout, proc.stderr[-2000:]
    meta = _toml(ROOT / "pyproject.toml")["project"]
    assert not [d for d in meta["dependencies"] if d.lower().startswith("pyyaml")]
