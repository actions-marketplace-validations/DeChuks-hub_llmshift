"""The daily lifecycle-data update: reading the providers' deprecation tables and
merging them into models.json without ever losing curated data."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TODAY = "2026-10-22"


@pytest.fixture
def updater(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("update_registry", ROOT / "scripts" / "update_registry.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "update_registry", module)  # dataclasses look their module up there
    spec.loader.exec_module(module)
    models = tmp_path / "models.json"
    models.write_bytes((ROOT / "tests" / "data" / "models-2026-10-21.json").read_bytes())  # frozen: the shipped copy moves daily
    monkeypatch.setattr(module, "MODELS", models)
    monkeypatch.setattr(module, "STATE", tmp_path / "sources.json")
    monkeypatch.setattr(module._watcher(), "LINES", tmp_path / "reviewed-lines.json")
    return module


def _table(header, rows, heading=""):
    cells = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<h3>{heading}</h3><table><tr>{cells}</tr>{body}</table>"


OPENAI = "<html><body><main>" + _table(
    ["Shutdown date", "Model / system", "Recommended replacement"],
    [["January 14, 2027", "GPT-4o (gpt-4o-2024-08-06, gpt-4o-2024-11-20)", "gpt-6-luna"],
     ["March 3, 2027", "gpt-4.9-preview-2026", "<a>gpt-6-sol</a>or a newer model"],  # a new deprecation
     ["Jan 6, 2027", "tts-1", "gpt-realtime-2.1-mini"],
     ["Dec 1, 2026", "gpt-image-1.5", "gpt-image-2.5-sunburst or gpt-image-2.5-flare"],
     ["2024-10-28", "babbage-002 fine-tuning", "gpt-4o-mini"],  # an older fine-tuning row
     ["Oct 1, 2026", "gpt-5.4-cyber", "The most capable cyber model available to you."]],
    heading="2026-10-14: GPT-4o and GPT-4o mini",
) + _table(["Date", "Update"], [["June 3, 2026", "Deprecation announced for the Evals platform."]]) + "</main></body></html>"

ANTHROPIC = "<html><body>" + _table(
    ["API model name", "Current state", "Deprecated", "Tentative retirement date"],
    [["claude-fable-5-1", "Active", "N/A", "Not sooner than September 1, 2027"],
     ["claude-sonnet-4-5-20250929", "Deprecated", "September 30, 2026", "December 15, 2026"],  # date moved
     ["claude-3-5-haiku-20241022", "Retired", "October 28, 2025", "February 19, 2026"],
     ["claude-opus-4-8", "Active", "N/A", "Not sooner than May 1, 2027"],
     ["claude-haiku-4-5-20251001", "Active", "N/A", "Not sooner than October 15, 2026"]],
) + "</body></html>"


# Google's Gemini API docs moved to AI Studio in November 2026: the page is a JavaScript app, and
# its text is Markdown inside <noscript>, with Markdoc tables (one "* cell" line per cell, rows
# split by "---"). A cut-down copy of the real page.
GEMINI_MD = """# Gemini deprecations
The shutdown dates in the table are the "earliest possible dates".
## Gemini 3 models {% id="gemini-3-models" %}
{% table %}
* Model
* Release date
* Shutdown date
* Recommended replacement
---
* `gemini-3.8-flash`
* September 2, 2026
* No shutdown date announced
*
---
* `gemini-3.7-flash`
* August 13, 2026
* N/A
* `gemini-3.8-flash` (`gemini-3.7-flash` is no longer available in Google AI Studio; the endpoint
  is not shutting down and automatically routes API requests to `gemini-3.8-flash`)
---
* `gemini-3.1-flash-lite`
* May 7, 2026
* May 7, 2027
* `gemini-3.5-flash-lite`
{% /table %}
### Preview models {% id="gemini-3-preview-models" %}
{% table %}
* Model
* Release date
* Shutdown date
* Recommended replacement
---
* `gemini-3.1-flash-tts-preview`
* February 26, 2026
* November 17, 2026
* `gemini-3.8-flash-tts` or `gemini-3.8-flash-lite-tts`
---
* `gemini-3-pro-preview`
* November 18, 2025
* March 9, 2026
* `gemini-3.1-pro-preview`
---
* `gemini-omni-flash-preview`
* June 30, 2026
* October 22, 2026
* `gemini-omni-1.2-flash` or the GA models on the [Agent Platform](https://docs.cloud.google.com/models/veo/3-1-generate)
{% /table %}
"""
GEMINI = (
    '<!doctype html><html lang="en-US"><head><base href="https://aistudio.google.com/"><title>Google AI Studio</title>'
    '<script nonce="n1">var build = "W316";</script></head><body><noscript><pre>' + GEMINI_MD.replace('"', "&quot;")
    + '</pre></noscript><ms-app></ms-app><script src="https://www.gstatic.com/app.js"></script></body></html>'
)


def test_markdown_tables_in_a_javascript_page_become_rows(updater):
    rows = updater.parse_page(GEMINI, "google", "https://x/google", ("model",), ("shutdown",), ("replacement",))
    by = {r.id: r for r in rows}
    assert set(by) == {"gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.1-flash-lite", "gemini-3.1-flash-tts-preview", "gemini-3-pro-preview",
                       "gemini-omni-flash-preview"}  # an id written as code counts without a version number
    omni = by["gemini-omni-flash-preview"]
    assert omni.retires == "2026-10-22" and omni.replacement == "gemini-omni-1.2-flash"  # not "3-1-generate" from the link
    tts = by["gemini-3.1-flash-tts-preview"]
    assert tts.retires == "2026-11-17" and tts.replacement == "gemini-3.8-flash-tts"  # "x or y": the first
    assert by["gemini-3.7-flash"].retires is None and by["gemini-3.7-flash"].replacement == "gemini-3.8-flash"  # not the id in the note
    assert by["gemini-3.8-flash"].retires is None and by["gemini-3.8-flash"].replacement is None  # "No shutdown date announced"
    assert by["gemini-3-pro-preview"].retires == "2026-03-09" and by["gemini-3.1-flash-lite"].retires == "2027-05-07"


def test_main_reads_the_gemini_page_again(updater, tmp_path, capsys):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "google.html").write_text(GEMINI, encoding="utf-8")
    assert updater.main(["--pages", str(pages), "--today", TODAY]) in (0, 4)
    assert "google: 6 rows" in capsys.readouterr().out


def test_tables_become_rows(updater):
    rows = updater.parse_page(OPENAI, "openai", "https://x/openai", ("model",), ("shutdown",), ("replacement", "substitute"))
    by = {r.id: r for r in rows}
    assert set(by) == {"gpt-4o-2024-08-06", "gpt-4o-2024-11-20", "gpt-4.9-preview-2026", "tts-1", "gpt-image-1.5", "babbage-002", "gpt-5.4-cyber"}
    assert by["gpt-4o-2024-08-06"].retires == "2027-01-14" and by["gpt-4o-2024-08-06"].deprecated == "2026-10-14"  # from the heading
    assert by["gpt-4.9-preview-2026"].replacement == "gpt-6-sol"  # "<a>gpt-6-sol</a>or" stays two words
    assert by["gpt-image-1.5"].replacement == "gpt-image-2.5-sunburst" and by["gpt-5.4-cyber"].replacement is None
    rows = updater.parse_page(ANTHROPIC, "anthropic", "https://x/anthropic", ("model",), ("retirement",), ("replacement",))
    by = {r.id: r for r in rows}
    assert by["claude-fable-5-1"].retires is None and by["claude-fable-5-1"].deprecated is None  # "Not sooner than" is no date
    assert by["claude-sonnet-4-5-20250929"].retires == "2026-12-15" and by["claude-sonnet-4-5-20250929"].deprecated == "2026-09-30"
    assert updater.parse_date("October 13, 2027 or later") is None and updater.parse_date("Feb 30, 2027") is None
    assert updater.model_ids("fine-tuning of gpt-3.5-turbo-0125 | gpt-3.5-turbo") == ["gpt-3.5-turbo-0125", "gpt-3.5-turbo"]


def test_merge_is_conservative(updater):
    data = json.loads(updater.MODELS.read_text(encoding="utf-8"))
    rows = updater.parse_page(OPENAI, "openai", "https://x/openai", ("model",), ("shutdown",), ("replacement", "substitute"))
    rows += updater.parse_page(ANTHROPIC, "anthropic", "https://x/anthropic", ("model",), ("retirement",), ("replacement",))
    rows.append(updater.Row("openai", "gpt-0-legacy-2023", "2024-01-01", None, None, "https://x/openai"))  # gone long ago
    by = lambda d: {(m["provider"], m["id"]): m for m in d["models"]}  # noqa: E731
    old = by(json.loads(json.dumps(data)))
    changes = updater.merge(data, rows, TODAY)
    new = by(data)
    assert new[("openai", "gpt-4.9-preview-2026")]["retires"] == "2027-03-03" and new[("openai", "gpt-4.9-preview-2026")]["status"] == "deprecated"
    assert ("openai", "gpt-0-legacy-2023") not in new  # past shutdowns are not added
    assert new[("anthropic", "claude-sonnet-4-5-20250929")]["retires"] == "2026-12-15"  # a moved date is followed
    assert new[("anthropic", "claude-3-5-haiku-20241022")] == old[("anthropic", "claude-3-5-haiku-20241022")]  # retired: untouched
    assert new[("anthropic", "claude-fable-5-1")] == old[("anthropic", "claude-fable-5-1")]  # "not sooner than": untouched
    assert new[("openai", "babbage-002")] == old[("openai", "babbage-002")]  # old fine-tuning row: history stays
    assert len(data["models"]) == len(old) + 1 and all(k in new for k in old)  # nothing removed
    assert len(changes) == 2 and any("added openai/gpt-4.9-preview-2026" in c for c in changes)


def test_layout_round_trips(updater):
    for path in (updater.MODELS, ROOT / "src" / "llmshift" / "data" / "models.json"):
        raw = path.read_bytes()
        assert updater.dump(json.loads(raw)) == raw, path  # automatic updates make one-line diffs


def test_the_shipped_data_is_valid_today(monkeypatch):
    # The other tests read a frozen copy; this one checks the data users get, on the real date.
    from llmshift import registry

    monkeypatch.delenv("LLMSHIFT_BUNDLED_REGISTRY")
    monkeypatch.delenv("LLMSHIFT_TODAY")
    assert registry.registry_problems(registry.bundled_registry(), min_models=600) == []


def test_main_writes_only_valid_reviewed_sized_updates(updater, tmp_path, capsys):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "openai.html").write_text(OPENAI, encoding="utf-8")
    (pages / "anthropic.html").write_text(ANTHROPIC, encoding="utf-8")
    (pages / "google.html").write_text("<html><p>moved</p></html>", encoding="utf-8")  # unreadable: left to the source watch
    before = updater.MODELS.read_bytes()
    assert updater.main(["--pages", str(pages), "--today", TODAY]) == 0 and updater.MODELS.read_bytes() == before  # dry run
    out = capsys.readouterr().out
    assert "2 change(s)" in out and "google: only 0 rows" in out
    assert updater.main(["--pages", str(pages), "--today", TODAY, "--max-changes", "1", "--write"]) == 3
    assert updater.MODELS.read_bytes() == before  # too many changes at once: nothing written
    summary = tmp_path / "summary.md"
    assert updater.main(["--pages", str(pages), "--today", TODAY, "--write", "--summary", str(summary)]) == 0
    data = json.loads(updater.MODELS.read_text(encoding="utf-8"))
    assert data["updated"] == TODAY and "added openai/gpt-4.9-preview-2026" in summary.read_text(encoding="utf-8")
    state = json.loads((tmp_path / "sources.json").read_text(encoding="utf-8"))
    assert set(state["pages"]) == {"https://developers.openai.com/api/docs/deprecations", "https://platform.claude.com/docs/en/about-claude/model-deprecations"}
    reviewed = json.loads((tmp_path / "reviewed-lines.json").read_text(encoding="utf-8"))  # the next issue diffs against these
    assert set(reviewed) == set(state["pages"]) and "Jan 6, 2027 | tts-1 | gpt-realtime-2.1-mini" in reviewed["https://developers.openai.com/api/docs/deprecations"]
    capsys.readouterr()
    assert updater.main(["--pages", str(pages), "--today", TODAY]) == 0 and "No changes" in capsys.readouterr().out  # idempotent


def test_an_added_field_keeps_the_usual_field_order(updater):
    data = {"updated": "2026-10-21", "models": [{"id": "tts-1", "provider": "openai", "status": "deprecated", "retires": "2027-01-06", "source": "s"}]}
    rows = [updater.Row("openai", "tts-1", "2027-01-06", "gpt-realtime-2.1-mini", None, "s")]
    assert updater.merge(data, rows, TODAY) == ["openai/tts-1: replacement none -> gpt-realtime-2.1-mini"]
    assert list(data["models"][0]) == ["id", "provider", "status", "retires", "replacement", "source"]


def test_a_newly_scheduled_date_also_keeps_the_field_order(updater):
    data = {"updated": "2026-10-21", "models": [{"id": "gemini-3.8-flash-tts", "provider": "google", "status": "active", "source": "s", "note": "rel 2026-09-22"}]}
    rows = [updater.Row("google", "gemini-3.8-flash-tts", "2027-09-22", None, None, "s")]
    assert updater.merge(data, rows, TODAY) == ["google/gemini-3.8-flash-tts: retires not scheduled -> 2027-09-22"]
    assert data["models"][0] == {"id": "gemini-3.8-flash-tts", "provider": "google", "status": "deprecated", "retires": "2027-09-22", "source": "s", "note": "rel 2026-09-22"}
    assert list(data["models"][0]) == ["id", "provider", "status", "retires", "source", "note"]


@pytest.mark.parametrize("text", ["No earlier than January 14, 2027", "Earliest: 2027-01-14", "not before Jan 14, 2027", "2027-01-14 (tentative)"])
def test_vague_dates_are_not_read_as_shutdown_days(updater, text):
    assert updater.parse_date(text) is None


def test_earlier_or_past_dates_wait_for_a_person(updater):
    data = {"updated": "2026-10-21", "models": [
        {"id": "a-model", "provider": "openai", "status": "deprecated", "retires": "2027-06-01", "source": "s"},
        {"id": "b-model", "provider": "openai", "status": "deprecated", "retires": "2027-06-01", "source": "s"},
        {"id": "c-model", "provider": "openai", "status": "deprecated", "retires": "2027-06-01", "source": "s"},
    ]}
    rows = [updater.Row("openai", "a-model", "2027-03-01", None, None, "s"),  # moved earlier
            updater.Row("openai", "b-model", "2026-10-01", None, None, "s"),  # already past
            updater.Row("openai", "c-model", "2027-09-01", None, None, "s")]  # moved later: applied
    review: list = []
    assert updater.merge(data, rows, TODAY, review) == ["openai/c-model: retires 2027-06-01 -> 2027-09-01"]
    assert [m["retires"] for m in data["models"]] == ["2027-06-01", "2027-06-01", "2027-09-01"]
    assert len(review) == 2 and "a-model" in review[0] and "b-model" in review[1]


def test_saved_pages_that_are_not_utf8_are_read(updater, tmp_path, capsys):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "openai.html").write_bytes(OPENAI.encode("utf-8") + b"\xff\xfe stray bytes")
    assert updater.main(["--pages", str(pages), "--today", TODAY]) in (0, 3, 4)
    assert "Traceback" not in capsys.readouterr().err
