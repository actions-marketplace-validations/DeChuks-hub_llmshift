"""The live retirement feed: `llmshift update`, `digest` and `calendar`."""

from __future__ import annotations

import copy
import html
import json
import re

import pytest

from llmshift import cli, feed, pricing, registry
from llmshift.registry import Registry


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A private LLMSHIFT_HOME; the price table cache is reset around each test."""
    path = tmp_path / "home"
    monkeypatch.setenv("LLMSHIFT_HOME", str(path))
    monkeypatch.setattr(cli, "EXTENSIONS", ("feed",))
    pricing.reload()
    yield path
    pricing.reload()


def newer_registry(**changes) -> dict:
    data = copy.deepcopy(registry.bundled_registry())
    data["updated"] = changes.pop("updated", "2026-10-21")
    data.update(changes)
    return data


def serve(server, path: str, payload) -> str:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    server.route("GET", path, lambda r: (200, body))
    return server.url + path


# ------------------------------------------------------------------ update


def test_update_downloads_newer_data_and_registry_uses_it(home, server, capsys):
    data = newer_registry(updated="2026-10-21")
    old = next(e for e in data["models"] if e["id"] == "gpt-4o-mini")
    old["retires"] = "2027-03-01"
    data["models"].append({"id": "acme-model-1", "provider": "openai", "status": "deprecated", "retires": "2026-12-01", "source": "https://example.com"})
    url = serve(server, "/stale.json", data)
    assert cli.main(["update", "--url", url]) == 0  # same date as the bundled data: nothing to do
    assert "up to date" in capsys.readouterr().out and not (home / "models.json").exists()

    data["updated"] = "2026-10-22"
    url = serve(server, "/models.json", data)
    assert cli.main(["update", "--url", url, "--check"]) == 0
    assert "newer lifecycle data is available: 2026-10-22" in capsys.readouterr().out and not (home / "models.json").exists()

    assert cli.main(["update", "--url", url]) == 0
    out = capsys.readouterr().out
    assert "lifecycle data updated: 2026-10-21 -> 2026-10-22" in out and "1 new, 1 changed" in out
    assert "new: acme-model-1 (openai), retires 2026-12-01" in out and "retires 2027-01-14 -> 2027-03-01" in out
    reg = Registry.load()
    assert reg.updated == "2026-10-22" and reg.origin == str(home / "models.json")
    assert reg.lookup("acme-model-1").status == "retiring" and str(reg.lookup("gpt-4o-mini").retires) == "2027-03-01"
    assert cli.main(["update", "--url", url]) == 0 and "up to date (2026-10-22)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(models=d["models"][:10]), "only 10 models"),
        (lambda d: d.update(schema=2), "needs a newer llmshift"),
        (lambda d: d.update(updated="2030-01-01"), "in the future"),
        (lambda d: d.update(updated="soon"), "invalid 'updated'"),
        (lambda d: d["models"][3].update(status="vanished"), "unknown status"),
        (lambda d: d["models"][5].update(retires="31/12/2026"), "not a date"),
        (lambda d: d["models"].append("not an entry"), "not an object"),
        (lambda d: d.pop("models"), "'models' must be a list"),
    ],
)
def test_update_rejects_bad_data(home, server, capsys, mutate, message):
    data = newer_registry(updated="2026-10-22")
    mutate(data)
    url = serve(server, "/bad.json", data)
    assert cli.main(["update", "--url", url]) == 1
    assert message in capsys.readouterr().err
    assert not (home / "models.json").exists() and Registry.load().origin == "bundled"


def test_update_handles_download_problems(home, server, capsys):
    url = serve(server, "/garbage.json", b"<html>not json</html>")
    assert cli.main(["update", "--url", url]) == 1 and "did not return valid JSON" in capsys.readouterr().err
    server.route("GET", "/missing.json", lambda r: (404, {"error": {"message": "Not Found"}}))
    assert cli.main(["update", "--url", server.url + "/missing.json"]) == 1 and "could not download" in capsys.readouterr().err
    local = home.parent / "local.json"
    local.write_text(json.dumps(newer_registry(updated="2026-10-22")), encoding="utf-8")
    assert cli.main(["update", "--url", str(local)]) == 0 and (home / "models.json").exists()


def test_registry_ignores_broken_or_older_caches(home):
    home.mkdir(parents=True)
    cache = home / "models.json"
    cache.write_bytes(b"\x00\x01 corrupt")
    assert Registry.load().origin == "bundled"
    cache.write_text(json.dumps(newer_registry(updated="2026-01-01")), encoding="utf-8")
    assert Registry.load().origin == "bundled"  # older than the bundled data
    broken = newer_registry(updated="2026-10-22")
    broken["models"][0] = {"id": ""}
    cache.write_text(json.dumps(broken), encoding="utf-8")
    assert Registry.load().origin == "bundled"
    cache.write_text(json.dumps(newer_registry(updated="2026-10-30")), encoding="utf-8")
    assert Registry.load().origin == "bundled"  # dated in the future (today is 2026-10-21 in tests)
    cache.write_text("\ufeff" + json.dumps(newer_registry(updated="2026-10-22")), encoding="utf-8")
    assert Registry.load().origin == str(cache)


def litellm_table(gpt4o_input: float) -> dict:
    table = {f"vendor/model-{i}": {"mode": "chat", "input_cost_per_token": i * 1e-8, "output_cost_per_token": i * 2e-8} for i in range(600)}
    table["gpt-4o"] = {"mode": "chat", "input_cost_per_token": gpt4o_input, "output_cost_per_token": 1e-05}
    table["dall-e-3"] = {"mode": "image_generation", "input_cost_per_token": 1.0}
    table["sample_spec"] = {"mode": "chat", "input_cost_per_token": 0.0}
    return table


def test_update_prices(home, server, monkeypatch, capsys):
    monkeypatch.setenv("LLMSHIFT_TODAY", "2026-11-02")
    reg_url = serve(server, "/models.json", newer_registry(updated="2026-10-21"))
    prices_url = serve(server, "/prices.json", litellm_table(3e-06))
    assert pricing.price_for("openai:gpt-4o").input == 2.5
    assert cli.main(["update", "--url", reg_url, "--prices", "--prices-url", prices_url, "--check"]) == 0
    assert "newer prices are available" in capsys.readouterr().out and pricing.table_origin() == "bundled"
    assert cli.main(["update", "--url", reg_url, "--prices", "--prices-url", prices_url]) == 0
    assert "prices updated: 2026-10-22 -> 2026-11-02" in capsys.readouterr().out
    price = pricing.price_for("openai:gpt-4o")
    assert price.input == 3.0 and "2026-11-02" in price.source and pricing.table_origin() == str(home / "pricing.json")
    assert pricing.price_for("dall-e-3") is None  # only token-priced chat/embedding models are kept
    assert cli.main(["update", "--url", reg_url, "--prices", "--prices-url", prices_url]) == 0
    assert "prices are up to date" in capsys.readouterr().out
    few = {"gpt-4o": {"mode": "chat", "input_cost_per_token": 1e-6, "output_cost_per_token": 1e-6}}
    assert cli.main(["update", "--url", reg_url, "--prices", "--prices-url", serve(server, "/few.json", few)]) == 1
    assert "only 1 prices" in capsys.readouterr().err


def test_corrupt_or_stale_price_cache_is_ignored(home, monkeypatch):
    home.mkdir(parents=True)
    (home / "pricing.json").write_text('{"fetched": "2026-11-01", "prices": {"gpt-4o": ["x", 1]}}', encoding="utf-8")
    pricing.reload()
    assert pricing.table_origin() == "bundled" and pricing.price_for("openai:gpt-4o").input == 2.5
    (home / "pricing.json").write_text(json.dumps({"fetched": "2026-01-01", "prices": {"gpt-4o": [9.0, 9.0]}}), encoding="utf-8")
    pricing.reload()
    assert pricing.table_origin() == "bundled"


def test_update_json_output_and_unwritable_home(home, server, monkeypatch, capsys, tmp_path):
    url = serve(server, "/models.json", newer_registry(updated="2026-10-22"))
    blocker = tmp_path / "file-not-folder"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("LLMSHIFT_HOME", str(blocker / "sub"))
    assert cli.main(["update", "--url", url]) == 1 and "cannot write" in capsys.readouterr().err
    monkeypatch.setenv("LLMSHIFT_HOME", str(home))
    assert cli.main(["update", "--url", url, "--format", "json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result[0]["status"] == "updated" and result[0]["latest"] == "2026-10-22"


# ------------------------------------------------------------------ digest


@pytest.fixture
def repos(tmp_path):
    a = tmp_path / "shop"
    (a / "app").mkdir(parents=True)
    (a / "app" / "triage.py").write_text('MODEL = "gpt-4o-2024-05-13"\nclient.chat.completions.create(model=MODEL)\n', encoding="utf-8")
    (a / "app" / "summary.py").write_text('client.messages.create(model="claude-3-5-haiku-20241022")\n', encoding="utf-8")
    b = tmp_path / "support"
    b.mkdir()
    (b / "bot.py").write_text('client.chat.completions.create(model="gpt-4o-2024-05-13")\nFALLBACK = "gpt-4o-mini"\n', encoding="utf-8")
    return a, b


def test_digest_message(home, repos, tmp_path, capsys):
    a, b = repos
    out = tmp_path / "digest.json"
    assert cli.main(["digest", str(a), str(b), "--within", "30", "--alert", f"file:{out}"]) == 0
    printed = capsys.readouterr().out
    assert "2 models in your repos need attention: 1 retired, 1 retiring within 30 days" in printed
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["severity"] == "critical" and payload["lines"][0].startswith("claude-3-5-haiku-20241022 (anthropic): retired on 2026-02-19")
    gpt = payload["lines"][1]
    assert gpt.startswith("gpt-4o-2024-05-13 (openai): retires 2026-10-23 (in 2 days); replacement: gpt-5.6-sol")
    assert "bot.py:1" in gpt and "triage.py:1" in gpt  # merged across both repositories
    assert "gpt-4o-mini" not in json.dumps(payload)  # retires in 85 days: outside the window
    assert cli.main(["digest", str(a), str(b), "--within", "120", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["counts"] == {"retired": 1, "retiring": 2, "deprecated": 0}


def test_digest_quiet_dry_run_and_fail_on(home, repos, tmp_path, capsys):
    a, _ = repos
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "x.py").write_text("print('no models')\n", encoding="utf-8")
    out = tmp_path / "quiet.json"
    assert cli.main(["digest", str(clean), "--alert", f"file:{out}"]) == 0
    assert "No models in your code" in capsys.readouterr().out and not out.exists()
    assert cli.main(["digest", str(clean), "--alert", f"file:{out}", "--always"]) == 0 and out.exists()
    out.unlink()
    assert cli.main(["digest", str(a), "--alert", f"file:{out}", "--dry-run"]) == 0 and not out.exists()
    assert cli.main(["digest", str(a), "--fail-on", "retired"]) == 1
    assert cli.main(["digest", str(clean), "--fail-on", "retiring"]) == 0
    assert cli.main(["digest", str(tmp_path / "nope")]) == 2
    assert cli.main(["digest", str(a), "--format", "markdown"]) == 0 and capsys.readouterr().out.count("## ") >= 1


def test_digest_reports_failed_delivery(home, repos, capsys):
    a, _ = repos
    assert cli.main(["digest", str(a), "--alert", "slack:http://127.0.0.1:9/hook"]) == 1
    assert "digest via slack failed" in capsys.readouterr().err


# ---------------------------------------------------------------- calendar


def unfold(ics: str) -> list[str]:
    return re.sub(r"\r\n[ \t]", "", ics).split("\r\n")


def unescape(value: str) -> str:
    return re.sub(r"\\([\\;,nN])", lambda m: "\n" if m.group(1) in "nN" else m.group(1), value)


def test_calendar_site_and_ics(home, tmp_path, capsys):
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps([{"id": "acme-chat, v2; beta", "provider": "openai", "status": "deprecated", "retires": "2026-11-15",
                                  "replacement": "acme-chat-3", "note": "Long note " + "with many words, commas; and semicolons " * 4,
                                  "source": "https://example.com/deprecations#acme"}]), encoding="utf-8")
    site = tmp_path / "site"
    assert cli.main(["calendar", "--out", str(site), "--registry", str(extra)]) == 0
    assert "upcoming" in capsys.readouterr().out
    page = (site / "index.html").read_text(encoding="utf-8")
    assert "gpt-4o-2024-05-13" in page and "acme-chat, v2; beta" in page and "<script src" not in page
    assert 'data-date="2026-10-23"' in page and "Retired in the last 90 days" in page
    raw = (site / "retirements.ics").read_bytes()
    assert raw.endswith(b"END:VCALENDAR\r\n") and b"\n" not in raw.replace(b"\r\n", b"")
    physical = raw.split(b"\r\n")[:-1]
    assert all(len(line) <= 75 for line in physical)
    assert any(line.startswith(b" ") for line in physical)  # long lines were folded
    lines = unfold(raw.decode("utf-8"))
    events = [i for i, line in enumerate(lines) if line == "BEGIN:VEVENT"]
    data = json.loads((site / "retirements.json").read_text(encoding="utf-8"))
    assert len(events) == len(data["upcoming"]) + len(data["recent"])
    uids = [line for line in lines if line.startswith("UID:")]
    assert len(uids) == len(set(uids)) == len(events)
    summary = next(line for line in lines if line.startswith("SUMMARY:acme"))
    assert summary == "SUMMARY:acme-chat\\, v2\\; beta retires (openai)" and unescape(summary[8:]) == "acme-chat, v2; beta retires (openai)"
    block = lines[lines.index(summary) - 5: lines.index(summary) + 6]
    assert "DTSTART;VALUE=DATE:20261115" in block and "DTEND;VALUE=DATE:20261116" in block
    description = next(line for line in block if line.startswith("DESCRIPTION:"))
    assert "Replacement: acme-chat-3" in unescape(description) and "\\," in description and "\\n" in description
    assert "URL:https://example.com/deprecations#acme" in block
    assert all(re.fullmatch(r"DTSTAMP:\d{8}T\d{6}Z", line) for line in lines if line.startswith("DTSTAMP"))


def test_fold_ics_line_respects_multibyte_characters():
    line = "DESCRIPTION:" + "naïve 日本語 \U0001F600 " * 20
    folded = feed.fold_ics_line(line)
    parts = folded.split("\r\n")
    assert all(len(p.encode("utf-8")) <= 75 for p in parts)
    assert all(p.startswith(" ") for p in parts[1:])
    assert "".join([parts[0]] + [p[1:] for p in parts[1:]]) == line
    assert feed.fold_ics_line("SHORT:x") == "SHORT:x"


def test_calendar_rejects_a_file_as_output(home, tmp_path):
    target = tmp_path / "taken"
    target.write_text("x", encoding="utf-8")
    assert cli.main(["calendar", "--out", str(target)]) == 2
    assert cli.main(["calendar", "--out", str(tmp_path / "ok"), "--days-back", "-1"]) == 2


def test_calendar_counts_visits_with_goatcounter_only_when_asked(home, tmp_path):
    plain, counted, hosted = tmp_path / "plain", tmp_path / "counted", tmp_path / "hosted"
    assert cli.main(["calendar", "--out", str(plain), "--goatcounter", ""]) == 0
    assert "goatcounter" not in (plain / "index.html").read_text(encoding="utf-8").lower()
    assert cli.main(["calendar", "--out", str(counted), "--goatcounter", " LLMShift "]) == 0
    page = (counted / "index.html").read_text(encoding="utf-8")
    script = re.search(r'<script data-goatcounter="([^"]*)" async src="([^"]*)" crossorigin="anonymous" '
                       r'integrity="sha384-[A-Za-z0-9+/]{64}"></script>\n</body>', page)
    assert script and script.groups() == ("https://llmshift.goatcounter.com/count", "https://gc.zgo.at/count.v5.js")
    assert "which sets no cookies" in page
    assert cli.main(["calendar", "--out", str(hosted), "--goatcounter", "https://stats.example.org/count"]) == 0
    assert 'data-goatcounter="https://stats.example.org/count"' in (hosted / "index.html").read_text(encoding="utf-8")
    for bad in ('x" onload="alert(1)', "http://stats.example.org/count", "https://stats.example.org/count?<b>", "a", "my_site"):
        assert cli.main(["calendar", "--out", str(tmp_path / "bad"), "--goatcounter", bad]) == 2
    assert not (tmp_path / "bad").exists()  # refused before anything was written


def test_calendar_writes_a_page_per_model_and_files_for_search_engines(home, tmp_path, capsys):
    import xml.etree.ElementTree as ET

    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps([
        {"id": "Acme <beta> & co", "provider": "openai", "status": "deprecated", "retires": "2026-11-15",
         "replacement": "acme-chat-3", "source": "https://example.com/deprecations#acme"},
        {"id": "acme-beta-co", "provider": "openai", "status": "active"},
    ]), encoding="utf-8")
    site = tmp_path / "site"
    assert cli.main(["calendar", "--out", str(site), "--registry", str(extra), "--base-url", "https://example.org/cal"]) == 0
    assert "model pages" in capsys.readouterr().out
    # One page per model, at an address that survives IDs that differ only in punctuation (IDs are lower-cased).
    first = (site / "models/openai/acme-beta-co/index.html").read_text(encoding="utf-8")
    second = (site / "models/openai/acme-beta-co-2/index.html").read_text(encoding="utf-8")
    assert "<h1><code>acme &lt;beta&gt; &amp; co</code></h1>" in first and "<beta>" not in first
    assert "<h1><code>acme-beta-co</code></h1>" in second and "has not announced a shutdown date" in second
    assert '<link rel="canonical" href="https://example.org/cal/models/openai/acme-beta-co/">' in first
    assert "uvx llmshift scan" in first and "llmshift compare --to acme-chat-3" in first
    ld = json.loads(re.search(r'<script type="application/ld\+json">(.*?)</script>', first, re.S).group(1))
    assert ld["@type"] == "FAQPage" and ld["mainEntity"][1]["acceptedAnswer"]["text"] == "OpenAI names acme-chat-3 as the replacement."
    index = (site / "index.html").read_text(encoding="utf-8")
    assert '<a href="models/openai/acme-beta-co/"><code>acme &lt;beta&gt; &amp; co</code></a>' in index
    assert '<link rel="canonical" href="https://example.org/cal/">' in index and '<a href="feed.xml">RSS</a>' in index
    # A model that is gone shows the error people search for.
    data = json.loads((site / "retirements.json").read_text(encoding="utf-8"))
    gone = next(m for m in data["recent"] if m["provider"] in ("openai", "anthropic", "google") and not m["model"].startswith("/"))
    page = (site / "models" / feed._slug(gone["provider"]) / feed._slug(gone["model"]) / "index.html").read_text(encoding="utf-8")
    assert "The error you get now" in page and html.escape(gone["model"]) in page
    assert '<a href="../../../models/' + feed._slug(gone["provider"]) + '/">' in page  # breadcrumb back to the provider page
    pages = sorted(p.relative_to(site).parent.as_posix() for p in site.rglob("index.html"))
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [loc.text for loc in ET.parse(site / "sitemap.xml").getroot().findall("s:url/s:loc", ns)]  # nosec B314  # llmshift's own output
    assert len(locs) == len(pages) and all(loc.startswith("https://example.org/cal/") for loc in locs)
    assert "https://example.org/cal/models/openai/acme-beta-co/" in locs
    assert "Sitemap: https://example.org/cal/sitemap.xml" in (site / "robots.txt").read_text(encoding="utf-8")
    items = ET.parse(site / "feed.xml").getroot().findall("channel/item")  # nosec B314  # llmshift's own output
    acme = next(i for i in items if i.find("title").text.startswith("acme <beta> & co (OpenAI) retires on 2026-11-15"))
    assert acme.find("link").text == "https://example.org/cal/models/openai/acme-beta-co/"
    within_a_year = [m for m in data["upcoming"] if m["days_left"] <= 365]
    assert len({i.find("guid").text for i in items}) == len(items) == len(within_a_year) + len(data["recent"])
    llms = (site / "llms.txt").read_text(encoding="utf-8")
    assert llms.startswith("# AI model retirement calendar\n") and "](https://example.org/cal/models/openai/)" in llms
    assert "uvx llmshift scan" in llms


def test_calendar_search_files_need_an_https_base_url(home, tmp_path, capsys):
    plain = tmp_path / "plain"
    assert cli.main(["calendar", "--out", str(plain)]) == 0
    assert "--base-url" in capsys.readouterr().out
    assert not any((plain / name).exists() for name in ("sitemap.xml", "robots.txt", "feed.xml"))
    assert (plain / "models" / "openai" / "index.html").exists()
    assert 'rel="canonical"' not in (plain / "models" / "index.html").read_text(encoding="utf-8")
    assert "](models/openai/)" in (plain / "llms.txt").read_text(encoding="utf-8")
    for bad in ("http://example.org/", 'https://example.org/"><script>', "javascript:alert(1)", "example.org", "https://exa mple.org/"):
        assert cli.main(["calendar", "--out", str(tmp_path / "bad"), "--base-url", bad]) == 2
    assert not (tmp_path / "bad").exists()  # refused before anything was written


def test_bundled_data_passes_the_update_validation():
    data = registry.bundled_registry()
    assert registry.registry_problems(data, min_models=len(data["models"])) == []
    assert pricing.prices_problems(pricing._bundled_prices(), min_prices=500) == []


def test_user_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMSHIFT_HOME", str(tmp_path / "custom"))
    assert registry.user_data_dir() == tmp_path / "custom"
    monkeypatch.delenv("LLMSHIFT_HOME")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    expected = tmp_path / ("local" if registry.os.name == "nt" else "xdg") / "llmshift"
    assert registry.user_data_dir() == expected


# ------------------------------------------------- registry source monitor


@pytest.fixture
def watcher(tmp_path, monkeypatch):
    import importlib.util

    script = registry.Path(__file__).resolve().parents[1] / "scripts" / "check_registry_sources.py"
    spec = importlib.util.spec_from_file_location("check_registry_sources", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "STATE", tmp_path / "sources.json")
    return module


def test_watcher_text_and_change_detection(watcher, monkeypatch):
    page = (
        "<html><head><title>x</title><script>var build = 12345;</script></head><body><nav>menu</nav>"
        "<main><h1>Deprecations</h1><p>gpt-4 &amp; friends retire 2026-10-23</p></main><footer>(c)</footer></body></html>"
    )
    assert watcher.page_text(page) == "Deprecations gpt-4 & friends retire 2026-10-23"
    assert watcher.page_text(page.replace("12345", "67890")) == watcher.page_text(page)  # build ids do not count
    pages = {"https://a.example/x": "<main>one</main>", "https://b.example/y": "<main>two</main>"}

    def fake_fetch(url, timeout=30, attempts=2):
        if url not in pages:
            raise RuntimeError("timed out")
        return pages[url]

    monkeypatch.setattr(watcher, "fetch", fake_fetch)
    first = watcher.check(list(pages) + ["https://c.example/z"], {})
    assert sorted(first["new"]) == sorted(pages) and first["unreachable"][0]["url"] == "https://c.example/z"
    pages["https://b.example/y"] = "<main>two, edited</main>"
    second = watcher.check(list(pages), {"pages": first["current"]})
    assert second["changed"] == ["https://b.example/y"] and second["same"] == ["https://a.example/x"]
    urls = watcher.source_urls()
    assert urls and all("#" not in u and u.startswith("http") for u in urls)


def test_watcher_main_exit_codes(watcher, monkeypatch, capsys):
    monkeypatch.setattr(watcher, "source_urls", lambda: ["https://a.example/x"])
    monkeypatch.setattr(watcher, "fetch", lambda url, **kw: (_ for _ in ()).throw(RuntimeError("offline")))
    assert watcher.main([]) == 1 and "network problem" in capsys.readouterr().err
    monkeypatch.setattr(watcher, "fetch", lambda url, **kw: "<main>ok</main>")
    assert watcher.main([]) == 0 and watcher.STATE.exists()  # the first run records the pages
    assert "recorded 1 pages" in capsys.readouterr().out
    assert watcher.main([]) == 0 and "1 unchanged" in capsys.readouterr().out


def test_watcher_issue_is_opened_then_updated_once(watcher, server, monkeypatch):
    server.httpd.RequestHandlerClass.do_PATCH = lambda self: self._handle("PATCH")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/llmshift")
    monkeypatch.setenv("GITHUB_API_URL", server.url)
    issues: list = []

    def create(req):
        issues.append({"number": 7, "title": req["json"]["title"], "body": req["json"]["body"]})
        return (201, issues[-1])

    def edit(req):
        issues[0].update(req["json"])
        return issues[0]

    server.route("GET", "/repos/acme/llmshift/issues", lambda req: [dict(i) for i in issues])
    server.route("POST", "/repos/acme/llmshift/issues/7/comments", lambda req: (201, {"id": 1}))
    server.route("POST", "/repos/acme/llmshift/issues", create)
    server.route("PATCH", "/repos/acme/llmshift/issues/7", edit)
    result = {"changed": ["https://a.example/x"], "new": [], "unreachable": []}
    assert watcher.report_issue(result) == "opened issue #7"
    assert watcher.report_issue(result) == "issue #7 already lists these pages"
    result["changed"].append("https://b.example/y")
    assert watcher.report_issue(result) == "updated issue #7" and "https://b.example/y" in issues[0]["body"]
    methods = [r["method"] for r in server.requests]
    assert methods.count("POST") == 2 and methods.count("PATCH") == 1  # one issue, one comment, one edit
    # Once every page matches the reviewed data again, the issue closes itself.
    assert watcher.close_issue() == "closed issue #7"
    assert server.requests[-1]["method"] == "PATCH" and server.requests[-1]["json"]["state"] == "closed"
    issues.clear()
    assert watcher.close_issue() == "no open issue"
