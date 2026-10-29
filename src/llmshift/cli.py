"""Command line interface: llmshift <command> [options]."""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from . import __version__
from . import console as con
from .config import SAMPLE_CONFIG, ConfigError, load_config
from .errors import EXIT_FAIL, EXIT_INTERRUPTED, EXIT_OK, EXIT_USAGE, UsageError
from .textutil import json_loads, plural, read_text, report_json

# Commands in their own modules; each provides register(subparsers).
EXTENSIONS: tuple[str, ...] = ("calibrate", "pick", "safety", "shadow", "mcp", "prcheck", "upgrade", "embedmig", "feed")


# ------------------------------------------------------------------ helpers


def _timestamp() -> str:
    """For report file names; down to the millisecond, so two runs in the same second
    (a script, two CI steps) do not overwrite each other's reports."""
    now = _dt.datetime.now()
    return now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"


def _write(path: str | Path, text: str) -> Path:
    from .testset import atomic_write

    p = Path(path)
    try:
        atomic_write(p, text)
    except OSError as exc:
        raise UsageError(f"cannot write {path}: {exc.strerror or exc}") from None
    return p


def _rel(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _step_summary(markdown: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with contextlib.suppress(OSError):
        with open(path, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(markdown + "\n")


def _github_output(name: str, value: Any) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    text = str(value)
    with contextlib.suppress(OSError):
        with open(path, "a", encoding="utf-8", errors="replace") as fh:
            if "\n" in text or "\r" in text:  # a line break would start another output: the multi-line form
                import secrets

                delimiter = f"llmshift_{secrets.token_hex(8)}"
                fh.write(f"{name}<<{delimiter}\n{text}\n{delimiter}\n")
            else:
                fh.write(f"{name}={text}\n")


def _cache(cfg: dict, args: argparse.Namespace) -> Any:
    from .cache import ResponseCache

    enabled = bool(cfg.get("cache", True)) and not getattr(args, "no_cache", False)
    return ResponseCache(cfg.get("cache_dir") or ".llmshift/cache", enabled=enabled)


def _tests_path(cfg: dict, args: argparse.Namespace) -> str:
    return getattr(args, "tests", None) or cfg.get("tests") or ".llmshift/tests.jsonl"


def _load_cases(cfg: dict, args: argparse.Namespace) -> tuple[list, str]:
    from .testset import load_tests

    path = _tests_path(cfg, args)
    if not Path(path).exists():
        raise UsageError(f"test file {path} not found. Create one with `llmshift snapshot <logs>` or `llmshift import openai-evals ...`.")
    groups = set(getattr(args, "group", None) or [])
    tags = set(getattr(args, "tag", None) or [])
    ids = set(getattr(args, "case", None) or [])
    limit = getattr(args, "limit", None)

    def keep(c: Any) -> bool:
        return (not groups or c.group in groups or any(g in (c.group or "") for g in groups)) and (not tags or bool(tags & set(c.tags)))

    # --case names ids as they are after duplicates are renamed, so it reads the whole file.
    cases = load_tests(path, limit=int(limit) if limit and not ids else None, keep=keep)
    if ids:
        cases = [c for c in cases if c.id in ids]
    if limit:
        cases = cases[: int(limit)]
    if not cases:
        raise UsageError(f"no test cases selected from {path}")
    # Settings the code no longer sends (temperature for a model that rejects it) leave the tests too.
    drop = list(getattr(args, "drop_param", None) or []) + [str(n).strip() for n in cfg.get("drop_params") or []]
    for case in cases if drop else ():
        params = case.input.get("params")
        if isinstance(params, dict):
            for name in drop:
                params.pop(name, None)
    return cases, path


def rejected_and_other(result: Any) -> tuple[list[str], int]:
    """(settings the model rejects as the requests send them, cases that changed for other
    reasons): the first need a code change, only the second a prompt change."""
    refused: set[str] = set()
    other = 0
    for c in result.cases:
        failing = [r for r in c.checks if r.status in ("fail", "error") and r.severity != "info"]
        refused.update(item.split("=", 1)[0] for r in failing if r.type == "request" for item in (r.details or {}).get("rejected") or [])
        if c.status in ("fail", "warn") and any(r.type != "request" for r in failing):
            other += 1
    return sorted(refused), other


def _overrides(args: argparse.Namespace) -> dict:
    out = {}
    if getattr(args, "temperature", None) is not None:
        out["temperature"] = args.temperature
    if getattr(args, "max_tokens", None) is not None:
        out["max_completion_tokens"] = args.max_tokens
    return out


def _severity(cfg: dict, args: argparse.Namespace) -> dict:
    sev = dict(cfg.get("severity") or {})
    for item in getattr(args, "severity", None) or []:
        if "=" not in item:
            raise UsageError(f"--severity expects check=level, got {item!r}")
        k, v = item.split("=", 1)
        if v not in ("error", "warn", "info"):
            raise UsageError(f"--severity level must be error, warn or info (got {v!r})")
        sev[k.strip()] = v.strip()
    return sev


def _load_compare_json(path: str) -> Any:
    """A results file written by `compare --json` (or the default report JSON)."""
    from .compare import CompareResult

    try:
        data = json_loads(read_text(path))
    except FileNotFoundError:
        raise UsageError(f"{path} not found") from None
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise UsageError(f"{path} is not valid JSON ({exc})") from None
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or not isinstance(data.get("summary"), dict):
        raise UsageError(f"{path} is not a compare result (write one with `llmshift compare --json FILE`)")
    try:
        return CompareResult.from_dict(data)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise UsageError(f"{path}: unexpected compare result ({exc})") from None


def _refuse_overwrite(paths: list, force: bool, elsewhere: str = "-o") -> None:
    existing = [str(x) for x in paths if Path(x).exists()]
    if existing and not force:
        raise UsageError(f"{', '.join(existing[:3])} already exist{'s' if len(existing) == 1 else ''}; pass --force to replace, or {elsewhere} to write elsewhere")


def _raw_archive_path(args: argparse.Namespace, cfg: dict) -> Path:
    """Where `import --save-raw` keeps the API responses: the file given, else a dated file beside the tests."""
    if args.save_raw:
        return Path(args.save_raw)
    if args.split:
        folder = Path(args.split)
    else:
        out = Path(args.output or cfg.get("tests") or ".llmshift/tests.jsonl")
        folder = out if out.is_dir() else out.parent
    return folder / f"openai-evals-raw-{_dt.date.today().isoformat()}.json"


def _positive(value: Any, what: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise UsageError(f"{what} must be a whole number, got {value!r}") from None
    if number < 1:
        raise UsageError(f"{what} must be at least 1, got {number}")
    return number


def _days(text: str) -> int:
    """argparse type for a number of days (zero or more)."""
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number of days, got {text!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be zero or more days, got {number}")
    return number


# argparse types: a wrong number is refused before anything runs (or is paid for).
def rate_arg(text: str) -> float:
    """A fraction from 0 to 1, as in --min-pass-rate 0.95."""
    try:
        number = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a fraction from 0 to 1, e.g. 0.95, got {text!r}") from None
    if not 0.0 <= number <= 1.0:  # also refuses nan
        hint = f" (for {number:g}% write {number / 100:g})" if 1 < number <= 100 else ""
        raise argparse.ArgumentTypeError(f"must be a fraction from 0 to 1, got {text}{hint}")
    return number


def model_arg(text: str) -> str:
    """A model spec. An empty one (`--to "$CANDIDATE"` with the variable unset) is refused:
    quietly testing the configured model instead would report on the wrong model."""
    value = text.strip()
    if not value:
        raise argparse.ArgumentTypeError("is empty (an unset variable?); leave the option out to use the model in llmshift.yaml")
    return value


def param_arg(text: str) -> str:
    """A request setting's name, as test files store it (temperature, top_p, tool_choice)."""
    value = text.strip()
    if not re.fullmatch(r"[A-Za-z_][\w.-]*", value):
        raise argparse.ArgumentTypeError(f"expected a setting name such as temperature, got {text!r}")
    return value


def models_arg(text: str) -> str:
    """One or more comma-separated model specs; see model_arg."""
    if not any(part.strip() for part in text.split(",")):
        raise argparse.ArgumentTypeError("is empty (an unset variable?); leave the option out to use the models in llmshift.yaml")
    return text


def temperature_arg(text: str) -> float:
    """A sampling temperature from 0 to 2, the widest range providers accept (Anthropic's
    provider keeps it within its own 0 to 1)."""
    try:
        number = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number from 0 to 2, e.g. 0.7, got {text!r}") from None
    if not 0.0 <= number <= 2.0:  # also refuses nan
        raise argparse.ArgumentTypeError(f"must be from 0 to 2, got {text}")
    return number


def positive_int(text: str) -> int:
    """A whole number of 1 or more (a count, a limit, a number of requests)."""
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


def show_arg(text: str) -> str:
    """How many cases to print: a number from 0, or "all"."""
    value = text.strip().lower()
    if value == "all" or (value.isdigit() and int(value) >= 0):
        return value
    raise argparse.ArgumentTypeError(f"expected a number or 'all', got {text!r}")


def _show(value: str | None) -> int:
    if value in (None, ""):
        return 5
    if str(value).lower() == "all":
        return -1
    return int(value)


# ------------------------------------------------------------------- init

WORKFLOW_PR = """name: llmshift
on:
  pull_request:
  workflow_dispatch:

jobs:
  llmshift:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    env:
      # Pull requests from forks and Dependabot get no secrets: compare is skipped there instead of failing.
      LLMSHIFT_NO_SECRETS: ${{ github.event_name == 'pull_request' && (github.event.pull_request.head.repo.full_name != github.repository || github.event.pull_request.user.login == 'dependabot[bot]') }}
    steps:
      - uses: actions/checkout@v5
        with:
          persist-credentials: false
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install llmshift
      # Fails the job if the code calls a retired model; annotates retiring ones on the diff.
      - run: llmshift scan --fail-on retired
      # Re-runs the test set against the candidate model (needs your provider key as a secret):
      # the repository variable LLMSHIFT_CANDIDATE if set, else candidate: in llmshift.yaml.
      - run: echo "::notice::llmshift compare skipped - pull requests from forks and Dependabot get no secrets"
        if: ${{ hashFiles('.llmshift/tests.jsonl') != '' && env.LLMSHIFT_NO_SECRETS == 'true' }}
      - run: llmshift compare ${LLMSHIFT_CANDIDATE:+--to "$LLMSHIFT_CANDIDATE"}
        if: ${{ hashFiles('.llmshift/tests.jsonl') != '' && env.LLMSHIFT_NO_SECRETS != 'true' }}
        env:
          LLMSHIFT_CANDIDATE: ${{ vars.LLMSHIFT_CANDIDATE }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      - uses: actions/upload-artifact@v6
        if: always()
        with:
          name: llmshift-reports
          path: .llmshift/reports
          if-no-files-found: ignore
"""

WORKFLOW_WATCH = """name: llmshift-watch
on:
  schedule:
    - cron: "17 3 * * *"   # nightly at 03:17 UTC
  workflow_dispatch:

jobs:
  watch:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      issues: write          # for the github alert
    steps:
      - uses: actions/checkout@v5
        with:
          persist-credentials: false
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install llmshift
      # Keep run history between nightly runs.
      - uses: actions/cache@v5
        with:
          path: .llmshift/watch
          key: llmshift-watch-${{ github.run_id }}
          restore-keys: llmshift-watch-
      # The repository variable LLMSHIFT_MODEL if set, else watch.model (or candidate:) in llmshift.yaml.
      - run: llmshift watch ${LLMSHIFT_MODEL:+--model "$LLMSHIFT_MODEL"} --alert github
        env:
          LLMSHIFT_MODEL: ${{ vars.LLMSHIFT_MODEL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
"""


def _init_models(baseline: str | None, candidate: str | None) -> tuple[str | None, str | None]:
    """init's defaults where --baseline / --candidate are not given: the model this project's
    code calls most, and the replacement its provider lists for it when it is going away."""
    if baseline and candidate:
        return baseline, candidate
    try:
        from collections import Counter

        from .providers import known_providers
        from .scanner import scan_paths

        # a sample is enough to see the model in use; a huge folder must not stall init
        known = [f.lookup for f in scan_paths(["."], max_files=2000).findings if f.lookup is not None and f.lookup.status != "unknown"]
        top = Counter(lk.model for lk in known).most_common(1)
    except Exception:  # an unreadable folder: no defaults rather than no config
        return baseline, candidate
    if not top:
        return baseline, candidate
    lk = next(lk for lk in known if lk.model == top[0][0])
    prefix = lk.provider.lower() if lk.provider and lk.provider.lower() in known_providers(None) else None

    def spec(model: str) -> str:
        return f"{prefix}:{model}" if prefix else model

    if lk.status in ("retired", "retiring", "deprecated") and lk.replacement and not candidate:
        candidate = spec(lk.replacement)
    return baseline or spec(lk.model), candidate


def cmd_init(args: argparse.Namespace, cfg: dict) -> int:
    path = Path(args.path or "llmshift.yaml")
    if path.exists() and not args.force:
        con.out(f"{path} already exists (use --force to overwrite).")
    else:
        baseline, candidate = _init_models(args.baseline, args.candidate)
        _write(path, SAMPLE_CONFIG.format(
            baseline=baseline or "provider:model", baseline_off="" if baseline else "# ",
            candidate=candidate or "provider:model", candidate_off="" if candidate else "# ",
        ))
        con.out(f"{con.style(con.sym('pass'), 'green')} wrote {_rel(path)}" + (f" (baseline {baseline}, the model your code calls most)" if baseline and not args.baseline else ""))
        if not candidate:
            con.out("  Set candidate: in it, or pass --to to compare and fix, once you know the model to move to.")
    if args.github:
        for name, text in (("llmshift.yml", WORKFLOW_PR), ("llmshift-watch.yml", WORKFLOW_WATCH)):
            wf = Path(".github/workflows") / name
            if wf.exists() and not args.force:
                con.out(f"{wf} already exists (use --force to overwrite).")
                continue
            _write(wf, text)
            con.out(f"{con.style(con.sym('pass'), 'green')} wrote {_rel(wf)}")
    gi = Path(".gitignore")
    with contextlib.suppress(OSError):
        # Bytes: the file may not be UTF-8, and its encoding must be left as it is.
        existing = gi.read_bytes() if gi.exists() else b""
        if b".llmshift/cache" not in existing:
            with gi.open("ab") as fh:
                fh.write((b"\n" if existing and not existing.endswith(b"\n") else b"") + b"# llmshift\n.llmshift/cache/\n.llmshift/reports/\n")
            con.out(f"{con.style(con.sym('pass'), 'green')} added .llmshift/cache and .llmshift/reports to .gitignore")
    con.out("")
    con.out("Next steps:")
    con.out("  1. llmshift scan                         # which models your code calls, and when they retire")
    con.out("  2. llmshift snapshot logs/               # turn real usage logs into .llmshift/tests.jsonl")
    con.out("  3. llmshift compare --to <new model>     # see exactly what changes")
    con.out("  4. llmshift fix --to <new model>         # get prompt changes that close the gaps")
    return EXIT_OK


# ------------------------------------------------------------------- scan


def cmd_scan(args: argparse.Namespace, cfg: dict) -> int:
    from . import report
    from .registry import Registry
    from .scanner import scan_paths, should_fail

    sc = cfg.section("scan") if hasattr(cfg, "section") else cfg.get("scan", {})
    configured = sc.get("registry") or []
    registry_files = list(args.registry or []) + (list(configured) if isinstance(configured, list) else [configured])
    registry = Registry.load(registry_files)
    warn_days = args.warn_days if args.warn_days is not None else int(sc.get("warn_days", 180))
    result = scan_paths(
        args.paths or sc.get("paths") or ["."],
        registry=registry,
        include_docs=args.include_docs or bool(sc.get("include_docs")),
        exclude=list(sc.get("exclude") or []) + list(args.exclude or []),
        warn_days=warn_days,
    )
    fmt = args.format
    if fmt == "json":
        text = report_json(result.to_dict())
    elif fmt == "markdown":
        text = report.scan_markdown(result)
    elif fmt == "sarif":
        text = json.dumps(report.scan_sarif(result), indent=2)
    elif args.output:
        with con.no_color():  # a file gets plain text, not terminal colour codes
            text = report.scan_terminal(result, verbose=args.all)
    else:
        text = report.scan_terminal(result, verbose=args.all)
    if args.output:
        _write(args.output, text + ("\n" if not text.endswith("\n") else ""))
        con.out(f"wrote {_rel(args.output)}", err=True)
    else:
        con.out(text)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # Annotations are workflow commands on stdout; they would corrupt JSON/SARIF printed there.
        if args.output or fmt not in ("json", "sarif"):
            for line in report.github_annotations(result):
                con.out(line)
        _step_summary(report.scan_markdown(result))
    fail_on = args.fail_on or sc.get("fail_on") or "retired"
    counts = result.counts()
    _github_output("retired", counts["retired"])
    _github_output("retiring", counts["retiring_soon"])
    if fmt == "table" and not args.output:
        if counts["retired"] or counts["retiring_soon"]:
            from .providers import infer_provider

            # Name the replacement of the most urgent model when `compare --to` takes it as written
            # (some replacements are an API or an SDK, a few need a provider: prefix).
            replacement = next(
                (m["lookup"].replacement for m in result.models()
                 if m["lookup"] and m["lookup"].status in ("retired", "retiring") and m["lookup"].replacement
                 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:@-]*", m["lookup"].replacement)
                 and infer_provider(m["lookup"].replacement)),
                "<replacement>",
            )
            con.out("")
            con.out(con.style("Next:", "bold") + " snapshot real traffic and check the replacement before the shutdown date:")
            con.out(f"  llmshift snapshot <your-logs>  &&  llmshift compare --to {replacement}")
            con.out(con.style("Then keep it checked on every pull request: llmshift init --github", "grey"))
    if should_fail(result, fail_on):
        if fmt == "table" and not args.output and not getattr(args, "in_demo", False):
            con.out(con.style(f"\nExit code 1: found models matching --fail-on {fail_on} (in CI this fails the job).", "red"), err=True)
        return EXIT_FAIL
    return EXIT_OK


def cmd_models(args: argparse.Namespace, cfg: dict) -> int:
    from .registry import Registry, today

    reg = Registry.load(args.registry or [])
    found = reg.search(args.query or "", provider=args.provider, status=args.status, warn_days=args.warn_days)
    if args.retiring_within is not None:
        found = [lk for lk in found if lk.status == "retiring" and lk.days_left is not None and lk.days_left <= args.retiring_within]
    if args.format == "json":
        con.out(json.dumps([lk.to_dict() for lk in found], indent=2))
        return EXIT_OK
    if not found:
        con.out("No matching models in the registry.")
        return EXIT_OK
    rows = []
    for lk in found[: args.limit]:
        st = {"retired": ("red",), "retiring": ("yellow",), "deprecated": ("yellow",), "active": ("green",)}.get(lk.status, ())
        rows.append([lk.model, lk.provider, con.style(lk.status, *st), lk.retires.isoformat() if lk.retires else "", lk.replacement or "", (lk.note or "")[:60]])
    con.out(con.table(rows, ["Model", "Provider", "Status", "Retires", "Replacement", "Note"], max_widths=[40, 13, 10, 10, 26, 60]))
    if len(found) > args.limit:
        con.out(f"... {len(found) - args.limit} more (use --limit)")
    con.out(con.style(f"Lifecycle data updated {reg.updated}. Status is computed for {today().isoformat()}.", "grey"))
    return EXIT_OK


# --------------------------------------------------------------- snapshot


def cmd_snapshot(args: argparse.Namespace, cfg: dict) -> int:
    from .providers import resolve
    from .snapshot import build_snapshot
    from .testset import append_tests, load_tests, save_tests

    sc = cfg.get("snapshot") or {}
    baseline = resolve(args.baseline_model, cfg) if args.baseline_model else None
    result = build_snapshot(
        args.sources,
        models=args.model,
        since=args.since,
        until=args.until,
        max_cases=args.max_cases or int(sc.get("max_cases", 200)),
        per_group=args.per_group or int(sc.get("per_group", 25)),
        seed=int(sc.get("seed", 7)) if args.seed is None else args.seed,
        redact_secrets=not args.no_redact and bool(sc.get("redact_secrets", True)),
        redact_pii=args.redact_pii or bool(sc.get("redact_pii", False)),
        include_errors=args.include_errors,
        include_truncated=args.include_truncated,
        baseline_provider=baseline,
        regenerate=args.regenerate,
        judge=args.with_judge_checks,
        concurrency=args.concurrency or int(cfg.get("concurrency", 4)),
        cache=_cache(cfg, args),
        max_records=_positive(args.max_records if args.max_records is not None else sc.get("max_records", 100_000), "--max-records"),
    )
    st = result.stats
    out_path = args.output or cfg.get("tests") or ".llmshift/tests.jsonl"
    if Path(out_path).is_dir():
        out_path = str(Path(out_path) / "tests.jsonl")
    con.out(con.style("LLMShift snapshot", "bold"))
    con.out(f"  read {st.records} records from {st.files} file(s); recognised {st.parsed} LLM calls" + (f" ({', '.join(f'{k}: {v}' for k, v in st.formats.most_common())})" if st.formats else ""))
    if st.sampled_out:
        con.out(con.style(f"  kept a fair sample per prompt template: {st.sampled_out:,} call(s) left out to stay within --max-records", "grey"))
    for reason, n in list(st.skipped.most_common()) + list(result.filtered.most_common()):
        example = st.examples.get(reason)
        con.out(con.style(f"  skipped {n}: {reason}" + (f"  e.g. {con.truncate(example, 90)}" if example else ""), "grey"))
    if result.redactions:
        con.out(f"  redacted {sum(result.redactions.values())} value(s): " + ", ".join(f"{k.lower()} x{v}" for k, v in result.redactions.items()))
    con.out(f"  {result.unique} unique requests ({result.duplicates} duplicates removed) in {result.groups} prompt template(s)")
    if result.generated or result.generation_errors:
        con.out(f"  generated {result.generated} baseline output(s) with {baseline.spec if baseline else '?'}" + (f", {result.generation_errors} failed" if result.generation_errors else ""))
    if not result.cases:
        con.fail("no test cases were created.")
        if st.parsed == 0:
            con.out("  None of the records looked like LLM calls. See `llmshift snapshot --help` for supported formats.", err=True)
        return EXIT_FAIL
    if args.append and Path(out_path).exists():
        existing = load_tests(out_path)
        known = {c.id for c in existing}
        new = [c for c in result.cases if c.id not in known]
        append_tests(new, out_path, existing)
        con.out(
            f"{con.style(con.sym('pass'), 'green')} added {len(new)} new test case(s) to {_rel(out_path)} ({len(existing) + len(new)} total"
            + (f"; {len(result.cases) - len(new)} were already there)" if len(new) < len(result.cases) else ")")
        )
    else:
        save_tests(result.cases, out_path)
        con.out(f"{con.style(con.sym('pass'), 'green')} wrote {len(result.cases)} test case(s) to {_rel(out_path)}")
    checks = ", ".join(f"{k} x{v}" for k, v in result.check_counts.most_common())
    con.out(con.style(f"  inferred checks: {checks}", "grey"))
    sizes = sorted(result.group_sizes.items(), key=lambda kv: -kv[1])[:8]
    if len(sizes) > 1:
        con.out(con.style("  templates: " + ", ".join(f"{g} ({n})" for g, n in sizes), "grey"))
    con.out("")
    con.out(f"Next: llmshift compare --to <new model>{'' if out_path == (cfg.get('tests') or '.llmshift/tests.jsonl') else f' --tests {out_path}'}")
    return EXIT_OK


# ---------------------------------------------------------------- compare


def _candidate(args: argparse.Namespace, cfg: dict) -> str:
    spec = getattr(args, "to", None) or cfg.get("candidate")
    if not spec:
        raise UsageError("which model do you want to test? Pass --to provider:model (e.g. --to openai:gpt-5.6-sol) or set candidate in llmshift.yaml")
    return spec


_REPORT_SUFFIX = {"html": ".html", "json": ".json", "markdown": ".md", "junit": ".junit.xml"}


def _write_reports(result: Any, args: argparse.Namespace, cfg: dict, prefix: str) -> list[str]:
    """Write the requested reports. Each flag takes an optional file name; without one the
    report goes to the reports directory. HTML and JSON are written by default (--no-report
    turns that off)."""
    from . import report

    requested = {kind: getattr(args, kind, None) for kind in _REPORT_SUFFIX}
    no_report = bool(getattr(args, "no_report", False))
    reports_dir = Path(getattr(args, "report_dir", None) or cfg.get("reports_dir") or ".llmshift/reports")
    stamp = _timestamp()
    render = {
        "html": lambda: report.compare_html(result),
        "json": lambda: report_json(result.to_dict()) + "\n",
        "markdown": lambda: report.compare_markdown(result),
        "junit": lambda: report.compare_junit(result),
    }
    written = []
    for kind, value in requested.items():
        if value:  # an explicit file name
            path: Path | None = Path(value)
        elif value == "" or (kind in ("html", "json") and not no_report):
            path = reports_dir / f"{prefix}-{stamp}{_REPORT_SUFFIX[kind]}"
        else:
            path = None
        if path is not None:
            written.append(str(_write(path, render[kind]())))
    return written


def cmd_compare(args: argparse.Namespace, cfg: dict) -> int:
    from . import report
    from .compare import run_compare

    cases, path = _load_cases(cfg, args)
    candidate = _candidate(args, cfg)
    min_rate = args.min_pass_rate if args.min_pass_rate is not None else cfg.get("min_pass_rate")
    con.out(con.style(f"Running {len(cases)} test case(s) on {candidate}...", "grey"), err=True)
    result = run_compare(
        cases,
        candidate,
        baseline=args.baseline_model,
        judge=args.judge or cfg.get("judge"),
        embedder=args.embedder or cfg.get("embedder"),
        config=cfg,
        concurrency=args.concurrency or int(cfg.get("concurrency", 4)),
        cache=_cache(cfg, args),
        allow_python=args.allow_python or bool(cfg.get("allow_python_checks")),
        severity=_severity(cfg, args),
        overrides=_overrides(args),
        tests_path=path,
        min_pass_rate=float(min_rate) if min_rate is not None else None,
        repeat=_positive(args.repeat if args.repeat is not None else cfg.get("repeat", 1), "--repeat"),
        judge_mode=args.judge_mode or cfg.get("judge_mode") or "auto",
    )
    if result.judge:
        from .calibrate import calibration_note

        note = calibration_note(result.judge, cfg)
        if note:
            result.insights.append(note)
    con.out(report.compare_terminal(result, show=_show(args.show), verbose=args.verbose))
    written = _write_reports(result, args, cfg, "compare")
    if written:
        con.out("")
        con.out("Reports: " + ", ".join(_rel(p) for p in written))
    _step_summary(report.compare_markdown(result))
    _github_output("pass_rate", result.summary.get("pass_rate"))
    s = result.summary
    refused, other = rejected_and_other(result)
    if refused:
        flags = " ".join(f"--drop-param {n}" for n in refused if n != "assistant prefill")
        con.out(con.style(f"Next: remove {', '.join(refused)} from the calls (llmshift upgrade lists where), then re-run" + (f" with {flags}" if flags else ""), "grey"))
    if other:
        con.out(con.style(f"Next: llmshift fix --to {candidate}  (suggests prompt changes for the {other} changed case(s))", "grey"))
    if result.inconclusive:  # what could be checked passed, or nothing could be: no verdict
        return EXIT_USAGE
    if s.get("evaluated", 0) == 0 or (result.min_pass_rate is not None and not result.passed):
        return EXIT_FAIL
    return EXIT_OK


# -------------------------------------------------------------------- fix


def cmd_fix(args: argparse.Namespace, cfg: dict) -> int:
    from .fix import fix_markdown, patched_cases, run_fix
    from .testset import save_tests

    cases, path = _load_cases(cfg, args)
    candidate = _candidate(args, cfg)
    fx = cfg.get("fix") or {}
    previous = None
    if args.from_results:
        previous = _load_compare_json(args.from_results)
        if previous.candidate.split(":", 1)[-1] != candidate.split(":", 1)[-1]:
            con.warn(f"{args.from_results} was produced for {previous.candidate}, not {candidate}; re-running compare instead")
            previous = None
    con.out(con.style(f"Looking for prompt changes so {candidate} matches the baseline ({plural(len(cases), 'case')})...", "grey"), err=True)
    result = run_fix(
        cases,
        candidate,
        config=cfg,
        judge=args.judge or cfg.get("judge"),
        optimizer=args.optimizer or fx.get("optimizer"),
        concurrency=args.concurrency or int(cfg.get("concurrency", 4)),
        cache=_cache(cfg, args),
        allow_python=args.allow_python or bool(cfg.get("allow_python_checks")),
        severity=_severity(cfg, args),
        rounds=args.rounds or int(fx.get("rounds", 3)),
        max_cases_per_group=args.max_cases_per_group or int(fx.get("max_cases_per_group", 12)),
        target=float(fx["target_pass_rate"]) if fx.get("target_pass_rate") is not None else 1.0,  # null: the default
        source_paths=args.source or ["."],
        previous=previous,
        validate=not args.no_validate,
    )
    con.out(con.style("LLMShift fix", "bold") + f"  {candidate}")
    for note in result.notes:
        con.out(con.style(f"  {con.sym('warn')} {note}", "yellow"))
    if not result.groups:
        con.out(f"  {con.style(con.sym('pass'), 'green')} nothing to fix: every case already matches the baseline.")
        return EXIT_OK
    def matched(st: dict) -> str:
        detail = []
        if st.get("failed"):
            detail.append(f"{st['failed']} failing")
        if st.get("changed"):
            detail.append(f"{st['changed']} changed")
        if st.get("errors"):
            detail.append(plural(st["errors"], "error"))
        return f"{st.get('passed', 0)} of {st.get('cases', 0)} match" + (f" ({', '.join(detail)})" if detail else "")

    for g in result.groups:
        con.out("")
        where = f"  (prompt at {', '.join(g.locations)})" if g.locations else ""
        con.out(con.style(g.group, "bold") + con.style(where, "grey"))
        before = matched(g.before)
        if g.suggested_prompt:
            con.out(f"  before: {before}   after: {con.style(matched(g.after), 'green', 'bold')}   via {g.strategy}")
            if g.added:
                con.out("  Add to the system prompt:")
                for line in g.added:
                    con.out(con.style(f"    + {line}", "green"))
            elif g.strategy.startswith("optimizer"):
                con.out("  The optimizer rewrote the prompt; see the report for the full text and diff.")
            if "example" in g.strategy:
                con.out(con.style("    + (plus a worked example of the expected output - see the report)", "green"))
        else:
            con.out(f"  before: {before}   no prompt change helped")
        for item in g.remaining[:4]:
            con.out(con.style(f"  still different: {con.truncate(item, con.width() - 22)}", "yellow"))
        for note in g.notes:
            con.out(con.style(f"  note: {note}", "yellow"))
    con.out("")
    b, a = result.before, result.after
    if a:
        con.out(f"Overall pass rate: {b['pass_rate']:.0%} {con.sym('arrow')} {con.style(format(a['pass_rate'], '.0%'), 'bold')} with the suggested prompts "
                f"({b['passed']} {con.sym('arrow')} {a['passed']} unchanged, {b['failed']} {con.sym('arrow')} {a['failed']} failing)")
    reports_dir = Path(args.report_dir or cfg.get("reports_dir") or ".llmshift/reports")
    md_path = Path(args.out) if args.out else reports_dir / f"fix-{_timestamp()}.md"
    _write(md_path, fix_markdown(result))
    json_path = md_path.with_suffix(".json")
    _write(json_path, report_json(result.to_dict()) + "\n")
    con.out(f"Full prompts and diffs: {_rel(md_path)} (and {_rel(json_path)})")
    if args.write_tests:
        save_tests(patched_cases(cases, result), args.write_tests)
        con.out(f"Test set with the suggested prompts: {_rel(args.write_tests)}  (check it with: llmshift compare --tests {args.write_tests} --to {candidate})")
    _step_summary(fix_markdown(result))
    con.out(con.style("Nothing in your code was changed. Apply the prompts you agree with, then re-run `llmshift compare`.", "grey"))
    return EXIT_OK


# ------------------------------------------------------------------ watch


def cmd_watch(args: argparse.Namespace, cfg: dict) -> int:
    from . import report
    from .alerts import SEVERITY_RANK, check_destinations
    from .runner import RunAborted
    from .watch import parse_every, run_watch, seconds_until, sleep_until_next

    w = cfg.get("watch") or {}
    model = args.model or w.get("model") or cfg.get("candidate")
    if not model:
        raise UsageError("which model should be watched? Pass --model provider:model or set watch.model in llmshift.yaml")
    if args.every and args.at:
        raise UsageError("use either --every or --at, not both")
    if args.every:
        parse_every(args.every)  # fail fast on a typo, before the first run
    if args.at:
        seconds_until(args.at)
    alerts = list(w.get("alerts") or []) + list(args.alert or [])
    problems = check_destinations(alerts)
    if problems and not args.dry_run:  # before the run, not after paying for it
        raise UsageError("alerts could not be delivered: " + "; ".join(problems))
    for problem in problems:
        con.warn(f"alert: {problem}")
    fail_on = args.fail_on or w.get("fail_on", "warning")
    # Threshold: --min-pass-rate, else watch.min_pass_rate, else the project-wide min_pass_rate,
    # each only when set in the config file (null switches it off). Without one, watch alerts on drift.
    min_rate = args.min_pass_rate
    if min_rate is None and hasattr(cfg, "user_value"):
        for keys in (("watch", "min_pass_rate"), ("min_pass_rate",)):
            found, value = cfg.user_value(*keys)
            if found:
                min_rate = value
                break
    elif min_rate is None:
        min_rate = w.get("min_pass_rate")
    loop = bool(args.every or args.at)
    exit_code = EXIT_OK
    while True:
        cases, path = _load_cases(cfg, args)
        con.out(con.style(f"Watching {model}: running {len(cases)} test case(s) against the live model...", "grey"), err=True)
        try:
            outcome = run_watch(
                cases,
                model,
                history_dir=args.history or w.get("history", ".llmshift/watch"),
                config=cfg,
                judge=args.judge or cfg.get("judge"),
                concurrency=args.concurrency or int(cfg.get("concurrency", 4)),
                allow_python=args.allow_python or bool(cfg.get("allow_python_checks")),
                severity=_severity(cfg, args),
                min_pass_rate=float(min_rate) if min_rate is not None else None,
                max_pass_rate_drop=args.max_drop if args.max_drop is not None else float(w.get("max_pass_rate_drop", 0.05)),
                max_similarity_drop=float(w.get("max_similarity_drop", 0.10)),
                retirement_warn_days=int(w.get("retirement_warn_days", 30)),
                alerts=alerts,
                alert_on=args.alert_on or w.get("alert_on", "warning"),
                dry_run=args.dry_run,
            )
        except RunAborted as exc:
            if not loop:
                raise
            _watch_outage(model, exc, alerts, args, w)  # an outage is what a watch reports; it keeps watching
            nxt = f"at {args.at}" if args.at else f"in {args.every}"
            con.out(con.style(f"  next check {nxt} (Ctrl+C to stop)", "grey"))
            sleep_until_next(args.every, args.at)
            continue
        s = outcome.summary
        _github_output("pass_rate", s.get("pass_rate"))  # the action's pass-rate output, as for compare
        con.out(con.style("LLMShift watch", "bold") + f"  {outcome.model}  run {outcome.run_id}")
        sim = f", similarity to baseline {s['similarity']:.2f}" if s.get("similarity") is not None else ""
        checked = s.get("total", 0) - s.get("unchecked", 0)  # the cases the pass rate is over: a request with no answer counts
        unchecked = f", {s['unchecked']} not checked" if s.get("unchecked") else ""
        rate = f"pass rate {s['pass_rate']:.0%} ({s['passed'] + s['warned']}/{checked})" if checked > 0 else "pass rate n/a (no case could be checked)"
        con.out(f"  {rate}, {s['failed']} failing, {plural(s['errors'], 'error')}{unchecked}{sim}")
        if outcome.previous:
            prev = outcome.previous.get("summary", {})
            con.out(con.style(f"  previous run {outcome.previous.get('run_id')}: pass rate {prev.get('pass_rate', 0):.0%}", "grey"))
        else:
            con.out(con.style("  first run for this model: it becomes the reference for drift detection", "grey"))
        if not outcome.drifts:
            con.out(f"  {con.style(con.sym('pass') + ' no drift detected', 'green')}")
        for d in outcome.drifts:
            st = {"critical": ("red", "bold"), "warning": ("yellow",), "info": ("cyan",)}[d.severity]
            con.out(con.style(f"  {d.severity.upper():<8} {d.title}", *st) + (f"\n           {d.detail}" if d.detail else ""))
        for note in outcome.notes:
            con.out(con.style(f"  note: {note}", "yellow"))
        for name, err in outcome.deliveries:
            if err:
                con.warn(f"alert via {name} failed: {err}")
            else:
                con.out(con.style(f"  alert sent via {name}", "grey"))
        if outcome.drifts and not alerts:
            con.out(con.style("  (no alert destinations configured: add --alert slack:<webhook> or watch.alerts in llmshift.yaml)", "grey"))
        if not args.dry_run:
            con.out(con.style(f"  history: {_rel(outcome.history_file)}", "grey"))
        if outcome.compare is not None:
            md = report.compare_markdown(outcome.compare, title=f"LLMShift watch: {outcome.model}")
            _step_summary(md)
            if args.html is not None:
                html_path = args.html or Path(cfg.get("reports_dir") or ".llmshift/reports") / f"watch-{_timestamp()}.html"
                _write(html_path, report.compare_html(outcome.compare))
                con.out(con.style(f"  report: {_rel(html_path)}", "grey"))
        worst = outcome.worst
        exit_code = EXIT_FAIL if fail_on != "never" and worst != "ok" and SEVERITY_RANK.get(worst, 0) >= SEVERITY_RANK.get(fail_on, 1) else EXIT_OK
        if not loop:
            return exit_code
        nxt = f"at {args.at}" if args.at else f"in {args.every}"
        con.out(con.style(f"  next check {nxt} (Ctrl+C to stop)", "grey"))
        sleep_until_next(args.every, args.at)


def _watch_outage(model: str, exc: Exception, alerts: list, args: argparse.Namespace, w: dict) -> None:
    from .alerts import AlertMessage, deliver, run_url

    con.fail(f"every request failed: {exc}")
    if not alerts or args.dry_run:
        return
    msg = AlertMessage(title=f"{model}: every request failed", severity="critical", model=str(model), lines=[str(exc)], run_url=run_url())
    for name, err in deliver(alerts, msg, default_min_severity=args.alert_on or w.get("alert_on", "warning")):
        if err:
            con.warn(f"alert via {name} failed: {err}")
        else:
            con.out(con.style(f"  alert sent via {name}", "grey"))


# ----------------------------------------------------------------- import


def cmd_import(args: argparse.Namespace, cfg: dict) -> int:
    from .importers import openai_evals as oe
    from .infer import infer_checks, merge_checks
    from .testset import atomic_write, save_tests

    if args.source_kind != "openai-evals":
        raise UsageError(f"unknown import source {args.source_kind!r}")
    from_api = bool(args.api or args.eval_id or args.all)
    raw_path = None
    if args.save_raw is not None:
        if not from_api:
            raise UsageError("--save-raw works with --api (an exported file is already the raw copy)")
        raw_path = _raw_archive_path(args, cfg)
        _refuse_overwrite([raw_path], args.force, "--save-raw FILE")  # before any API call, not after the whole download
    if from_api:
        con.out(con.style(f"Reading evals from the OpenAI API (the platform turns read-only on {oe.READ_ONLY_DATE} and shuts down on {oe.SHUTDOWN_DATE})...", "grey"), err=True)
        result = oe.import_from_api(args.eval_id, all_evals=args.all, max_items=args.max_items, name_filter=args.eval, progress=lambda m: con.out(con.style(f"  {m}", "grey"), err=True), keep_raw=raw_path is not None)
    elif args.source:
        src = Path(args.source)
        if not src.exists():
            raise UsageError(f"{args.source} not found")
        if src.is_dir() or src.suffix.lower() in (".yaml", ".yml"):
            result = oe.import_oss_registry(args.source, eval_names=args.eval, max_items=args.max_items)
        else:
            result = oe.import_from_export(args.source, data=args.data, max_items=args.max_items)
    else:
        raise UsageError("pass a source: --api (with --all or --eval-id), an exported eval JSON file, or an openai/evals registry directory")
    if raw_path is not None:  # saved even when nothing converts: the graders and data sources are the record
        archive = {"source": "OpenAI Evals API", "saved_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "llmshift": __version__, "responses": result.raw}
        atomic_write(raw_path, json.dumps(archive, indent=1, ensure_ascii=False) + "\n")
        con.out(f"{con.style(con.sym('pass'), 'green')} saved {plural(len(result.raw), 'API response')} as read to {_rel(raw_path)}")
    if args.auto_checks:
        for case in result.cases:
            if case.has_baseline:
                inferred = infer_checks(case.messages, case.params, case.baseline_output, case.baseline_tool_calls)
                inferred = [dict(c, severity="warn") for c in inferred if c.get("type") not in ("not_empty",)]
                case.checks = merge_checks(case.checks, inferred)
    con.out(con.style("LLMShift import: OpenAI Evals", "bold"))
    rows = []
    for ev in result.evals:
        checks = ", ".join(f"{k} x{v}" for k, v in ev.checks.most_common())
        rows.append([ev.name, str(ev.cases), str(ev.with_baseline), checks or "-"])
    if rows:
        con.out(con.table(rows, ["Eval", "Cases", "With baseline", "Checks"], max_widths=[40, 6, 13, 60]))
    for w in result.warnings[:20]:
        con.warn(w)
    if len(result.warnings) > 20:
        con.warn(f"... {len(result.warnings) - 20} more warnings")
    for w in oe.grader_model_warnings(result.cases):
        con.warn(w)
    if not result.cases:
        con.fail("nothing was imported")
        return EXIT_FAIL
    if not args.no_verify:
        agreement = oe.verify_against_recorded(result.cases, allow_python=args.allow_python)
        if agreement.get("graded"):
            by = ", ".join(f"{k.split(':')[-1]} {v['agree']}/{v['total']}" for k, v in agreement["by_grader"].items())
            con.out(f"  Fidelity: the imported graders reproduce OpenAI's recorded verdicts on {agreement['agree']}/{agreement['graded']} graded outputs ({agreement['rate']:.0%}; {by})")
    if args.split:
        out_dir = Path(args.split)
        by_group: dict[str, list] = {}
        for c in result.cases:
            by_group.setdefault(c.group or "eval", []).append(c)
        _refuse_overwrite([out_dir / f"{g}.jsonl" for g in by_group], args.force)
        for g, cs in by_group.items():
            save_tests(cs, out_dir / f"{g}.jsonl")
        con.out(f"{con.style(con.sym('pass'), 'green')} wrote {len(result.cases)} test case(s) in {len(by_group)} file(s) under {_rel(out_dir)}")
        target = str(out_dir)
    else:
        out = args.output or cfg.get("tests") or ".llmshift/tests.jsonl"
        if Path(out).is_dir():
            out = str(Path(out) / "tests.jsonl")
        _refuse_overwrite([out], args.force)
        save_tests(result.cases, out)
        con.out(f"{con.style(con.sym('pass'), 'green')} wrote {len(result.cases)} test case(s) to {_rel(out)}")
        target = out
    needs_judge = sum(1 for c in result.cases for ch in c.checks if ch.get("type") in ("llm_label", "llm_score", "modelgraded"))
    con.out("")
    con.out(f"Next: llmshift compare --tests {target} --to <model>" + ("  --judge <model>" if needs_judge else ""))
    if any(ch.get("type") == "python" for c in result.cases for ch in c.checks):
        con.out(con.style("  This set contains python graders: add --allow-python to run them (they execute the grader code locally).", "grey"))
    return EXIT_OK


# ------------------------------------------------------------------ report


def cmd_report(args: argparse.Namespace, cfg: dict) -> int:
    from . import report

    result = _load_compare_json(args.results)
    con.out(report.compare_terminal(result, show=_show(args.show), verbose=args.verbose))
    base = Path(args.results)
    stem = base.with_suffix("") if base.suffix.lower() == ".json" else base
    for value, kind, text in ((args.html, "html", report.compare_html), (args.markdown, "markdown", report.compare_markdown), (args.junit, "junit", report.compare_junit)):
        if value is None:
            continue
        path = Path(value) if value else Path(str(stem) + _REPORT_SUFFIX[kind])  # default: next to the results file
        _write(path, text(result))
        con.out(f"wrote {_rel(path)}")
    return EXIT_OK


# -------------------------------------------------------------------- demo


def cmd_demo(args: argparse.Namespace, cfg: dict) -> int:
    from .config import load_config as _load
    from .demo import create_project

    try:
        root = create_project(args.dir, force=args.force)
    except FileExistsError as exc:
        raise UsageError(str(exc)) from None
    con.out(con.style("LLMShift demo", "bold") + "  (offline: simulated models, no API keys needed)")
    con.out(f"Created a demo project in {root}")
    con.out("The Acme support app calls gpt-4o-2024-05-13 and claude-3-5-haiku-20241022. Let's switch it to a new model.\n")
    cwd = os.getcwd()
    os.chdir(root)
    try:
        dcfg = _load(str(root / "llmshift.yaml"))
        steps = [
            ("1/5  llmshift scan", ["scan"]),
            ("2/5  llmshift snapshot logs/", ["snapshot", "logs"]),
            ("3/5  llmshift compare --to mock:nextgen", ["compare", "--to", "mock:nextgen", "--show", "3"]),
            ("4/5  llmshift fix --to mock:nextgen", ["fix", "--to", "mock:nextgen", "--write-tests", ".llmshift/tests.fixed.jsonl"]),
            ("5/5  llmshift pick --models mock:nextgen,mock:frugal --volume 100k", ["pick", "--models", "mock:nextgen,mock:frugal", "--volume", "100k"]),
        ]
        parser = build_parser()
        for title, argv in steps:
            con.out(con.style(con.sym("bar") * 4 + f" {title} " + con.sym("bar") * max(4, 60 - len(title)), "cyan", "bold"))
            sub = parser.parse_args(argv)
            sub.in_demo = True
            sub.func(sub, dcfg)
            con.out("")
        con.out(con.style("Demo complete.", "bold"))
        con.out(f"  Project: {root}")
        con.out(f"  Open the HTML report in {root / '.llmshift' / 'reports'} to browse every diff.")
        con.out("  With real models: llmshift snapshot <your logs> && llmshift compare --to openai:gpt-5.6-sol")
    finally:
        os.chdir(cwd)
    return EXIT_OK


# ------------------------------------------------------------------ doctor


def cmd_doctor(args: argparse.Namespace, cfg: dict) -> int:
    import platform

    from .registry import Registry

    problems = 0
    con.out(con.style("LLMShift doctor", "bold"))
    con.out(f"  llmshift {__version__} on Python {platform.python_version()} ({platform.system()})")
    cfg_path = getattr(cfg, "path", None)
    if cfg_path is None:
        con.out("  config: none found (defaults in use; run `llmshift init`)")
    else:
        con.out(f"  config: {cfg_path}" + (" ([tool.llmshift])" if Path(cfg_path).name == "pyproject.toml" else ""))
    reg = Registry.load()
    con.out(f"  lifecycle registry: {len(reg.entries)} models, updated {reg.updated}" + ("" if reg.origin == "bundled" else f" (downloaded: {reg.origin})"))
    from . import pricing

    origin = pricing.table_origin()
    con.out(f"  model prices: {pricing.table_date()}" + ("" if origin == "bundled" else f" (downloaded: {origin})") + "; refresh both with `llmshift update --prices`")
    tests = cfg.get("tests") or ".llmshift/tests.jsonl"
    if Path(tests).exists():
        from .testset import load_tests

        try:
            n = len(load_tests(tests))
            con.out(f"  tests: {tests} ({plural(n, 'case')})")
        except Exception as exc:
            problems += 1
            con.out(con.style(f"  tests: {tests} is invalid: {exc}", "red"))
    else:
        con.out(f"  tests: {tests} (not created yet)")
    keys = [
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("GEMINI_API_KEY", "gemini"),
        ("GOOGLE_API_KEY", "gemini (alt)"),
        ("AZURE_OPENAI_API_KEY", "azure"),
        ("MISTRAL_API_KEY", "mistral"),
        ("OPENROUTER_API_KEY", "openrouter"),
        ("GROQ_API_KEY", "groq"),
        ("DEEPSEEK_API_KEY", "deepseek"),
        ("XAI_API_KEY", "xai"),
    ]
    present = [f"{p}" for env, p in keys if os.environ.get(env)]
    con.out(f"  API keys found for: {', '.join(present) if present else 'none (only mock:, python: and local providers will work)'}")
    for spec in [cfg.get("baseline"), cfg.get("candidate"), cfg.get("judge")]:
        if spec:
            from .providers import resolve
            from .providers.http import display_url

            try:
                p = resolve(spec, cfg)
                base_url = getattr(p, "base_url", "")
                con.out(f"  {spec}: {p.__class__.__name__} ({display_url(base_url) if base_url else p.kind})")
            except Exception as exc:
                problems += 1
                con.out(con.style(f"  {spec}: {exc}", "red"))
    if problems:  # a CI step running doctor fails on what it found
        con.out(con.style(f"  {plural(problems, 'problem')} found", "red", "bold"))
        return EXIT_FAIL
    return EXIT_OK


def cmd_cache(args: argparse.Namespace, cfg: dict) -> int:
    from .cache import ResponseCache

    cache = ResponseCache(cfg.get("cache_dir") or ".llmshift/cache")
    n = cache.clear()
    con.out(f"removed {n} cached response(s) from {cache.dir}")
    return EXIT_OK


# ------------------------------------------------------------------ parser


def _add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tests", help="test file or directory (default: tests in llmshift.yaml, else .llmshift/tests.jsonl)")
    p.add_argument("--concurrency", type=positive_int, help="parallel requests (default 4)")
    p.add_argument("--limit", type=positive_int, help="only run the first N cases")
    p.add_argument("--group", action="append", help="only cases from this prompt template (repeatable)")
    p.add_argument("--tag", action="append", help="only cases with this tag (repeatable)")
    p.add_argument("--case", action="append", help="only this case id (repeatable)")
    p.add_argument("--drop-param", action="append", metavar="NAME", type=param_arg, help="leave this setting out of every test request, e.g. temperature once your code no longer sends it (repeatable; also drop_params: in llmshift.yaml)")
    p.add_argument("--judge", metavar="MODEL", type=model_arg, help="model for LLM-judged checks and imported model graders, e.g. openai:gpt-5.6-terra")
    p.add_argument("--judge-mode", choices=["auto", "always"], help="auto (default): ask the judge only when the rule-based checks cannot decide")
    p.add_argument("--allow-python", action="store_true", help="run imported python graders (executes their code locally)")
    p.add_argument("--severity", action="append", metavar="CHECK=LEVEL", help="override a check's severity, e.g. tone=error or facts=warn")
    p.add_argument("--no-cache", action="store_true", help="do not reuse cached model responses")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmshift",
        description="Switch AI models without breaking your product.",
        epilog="Run `llmshift demo` for an offline walkthrough. Docs: llmshift <command> --help",
    )
    parser.add_argument("--version", action="version", version=f"llmshift {__version__}")
    parser.add_argument("--config", help="path to llmshift.yaml")
    parser.add_argument("--no-color", action="store_true", help="disable coloured output")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("init", help="create llmshift.yaml (and GitHub workflows with --github)")
    p.add_argument("--candidate", type=model_arg, help="the model you are moving to")
    p.add_argument("--baseline", type=model_arg, help="the model you use today")
    p.add_argument("--github", action="store_true", help="also write .github/workflows for PR checks and nightly watch")
    p.add_argument("--path", help="where to write the config (default llmshift.yaml)")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("scan", help="find every model your code calls and warn about retirement dates")
    p.add_argument("paths", nargs="*", help="files or directories (default: .)")
    p.add_argument("--format", choices=["table", "json", "markdown", "sarif"], default="table")
    p.add_argument("--output", "-o", help="write the report to a file")
    p.add_argument("--fail-on", choices=["retired", "retiring", "deprecated", "unknown", "never"], help="exit 1 when models in this state are found (default: retired)")
    p.add_argument("--warn-days", type=_days, help="treat models retiring within N days as urgent (default 180)")
    p.add_argument("--include-docs", action="store_true", help="also scan Markdown/text files")
    p.add_argument("--exclude", action="append", help="glob of paths to skip (repeatable)")
    p.add_argument("--registry", action="append", help="extra lifecycle registry JSON (overrides bundled entries)")
    p.add_argument("--all", action="store_true", help="also list each healthy model's locations")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("models", help="look up model retirement dates and replacements")
    p.add_argument("query", nargs="?", help="substring of a model id, e.g. gpt-4o or claude-3")
    p.add_argument("--provider", help="openai, anthropic, google, google-vertex, mistral, cohere, xai, deepseek")
    p.add_argument("--status", choices=["active", "retiring", "deprecated", "retired"])
    p.add_argument("--retiring-within", type=_days, metavar="DAYS")
    p.add_argument("--warn-days", type=_days, default=180)
    p.add_argument("--registry", action="append")
    p.add_argument("--limit", type=positive_int, default=60)
    p.add_argument("--format", choices=["table", "json"], default="table")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser(
        "snapshot",
        help="turn real usage logs into a test set (no hand-written tests)",
        description="Turn LLM usage logs into a test set. Accepts JSONL/JSON/CSV (optionally .gz) in many formats: OpenAI chat/responses/batch, "
        "Anthropic, Gemini, Helicone, Langfuse/LangSmith-style exports, chat datasets, generic prompt/response rows, and llmshift.record logs.",
    )
    p.add_argument("sources", nargs="+", help="log files, directories or globs")
    p.add_argument("--output", "-o", help="test file to write (default .llmshift/tests.jsonl)")
    p.add_argument("--model", action="append", type=model_arg, help="only calls to this model (glob, repeatable), e.g. 'gpt-4o*'")
    p.add_argument("--since", help="only calls on/after this date (YYYY-MM-DD or ISO-8601)")
    p.add_argument("--until", help="only calls on/before this date")
    p.add_argument("--max-cases", type=positive_int, help="maximum test cases (default 200)")
    p.add_argument("--per-group", type=positive_int, help="maximum cases per prompt template (default 25)")
    p.add_argument("--seed", type=int, help="sampling seed")
    p.add_argument("--redact-pii", action="store_true", help="also replace e-mails, phone numbers, card numbers, IPs with placeholders")
    p.add_argument("--no-redact", action="store_true", help="do not redact API keys and tokens")
    p.add_argument("--include-errors", action="store_true", help="keep calls that failed")
    p.add_argument("--include-truncated", action="store_true", help="keep outputs cut off by the token limit")
    p.add_argument("--baseline-model", metavar="MODEL", type=model_arg, help="generate baseline outputs with this model when logs lack them")
    p.add_argument("--regenerate", action="store_true", help="re-generate every baseline with --baseline-model")
    p.add_argument("--with-judge-checks", action="store_true", help="add an LLM-judged equivalence check to free-text cases")
    p.add_argument("--append", action="store_true", help="add to an existing test file instead of replacing it")
    p.add_argument("--max-records", type=positive_int, metavar="N", help="calls held in memory (default 100000); larger logs keep a fair sample per prompt template")
    p.add_argument("--concurrency", type=positive_int)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("compare", help="run the tests on a new model and show exactly what changed")
    p.add_argument("--to", "--model", "--candidate", dest="to", metavar="MODEL", type=model_arg, help="the new model, e.g. openai:gpt-5.6-sol, anthropic:claude-sonnet-5-5")
    p.add_argument("--baseline-model", metavar="MODEL", type=model_arg, help="also run the old model live; failures it shares are treated as noise")
    p.add_argument("--embedder", metavar="MODEL", type=model_arg, help="embedding model for cosine similarity checks, e.g. openai:text-embedding-3-small")
    p.add_argument("--min-pass-rate", type=rate_arg, help="exit 1 below this pass rate (default from config: 0.95)")
    p.add_argument("--temperature", type=temperature_arg, help="override temperature for every request (0 to 2)")
    p.add_argument("--max-tokens", type=positive_int, help="override the output token limit for every request")
    p.add_argument("--show", type=show_arg, help="how many changed cases to print (number or 'all', default 5)")
    p.add_argument("--verbose", "-v", action="store_true", help="print every changed case with full diffs and skipped checks")
    p.add_argument("--html", nargs="?", const="", metavar="FILE", help="HTML report (default: in the reports directory)")
    p.add_argument("--json", nargs="?", const="", metavar="FILE", help="JSON results (default: in the reports directory)")
    p.add_argument("--markdown", nargs="?", const="", metavar="FILE", help="also write a Markdown summary, e.g. for PR comments")
    p.add_argument("--junit", nargs="?", const="", metavar="FILE", help="also write JUnit XML for CI test views")
    p.add_argument("--report-dir", help="where default reports go (default .llmshift/reports)")
    p.add_argument("--no-report", action="store_true", help="do not write the default HTML/JSON reports")
    p.add_argument("--repeat", type=positive_int, metavar="N", help="run each case N times: finds flaky cases and narrows the pass-rate range")
    _add_run_options(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("fix", help="suggest prompt changes until the new model matches the old one")
    p.add_argument("--to", "--model", "--candidate", dest="to", metavar="MODEL", type=model_arg, help="the new model")
    p.add_argument("--optimizer", metavar="MODEL", type=model_arg, help="model that rewrites prompts (optional; rule-based fixes work without it)")
    p.add_argument("--rounds", type=positive_int, help="improvement rounds per prompt template (default 3)")
    p.add_argument("--max-cases-per-group", type=positive_int, help="cases used to score each candidate prompt (default 12)")
    p.add_argument("--from-results", help="reuse a compare JSON instead of re-running the baseline comparison")
    p.add_argument("--write-tests", help="also write a copy of the test set with the suggested prompts applied")
    p.add_argument("--out", help="Markdown report path (a .json is written next to it)")
    p.add_argument("--report-dir")
    p.add_argument("--source", action="append", help="where to look for the prompts in your code (default: .)")
    p.add_argument("--no-validate", action="store_true", help="skip the final full re-run with all suggested prompts")
    _add_run_options(p)
    p.set_defaults(func=cmd_fix)

    p = sub.add_parser("watch", help="check the live model on a schedule and alert when its behaviour drifts")
    p.add_argument("--model", metavar="MODEL", type=model_arg, help="the live model to watch, e.g. openai:gpt-5.6-sol")
    p.add_argument("--history", help="history directory (default .llmshift/watch)")
    p.add_argument("--alert", action="append", help="slack:<url> | discord:<url> | teams:<url> | webhook:<url> | github | email:<addr> | file:<path>")
    p.add_argument("--alert-on", choices=["info", "warning", "critical"], help="lowest drift severity that sends alerts (default warning)")
    p.add_argument("--fail-on", choices=["info", "warning", "critical", "never"], help="exit 1 at or above this drift severity (default warning)")
    p.add_argument("--min-pass-rate", type=rate_arg, help="alert (critical) below this pass rate")
    p.add_argument("--max-drop", type=rate_arg, help="alert when the pass rate drops by this much (default 0.05)")
    p.add_argument("--every", help="keep running: repeat every interval, e.g. 24h, 6h, 30m")
    p.add_argument("--at", help="keep running: once a day at HH:MM local time")
    p.add_argument("--dry-run", action="store_true", help="do not write history")
    p.add_argument("--html", nargs="?", const="", metavar="FILE", help="also write an HTML report of this run (default: in the reports directory)")
    _add_run_options(p)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("import", help="import tests from other tools (OpenAI Evals)")
    p.add_argument("source_kind", choices=["openai-evals"], help="what to import")
    p.add_argument("source", nargs="?", help="exported eval JSON, or an openai/evals registry directory / YAML file")
    p.add_argument("--api", action="store_true", help="read evals from the OpenAI API (needs OPENAI_API_KEY)")
    p.add_argument("--all", action="store_true", help="with --api: import every eval in the project")
    p.add_argument("--eval-id", action="append", help="with --api: import this eval id (repeatable)")
    p.add_argument("--eval", action="append", help="only evals whose name matches this glob (repeatable)")
    p.add_argument("--data", help="data file for an exported eval, as {\"item\": ...} JSONL")
    p.add_argument("--max-items", type=positive_int, help="at most N items per eval")
    p.add_argument("--output", "-o", help="test file to write (default .llmshift/tests.jsonl)")
    p.add_argument("--split", help="write one test file per eval into this directory")
    p.add_argument("--auto-checks", action="store_true", help="also infer format/fact/tone checks from recorded outputs (as warnings)")
    p.add_argument("--allow-python", action="store_true", help="run python graders during the fidelity check")
    p.add_argument("--no-verify", action="store_true", help="skip re-grading recorded outputs to check fidelity")
    p.add_argument("--save-raw", nargs="?", const="", metavar="FILE", help="with --api: also keep every API response as read, in a dated JSON file next to the tests (or FILE)")
    p.add_argument("--force", action="store_true", help="replace existing test files")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("report", help="re-render a saved compare JSON (terminal, HTML, Markdown, JUnit)")
    p.add_argument("results", help="JSON written by compare")
    p.add_argument("--html", nargs="?", const="", metavar="FILE", help="write an HTML report (default: next to the results file)")
    p.add_argument("--markdown", nargs="?", const="", metavar="FILE", help="write a Markdown summary (default: next to the results file)")
    p.add_argument("--junit", nargs="?", const="", metavar="FILE", help="write JUnit XML (default: next to the results file)")
    p.add_argument("--show", type=show_arg)
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("demo", help="offline end-to-end walkthrough with simulated models")
    p.add_argument("--dir", help="where to create the demo project (default: a temp directory)")
    p.add_argument("--force", action="store_true", help="allow --dir to be a directory that is not empty")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("doctor", help="check configuration, API keys and the test file")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("cache", help="manage the response cache")
    p.add_argument("action", choices=["clear"])
    p.set_defaults(func=cmd_cache)

    from . import calibrate, embedmig, feed, mcp, pick, prcheck, safety, shadow, upgrade

    modules = {"calibrate": calibrate, "pick": pick, "safety": safety, "shadow": shadow, "mcp": mcp, "prcheck": prcheck, "upgrade": upgrade, "embedmig": embedmig, "feed": feed}
    for name in EXTENSIONS:
        modules[name].register(sub)
    return parser


def _global_options_first(argv: list[str]) -> list[str]:
    """`llmshift compare --config x.yaml` as well as `llmshift --config x.yaml compare`: the
    global options are moved in front of the subcommand before parsing (giving every
    subcommand its own copy would make abbreviations like --no-c ambiguous)."""
    front: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":  # what follows is the subcommand's, as it is
            rest.extend(argv[i:])
            break
        if arg == "--no-color" or arg.startswith("--config="):
            front.append(arg)
        elif arg == "--config" and i + 1 < len(argv):
            front += [arg, argv[i + 1]]
            i += 1
        else:
            rest.append(arg)
        i += 1
    return front + rest


def main(argv: list[str] | None = None) -> int:
    con.setup_streams()
    parser = build_parser()
    args = parser.parse_args(_global_options_first(list(sys.argv[1:] if argv is None else argv)))
    if args.no_color:
        con.set_color(False)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    from .providers.base import ProviderError
    from .runner import RunAborted
    from .testset import TestFileError

    try:
        try:
            cfg = load_config(args.config)
        except ConfigError:
            if args.command != "mcp":
                raise
            cfg = {}  # the server reads the config for every tool call, where it works: tools report the problem
        if args.command != "mcp":  # the server's output is the protocol
            for warning in getattr(cfg, "warnings", []):
                con.warn(warning)
        return int(args.func(args, cfg) or 0)
    except KeyboardInterrupt:
        con.err("\ninterrupted")
        return EXIT_INTERRUPTED
    except RunAborted as exc:
        con.fail(f"every request failed: {exc}")
        con.out("  Check the model name and that the provider's API key environment variable is set (`llmshift doctor`).", err=True)
        return EXIT_USAGE
    except (UsageError, ConfigError, TestFileError, ProviderError, FileNotFoundError, ValueError) as exc:
        con.fail(str(exc))
        if os.environ.get("LLMSHIFT_DEBUG"):
            traceback.print_exc()
        return EXIT_USAGE
    except OSError as exc:  # unreadable or unwritable files, full disks
        con.fail(f"{exc.filename}: {exc.strerror or exc}" if getattr(exc, "filename", None) else str(exc))
        if os.environ.get("LLMSHIFT_DEBUG"):
            traceback.print_exc()
        return EXIT_USAGE
    except Exception as exc:  # pragma: no cover - last resort
        con.fail(f"unexpected {exc.__class__.__name__}: {exc}")
        if os.environ.get("LLMSHIFT_DEBUG"):
            traceback.print_exc()
        else:
            con.out("  Re-run with LLMSHIFT_DEBUG=1 for a traceback.", err=True)
        return EXIT_FAIL


def entry() -> None:
    """Console-script entry point."""
    try:
        code = main()
    except SystemExit:  # --help and --version end inside argparse
        con.finish_output()
        raise
    con.finish_output()
    if code == EXIT_INTERRUPTED:
        # Worker threads may still be blocked in network reads: exit without waiting for them.
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        os._exit(code)
    sys.exit(code)


if __name__ == "__main__":  # pragma: no cover
    entry()
