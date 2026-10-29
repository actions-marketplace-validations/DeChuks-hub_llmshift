"""`llmshift judge label` and `llmshift judge calibrate`: check the AI judge against you.

Label about 20 old/new answer pairs yourself (`label` walks you through pairs from a
compare run and saves each answer at once). `calibrate` asks the judge about the same
pairs and reports how often it agrees with you, and whether it errs on the lenient or
the strict side. Every later `compare` that uses that judge quotes the agreement.

Labels are JSON lines you can also write by hand:
    {"id": "c_1", "messages": [...], "reference": "old answer", "candidate": "new answer", "label": "pass"}
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import console as con
from .errors import EXIT_FAIL, EXIT_OK, UsageError
from .textutil import json_loads, jsonl_lines, read_text

MIN_LABELS = 5
VALID = ("pass", "fail")


def _home(cfg: dict) -> Path:
    """The .llmshift folder next to the cache (config paths are already project-relative)."""
    return Path(cfg.get("cache_dir") or ".llmshift/cache").parent


def labels_path(cfg: dict, override: str | None = None) -> Path:
    return Path(override) if override else _home(cfg) / "judge-labels.jsonl"


def calibration_path(cfg: dict) -> Path:
    return _home(cfg) / "judge-calibration.json"


def load_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(jsonl_lines(read_text(path)), 1):
        if not line.strip():
            continue
        try:
            rec = json_loads(line)
        except (ValueError, RecursionError) as exc:
            raise UsageError(f"{path}:{n}: invalid JSON ({exc})") from None
        if not isinstance(rec, dict) or rec.get("label") not in VALID or not isinstance(rec.get("candidate"), str):
            raise UsageError(f"{path}:{n}: a label needs \"candidate\" (text) and \"label\": \"pass\" or \"fail\"")
        rec.setdefault("reference", "")
        if not isinstance(rec.get("messages"), list) or not rec["messages"]:
            rec["messages"] = [{"role": "user", "content": str(rec.get("input") or "")}]
        out.append(rec)
    return out


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _latest_results(cfg: dict) -> Path:
    reports = Path(cfg.get("reports_dir") or ".llmshift/reports")
    found = sorted(reports.glob("compare-*.json"), key=lambda p: p.stat().st_mtime) if reports.is_dir() else []
    if not found:
        raise UsageError(f"no compare results in {reports}; run `llmshift compare --to <model>` first or pass --results FILE")
    return found[-1]


def candidate_pairs(results_path: Path, count: int, seed: int, already: set[str]) -> list[dict]:
    """Pairs worth labelling: answered cases whose new answer differs from the old one,
    mixing failed, changed and passing cases."""
    from .cli import _load_compare_json
    from .testset import load_tests

    result = _load_compare_json(str(results_path))
    full: dict[str, Any] = {}
    if result.tests_path and Path(result.tests_path).exists():
        try:
            full = {c.id: c for c in load_tests(result.tests_path)}
        except Exception:
            full = {}
    pool: dict[str, list] = {"fail": [], "warn": [], "pass": []}
    for c in result.cases:
        if c.status not in pool or c.id in already or not c.baseline_output or not c.candidate_output:
            continue
        if c.baseline_output.strip() == c.candidate_output.strip():
            continue
        pool[c.status].append(c)
    rng = random.Random(seed)  # nosec B311  # seeded on purpose: the same sample on every run; nothing secret
    for items in pool.values():
        rng.shuffle(items)
    picked = []
    while len(picked) < count and any(pool.values()):
        for status in ("fail", "warn", "pass"):
            if pool[status] and len(picked) < count:
                picked.append(pool[status].pop())
    out = []
    for c in picked:
        case = full.get(c.id)
        messages = case.messages if case is not None else [{"role": "user", "content": c.input_preview}]
        out.append({"id": c.id, "group": c.group, "messages": messages, "reference": c.baseline_output, "candidate": c.candidate_output, "status": c.status})
    return out


def _show(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def cmd_label(args: Any, cfg: dict) -> int:
    from . import messages as msgs

    path = labels_path(cfg, args.labels)
    existing = load_labels(path)
    results = Path(args.results) if args.results else _latest_results(cfg)
    pairs = candidate_pairs(results, max(0, args.count - len(existing)) if not args.more else args.count, args.seed, {r.get("id") for r in existing})
    if not pairs:
        con.out(f"Nothing new to label: {len(existing)} label(s) in {path}." if existing else "No answered cases with changed output in those results.")
        return EXIT_OK
    if not sys.stdin.isatty():
        raise UsageError(
            "labelling is interactive; run it in a terminal, or write the labels file by hand (one JSON object per line with "
            '"messages", "reference", "candidate" and "label": "pass" or "fail")'
        )
    con.out(con.style(f"Label {len(pairs)} pair(s) from {results.name}: is the new answer as good as the old one?", "bold"))
    con.out(con.style("  y = yes, same facts and meaning   n = no, it is worse or different   s = skip   q = quit (labels are saved as you go)", "grey"))
    done = 0
    for i, pair in enumerate(pairs, 1):
        con.out("")
        con.out(con.style(f"Pair {i} of {len(pairs)}  ({pair['id']}, {pair['group']})", "bold"))
        con.out("  Task: " + _show(msgs.last_user_text(pair["messages"]) or msgs.system_prompt(pair["messages"]), 600))
        con.out(con.style("  Old answer:", "grey") + "\n    " + _show(pair["reference"], 1200).replace("\n", "\n    "))
        con.out(con.style("  New answer:", "grey") + "\n    " + _show(pair["candidate"], 1200).replace("\n", "\n    "))
        while True:
            try:
                answer = input("  Same quality? [y/n/s/q] ").strip().lower()
            except EOFError:
                answer = "q"
            if answer in ("y", "yes", "n", "no", "s", "skip", "q", "quit"):
                break
        if answer.startswith("q"):
            break
        if answer.startswith("s"):
            continue
        record = {k: pair[k] for k in ("id", "group", "messages", "reference", "candidate")}
        record["label"] = "pass" if answer.startswith("y") else "fail"
        record["labelled_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        _append(path, record)
        done += 1
    total = len(existing) + done
    con.out("")
    con.out(f"{con.style(con.sym('pass'), 'green')} saved {done} label(s); {total} in {path}")
    if total >= MIN_LABELS:
        con.out("Next: llmshift judge calibrate --judge <model>")
    else:
        con.out(f"Label at least {MIN_LABELS} pairs (about 20 is better) before calibrating.")
    return EXIT_OK


@dataclass
class Calibration:
    judge: str
    labels: str
    n: int = 0
    agree: int = 0
    too_lenient: int = 0  # the judge passed what you failed
    too_strict: int = 0  # the judge failed what you passed
    errors: int = 0
    kappa: float | None = None
    disagreements: list = field(default_factory=list)
    created_at: str = ""
    error: str | None = None  # why the judge gave no verdict (not saved)

    @property
    def rate(self) -> float:
        return self.agree / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {
            "judge": self.judge, "labels": self.labels, "n": self.n, "agree": self.agree, "rate": round(self.rate, 4),
            "kappa": None if self.kappa is None else round(self.kappa, 4), "too_lenient": self.too_lenient,
            "too_strict": self.too_strict, "errors": self.errors, "disagreements": self.disagreements[:20], "created_at": self.created_at,
        }


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Agreement beyond chance between judge and labels (1 = perfect, 0 = chance)."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = sum((sum(1 for a, _ in pairs if a == v) / n) * (sum(1 for _, b in pairs if b == v) / n) for v in VALID)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def run_calibration(labels: list[dict], judge: Any, *, labels_name: str = "", cache: Any = None, concurrency: int = 4) -> Calibration:
    from .checks import CheckContext, Sample, evaluate
    from .testset import TestCase

    ctx = CheckContext(judge=judge, judge_cache=cache, judge_mode="always")
    check = {"type": "llm_judge", "always": True}

    def verdict(rec: dict) -> str:
        case = TestCase(id=str(rec.get("id") or "label"), input={"messages": rec["messages"], "params": {}}, baseline={"output": rec.get("reference") or ""}, checks=[check])
        return evaluate(check, Sample(text=rec["candidate"]), case, ctx).status

    from .runner import interruptible_map

    verdicts = interruptible_map(verdict, labels, concurrency)
    cal = Calibration(judge=judge.spec, labels=labels_name, created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    compared = []
    for rec, got in zip(labels, verdicts):
        if got not in VALID:
            cal.errors += 1
            continue
        compared.append((got, rec["label"]))
        if got == rec["label"]:
            cal.agree += 1
            continue
        if got == "pass":
            cal.too_lenient += 1
        else:
            cal.too_strict += 1
        cal.disagreements.append({"id": rec.get("id"), "label": rec["label"], "judge": got, "candidate": _show(rec["candidate"], 200)})
    cal.n = len(compared)
    cal.kappa = cohen_kappa(compared)
    cal.error = ctx.judge_error
    return cal


def cmd_calibrate(args: Any, cfg: dict) -> int:
    from .cli import _cache
    from .providers import resolve

    spec = args.judge or cfg.get("judge")
    if not spec:
        raise UsageError("which judge? pass --judge provider:model or set judge in llmshift.yaml")
    path = labels_path(cfg, args.labels)
    labels = load_labels(path)
    if len(labels) < MIN_LABELS:
        raise UsageError(f"{path} has {len(labels)} label(s); label at least {MIN_LABELS} pairs (about 20 is better) with `llmshift judge label`")
    judge = resolve(spec, cfg)
    con.out(con.style(f"Asking {judge.spec} about {len(labels)} labelled pair(s)...", "grey"), err=True)
    cal = run_calibration(labels, judge, labels_name=str(path), cache=_cache(cfg, args), concurrency=args.concurrency or int(cfg.get("concurrency", 4)))
    if not cal.n:  # nothing to measure: an earlier check of this judge stays as it was
        con.fail(f"{judge.spec} gave no usable verdict on {len(labels)} pair(s)" + (f": {cal.error}" if cal.error else ""))
        return EXIT_FAIL
    out = calibration_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    judges = _calibrations(out)
    judges[cal.judge] = cal.to_dict()  # one result per judge: checking another judge keeps this one
    out.write_bytes((json.dumps({"judges": judges}, indent=2, ensure_ascii=False) + "\n").encode("utf-8", "replace"))
    if args.format == "json":
        con.out(json.dumps(cal.to_dict(), indent=2, ensure_ascii=False))
    else:
        con.out(con.style("LLMShift judge calibration", "bold") + f"  {cal.judge}")
        if cal.n:
            con.out(f"  agrees with your labels on {cal.agree} of {cal.n} pairs ({cal.rate:.0%})" + (f", kappa {cal.kappa:.2f}" if cal.kappa is not None else ""))
            if cal.too_lenient:
                con.out(f"  too lenient {cal.too_lenient}x: passed an answer you marked as worse")
            if cal.too_strict:
                con.out(f"  too strict {cal.too_strict}x: failed an answer you accepted")
        if cal.errors:
            why = f": {cal.error}" if cal.error else " (no verdict in the reply)"
            con.out(con.style(f"  {cal.errors} pair(s) got no verdict from the judge{why}", "yellow"))
        for d in cal.disagreements[:5]:
            con.out(con.style(f"    {d['id']}: you said {d['label']}, the judge said {d['judge']} - {d['candidate']}", "grey"))
        con.out(con.style(f"  saved to {out}; `llmshift compare` will quote this agreement", "grey"))
    if not cal.n or cal.rate + 1e-9 < args.min_agreement:
        if cal.n:
            con.warn(f"agreement {cal.rate:.0%} is below {args.min_agreement:.0%}: add a rubric to the llm_judge checks or use a stronger judge")
        return EXIT_FAIL
    return EXIT_OK


def _calibrations(path: Path) -> dict:
    """{judge spec: calibration} from the calibration file (also the older one-judge format)."""
    try:
        data = json_loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError, RecursionError):
        return {}
    if isinstance(data, dict) and isinstance(data.get("judges"), dict):
        return {str(k): v for k, v in data["judges"].items() if isinstance(v, dict)}
    if isinstance(data, dict) and data.get("judge"):
        return {str(data["judge"]): data}
    return {}


def calibration_note(judge_spec: str, cfg: dict) -> str | None:
    """The line `compare` adds when a judge was used."""
    judges = _calibrations(calibration_path(cfg))
    if not judges:
        return "The judge has not been checked against your own judgement yet: run `llmshift judge label`, then `llmshift judge calibrate`."
    data = judges.get(judge_spec)
    if data is None:
        return f"Your labels were checked with {', '.join(sorted(judges))}; run `llmshift judge calibrate --judge {judge_spec}` to check this judge."
    rate, n = data.get("rate"), data.get("n")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (rate, n)):
        return f"The saved check of judge {judge_spec} cannot be read; run `llmshift judge calibrate --judge {judge_spec}` again."
    when = str(data.get("created_at") or "")[:10]
    note = f"Judge {judge_spec} agrees with your labels on {data.get('agree', 0)} of {n} pairs ({rate:.0%}; checked {when})."
    if rate < 0.8:
        note += " Treat its verdicts with care."
    return note


def register(sub: Any) -> None:
    from .cli import _add_run_options

    p = sub.add_parser("judge", help="check the AI judge against your own labels (label, calibrate)")
    jsub = p.add_subparsers(dest="judge_command", metavar="<label|calibrate>")
    jsub.required = True
    lp = jsub.add_parser("label", help="label old/new answer pairs from a compare run (interactive)")
    lp.add_argument("--results", help="compare JSON (default: the newest in the reports directory)")
    lp.add_argument("--labels", help="labels file (default .llmshift/judge-labels.jsonl)")
    lp.add_argument("--count", type=int, default=20, help="how many labels to collect in total (default 20)")
    lp.add_argument("--more", action="store_true", help="label --count more pairs even if you already have enough")
    lp.add_argument("--seed", type=int, default=7)
    lp.set_defaults(func=cmd_label)
    cp = jsub.add_parser("calibrate", help="ask the judge about your labelled pairs and report the agreement")
    cp.add_argument("--labels", help="labels file (default .llmshift/judge-labels.jsonl)")
    cp.add_argument("--min-agreement", type=float, default=0.8, help="exit 1 below this agreement (default 0.8)")
    cp.add_argument("--format", choices=["table", "json"], default="table")
    _add_run_options(cp)
    cp.set_defaults(func=cmd_calibrate)
