"""Checks that need a model or code to grade: LLM judges, OpenAI Evals graders
(label_model, score_model, python, multi) and openai/evals model-graded specs.

Judge selection: an explicit --judge always wins (useful when a grader's own
model is being retired); otherwise a grader that names its model uses it;
otherwise the check is skipped with a clear message.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
import string
import subprocess  # nosec B404  # python graders run in a child Python process (see SECURITY.md)
import sys
import tempfile
import threading
from typing import Any

from .formats import parse_json_lenient
from .messages import content_text
from .templating import render
from .textutil import shorten

# --------------------------------------------------------------- judge calls


def _judge_for(check: dict, ctx: Any) -> Any:
    if ctx.judge is not None:
        return ctx.judge
    model = check.get("model")
    if model and ctx.resolve_model is not None:
        return ctx.resolve_model(model)
    return None


def _no_judge(check: dict) -> Any:
    from .checks import SKIP

    model = check.get("model")
    hint = f" (grader model {model} is not available)" if model else ""
    return SKIP(f"needs a judge model{hint}: pass --judge provider:model", reason="needs_judge")


def ask(provider: Any, messages: list[dict], ctx: Any, params: dict | None = None) -> str:
    """One deterministic judge call, cached."""
    request = {"messages": messages, "params": {"temperature": 0, **(params or {})}}
    cache = getattr(ctx, "judge_cache", None)
    key = None
    if cache is not None and cache.enabled:
        key = cache.key(provider.identity(), request, "judge")
        hit = cache.get(key)
        if hit is not None:
            return hit.text
    try:
        resp = provider.complete(request)
    except Exception as exc:
        with _COUNT_LOCK:
            ctx.judge_failed = getattr(ctx, "judge_failed", 0) + 1
            if not getattr(ctx, "judge_error", None):
                ctx.judge_error = f"{exc}"
        raise
    with _COUNT_LOCK:  # checks run in worker threads
        ctx.judge_calls += 1
    if key is not None:
        cache.put(key, resp)
    return resp.text or ""


_COUNT_LOCK = threading.Lock()
_NUM = r"(-?\d+(?:\.\d+)?)"


def score_from_text(reply: str, lo: float, hi: float) -> float:
    """A score from a free-text reply: "score: 3", "3/10", "3 out of 10", else the
    last in-range number; 0 when there is none (OpenAI's rule for non-numeric output)."""
    # Bounded spaces, and a number from the start of its digits: with \s* and a number from
    # any digit, a reply of "score" and 40,000 spaces (or 40,000 digits) took 20 to 40 s.
    for pattern in (
        rf"(?i)\b(?:score|result|rating|grade)\b\s{{0,10}}(?:(?:is|of|:|=)\s{{0,10}})?{_NUM}",
        rf"(?<![\d.]){_NUM}\s{{0,10}}(?:/|out of)\s{{0,10}}\d+(?:\.\d+)?",
    ):
        m = re.search(pattern, reply)
        if m:
            return float(m.group(1))
    in_range = [float(x) for x in re.findall(_NUM, reply) if lo <= float(x) <= hi]
    return in_range[-1] if in_range else 0.0


def find_label(reply: str, labels: list[str]) -> str | None:
    """The label a free-text reply settles on: whole words only ("incorrect" is not
    "correct"), the last one mentioned, the longest when several match there."""
    hits: list[tuple[int, int, str]] = []
    for lab in labels:
        for m in re.finditer(r"(?<![\w-])" + re.escape(lab) + r"(?![\w-])", reply, re.I):
            hits.append((m.start(), len(lab), lab))
    if not hits:
        return None
    return max(hits)[2]


def _messages_from_template(template: Any, namespaces: dict) -> list[dict]:
    rendered = render(template, namespaces)
    if isinstance(rendered, str):
        return [{"role": "user", "content": rendered}]
    out = []
    for m in rendered or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        if role not in ("system", "developer", "user", "assistant"):
            role = "user"
        out.append({"role": role, "content": content_text(m.get("content"))})
    return out or [{"role": "user", "content": ""}]


# ------------------------------------------------------------------ llm_judge

JUDGE_SYSTEM = (
    "You are a meticulous evaluator. A product is switching from one AI model to another. "
    "You compare the new model's answer with the reference answer the old model gave for the same input, "
    "following the criteria exactly. Reply with JSON only."
)
DEFAULT_CRITERIA = (
    "PASS if the new answer conveys the same facts, decisions and meaning as the reference and does not contradict it. "
    "Wording, length and formatting may differ unless the criteria below say otherwise. "
    "FAIL if a fact, number, name, decision or recommendation differs, is missing, or if the new answer refuses."
)
# A verdict word in a reply in prose, and what negates it: "not", "n't", "never", "cannot" or
# "unable to" up to three words before (not across "and" or "but"), or "no" right before.
# "does not pass" is a fail and "doesn't fail" a pass; "not only passes" is not negated.
_PROSE_VERDICT_RE = re.compile(
    r"(\bno\s+|(?:\b(?:not|never|cannot|unable\s+to)|n[’']t)\s+(?!only\b)(?:(?!(?:and|but|or|so|yet)\b)\w+\s+){0,3}?)?"
    r"\b(pass|fail)(?:ed|es|s|ing)?\b",
    re.I,
)


def c_llm_judge(check: dict, sample: Any, case: Any, ctx: Any) -> Any:
    from .checks import FAIL, PASS, Outcome

    judge = _judge_for(check, ctx)
    if judge is None:
        return _no_judge(check)
    task = "\n\n".join(f"[{m.get('role')}] {content_text(m.get('content'))}" for m in case.messages)[-6000:]
    reference = check.get("reference", case.baseline_output)
    criteria = check.get("rubric") or check.get("criteria") or DEFAULT_CRITERIA
    parts = [f"[Task given to the model]\n{task}"]
    if reference:
        parts.append(f"[Reference answer from the old model]\n{reference[:6000]}")
    parts.append(f"[New answer to evaluate]\n{(sample.text or json.dumps(sample.tool_calls))[:6000]}")
    parts.append(f"[Criteria]\n{criteria}")
    parts.append('Return exactly: {"verdict": "pass" or "fail", "reason": "<one sentence>", "differences": ["<short difference>"]}')
    reply = ask(judge, [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}], ctx,
                {"response_format": {"type": "json_object"}})
    data = parse_json_lenient(reply)
    verdict = None
    reason = ""
    diffs: list = []
    if isinstance(data, dict):
        verdict = str(data.get("verdict", "")).strip().lower()
        reason = str(data.get("reason", ""))
        diffs = [str(d) for d in (data.get("differences") or []) if d][:6]
    if verdict not in ("pass", "fail"):
        explicit = re.search(r"(?i)\bverdict\W{0,5}(pass|fail)", reply)
        prose = _PROSE_VERDICT_RE.search(reply)
        # An explicit "Verdict: FAIL" wins; otherwise the first verdict word, since later
        # mentions tend to be hypothetical ("it would pass only if ...").
        if explicit:
            verdict = explicit.group(1).lower()
        elif prose:
            verdict = prose.group(2).lower()
            if prose.group(1):  # "does not pass", "doesn't fail"
                verdict = "fail" if verdict == "pass" else "pass"
        else:
            verdict = None
    if verdict is None:
        return Outcome("error", f"judge reply not understood: {shorten(reply, 120)}")
    details = {"judge": judge.spec, "reason": reason, "differences": diffs}
    if verdict == "pass":
        return PASS(reason, **details)
    return FAIL("; ".join(diffs) if diffs else (reason or "judge found a difference"), **details)


# ------------------------------------------------------------------ llm_label


def c_llm_label(check: dict, sample: Any, case: Any, ctx: Any) -> Any:
    """OpenAI Evals `label_model` grader."""
    from .checks import FAIL, PASS, Outcome

    judge = _judge_for(check, ctx)
    if judge is None:
        return _no_judge(check)
    labels = [str(x) for x in check.get("labels") or []]
    passing = {str(x) for x in check.get("passing_labels") or []}
    if not labels:
        return Outcome("error", "label grader has no labels")
    messages = _messages_from_template(check.get("input") or [], {"item": case.vars, "sample": sample.namespace()})
    instruction = "Respond with a JSON object {\"label\": \"<label>\"} where <label> is exactly one of: " + ", ".join(labels) + "."
    messages[-1]["content"] = (messages[-1]["content"] + "\n\n" + instruction).strip()
    schema = {"type": "object", "properties": {"label": {"type": "string", "enum": labels}}, "required": ["label"], "additionalProperties": False}
    params = {"response_format": {"type": "json_schema", "json_schema": {"name": "label", "schema": schema, "strict": True}}}
    params.update(_sampling(check))
    reply = ask(judge, messages, ctx, params)
    label = None
    data = parse_json_lenient(reply)
    if isinstance(data, dict) and str(data.get("label")) in labels:
        label = str(data["label"])
    if label is None:
        label = find_label(reply, labels)
    if label is None:
        return Outcome("error", f"grader reply has no valid label: {shorten(reply, 100)}")
    if label in passing:
        return PASS(f"label {label!r}", label=label)
    return FAIL(f"label {label!r} (passing: {', '.join(sorted(passing))})", label=label)


def _sampling(check: dict) -> dict:
    sp = check.get("sampling_params") or {}
    out = {}
    for key in ("temperature", "top_p", "seed", "reasoning_effort"):
        if sp.get(key) is not None:
            out[key] = sp[key]
    mt = sp.get("max_completion_tokens") or sp.get("max_completions_tokens") or sp.get("max_tokens")
    if mt:
        out["max_completion_tokens"] = mt
    return out


# ------------------------------------------------------------------ llm_score


def c_llm_score(check: dict, sample: Any, case: Any, ctx: Any) -> Any:
    """OpenAI Evals `score_model` grader."""
    from .checks import FAIL, PASS

    judge = _judge_for(check, ctx)
    if judge is None:
        return _no_judge(check)
    lo, hi = (check.get("range") or [0, 1])[:2]
    lo, hi = float(lo), float(hi)
    threshold = float(check["pass_threshold"]) if check.get("pass_threshold") is not None else (lo + hi) / 2
    messages = _messages_from_template(check.get("input") or [], {"item": case.vars, "sample": sample.namespace()})
    instruction = f'Respond with a JSON object {{"steps": [{{"description": "...", "conclusion": "..."}}], "result": <a number from {lo:g} to {hi:g}>}}.'
    messages[-1]["content"] = (messages[-1]["content"] + "\n\n" + instruction).strip()
    schema = {
        "type": "object",
        "properties": {
            "steps": {"type": "array", "items": {"type": "object", "properties": {"description": {"type": "string"}, "conclusion": {"type": "string"}}, "required": ["description", "conclusion"], "additionalProperties": False}},
            "result": {"type": "number"},
        },
        "required": ["steps", "result"],
        "additionalProperties": False,
    }
    params = {"response_format": {"type": "json_schema", "json_schema": {"name": "score", "schema": schema, "strict": True}}}
    params.update(_sampling(check))
    reply = ask(judge, messages, ctx, params)
    data = parse_json_lenient(reply)
    score = None
    if isinstance(data, dict):
        try:
            score = float(data.get("result"))
        except (TypeError, ValueError):
            score = None
    if score is None or not math.isfinite(score):
        score = score_from_text(reply, lo, hi)
    score = max(lo, min(hi, score))
    norm = (score - lo) / (hi - lo) if hi > lo else score
    msg = f"score {score:g} (pass >= {threshold:g})"
    return PASS(msg, score=norm, raw_score=score) if score >= threshold else FAIL(msg, score=norm, raw_score=score)


# -------------------------------------------------------------- modelgraded

ANSWER_PROMPTS = {
    "classify": "Answer the question by printing only a single choice from {choices} (without quotes or punctuation) corresponding to the correct answer with no other text.",
    "classify_cot": "First, answer by printing a single choice from {choices} (without quotes or punctuation) corresponding to the correct answer. Then, from the next line, explain your reasonings step by step.",
    "cot_classify": "First, write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Then print only a single choice from {choices} (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the answer by itself on a new line.\n\nReasoning:",
}
INVALID = "__invalid__"


def chat_to_text(prompt: Any) -> str:
    """openai/evals chat_prompt_to_text_prompt(for_completion=False)."""
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return str(prompt)
    if len(prompt) == 1 and isinstance(prompt[0], dict):
        return content_text(prompt[0].get("content"))
    prefixes = {"system": "", "example_user": "User: ", "example_assistant": "Assistant: "}
    out = ""
    for msg in prompt:
        if not isinstance(msg, dict):
            continue
        role = msg.get("name") or msg.get("role") or "user"
        out += prefixes.get(role, role.capitalize() + ": ") + content_text(msg.get("content")) + "\n"
    return out.lstrip()


class _Lenient(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _get_choice(text: str, eval_type: str, match_fn: str, choices: list[str]) -> str:
    fns = {
        "include": lambda x, y: x in y,
        "exact": lambda x, y: x == y,
        "endswith": lambda x, y: x.endswith(y),
        "starts_or_endswith": lambda x, y: x.startswith(y) or x.endswith(y),
    }
    fn = fns.get(match_fn or "starts_or_endswith", fns["starts_or_endswith"])
    lines = text.strip().split("\n")
    if (eval_type or "").startswith("cot_classify"):
        lines = lines[::-1]
    punct = set(string.punctuation)
    for line in lines:
        line = "".join(c for c in line.strip() if c not in punct)
        if not line:
            continue
        for choice in choices:
            if fn(line, choice):
                return choice
    return INVALID


def c_modelgraded(check: dict, sample: Any, case: Any, ctx: Any) -> Any:
    """openai/evals ModelBasedClassify (e.g. the `fact` and `closedqa` specs)."""
    from .checks import FAIL, PASS, Outcome

    judge = _judge_for(check, ctx)
    if judge is None:
        return _no_judge(check)
    choices = check.get("choice_strings") or []
    if isinstance(choices, str):
        choices = list(choices)
    if not choices:
        return Outcome("error", "model-graded spec has no choice_strings")
    values = _Lenient()
    for key, value in (case.vars or {}).items():
        values[key] = chat_to_text(value) if key == "input" else (repr(value) if isinstance(value, list) else str(value))
    values.update({k: str(v) for k, v in (check.get("args") or {}).items()})
    out_key = (check.get("input_outputs") or {"input": "completion"}).get("input", "completion")
    values[out_key] = sample.text
    prompt = check.get("prompt") or ""
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt.format_map(values)}]
    else:
        messages = [{"role": m.get("role", "user"), "content": str(m.get("content", "")).format_map(values)} for m in prompt]
    eval_type = check.get("eval_type")
    if eval_type in ANSWER_PROMPTS:
        choice_text = " or ".join(f'"{c}"' for c in choices)
        messages[-1]["content"] += "\n\n" + ANSWER_PROMPTS[eval_type].format(choices=choice_text)
    reply = ask(judge, messages, ctx, {"max_tokens": 1024})
    choice = _get_choice(reply, check.get("spec_eval_type") or eval_type or "classify", check.get("match_fn"), choices)
    scores = check.get("choice_scores")
    score = None
    if scores == "from_strings":
        try:
            score = float(choice)
        except ValueError:
            score = None
    elif isinstance(scores, dict) and scores:
        score = float(scores[choice]) if choice in scores else min(float(v) for v in scores.values())
    passing = check.get("passing_choices")
    if passing is not None:
        ok = choice in passing
    elif score is not None and isinstance(scores, dict):
        vals = [float(v) for v in scores.values()]
        ok = score >= float(check.get("pass_threshold", (min(vals) + max(vals)) / 2))
    else:
        ok = choice != INVALID
    msg = f"grader chose {choice!r}" if choice != INVALID else f"grader reply not parsable: {shorten(reply, 80)}"
    return PASS(msg, score=score, choice=choice) if ok else FAIL(msg, score=score if score is not None else 0.0, choice=choice)


# --------------------------------------------------------------------- python

# The grader's source is written to a file and imported like any module, in a separate
# Python process, so llmshift never hands it to Python's exec or eval.
_GRADER_MODULE = "llmshift_grader"
_RUNNER = r"""
import json, math, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
payload = json.loads(sys.stdin.read())
try:
    import llmshift_grader
    grade = getattr(llmshift_grader, "grade", None)
    if not callable(grade):
        raise ValueError("the source defines no grade(sample, item) function")
    value = float(grade(payload["sample"], payload["item"]))
    if not math.isfinite(value):
        raise ValueError("grade() returned a non-finite number")
    out = {"score": value}
except Exception as exc:
    out = {"error": type(exc).__name__ + ": " + str(exc)}
sys.stdout.write("\n__LLMSHIFT__" + json.dumps(out))
"""


def c_python(check: dict, sample: Any, case: Any, ctx: Any) -> Any:
    from .checks import FAIL, PASS, SKIP

    if not ctx.allow_python:
        return SKIP("python grader not run: pass --allow-python (or set allow_python_checks: true) to execute grader code", reason="python_disabled")
    ns = sample.namespace()
    ns["output_audio"] = {}
    payload = json.dumps({"sample": ns, "item": case.vars or {}}, ensure_ascii=False, default=str)
    threshold = float(check.get("pass_threshold", 0.5))
    try:
        with tempfile.TemporaryDirectory(prefix="llmshift-grader-") as tmp:
            with open(os.path.join(tmp, _GRADER_MODULE + ".py"), "w", encoding="utf-8") as fh:
                fh.write(check.get("source") or "")
            proc = subprocess.run(  # nosec B603  # this Python, by full path, with a fixed runner; no shell
                [sys.executable, "-c", _RUNNER],
                input=payload,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(check.get("timeout", 120)),
                cwd=tmp,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
    except subprocess.TimeoutExpired:
        return FAIL("python grader timed out (scored 0)", score=0.0)
    marker = proc.stdout.rfind("__LLMSHIFT__")
    if marker < 0:
        return FAIL(f"python grader crashed (scored 0): {shorten(proc.stderr.strip(), 160)}", score=0.0)
    result = json.loads(proc.stdout[marker + len("__LLMSHIFT__") :])
    if "error" in result:
        return FAIL(f"python grader error (scored 0): {shorten(result['error'], 160)}", score=0.0)
    score = float(result["score"])
    msg = f"score {score:g} (pass >= {threshold:g})"
    return PASS(msg, score=score) if score >= threshold else FAIL(msg, score=score)


# ---------------------------------------------------------------------- multi

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.BitXor: operator.pow, ast.Mod: operator.mod}
_FUNCS = {"min": min, "max": max, "abs": abs, "floor": math.floor, "ceil": math.ceil, "exp": math.exp, "sqrt": math.sqrt, "log": math.log}


def safe_eval(expr: str, names: dict) -> float:
    """Evaluate an arithmetic formula over grader scores without Python's eval."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"unknown name {node.id!r} in formula")
            return float(names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return float(_BIN_OPS[type(node.op)](ev(node.left), ev(node.right)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = ev(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return float(_FUNCS[node.func.id](*[ev(a) for a in node.args]))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")

    return ev(ast.parse(expr.strip(), mode="eval"))


def c_multi(check: dict, sample: Any, case: Any, ctx: Any) -> Any:
    from .checks import FAIL, PASS, Outcome, evaluate

    graders = check.get("graders") or {}
    scores: dict[str, float] = {}
    for name, sub in graders.items():
        res = evaluate(sub, sample, case, ctx)
        if res.status == "skip":
            return Outcome("skip", f"sub-grader {name!r} skipped: {res.message}")
        scores[name] = res.score if res.score is not None else (1.0 if res.status == "pass" else 0.0)
    try:
        value = safe_eval(check.get("calculate_output") or "0", scores)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        return Outcome("error", f"cannot evaluate {check.get('calculate_output')!r}: {exc}")
    threshold = float(check.get("pass_threshold", 0.5))
    msg = f"score {value:.3g} from {', '.join(f'{k}={v:.2g}' for k, v in scores.items())}"
    return PASS(msg, score=value) if value >= threshold else FAIL(msg, score=value)
