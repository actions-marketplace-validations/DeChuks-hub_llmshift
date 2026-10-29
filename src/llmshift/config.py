"""Configuration: defaults, `llmshift.yaml` discovery and environment expansion.

Lookup order: --config PATH, then the nearest llmshift.yaml / llmshift.yml /
.llmshift.yaml / .llmshift/config.yaml or pyproject.toml with a
[tool.llmshift] table, in the current directory or a parent up to the git root
or your home directory (never a drive root, never a file another user owns:
see find_config). Relative paths in the file are relative to the file. Values
like ${VAR} or ${VAR:-default} are expanded from the environment.
"""

from __future__ import annotations

import copy
import difflib
import os
import re
from pathlib import Path
from typing import Any, Iterable

from .textutil import read_text

CONFIG_NAMES = ("llmshift.yaml", "llmshift.yml", ".llmshift.yaml", ".llmshift.yml", ".llmshift/config.yaml")

DEFAULTS: dict[str, Any] = {
    "tests": ".llmshift/tests.jsonl",
    "baseline": None,
    "candidate": None,
    "judge": None,
    "embedder": None,
    "concurrency": 4,
    "timeout": 120,
    "max_retries": 4,
    "cache": True,
    "cache_dir": ".llmshift/cache",
    "reports_dir": ".llmshift/reports",
    "min_pass_rate": 0.95,  # nosec B105  # a check status or setting name, not a password
    "allow_python_checks": False,
    "severity": {},
    "providers": {},
    "scan": {
        "paths": ["."],
        "exclude": [],
        "warn_days": 180,
        "fail_on": "retired",
        "include_docs": False,
        "registry": None,
    },
    "snapshot": {
        "max_cases": 200,
        "per_group": 25,
        "redact_secrets": True,
        "redact_pii": False,
        "seed": 7,
    },
    "fix": {
        "rounds": 3,
        "optimizer": None,
        "target_pass_rate": 1.0,  # nosec B105  # a check status or setting name, not a password
        "max_cases_per_group": 12,
    },
    "watch": {
        "history": ".llmshift/watch",
        "max_pass_rate_drop": 0.05,  # nosec B105  # a check status or setting name, not a password
        "min_pass_rate": None,  # nosec B105  # a check status or setting name, not a password
        "max_similarity_drop": 0.10,
        "retirement_warn_days": 30,
        "fail_on": "warning",
        "alerts": [],
    },
}

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
# Settings that hold paths (relative to the config file) and settings that are lists.
_PATH_KEYS = (("tests",), ("cache_dir",), ("reports_dir",), ("scan", "registry"), ("scan", "paths"), ("watch", "history"))
_LIST_KEYS = (("scan", "paths"), ("scan", "exclude"), ("watch", "alerts"))


class ConfigError(ValueError):
    pass


def _env_sub(m: re.Match) -> str:
    current = os.environ.get(m.group(1))
    if m.group(2) is not None:
        return current or m.group(2)  # ${VAR:-default}: also when VAR is empty, as in the shell
    return current or ""


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(_env_sub, value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def find_config(start: str | os.PathLike | None = None, *, stop: str | os.PathLike | None = None, notes: list | None = None) -> Path | None:
    """The nearest config file (or pyproject.toml with [tool.llmshift]) at or above `start`.

    A config can name programs to run, so the search stays inside what is yours, as git
    has done since CVE-2022-24765: it stops at the git root, at `stop` (the project an MCP
    server was started on) and at your home directory when `start` is inside it. It never
    reads a drive root or "/" (any user can create C:\\.llmshift), and on Linux and macOS it
    skips files that another user owns. On Windows, where owners are not cheap to check,
    a folder outside your home directory is searched above `start` only up to its git root.
    Skipped files are noted in `notes`."""
    here = Path(start or os.getcwd()).resolve()
    home = _home()
    in_home = home is not None and _within(here, home)
    limit = Path(stop).resolve() if stop else None
    chain = [here, *here.parents]
    if _windows_rules() and not in_home:
        root = next((d for d in chain if d.parent != d and (d / ".git").exists()), None)
        chain = chain[: chain.index(root) + 1] if root is not None else [here]
    for directory in chain:
        if directory != here and directory.parent == directory:
            break  # a drive root or "/": shared, and no project lives there
        for candidate in [directory / name for name in CONFIG_NAMES] + [directory / "pyproject.toml"]:
            if not candidate.is_file():
                continue
            if not _owned_by_you(candidate):
                if notes is not None:
                    notes.append(f"ignored {candidate}: another user owns it, and a config can name programs to run")
                continue
            if candidate.name == "pyproject.toml" and _load_pyproject(candidate) is None:
                continue
            return candidate
        if (directory / ".git").exists() or directory == limit or (in_home and directory == home):
            break
    return None


def _home() -> Path | None:
    try:
        return Path.home().resolve()
    except (RuntimeError, OSError):  # no home directory (some service accounts)
        return None


def _within(path: Path, folder: Path) -> bool:
    inner, outer = os.path.normcase(str(path)), os.path.normcase(str(folder)).rstrip("\\/")
    return inner == outer or inner.startswith(outer + os.sep)


def _windows_rules() -> bool:
    return os.name == "nt"


def _owned_by_you(path: Path) -> bool:
    """On Linux and macOS: the file is yours or root's. Root itself reads everything, as in
    CI containers where the checkout belongs to another user."""
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return True
    uid = geteuid()
    try:
        return uid == 0 or path.stat().st_uid in (uid, 0)
    except OSError:
        return False


_TOOL_TABLE_RE = re.compile(r"^\s*\[\s*tool\s*\.\s*[\"']?llmshift[\"']?\s*[\].]", re.M)


def _load_pyproject(pyproject: Path) -> dict | None:
    """The [tool.llmshift] table of a pyproject.toml, or None when it has none.
    Settings that exist but cannot be read are an error, never silently ignored."""
    try:
        text = read_text(pyproject)
    except (OSError, ValueError):
        return None
    if "llmshift" not in text:
        return None
    has_table = bool(_TOOL_TABLE_RE.search(text))
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:  # Python < 3.11: tomli is installed with llmshift there
        try:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef]
        except ImportError:
            if has_table:
                raise ConfigError(f"{pyproject}: reading [tool.llmshift] on Python < 3.11 needs the tomli package (pip install tomli)") from None
            return None
    try:
        data = tomllib.loads(text)
    except Exception as exc:
        if has_table:
            raise ConfigError(f"{pyproject}: could not parse TOML ({exc})") from None
        return None
    section = (data.get("tool") or {}).get("llmshift")
    return section if isinstance(section, dict) else None


class Config(dict):
    """A dict with attribute-free helpers; `path` is the file it came from and
    `user` holds only what that file set (before defaults were merged in)."""

    path: Path | None = None
    user: dict = {}
    warnings: list = []  # settings in the file that llmshift does not read

    def section(self, name: str) -> dict:
        value = self.get(name)
        return value if isinstance(value, dict) else {}

    def user_value(self, *keys: str) -> tuple[bool, Any]:
        """(was it set in the config file, its value) - a value may be set to null on purpose."""
        node: Any = self.user
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return False, None
            node = node[key]
        return True, node


def _get_path(data: dict, keys: tuple) -> tuple[dict | None, str]:
    node: Any = data
    for key in keys[:-1]:
        node = node.get(key) if isinstance(node, dict) else None
    return (node if isinstance(node, dict) else None), keys[-1]


def _normalize(data: dict) -> dict:
    """A single value where a list is expected becomes a one-item list."""
    for keys in _LIST_KEYS:
        node, key = _get_path(data, keys)
        if node is not None and key in node:
            value = node[key]
            if value is None:
                node[key] = []
            elif isinstance(value, (str, dict)):
                node[key] = [value]
    return data


def _rebase(cfg: dict, base: Path, user: dict) -> None:
    """Make relative paths from the config file work from any working directory.
    Project state (tests, cache, reports, history) lives next to the config; the
    default scan path stays the working directory."""
    cwd = Path(os.getcwd()).resolve()
    if base.resolve() == cwd:
        return

    def fix(value: Any) -> Any:
        if isinstance(value, list):
            return [fix(v) for v in value]
        if not isinstance(value, str) or not value.strip() or os.path.isabs(value) or "://" in value:
            return value
        joined = base / value
        try:
            return os.path.relpath(joined, cwd)
        except ValueError:  # another drive on Windows
            return str(joined)

    for keys in _PATH_KEYS:
        if keys == ("scan", "paths") and keys[-1] not in (_get_path(user, keys)[0] or {}):
            continue
        node, key = _get_path(cfg, keys)
        if node is not None and node.get(key):
            node[key] = fix(node[key])
    alerts = (cfg.get("watch") or {}).get("alerts") or []
    for dest in alerts if isinstance(alerts, list) else []:
        if isinstance(dest, dict) and dest.get("type") == "file" and dest.get("path"):
            dest["path"] = fix(dest["path"])


# Known settings: (kind, low, high). Values from ${ENV} arrive as text, so numbers and
# booleans written as text are converted; anything else wrong is reported by name.
_RULES: dict[tuple, tuple] = {
    ("concurrency",): ("int", 1, 256),
    ("timeout",): ("number", 0.001, 86400),
    ("max_retries",): ("int", 0, 50),
    ("min_pass_rate",): ("rate", 0, 1),
    ("repeat",): ("int", 1, 100),
    ("cache",): ("bool", None, None),
    ("allow_python_checks",): ("bool", None, None),
    ("judge_mode",): ("choice", ("auto", "always"), None),
    ("scan", "warn_days"): ("int", 0, 36500),
    ("scan", "include_docs"): ("bool", None, None),
    ("scan", "fail_on"): ("choice", ("retired", "retiring", "deprecated", "unknown", "never"), None),
    ("snapshot", "max_cases"): ("int", 1, 10_000_000),
    ("snapshot", "per_group"): ("int", 1, 10_000_000),
    ("snapshot", "max_records"): ("int", 1, 100_000_000),
    ("snapshot", "seed"): ("int", None, None),
    ("snapshot", "redact_secrets"): ("bool", None, None),
    ("snapshot", "redact_pii"): ("bool", None, None),
    ("fix", "rounds"): ("int", 1, 50),
    ("fix", "target_pass_rate"): ("rate", 0, 1),
    ("fix", "max_cases_per_group"): ("int", 1, 100_000),
    ("watch", "max_pass_rate_drop"): ("rate", 0, 1),
    ("watch", "min_pass_rate"): ("rate", 0, 1),
    ("watch", "max_similarity_drop"): ("rate", 0, 1),
    ("watch", "retirement_warn_days"): ("int", 0, 36500),
    ("watch", "fail_on"): ("choice", ("info", "warning", "critical", "never"), None),
    ("watch", "alert_on"): ("choice", ("info", "warning", "critical"), None),
}
_SECTIONS = ("scan", "snapshot", "fix", "watch", "providers", "severity", "pricing", "pick", "digest")
# Every setting llmshift reads; anything else in the file is a typo or a leftover, which
# is reported (and otherwise ignored) instead of silently changing nothing.
_TOP_KEYS = set(DEFAULTS) | set(_SECTIONS) | {"repeat", "judge_mode", "drop_params"}
_SECTION_KEYS = {
    "scan": {"paths", "exclude", "warn_days", "fail_on", "include_docs", "registry"},
    "snapshot": {"max_cases", "per_group", "max_records", "seed", "redact_secrets", "redact_pii"},
    "fix": {"rounds", "optimizer", "target_pass_rate", "max_cases_per_group"},
    "watch": {"model", "history", "min_pass_rate", "max_pass_rate_drop", "max_similarity_drop", "retirement_warn_days", "fail_on", "alert_on", "alerts"},
    "pick": {"models", "volume"},
    "digest": {"within", "alerts"},
}


def project_dir(cfg: Any) -> Path:
    """The folder the config file belongs to (the current folder without one)."""
    source = getattr(cfg, "path", None)
    if source is None:
        return Path.cwd()
    base = Path(source).resolve().parent
    return base.parent if base.name == ".llmshift" else base


def _shown(source: Path) -> str:
    """The config file as messages name it: relative to the current folder when inside it."""
    try:
        rel = os.path.relpath(source)
    except ValueError:  # another drive on Windows
        return str(source)
    return str(source) if rel.startswith("..") else rel


def _unknown(name: str, known: Iterable[str], where: str) -> str:
    close = difflib.get_close_matches(name.rsplit(".", 1)[-1], sorted(known), n=1, cutoff=0.75)
    hint = f"did you mean {close[0]!r}?" if close else "it is ignored"
    return f"{where}unknown setting {name!r} ({hint})"


def config_warnings(user: dict, source: Path | None) -> list[str]:
    """Settings in the file that llmshift does not read: a misspelt name, or a severity
    level or check type that does not exist."""
    from .checks import check_types, normalize_severity

    where = f"{_shown(source)}: " if source else ""
    out = [_unknown(str(key), _TOP_KEYS, where) for key in user if key not in _TOP_KEYS]
    for section, known in _SECTION_KEYS.items():
        values = user.get(section)
        if isinstance(values, dict):
            out += [_unknown(f"{section}.{key}", known, where) for key in values if key not in known]
    severity = user.get("severity")
    if isinstance(severity, dict):
        known_types = check_types()
        for ctype, level in severity.items():
            if str(ctype) not in known_types:
                out.append(_unknown(f"severity.{ctype}", known_types, where).replace("unknown setting", "unknown check type"))
            if normalize_severity(level) is None:
                out.append(f"{where}severity.{ctype} must be error, warn or info, got {level!r} (it is ignored)")
    return out


_TRUE, _FALSE = ("true", "yes", "on", "1"), ("false", "no", "off", "0")


def _coerce(value: Any, kind: str, low: Any, high: Any, name: str) -> Any:
    if value is None:
        if kind == "rate":
            return None  # null switches a threshold off
        raise ValueError(f"{name} must not be empty")
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE or text in _FALSE:
            return text in _TRUE
        raise ValueError(f"{name} must be true or false, got {value!r}")
    if kind == "choice":
        if str(value) not in low:
            raise ValueError(f"{name} must be one of {', '.join(low)}, got {value!r}")
        return str(value)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if kind == "int":
        if number != int(number):
            raise ValueError(f"{name} must be a whole number, got {value!r}")
        number = int(number)
    if (low is not None and number < low) or (high is not None and number > high):
        bounds = f"between {low} and {high}" if high is not None else f"at least {low}"
        raise ValueError(f"{name} must be {bounds}, got {value!r}")
    return number


def _validate(user: dict, source: Path | None) -> None:
    """Check (and convert) the known settings, so mistakes are reported where they are."""
    where = f"{_shown(source)}: " if source else ""
    for section in _SECTIONS:
        if section in user and user[section] is not None and not isinstance(user[section], dict):
            raise ConfigError(f"{where}'{section}' must be a mapping of settings, got {type(user[section]).__name__}")
    for keys, (kind, low, high) in _RULES.items():
        node, key = _get_path(user, keys)
        if node is None or key not in node:
            continue
        try:
            node[key] = _coerce(node[key], kind, low, high, ".".join(keys))
        except ValueError as exc:
            raise ConfigError(f"{where}{exc}") from None
    registry = (user.get("scan") or {}).get("registry")
    if registry is not None and not (isinstance(registry, str) or (isinstance(registry, list) and all(isinstance(v, str) for v in registry))):
        raise ConfigError(f"{where}scan.registry must be a registry file name or a list of them")
    for key in ("paths", "exclude"):
        values = (user.get("scan") or {}).get(key)
        if values is not None and not all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in values):
            raise ConfigError(f"{where}scan.{key} must be a list of paths or patterns")
    for key in ("tests", "baseline", "candidate", "judge", "embedder", "cache_dir", "reports_dir"):
        if user.get(key) is not None and not isinstance(user[key], (str, int, float)):
            raise ConfigError(f"{where}{key} must be text, got {type(user[key]).__name__}")
    drop = user.get("drop_params")
    if drop is not None and not (isinstance(drop, list) and all(isinstance(v, str) and v.strip() for v in drop)):
        raise ConfigError(f"{where}drop_params must be a list of setting names, e.g. [temperature, top_p]")


def load_config(path: str | os.PathLike | None = None, *, stop: str | os.PathLike | None = None) -> Config:
    """The config at `path`, or the nearest one (see find_config; `stop` bounds the search)."""
    data: dict | None = None
    source: Path | None = None
    skipped: list[str] = []
    if path:
        source = Path(path)
        if not source.is_file():
            raise ConfigError(f"config file not found: {source}")
    else:
        source = find_config(stop=stop, notes=skipped)
    if source is not None and source.name == "pyproject.toml":
        data = _load_pyproject(source)
        if data is None:
            raise ConfigError(f"{source}: no [tool.llmshift] table (or the file is not valid TOML)")
    elif source is not None:
        try:
            from . import yamlsafe

            data = yamlsafe.load(read_text(source)) or {}
        except Exception as exc:
            raise ConfigError(f"{source}: could not parse config ({exc})") from None
        if not isinstance(data, dict):
            raise ConfigError(f"{source}: the config must be a mapping of settings")
    user = _normalize(expand_env(copy.deepcopy(data or {})))
    _validate(user, source)
    cfg = Config(deep_merge(DEFAULTS, user))
    if source is not None:
        base = source.resolve().parent
        if base.name == ".llmshift":  # .llmshift/config.yaml belongs to the directory above
            base = base.parent
        _rebase(cfg, base, user)
    cfg.path = source
    cfg.user = user
    cfg.warnings = skipped + config_warnings(user, source)
    return cfg


SAMPLE_CONFIG = """\
# LLMShift configuration - https://pypi.org/project/llmshift/
# Model specs look like provider:model, e.g. openai:gpt-5, anthropic:claude-sonnet-4-5,
# gemini:gemini-2.5-flash, azure:my-deployment, ollama:llama3.1 (or a bare model id).

tests: .llmshift/tests.jsonl      # written by `snapshot` / `import`, read by compare/fix/watch
{baseline_off}baseline: {baseline}                 # the model you use today (used to re-run baselines if needed)
{candidate_off}candidate: {candidate}               # the model you are moving to
# judge: openai:gpt-5.6-terra        # optional LLM judge for meaning/tone checks
min_pass_rate: 0.95                  # compare/watch fail below this pass rate
concurrency: 4

scan:
  paths: ["."]
  exclude: []
  warn_days: 180                     # warn when a model retires within this many days
  fail_on: retired                   # retired | retiring | deprecated | never

snapshot:
  max_cases: 200
  per_group: 25                      # max cases per prompt template
  redact_secrets: true
  redact_pii: false

watch:
  history: .llmshift/watch
  max_pass_rate_drop: 0.05
  alerts: []
  #  - type: slack
  #    url: ${{SLACK_WEBHOOK_URL}}
  #  - type: github                  # opens an issue (needs GITHUB_TOKEN)

# Custom or OpenAI-compatible endpoints:
# providers:
#   local:
#     type: openai
#     base_url: http://localhost:8000/v1
#     api_key_env: LOCAL_API_KEY
"""
