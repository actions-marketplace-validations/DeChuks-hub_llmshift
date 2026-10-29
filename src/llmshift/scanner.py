"""`llmshift scan`: find every model your code calls and warn about retirement dates.

Detection layers:
* Python: AST analysis - `model=` style keyword arguments, models passed
  positionally to known constructors, {"model": ...} dicts, MODEL constants
  (resolved where they are used) and os.getenv("X_MODEL", "default") fallbacks.
* Other code (JS/TS, Go, Java, C#, Ruby, PHP, Rust, ...): string literals that
  are model IDs, and values assigned to model-like keys.
* Config (YAML, JSON, TOML, .env, INI, Terraform): model-like keys and IDs.
* Jupyter notebooks: code cells, with cell numbers in the report.
"""

from __future__ import annotations

import ast
import codecs
import fnmatch
import functools
import glob
import json
import os
import re
import subprocess  # nosec B404  # `git ls-files` only, by full path, no shell
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import __version__
from .registry import SEVERITY_ORDER, Lookup, Registry
from .fileio import find_program, long_path
from .textutil import source_lines

CODE_EXT = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".go", ".java", ".kt", ".kts", ".scala",
    ".rb", ".php", ".cs", ".rs", ".swift", ".dart", ".ex", ".exs", ".lua", ".r", ".jl", ".sh", ".bash", ".zsh", ".ps1",
    ".vue", ".svelte", ".astro", ".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".sql",
}
CONFIG_EXT = {".yaml", ".yml", ".json", ".jsonc", ".toml", ".ini", ".cfg", ".conf", ".env", ".properties", ".tf", ".tfvars", ".hcl", ".xml", ".prompt"}
DOC_EXT = {".md", ".mdx", ".rst", ".txt", ".adoc"}
SPECIAL_NAMES = {"dockerfile", "makefile", "procfile", ".env"}
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", ".env.d", "__pycache__", "dist", "build", ".tox", ".nox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages", ".next", ".nuxt", "coverage", ".idea", ".vscode",
    "target", "vendor", ".llmshift", ".terraform", ".gradle", "bower_components", ".cache", ".parcel-cache", ".turbo",
}
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "cargo.lock", "pipfile.lock", "composer.lock",
    "go.sum", "uv.lock", "bun.lockb", "gemfile.lock",
}
MAX_BYTES = 2_000_000
MAX_NOTEBOOK_BYTES = 50_000_000  # notebooks carry outputs (images); the limit applies to their code
MAX_LINE = 2000  # longer lines (minified code, one-line JSON) are only searched for model IDs

_KNOWN = [
    r"gpt-\d+(?:\.\d+)?o?(?:-[a-z0-9]+(?:\.\d+)?)*",
    r"chatgpt-4o-latest",
    r"(?:gpt-\d+(?:\.\d+)?-)?chat-latest",
    # o-series: lowercase and with a suffix here; a bare "o3" only counts in a model context
    r"(?-i:o[134](?:-(?:mini|pro|preview|deep-research))+(?:-\d{4}-\d{2}-\d{2})?|o[134]-\d{4}-\d{2}-\d{2})",
    r"(?:text|code)-(?:davinci|curie|babbage|ada|cushman)-(?:edit-)?\d{3}",
    r"(?:davinci|babbage)-002",
    r"text-(?:embedding|similarity|search|moderation)-[a-z0-9-]+",
    r"omni-moderation-[a-z0-9-]+",
    r"dall-e-[23]",
    r"whisper-1",
    r"tts-1(?:-hd)?(?:-\d{4})?",
    r"codex-mini-latest",
    r"computer-use-preview(?:-[\d-]+)?",
    r"sora-\d[a-z0-9.\-]*",
    r"ft:[a-z0-9.\-]+:[\w.\-:]*",
    r"(?:(?:us|eu|apac|global|us-gov|jp|au|ca)\.)?anthropic\.claude-[a-z0-9.\-]+(?::\d+)?",
    r"claude-(?:instant-)?(?:\d|opus|sonnet|haiku|fable|mythos|v\d)[a-z0-9.\-]*(?:@\d{8})?",
    r"(?:models/)?gemini-(?:\d|pro|ultra|flash|nano|exp|embedding|live|robotics)[a-z0-9.\-]*",
    r"(?:text|chat|code|codechat)-bison(?:-32k)?(?:@\d{3})?",
    r"textembedding-gecko(?:-multilingual)?(?:@\d{3})?",
    r"text-(?:multilingual-)?embedding-00\d",
    r"(?:open-)?(?:mistral|mixtral|codestral|ministral|pixtral|magistral|devstral|voxtral)-(?:\d|tiny|small|medium|large|embed|saba|ocr|latest|nemo|moderation|mamba|mini)[a-z0-9.\-]*",
    r"command-(?:r|a|light|nightly|xlarge|medium|r7b)(?:[a-z0-9.\-]*)",
    r"embed-(?:english|multilingual)(?:-light)?-v\d\.\d",
    r"rerank-(?:english|multilingual|v\d)[a-z0-9.\-]*",
    r"grok-(?:\d|beta|vision|code|imagine|voice|build)[a-z0-9.\-]*",
    r"deepseek-(?:chat|reasoner|coder|v\d|r\d|flash|prover|math|vl)[a-z0-9.\-]*",
    r"(?:meta-llama/)?(?:meta-)?llama-?\d[a-z0-9.:\-]*",
    r"(?:openrouter/)?(?:openai|anthropic|google|mistralai|meta-llama|deepseek|x-ai|cohere)/[a-z0-9][\w.\-:]*",
]
KNOWN_RE = re.compile(r"(?:" + "|".join(_KNOWN) + r")", re.I)
KNOWN_SEARCH_RE = re.compile(r"(?<![\w./@-])(?:" + "|".join(_KNOWN) + r")(?![\w@-])", re.I)
_NOT_MODEL_SUFFIX_RE = re.compile(r"-(?:encoder|tokenizer|tokenizers|index|python|node|js|sdk|cli|api|client|desktop|plugin|action|types|wrapper|proxy|server|app|bot)$", re.I)
_MODEL_KEY_RE = re.compile(
    r"^(?:[a-z0-9]+_)*(?:model|model_name|model_id|llm_model|chat_model|embedding_model|engine|deployment|deployment_name|deployment_id|azure_deployment)$",
    re.I,
)
_KEY_VALUE_RE = re.compile(
    # A key starts at a word start and is at most 64 long: a long word is not retried at every position.
    r"""(?<![\w])(?P<key>[A-Za-z_][\w-]{0,63})["']?\s*(?::|=|=>|:=)\s*(?P<q>["'`]?)(?P<val>[A-Za-z0-9][\w.:/@\-]{1,255})(?P=q)""",
)
_LITERAL_RE = re.compile(r"""(["'`])((?:(?!\1).){2,256}?)\1""")
_BAD_VALUES = {
    "true", "false", "none", "null", "nil", "auto", "default", "string", "str", "int", "float", "model", "models", "name",
    "id", "text", "json", "object", "any", "required", "optional", "undefined", "env", "self", "this", "value", "type",
    # sizes, tiers and stages: product or config words, never model IDs
    "full", "half", "lite", "light", "basic", "standard", "premium", "simple", "small", "medium", "large", "tiny",
    "mini", "xs", "xl", "xxl", "base", "custom", "fast", "slow", "test", "testing", "mock", "fake", "dummy", "example",
    "sample", "demo", "local", "remote", "primary", "secondary", "fallback", "main", "new", "old", "legacy", "current",
    "latest", "stable", "beta", "alpha", "dev", "prod", "production", "staging",
}
_FILE_EXT_RE = re.compile(r"\.(?:pt|pth|bin|onnx|gguf|ggml|json|pkl|pickle|h5|safetensors|joblib|tflite|mlmodel|zip|tar|gz|yaml|yml|txt|csv)$", re.I)
PY_MODEL_KWARGS = {
    "model", "model_name", "model_id", "modelId", "engine", "deployment_name", "deployment_id", "azure_deployment",
    "llm_model", "chat_model", "embedding_model", "model_version", "deployment",
}
PY_POSITIONAL_FUNCS = {
    "GenerativeModel", "init_chat_model", "OpenAIModel", "AnthropicModel", "GeminiModel", "GoogleModel", "MistralModel",
    "GroqModel", "CohereModel", "OpenAIChatModel", "OpenAIResponsesModel", "Agent", "LM", "completion", "acompletion",
    "embedding", "aembedding", "ChatOpenAI", "ChatAnthropic", "OpenAIChat", "get_model", "ChatLiteLLM", "LiteLLMModel",
}


def _snake(key: str) -> str:
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", key).replace("-", "_").lower()


_BAD_KEYS = {
    "v_model", "ng_model", "x_model", "wire_model", "data_model", "view_model", "pydantic_model", "db_model", "orm_model",
    "auth_user_model", "user_model", "parent_model", "related_model", "through_model", "proxy_model",
    "form_model", "serializer_model", "admin_model",
}
_LLM_KEY_RE = re.compile(
    r"(?:^|_)(?:llm|llms|lm|gpt|openai|anthropic|claude|gemini|genai|ai|chat|completion|completions|embedding|embeddings|judge|grader)(?:_|$)"
)
# Files that talk to an LLM somewhere; generic keys such as DEFAULT_MODEL only count in them.
# Searched in the lower-cased text (twice as fast as re.I over a whole file).
_LLM_HINT_RE = re.compile(
    r"openai|anthropic|gemini|genai|litellm|langchain|llama_?index|bedrock|vertexai|mistral|ollama|cohere|groq|deepseek"
    r"|chat\.completions|messages\.create|generate_?content|\bllms?\b|_llm|llm_|\bgpt|claude"
    r"|max_(?:completion_|output_)?tokens|\btop_p\b|system_?prompt"
)


@functools.lru_cache(maxsize=8192)
def model_key_strength(key: str) -> str | None:
    """How much a key says about its value: "strong" (model, OPENAI_MODEL, chat_model),
    "weak" (DEFAULT_MODEL, deployment: the value must look like a model or the file must
    use an LLM), "known" (engine: only known model IDs, since pandas and Django use it too)."""
    snake = _snake(key)
    if snake in _BAD_KEYS or not _MODEL_KEY_RE.match(snake):
        return None
    if snake.endswith("engine"):
        return "known"
    if snake in ("model", "model_name", "model_id", "azure_deployment") or _LLM_KEY_RE.search(snake):
        return "strong"
    return "weak"


def is_model_key(key: str) -> bool:
    return model_key_strength(key) is not None


def accepts_value(strength: str | None, value: str, llm_hint: bool) -> bool:
    """A value under a model-like key counts when it is a known model ID. Other names
    (a fine-tune, a deployment) count only in files that show signs of calling an LLM,
    so `model: full` in a size chart or a test fixture is not a finding."""
    if strength is None:
        return False
    if looks_like_known_model(value, in_model_context=True):
        return True
    return llm_hint and strength in ("strong", "weak") and plausible_model_value(value)


def has_llm_hint(text: str) -> bool:
    return bool(_LLM_HINT_RE.search(text.lower()))


_BARE_O_SERIES_RE = re.compile(r"o[134]")
# Cheap substring test run before the big regex (keeps large repositories fast). Every
# model id KNOWN_RE matches must contain one of these (tests/test_scan.py checks the
# lifecycle registry's ids).
_TRIGGERS = (
    "gpt", "o1", "o3", "o4", "claude", "gemini", "davinci", "babbage", "curie", "-ada-", "cushman", "embed", "moderation",
    "similarity", "search", "dall-e", "whisper", "tts-", "codex", "computer-use", "sora", "ft:", "bison", "gecko", "stral",
    "xtral", "command", "rerank", "grok", "deepseek", "llama", "chat-latest", "/",
)
_TRIGGER_RE = re.compile("|".join(map(re.escape, _TRIGGERS)))  # one search instead of a test per trigger


def has_trigger(text_lower: str) -> bool:
    return _TRIGGER_RE.search(text_lower) is not None


MAX_ID = 256  # a Bedrock inference-profile ARN alone is over 100 characters
# provider:model as pydantic-ai, LangChain's init_chat_model, LiteLLM and llmshift write it
_SPEC_PREFIX_RE = re.compile(
    r"^(?:openai(?:-chat|-responses)?|anthropic|gemini|google(?:-gla|-vertex|_genai|_vertexai)?|vertex(?:ai)?|azure(?:_openai)?"
    r"|mistral(?:ai)?|cohere|xai|deepseek|openrouter|bedrock(?:_converse)?|groq|ollama|together|fireworks|huggingface):",
    re.I,
)
_ARN_MODEL_RE = re.compile(r"^arn:aws[\w-]*:bedrock:[^/\s]*/(?P<model>[^/\s]+)$", re.I)
# Families whose real ids are always lower case: "GPT-4" or "Claude-3" in text is a label.
_LOWERCASE_FAMILY_RE = re.compile(r"(?i)^(?:gpt-|o\d|claude-|gemini-|text-|dall-e|whisper|tts-|chatgpt)")


def looks_like_known_model(value: str, *, in_model_context: bool = False) -> bool:
    v = value.strip()
    if not 2 <= len(v) <= MAX_ID or " " in v or "\n" in v:
        return False
    spec = _SPEC_PREFIX_RE.match(v)
    if spec:
        return looks_like_known_model(v[spec.end():], in_model_context=True)
    arn = _ARN_MODEL_RE.match(v)
    if arn:
        return looks_like_known_model(arn.group("model"), in_model_context=True)
    if len(v) > 120:
        return False
    if in_model_context and _BARE_O_SERIES_RE.fullmatch(v):
        return True
    if not has_trigger(v.lower()):
        return False
    return bool(KNOWN_RE.fullmatch(v)) and not _NOT_MODEL_SUFFIX_RE.search(v)


def is_label(value: str) -> bool:
    """A display name such as "GPT-4" or "Claude-3": the API ids of these families are
    always lower case, so a bare string with capitals is text, not a model id."""
    return value != value.lower() and bool(_LOWERCASE_FAMILY_RE.match(value))


def plausible_model_value(value: str) -> bool:
    v = value.strip()
    if not v or len(v) > MAX_ID or " " in v:
        return False
    if v.lower() in _BAD_VALUES or v.isdigit():
        return False
    if v.startswith(("http://", "https://", "$", "{", "%", "<", "/", "./", "../", "~")) or "{" in v or "}" in v:
        return False
    if _FILE_EXT_RE.search(v):
        return False
    if re.fullmatch(r"[A-Z][a-zA-Z]+", v):  # ClassName-like
        return False
    return bool(re.search(r"[a-z]", v, re.I))


@dataclass
class Finding:
    file: str
    line: int
    col: int
    model: str
    context: str
    kind: str  # call | string | config | env-default | constant
    call: str | None = None
    via: str | None = None
    cell: int | None = None
    lookup: Lookup | None = None
    file_line: int | None = None  # notebooks: the line in the .ipynb file itself

    @property
    def location(self) -> str:
        loc = f"{self.file}:{self.line}"
        if self.cell is not None:
            loc += f" (cell {self.cell})"
        return loc

    def to_dict(self) -> dict:
        d = {"file": self.file, "line": self.line, "col": self.col, "model": self.model, "kind": self.kind, "context": self.context}
        for key in ("call", "via", "cell", "file_line"):
            if getattr(self, key) is not None:
                d[key] = getattr(self, key)
        if self.lookup:
            d["lifecycle"] = self.lookup.to_dict()
        return d


@dataclass
class ScanResult:
    root: str
    files_scanned: int = 0
    findings: list[Finding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    registry_updated: str | None = None
    warn_days: int = 180

    def models(self) -> list[dict]:
        """One entry per distinct model, worst first."""
        groups: dict[str, list[Finding]] = {}
        for f in self.findings:
            groups.setdefault(f.model, []).append(f)
        out = []
        for model, items in groups.items():
            lk = items[0].lookup
            worst = max((i.lookup for i in items if i.lookup), key=lambda x: SEVERITY_ORDER.get(x.severity, 0), default=lk)
            out.append({"model": model, "lookup": worst, "findings": sorted(items, key=lambda i: (i.file, i.line))})
        return sorted(
            out,
            key=lambda m: (
                -SEVERITY_ORDER.get(m["lookup"].severity if m["lookup"] else "info", 0),
                m["lookup"].days_left if m["lookup"] and m["lookup"].days_left is not None else 10**6,
                m["model"],
            ),
        )

    def counts(self) -> dict:
        c = {"retired": 0, "retiring": 0, "retiring_soon": 0, "deprecated": 0, "alias": 0, "unknown": 0, "active": 0}
        for m in self.models():
            lk = m["lookup"]
            if lk is None:
                continue
            if lk.status == "retiring":
                c["retiring"] += 1
                if lk.days_left is not None and lk.days_left <= self.warn_days:
                    c["retiring_soon"] += 1
            elif lk.status in c:
                c[lk.status] += 1
            if lk.alias and lk.status == "active":
                c["alias"] += 1
        return c

    def to_dict(self) -> dict:
        return {
            "llmshift_version": __version__,
            "root": self.root,
            "files_scanned": self.files_scanned,
            "registry_updated": self.registry_updated,
            "warn_days": self.warn_days,
            "summary": self.counts(),
            "models": [
                {"model": m["model"], "lifecycle": m["lookup"].to_dict() if m["lookup"] else None, "locations": [f.location for f in m["findings"]]}
                for m in self.models()
            ],
            "findings": [f.to_dict() for f in self.findings],
            "errors": self.errors,
        }


# ------------------------------------------------------------ file walking


def _git_files(root: Path) -> list[Path] | None:
    exe = find_program("git")
    if exe is None:
        return None
    try:
        proc = subprocess.run(  # nosec B603  # git found by full path, arguments as a list, no shell
            [exe, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=str(root),
            stdin=subprocess.DEVNULL,  # never the caller's stdin (in `llmshift mcp` that is the protocol pipe)
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    names = [n for n in proc.stdout.decode("utf-8", "replace").split("\0") if n]
    return [root / n for n in names]


def _wanted(path: Path, include_docs: bool) -> bool:
    name = path.name.lower()
    if name in SKIP_FILES or name == "models.json" and path.parent.name == "data" and path.parent.parent.name == "llmshift":
        return False
    if name in SPECIAL_NAMES or name.startswith(".env") or name.startswith("docker-compose"):
        return True
    ext = path.suffix.lower()
    if ext == ".ipynb" or ext in CODE_EXT or ext in CONFIG_EXT:
        return True
    return include_docs and ext in DOC_EXT


# Directories never scanned, even when git tracks them; SKIP_DIRS only applies when walking.
# .llmshift holds llmshift's own reports, test sets and caches: recorded model names, not code.
_ALWAYS_SKIP = {".git", "node_modules", "site-packages", "__pycache__", ".llmshift"}
_GLOB_CHARS_RE = re.compile(r"[*?\[]")


def exclude_matcher(patterns: Iterable[str]):
    """Match --exclude patterns against a file's paths (relative to the scanned root, to the
    working directory, or absolute). A pattern without "/" matches a name at any depth;
    "**/x" also matches a top-level "x"; a directory pattern excludes what is inside it."""
    pats = []
    for raw in patterns or ():
        pat = str(raw or "").strip().replace("\\", "/")
        while pat.startswith("./"):
            pat = pat[2:]
        pat = pat.rstrip("/")
        if pat:
            pats.append(pat)

    def matches(path: str, pat: str) -> bool:
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(path, pat + "/*"):
            return True
        if pat.startswith("**/"):
            return matches(path, pat[3:])
        if "/" not in pat:
            return any(fnmatch.fnmatch(part, pat) for part in path.split("/"))
        return False

    def excluded(*paths: str | None) -> bool:
        return any(matches(path, pat) for path in paths if path for pat in pats)

    excluded.patterns = pats  # type: ignore[attr-defined]
    return excluded


def _expand(paths: Iterable[str]) -> list[str]:
    """Expand glob patterns the shell did not (Windows shells never do)."""
    out = []
    for raw in paths:
        if not Path(raw).exists() and _GLOB_CHARS_RE.search(raw):
            matches = sorted(glob.glob(raw, recursive=True))
            if not matches:
                raise FileNotFoundError(f"no files match {raw}")
            out.extend(matches)
        else:
            out.append(raw)
    return out


def iter_source_files(paths: Iterable[str], *, include_docs: bool = False, exclude: Iterable[str] = ()) -> Iterable[tuple[Path, Path]]:
    """Yield (root, file) pairs, honouring .gitignore when inside a git repo."""
    excluded = exclude_matcher(exclude)
    absolute = any(p.startswith("/") or re.match(r"^[A-Za-z]:/", p) for p in excluded.patterns)  # type: ignore[attr-defined]
    cwd = os.getcwd()

    def candidates(f: Path, rel: str | None) -> tuple:
        try:
            from_cwd = Path(os.path.relpath(f, cwd)).as_posix()
        except ValueError:  # another drive on Windows
            from_cwd = None
        return (rel, from_cwd, f.resolve().as_posix() if absolute else None)

    seen: set[Path] = set()
    for raw in _expand(paths):
        root = Path(raw)
        if root.is_file():  # named explicitly: scanned whatever its type, unless excluded
            rf = root.resolve()
            if rf not in seen and not excluded(root.name, *candidates(root, None)):
                seen.add(rf)
                yield root.parent, root
            continue
        if not root.is_dir():
            raise FileNotFoundError(f"no such file or directory: {raw}")
        files = _git_files(root)
        git_mode = bool(files)  # nothing listed: not a repo, or a directory git ignores
        if not git_mode:
            files = []
            # Walked in Windows' long form, so folders past 260 characters are listed too;
            # the files keep the path as given (root/...).
            top = long_path(root, always=True)
            for dirpath, dirnames, filenames in os.walk(top):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
                here = root / os.path.relpath(dirpath, top) if dirpath != top else root
                files.extend(here / f for f in filenames)
        skip = _ALWAYS_SKIP if git_mode else SKIP_DIRS
        for f in sorted(files or []):
            try:
                rel = f.relative_to(root).as_posix()
            except ValueError:
                rel = f.as_posix()
            if any(part in skip for part in Path(rel).parts[:-1]):
                continue
            if not _wanted(f, include_docs) or excluded(*candidates(f, rel)):
                continue
            rf = f.resolve()
            if rf in seen or not os.path.isfile(long_path(f)):
                continue
            seen.add(rf)
            yield root, f


def decode_text(data: bytes) -> str | None:
    """Decode a source file (UTF-8, or UTF-16 as written by some Windows tools); None if binary."""
    if data.startswith(codecs.BOM_UTF8):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    head = data[:4096]
    if b"\0" in head:
        odd, even = head[1::2], head[0::2]
        if len(head) >= 4 and odd.count(0) >= 0.9 * len(odd) and even.count(0) <= 0.1 * len(even):
            return data.decode("utf-16-le", errors="replace")
        if len(head) >= 4 and even.count(0) >= 0.9 * len(even) and odd.count(0) <= 0.1 * len(odd):
            return data.decode("utf-16-be", errors="replace")
        return None
    return data.decode("utf-8", errors="replace")


def _read_text(path: Path) -> tuple[str | None, str | None]:
    """(text, reason it was skipped). Binary files are skipped silently."""
    limit = MAX_NOTEBOOK_BYTES if path.suffix.lower() == ".ipynb" else MAX_BYTES
    try:
        size = os.stat(long_path(path)).st_size
        if size > limit:
            return None, f"larger than {limit // 1_000_000} MB"
        with open(long_path(path), "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return None, f"cannot read: {exc.strerror or exc}"
    return decode_text(data), None


_AZURE_HOST_RE = re.compile(r"\bopenai\.azure\.com\b")


def platform_hint(text: str) -> str | None:
    lower = text.lower()
    if "anthropicvertex" in lower or "vertexai" in lower or "aiplatform" in lower or "vertex_ai/" in lower:
        return "vertex"
    if "bedrock-runtime" in lower or "bedrockruntime" in lower or "bedrock_runtime" in lower or "anthropicbedrock" in lower:
        return "bedrock"
    if "azureopenai" in lower or "azure_endpoint" in lower or _AZURE_HOST_RE.search(lower):
        return "azure"
    return None


# --------------------------------------------------------------- python


_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


class _PyVisitor(ast.NodeVisitor):
    def __init__(self, lines: list[str], llm_hint: bool = True):
        self.lines = lines
        self.llm_hint = llm_hint
        # Names are looked up the way Python does: in the function (or class body) where they
        # are used, then the enclosing ones, then the module. A parameter, or a name a function
        # assigns something else, hides the same name elsewhere in the file.
        self.constants: dict[int, dict[str, tuple[str, int]]] = {}  # scope -> {name: (text, line)}
        self.shadowed: dict[int, set[str]] = {}  # scope -> parameters and other assigned names
        self.stack: list[ast.AST] = []
        self.default_names: dict[int, dict[str, str]] = {}  # scope -> {parameter: the name its default is}
        self.module_id = 0
        self.found: list[tuple[int, int, str, str, str | None, str | None]] = []  # line, col, value, kind, call, via
        self.claimed: set[tuple[int, int]] = set()
        self.docstrings: set[int] = set()

    def _call_name(self, node: ast.Call) -> str:
        try:
            return ast.unparse(node.func)  # type: ignore[attr-defined]
        except Exception:
            f = node.func
            parts = []
            while isinstance(f, ast.Attribute):
                parts.append(f.attr)
                f = f.value
            if isinstance(f, ast.Name):
                parts.append(f.id)
            return ".".join(reversed(parts))

    def _add(self, node: ast.AST, value: str, kind: str, call: str | None = None, via: str | None = None) -> None:
        line, col = getattr(node, "lineno", 0), getattr(node, "col_offset", 0)
        self.found.append((line, col + 1, value, kind, call, via))
        self.claimed.add((line, col))

    def _env_default(self, node: ast.AST) -> ast.Constant | None:
        # os.getenv("X", "default") / os.environ.get("X", "default")
        if isinstance(node, ast.Call) and len(node.args) >= 2:
            name = self._call_name(node)
            if name.endswith(("getenv", "environ.get", "get")) and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                return node.args[1]
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            last = node.values[-1]
            if isinstance(last, ast.Constant) and isinstance(last.value, str):
                return last
        return None

    def _require_known(self, key: str) -> bool:
        strength = model_key_strength(key) or ("strong" if key in PY_MODEL_KWARGS and key != "engine" else "known")
        return strength == "known" or not self.llm_hint  # see accepts_value()

    def _resolve(self, node: ast.AST, call: str | None, kind: str, require_known: bool) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.strip()
            if (looks_like_known_model(v) if require_known else plausible_model_value(v) or looks_like_known_model(v)):
                self._add(node, v, kind, call)
            return
        if isinstance(node, ast.Name):
            hit = self._lookup(node.id)
            if hit is not None:
                value, line = hit
                if looks_like_known_model(value) or (not require_known and plausible_model_value(value)):
                    self._add(node, value, kind, call, via=f"{node.id}, line {line}")
                return
        default = self._env_default(node)
        if default is not None:
            v = default.value.strip()
            if looks_like_known_model(v) or (not require_known and plausible_model_value(v)):
                self._add(default, v, "env-default", call)
            return
        if isinstance(node, ast.IfExp):
            self._resolve(node.body, call, kind, require_known)
            self._resolve(node.orelse, call, kind, require_known)

    def _lookup(self, name: str, stack: list | None = None) -> tuple[str, int] | None:
        stack = self.stack if stack is None else stack
        for i in range(len(stack) - 1, -1, -1):
            scope = stack[i]
            if isinstance(scope, ast.ClassDef) and i != len(stack) - 1:
                continue  # a method does not see the names of its class body
            hit = self.constants.get(id(scope), {}).get(name)
            if hit is not None:
                return hit
            ref = self.default_names.get(id(scope), {}).get(name)
            if ref is not None:  # def f(model=MODEL): MODEL as it is where the def stands (a class body counts)
                return self._lookup(ref, stack[:i])
            if name in self.shadowed.get(id(scope), ()):
                return None  # a parameter or another value: not the text assigned elsewhere
        return None

    def _collect(self, scope: ast.AST) -> None:
        """The names `scope` binds itself (not those of the functions and classes in it)."""
        names: dict[str, tuple[str, int]] = {}
        other: set[str] = set()
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = scope.args
            other.update(x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs, *(x for x in (a.vararg, a.kwarg) if x)])
            # A parameter's default is what the body uses when the caller passes nothing.
            positional = [*a.posonlyargs, *a.args]
            pairs = [*zip(positional[len(positional) - len(a.defaults):], a.defaults), *zip(a.kwonlyargs, a.kw_defaults)]
            refs: dict[str, str] = {}
            for arg, default in pairs:
                if isinstance(default, ast.Constant) and isinstance(default.value, str):
                    names[arg.arg] = (default.value.strip(), scope.lineno)
                elif isinstance(default, ast.Name):
                    refs[arg.arg] = default.id  # looked up where the def stands, when the body is read
            self.default_names[id(scope)] = refs
        todo = list(ast.iter_child_nodes(scope))
        while todo:
            n = todo.pop()
            if isinstance(n, _SCOPE_NODES):
                if hasattr(n, "name"):
                    other.add(n.name)
                continue
            todo.extend(ast.iter_child_nodes(n))
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                value = self._env_default(n.value) if n.value is not None and not isinstance(n.value, ast.Constant) else n.value
                # MODEL = "gpt-4o", and MODEL = os.getenv("MODEL", "gpt-4o"): the default is what runs
                text = value.value.strip() if isinstance(value, ast.Constant) and isinstance(value.value, str) else None
                for t in targets:
                    if isinstance(t, ast.Name):
                        if text is not None and (t.id not in names or n.lineno >= names[t.id][1]):
                            names[t.id] = (text, n.lineno)
                        elif text is None:
                            other.add(t.id)
            elif isinstance(n, (ast.AugAssign, ast.For, ast.AsyncFor, ast.NamedExpr)) and isinstance(n.target, ast.Name):
                other.add(n.target.id)
            elif isinstance(n, ast.withitem) and isinstance(n.optional_vars, ast.Name):
                other.add(n.optional_vars.id)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                other.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                other.update((alias.asname or alias.name).split(".")[0] for alias in n.names)
            elif type(n).__name__ in ("MatchAs", "MatchStar") and getattr(n, "name", None):  # match captures (3.10+)
                other.add(n.name)
            elif type(n).__name__ == "MatchMapping" and getattr(n, "rest", None):
                other.add(n.rest)
        self.constants[id(scope)] = names
        self.shadowed[id(scope)] = other

    def visit_Module(self, node: ast.Module) -> None:
        self.module_id = id(node)
        for stmt in ast.walk(node):  # breadth first: the module's names are known before its functions'
            if isinstance(stmt, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(stmt, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                    self.docstrings.add(id(body[0].value))
            if isinstance(stmt, (ast.Module, *_SCOPE_NODES)):
                self._collect(stmt)
        self.stack.append(node)
        self.generic_visit(node)
        self.stack.pop()

    def _visit_in(self, scope: ast.AST, inner: list, outer: list) -> None:
        """Decorators, defaults and base classes belong to the enclosing scope; the body to `scope`."""
        for n in outer:
            if n is not None:
                self.visit(n)
        self.stack.append(scope)
        try:
            for n in inner:
                self.visit(n)
        finally:
            self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        a = node.args
        annotations = [x.annotation for x in [*a.posonlyargs, *a.args, *a.kwonlyargs, *(x for x in (a.vararg, a.kwarg) if x)]]
        outer = [*node.decorator_list, *a.defaults, *a.kw_defaults, *annotations, node.returns, *getattr(node, "type_params", [])]
        self._visit_in(node, node.body, outer)  # Literal["gpt-4o"] in an annotation is read too

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_in(node, [node.body], [*node.args.defaults, *node.args.kw_defaults])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_in(node, node.body, [*node.decorator_list, *node.bases, *node.keywords, *getattr(node, "type_params", [])])

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
            if name and is_model_key(name):
                self._resolve(node.value, None, "constant", require_known=self._require_known(name))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        target = node.target
        name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
        if name and node.value is not None and is_model_key(name):
            self._resolve(node.value, None, "constant", require_known=self._require_known(name))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        short = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
        call = None  # the dotted name is only computed when a model is involved
        for kw in node.keywords:
            if kw.arg and (kw.arg in PY_MODEL_KWARGS or is_model_key(kw.arg)):
                call = call or self._call_name(node)
                self._resolve(kw.value, call, "call", require_known=self._require_known(kw.arg))
        if short in PY_POSITIONAL_FUNCS and node.args:
            self._resolve(node.args[0], call or self._call_name(node), "call", require_known=True)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and is_model_key(key.value) and value is not None:
                self._resolve(value, None, "call", require_known=self._require_known(key.value))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and id(node) not in self.docstrings:
            key = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            if key not in self.claimed and looks_like_known_model(node.value.strip()) and not is_label(node.value.strip()):
                self._add(node, node.value.strip(), "string")


def scan_python(text: str, llm_hint: bool | None = None) -> list[tuple[int, int, str, str, str | None, str | None]]:
    from .textutil import parse_python

    llm_hint = has_llm_hint(text) if llm_hint is None else llm_hint
    tree = parse_python(text)
    if tree is None:  # a syntax error, newer syntax than this Python, or too deep to parse
        return scan_generic(text, code=True, llm_hint=llm_hint)
    visitor = _PyVisitor(source_lines(text), llm_hint=llm_hint)
    try:
        visitor.visit(tree)
    except RecursionError:  # e.g. a 500-branch elif ladder: read it line by line instead
        return scan_generic(text, code=True, llm_hint=llm_hint)
    # The catch-all string pass runs during traversal; drop duplicates it produced.
    seen = set()
    out = []
    for item in sorted(visitor.found, key=lambda x: (x[0], x[1], {"call": 0, "env-default": 1, "constant": 2, "string": 3}.get(x[3], 4))):
        key = (item[0], item[2])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


# -------------------------------------------------------------- generic

_COMMENT_RE = re.compile(r"^\s*(//|#(?!!)|/\*|\*\s|\*$|<!--)")

# Comment syntax by file type: a model named in a comment is not one the code calls.
_HASH, _SLASH, _BLOCK, _SEMI, _DASH = "#", "//", "/*", ";", "--"
_HTML = "<!--"  # a block, like /* */
_HASH_AT_START = "#^"  # "#" only as a line's first character (MySQL comments; elsewhere in SQL it is an operator)
_C_LIKE = (_SLASH, _BLOCK)
_BLOCK_END = {_BLOCK: "*/", _HTML: "-->"}
_BEFORE_BLOCK = set(" \t;{}(),=:")  # where a /* can start a comment; elsewhere it is a glob or a regex (lists/*, /\/*/)
COMMENT_SYNTAX: dict[str, tuple[str, ...]] = {
    **dict.fromkeys((".yaml", ".yml", ".toml", ".env", ".sh", ".bash", ".zsh", ".rb", ".r", ".jl", ".ex", ".exs", ".ps1", ".tfvars", "dockerfile", "makefile", "procfile"), (_HASH,)),
    **dict.fromkeys((".ini", ".cfg", ".conf"), (_HASH, _SEMI)),
    ".properties": (_HASH, "!"),
    **dict.fromkeys((".tf", ".hcl"), (_HASH, *_C_LIKE)),
    **dict.fromkeys((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".go", ".java", ".kt", ".kts", ".scala", ".cs", ".rs", ".swift", ".dart",
                     ".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".jsonc", ".json5"), _C_LIKE),
    **dict.fromkeys((".vue", ".svelte", ".astro"), (_HTML, *_C_LIKE)),  # an HTML template around the script
    ".php": (_HASH, *_C_LIKE),
    ".sql": (_DASH, _BLOCK, _HASH_AT_START),
    ".lua": (_DASH,),
}


_OWN_MARKDOWN_RE = re.compile(r"\A(?:<!-- llmshift[^>]*-->\s*)?#{1,2} LLMShift\b")


def is_llmshift_output(path: Path, text: str) -> bool:
    """A report llmshift wrote (JSON, SARIF, JUnit, Markdown, HTML): the models and prompts
    it names are its findings, not what the code uses, so reading it would only echo
    earlier results."""
    suffix = path.suffix.lower()
    head = text[:4096]
    if suffix in (".md", ".markdown"):
        return bool(_OWN_MARKDOWN_RE.match(head.lstrip()))
    if suffix in (".html", ".htm"):
        return "<title>LLMShift" in head
    if suffix not in (".json", ".sarif", ".xml"):
        return False
    return '"llmshift_version"' in head or '<testsuites name="llmshift"' in head or bool(re.search(r'"driver":\s*\{\s*"name":\s*"llmshift"', head))


def comment_syntax(path: Path) -> tuple[str, ...] | None:
    name = path.name.lower()
    if name.startswith(".env"):
        return COMMENT_SYNTAX[".env"]
    return COMMENT_SYNTAX.get(path.suffix.lower()) or COMMENT_SYNTAX.get(name)


@functools.lru_cache(maxsize=None)
def _comment_stops(markers: tuple[str, ...]) -> re.Pattern:
    """The characters strip_comment looks at outside quotes: quotes and the first character
    of each comment marker. The text between two of them is copied in one piece."""
    return re.compile("[" + re.escape("\"'`" + "".join(sorted({m[0] for m in markers}))) + "]")


def strip_comment(line: str, markers: tuple[str, ...], state: dict) -> str:
    """The line without its comment (text outside quotes after a marker), and without the
    part inside a /* block */ that `state` carries from line to line. "#" and ";" count
    only at the start or after a space, so a URL's #fragment or a value's ";" stays."""
    out: list[str] = []
    i, n, quote = 0, len(line), ""
    stops = _comment_stops(markers)
    blocks = [m for m in (_BLOCK, _HTML) if m in markers]
    while i < n:
        if state.get("block"):  # the closing token of the open block
            end = line.find(state["block"], i)
            if end < 0:
                break
            close = len(state["block"])
            state["block"] = None
            out.append(" " * (end + close - i))  # keep the columns of what follows
            i = end + close
            continue
        if quote:  # copied up to the closing quote or an escape
            ends = [j for j in (line.find(quote, i), line.find("\\", i)) if j >= 0]
            j = min(ends) if ends else n
        else:  # copied up to the next quote or possible comment marker
            hit = stops.search(line, i)
            j = hit.start() if hit else n
        if j > i:
            out.append(line[i:j])
            i = j
            if i >= n:
                break
        ch = line[i]
        if quote:
            if ch == "\\":
                out.append(line[i : i + 2])
                i += 2
                continue
            quote = ""  # the closing quote
        elif ch in "\"'`":
            quote = ch
        else:
            opened = next((m for m in blocks if line.startswith(m, i)), None)
            # A block opens only where a comment can start, and only if the file closes it
            # somewhere: a "/*" in a path glob or a regex would otherwise hide the rest of the file.
            if opened and (i == 0 or line[i - 1] in _BEFORE_BLOCK) and _BLOCK_END[opened] in state.get("closers", (_BLOCK_END[opened],)):
                state["block"] = _BLOCK_END[opened]
                out.append(" " * len(opened))
                i += len(opened)
                continue
            for marker in markers:
                if marker in (_BLOCK, _HTML):
                    continue
                if marker == _HASH_AT_START:
                    if ch == "#" and not line[:i].strip():
                        return "".join(out)
                    continue
                if line.startswith(marker, i) and (marker not in (_HASH, _SEMI, "!") or i == 0 or line[i - 1].isspace()):
                    return "".join(out)
        out.append(ch)
        i += 1
    return "".join(out)


def scan_generic(
    text: str, *, code: bool, shell_script: bool = False, llm_hint: bool | None = None, comments: tuple[str, ...] | None = None
) -> list[tuple[int, int, str, str, str | None, str | None]]:
    """Scan non-Python text line by line. `shell_script`: values may be unquoted (MODEL=gpt-4o)
    and bare model IDs count (--model gpt-4o). `comments`: the file type's comment
    markers (see COMMENT_SYNTAX); text in comments is not scanned."""
    llm_hint = has_llm_hint(text) if llm_hint is None else llm_hint
    out = []
    block: dict = {"closers": {end for end in _BLOCK_END.values() if end in text}}  # blocks this file can close
    openers = [m for m in (_BLOCK, _HTML) if m in (comments or ())]
    # A line that starts with one of these is all comment and cannot open a block.
    whole_line = tuple("#" if m == _HASH_AT_START else m for m in (comments or ()) if m not in (_BLOCK, _HTML))
    for lineno, line in enumerate(source_lines(text), 1):
        if len(line) > MAX_LINE:
            if has_trigger(line.lower()):
                found: dict[str, int] = {}
                for m in KNOWN_SEARCH_RE.finditer(line):
                    val = m.group(0).rstrip(".-:")  # the end of a sentence is not part of the id
                    if not _NOT_MODEL_SUFFIX_RE.search(val) and not is_label(val):
                        found.setdefault(val, m.start() + 1)
                out.extend((lineno, col, val, "string" if code else "config", None, None) for val, col in found.items())
            continue
        if comments:
            if whole_line and not block.get("block") and line.lstrip().startswith(whole_line):
                continue
        elif code and _COMMENT_RE.match(line):
            continue
        lower = line.lower()
        keyed = "model" in lower or "engine" in lower or "deployment" in lower
        triggered = has_trigger(lower)
        if comments:
            # Taking out a comment only removes text, so a line without model words has
            # nothing to find; it is skipped unless it is in, or opens, a /* block */.
            if not keyed and not triggered and not block.get("block") and not any(m in line for m in openers):
                continue
            stripped = strip_comment(line, comments, block)
            if stripped != line:
                line, lower = stripped, stripped.lower()
                keyed = "model" in lower or "engine" in lower or "deployment" in lower
                triggered = has_trigger(lower)
        if not keyed and not triggered:
            continue
        hits: dict[str, tuple[int, str]] = {}
        model_ctx = False
        for m in _KEY_VALUE_RE.finditer(line) if keyed else ():
            key, quote, val = m.group("key"), m.group("q"), m.group("val")
            strength = model_key_strength(key)
            if strength is None:
                continue
            model_ctx = True
            if code and not quote and not shell_script:
                continue
            if accepts_value(strength, val, llm_hint):
                hits.setdefault(val, (m.start("val") + 1, "call" if code else "config"))
        for m in _LITERAL_RE.finditer(line) if triggered else ():
            val = m.group(2).strip()
            if looks_like_known_model(val) and (model_ctx or not is_label(val)):
                hits.setdefault(val, (m.start(2) + 1, "call" if (code and model_ctx) else ("string" if code else "config")))
        if (shell_script or not code) and triggered:
            for m in KNOWN_SEARCH_RE.finditer(line):
                val = m.group(0).rstrip(".-:")  # the end of a sentence is not part of the id
                if not _NOT_MODEL_SUFFIX_RE.search(val) and not is_label(val):
                    hits.setdefault(val, (m.start() + 1, "string" if code else "config"))
        for val, (col, kind) in hits.items():
            out.append((lineno, col, val, kind, None, None))
    return out


_SOURCE_KEY_RE = re.compile(r'^\s*"source"\s*:\s*(\[)?')


def _line_starts(pieces: list) -> list[int]:
    """For each line of "".join(pieces), the index of the piece it starts in."""
    starts: list[int] = []
    open_line = False
    for j, piece in enumerate(pieces):
        if not isinstance(piece, str):
            continue
        pos = 0
        while pos < len(piece):
            if not open_line:
                starts.append(j)
                open_line = True
            nl = piece.find("\n", pos)
            if nl == -1:
                break
            open_line = False
            pos = nl + 1
    return starts


def scan_notebook(text: str) -> list[tuple[int, int, str, str, str | None, str | None, int, int | None]]:
    """Scan code cells. Items carry the cell number and, when the notebook is stored one
    source line per JSON line (as Jupyter writes it), the line in the .ipynb file."""
    try:
        nb = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"not a valid notebook ({exc})") from None
    if not isinstance(nb, dict):
        raise ValueError("not a valid notebook (expected a JSON object)")
    cells = nb.get("cells")
    v3 = cells is None  # nbformat 3 keeps cells under worksheets, with "input" for code
    if v3:
        cells = [c for ws in nb.get("worksheets") or [] if isinstance(ws, dict) for c in ws.get("cells") or []]
    cells = [c for c in cells or [] if isinstance(c, dict)]
    key_lines = [] if v3 else [(n, bool(m.group(1))) for n, ln in enumerate(source_lines(text), 1) for m in [_SOURCE_KEY_RE.match(ln)] if m]
    if len(key_lines) != len(cells):
        key_lines = []
    code_cells = [(idx, cell) for idx, cell in enumerate(cells, 1) if cell.get("cell_type") == "code"]
    total = sum(len(str(cell.get("source") or cell.get("input") or "")) for _, cell in code_cells)
    if total > MAX_BYTES:
        raise ValueError(f"skipped: more than {MAX_BYTES // 1_000_000} MB of code")
    llm_hint = has_llm_hint(text)
    out = []
    for idx, cell in code_cells:
        raw = cell.get("source", cell.get("input")) or ""
        src = "".join(p for p in raw if isinstance(p, str)) if isinstance(raw, list) else str(raw)
        starts = _line_starts(raw) if isinstance(raw, list) else []
        key = key_lines[idx - 1] if key_lines else None
        # IPython magics/shell lines break ast.parse; blank them but keep numbering.
        cleaned = "\n".join("" if ln.lstrip().startswith(("%", "!")) else ln for ln in src.split("\n"))
        for item in scan_python(cleaned, llm_hint=llm_hint):
            file_line = None
            if key is not None:
                key_line, is_list = key
                if not is_list:
                    file_line = key_line
                elif 0 < item[0] <= len(starts):
                    file_line = key_line + 1 + starts[item[0] - 1]
            out.append(item + (idx, file_line))
    return out


# --------------------------------------------------------------- main


def scan_paths(
    paths: list[str] | None = None,
    *,
    registry: Registry | None = None,
    include_docs: bool = False,
    exclude: Iterable[str] = (),
    warn_days: int = 180,
    max_files: int | None = None,
) -> ScanResult:
    """Every model reference in `paths`; with `max_files`, the first that many files."""
    registry = registry or Registry.load()
    paths = list(paths or ["."])
    result = ScanResult(root=", ".join(paths), registry_updated=registry.updated, warn_days=warn_days)
    seen: set[tuple] = set()
    for _root, path in iter_source_files(paths, include_docs=include_docs, exclude=exclude):
        if max_files is not None and result.files_scanned >= max_files:
            break
        try:
            rel = Path(os.path.relpath(path)).as_posix()
            if rel.startswith("../"):
                rel = path.resolve().as_posix()
        except ValueError:  # different drive on Windows
            rel = path.resolve().as_posix()
        text, skipped = _read_text(path)
        if skipped:
            result.errors.append(f"{rel}: {skipped}")
        if text is None or is_llmshift_output(path, text):
            continue
        result.files_scanned += 1
        hint = platform_hint(text)
        ext = path.suffix.lower()
        try:
            if ext == ".ipynb":
                items = scan_notebook(text)
            elif ext in (".py", ".pyi"):
                items = [i + (None, None) for i in scan_python(text)]
            else:
                is_code = ext in CODE_EXT
                shell_script = ext in (".sh", ".bash", ".zsh")
                items = [i + (None, None) for i in scan_generic(text, code=is_code, shell_script=shell_script, comments=comment_syntax(path))]
        except RecursionError:
            result.errors.append(f"{rel}: too deeply nested to analyse")
            continue
        except Exception as exc:  # one odd file must not stop the scan
            message = str(exc) if isinstance(exc, ValueError) else f"could not analyse ({exc.__class__.__name__}: {exc})"
            result.errors.append(f"{rel}: {message}")
            continue
        lines = source_lines(text)
        for line, col, value, kind, call, via, cell, file_line in items:
            key = (rel, line, value, cell)
            if key in seen:
                continue
            seen.add(key)
            if cell is not None:
                context = ""
            else:
                context = lines[line - 1].strip()[:160] if 0 < line <= len(lines) else ""
            finding = Finding(file=rel, line=line, col=col, model=value, context=context, kind=kind, call=call, via=via, cell=cell, file_line=file_line)
            finding.lookup = registry.lookup(value, platform=hint, warn_days=warn_days)
            result.findings.append(finding)
    result.findings.sort(key=lambda f: (f.file, f.line, f.col))
    return result


FAIL_LEVELS = ("retired", "retiring", "deprecated", "unknown", "never")


def should_fail(result: ScanResult, fail_on: str) -> bool:
    if fail_on == "never":
        return False
    for m in result.models():
        lk = m["lookup"]
        if lk is None:
            continue
        if lk.status == "retired":
            return True
        if fail_on in ("retiring", "deprecated", "unknown") and lk.status == "retiring" and (lk.days_left or 0) <= result.warn_days:
            return True
        if fail_on in ("deprecated", "unknown") and lk.status in ("deprecated", "retiring"):
            return True
        if fail_on == "unknown" and lk.status == "unknown":
            return True
    return False
