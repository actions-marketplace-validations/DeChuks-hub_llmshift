"""Text utilities: normalisation, tokenisation, similarity metrics, diffs and
language detection. Pure standard library, deterministic."""

from __future__ import annotations

import ast
import codecs
import difflib
import hashlib
import json
import math
import re
import threading
import unicodedata
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

_TRANSLATE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "\u00a0": " ",
        " ": " ",
        " ": " ",
        "\u200b": "",
    }
)

_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿"
_TOKEN_RE = re.compile(rf"[{_CJK}]|[^\W_{_CJK}]+")
_WORD_RE = re.compile(rf"[{_CJK}]|[^\W_{_CJK}]+(?:['’][^\W_{_CJK}]+)*")

# Upper bound on tokens compared by LCS/METEOR (bit-parallel LCS makes this cheap).
MAX_LCS_TOKENS = 50000


def normalize(text: Any) -> str:
    """Lower-case, NFKC-normalise, straighten quotes/dashes and collapse whitespace."""
    s = unicodedata.normalize("NFKC", "" if text is None else str(text)).translate(_TRANSLATE)
    return re.sub(r"\s+", " ", s.lower()).strip()


def tokenize(text: Any) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))


def words(text: Any) -> list[str]:
    """Words as a human would count them (CJK characters count individually)."""
    return _WORD_RE.findall(unicodedata.normalize("NFKC", "" if text is None else str(text)))


def word_count(text: Any) -> int:
    return len(words(text))


def stable_hash(obj: Any, length: int = 12) -> str:
    try:
        data = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:  # keys of mixed types (YAML allows {1: a, "b": c}) cannot be sorted together
        data = json.dumps(_str_keys(obj), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8", "surrogatepass")).hexdigest()[:length]


def _str_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _str_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_str_keys(v) for v in obj]
    return obj


_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def has_surrogate_escape(raw: str) -> bool:
    """Cheap test before cleaning: JSON text that may decode to lone surrogates."""
    return "\\ud" in raw or "\\uD" in raw


def clean_surrogates(value: Any) -> Any:
    """Replace lone UTF-16 surrogates (JSON allows "\\ud83d" alone, and models emit them
    when a stream splits an emoji) with U+FFFD, so the text can be stored as UTF-8."""
    if isinstance(value, str):
        return _SURROGATE_RE.sub("\ufffd", value) if _SURROGATE_RE.search(value) else value
    if isinstance(value, list):
        return [clean_surrogates(v) for v in value]
    if isinstance(value, dict):
        return {clean_surrogates(k): clean_surrogates(v) for k, v in value.items()}
    return value


# Deeper JSON is refused everywhere. The C parser's own limit depends on the OS and the
# Python version (5000 levels parse on Linux with 3.12, not on Windows), and nested
# values are walked recursively afterwards.
_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")


def source_lines(text: str) -> list[str]:
    """Lines as Python, git and editors number them: split at \\n, \\r\\n and \\r only,
    not at the form feeds, U+2028 and others that str.splitlines() also splits at."""
    return _LINE_BREAK_RE.split(text)


def decode_text(raw: bytes, where: Any) -> str:
    """Bytes of a file you wrote as text: UTF-8 (with or without a BOM), or UTF-16 with a
    BOM (what Windows PowerShell 5.1 writes with `>`). Anything else is an error that names
    the file."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            raise ValueError(f"{where}: not valid UTF-16 text; save it as UTF-8") from None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{where}: not UTF-8 text (byte {exc.start} is not); save it as UTF-8") from None


def read_text(path: Any) -> str:
    """A text file you wrote (a test file, llmshift.yaml, labels), as decode_text reads it."""
    return decode_text(Path(path).read_bytes(), path)


def jsonl_lines(text: str) -> list[str]:
    """The lines of a JSON Lines file. Only "\\n" (or "\\r\\n") ends a line: str.splitlines()
    also splits at U+2028, U+0085 and other characters JSON keeps unescaped in strings."""
    return [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]


MAX_JSON_DEPTH = 200
_JSON_SCAN_RE = re.compile(r'["\[\]{}]')


def json_too_deep(text: str, start: int = 0, limit: int = MAX_JSON_DEPTH) -> bool:
    """Whether the JSON value at `start` nests more than `limit` levels. Linear time,
    whatever the input (strings are skipped with find, never with backtracking)."""
    if text.count("[", start) + text.count("{", start) <= limit:
        return False
    depth, i, search = 0, start, _JSON_SCAN_RE.search
    while True:
        m = search(text, i)
        if m is None:
            return False
        ch, i = m.group(), m.end()
        if ch == '"':
            while True:  # to the closing quote: one not preceded by an odd run of backslashes
                j = text.find('"', i)
                if j < 0:
                    return False
                k = j
                while k > i and text[k - 1] == "\\":
                    k -= 1
                i = j + 1
                if (j - k) % 2 == 0:
                    break
        elif ch in "[{":
            depth += 1
            if depth > limit:
                return True
        else:
            depth -= 1
            if depth <= 0:
                return False


def json_loads(text: str) -> Any:
    """json.loads for untrusted text with the same nesting limit on every platform;
    deeper nesting raises RecursionError, as json.loads does past its own limit."""
    if isinstance(text, str) and json_too_deep(text):
        raise RecursionError(f"JSON nested more than {MAX_JSON_DEPTH} levels deep")
    return json.loads(text)


_PARSE_LOCK = threading.Lock()  # catch_warnings swaps process-wide state before Python 3.14


def parse_python(source: str) -> ast.Module | None:
    """The syntax tree of Python code llmshift reads (never runs), or None when this
    Python cannot parse it: a syntax error, newer syntax, or nesting too deep. The
    compiler's warnings about that code ("invalid escape sequence" on 3.12+) are not
    llmshift's to print, and under PYTHONWARNINGS=error they must not fail the parse."""
    with _PARSE_LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return ast.parse(source)
        except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
            return None


def shorten(text: Any, limit: int = 120) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: max(0, limit - 3)] + "..."


def xml_escape(text: str) -> str:
    """XML text content, as xml.sax.saxutils.escape writes it (llmshift writes XML and never
    parses it, so it has no reason to import an XML parser package)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def quoteattr(text: str) -> str:
    """A quoted XML attribute value, as xml.sax.saxutils.quoteattr writes it."""
    text = xml_escape(text).replace("\n", "&#10;").replace("\r", "&#13;").replace("\t", "&#9;")
    if '"' not in text:
        return f'"{text}"'
    if "'" not in text:
        return f"'{text}'"
    return '"' + text.replace('"', "&quot;") + '"'


PROJECT_URL = "https://github.com/DeChuks-hub/llmshift"


def credit(what: str = "Generated by") -> str:
    """The last line of a Markdown report, pull request comment or issue: what wrote it,
    linked, so that whoever reads it can find the tool."""
    from . import __version__

    return f"<sub>{what} [llmshift]({PROJECT_URL}) {__version__}</sub>"


def plural(n: int, word: str, many: str | None = None) -> str:
    """'1 case', '3 cases' (or `many` for irregular words)."""
    return f"{n} {word if n == 1 else (many or word + 's')}"


def md_fence(body: str, lang: str = "") -> str:
    """`body` as a Markdown code block that no line of it can close (a prompt or diff
    with its own ``` would otherwise end the block and turn the rest into live Markdown)."""
    longest = max((len(run) for run in re.findall(r"`{3,}", body)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{show_bidi(body)}\n{fence}"


# Bidi embedding, override and isolate controls: "Trojan Source" text that displays in
# another order than it reads. Directional marks, which reorder nothing, are not among them.
_BIDI_RE = re.compile("[" + "".join(map(chr, (*range(0x202A, 0x202F), *range(0x2066, 0x206A)))) + "]")


def show_bidi(text: str) -> str:
    """The bidi controls in `text` shown as backslash-u escapes, as the terminal shows them.
    Inside JSON strings the escape is JSON's own, so the data reads back unchanged."""
    return _BIDI_RE.sub(lambda m: f"\\u{ord(m.group()):04x}", text)


def report_json(data: Any) -> str:
    """A report as JSON: indented, readable characters kept, bidi controls escaped."""
    return show_bidi(json.dumps(data, indent=2, ensure_ascii=False))


ZWSP = chr(0x200B)  # zero-width space: breaks a link or mention without showing
_LINK_SYNTAX_RE = re.compile(r"[\\\[\]()!]")
_AUTOLINK_RE = re.compile(r"(?i)\b(?:https?|ftp)(?=://)|\bwww(?=\.)")


def md_inert(text: str) -> str:
    """Model or user text that Markdown shows as written: [text](url) is no link, ![](url)
    no image (a tracking pixel in a PR comment) and a bare URL no autolink. Backslashes are
    escaped first, so the text cannot undo an escape. Newlines are kept."""
    text = _LINK_SYNTAX_RE.sub(lambda m: "\\" + m.group(), str(text))
    return _AUTOLINK_RE.sub(lambda m: m.group() + ZWSP, text)


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def rouge_n(candidate: str, reference: str, n: int = 1) -> float:
    ct, rt = tokenize(candidate), tokenize(reference)
    if not ct or not rt:
        return 1.0 if not ct and not rt else 0.0
    n = max(1, min(n, len(ct), len(rt)))  # short texts: use the longest order both support
    c, r = Counter(ngrams(ct, n)), Counter(ngrams(rt, n))
    overlap = sum((c & r).values())
    if not overlap:
        return 0.0
    return _f1(overlap / sum(c.values()), overlap / sum(r.values()))


def lcs_length(a: list[str], b: list[str]) -> int:
    """Exact longest-common-subsequence length (bit-parallel, Allison-Dix/Hyyro).

    Runs in O(len(a) * len(b) / word size), so even 20k-token texts take milliseconds.
    """
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    masks: dict[str, int] = {}
    for i, token in enumerate(b):
        masks[token] = masks.get(token, 0) | (1 << i)
    full = (1 << len(b)) - 1
    v = full
    for token in a:
        u = v & masks.get(token, 0)
        v = ((v + u) | (v - u)) & full
    return len(b) - bin(v).count("1")


def rouge_l(candidate: str, reference: str) -> float:
    c = tokenize(candidate)[:MAX_LCS_TOKENS]
    r = tokenize(reference)[:MAX_LCS_TOKENS]
    if not c or not r:
        return 1.0 if not c and not r else 0.0
    if c == r:
        return 1.0
    lcs = lcs_length(c, r)
    return _f1(lcs / len(c), lcs / len(r))


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Sentence BLEU with uniform weights, epsilon smoothing and brevity penalty."""
    c, r = tokenize(candidate), tokenize(reference)
    if not c or not r:
        return 1.0 if not c and not r else 0.0
    orders = [n for n in range(1, max_n + 1) if len(c) >= n]
    log_sum = 0.0
    for n in orders:
        cn, rn = Counter(ngrams(c, n)), Counter(ngrams(r, n))
        total = sum(cn.values())
        matches = sum((cn & rn).values())
        p = matches / total if matches else 0.1 / total
        log_sum += math.log(p)
    geo = math.exp(log_sum / len(orders))
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / len(c))
    return max(0.0, min(1.0, geo * bp))


def gleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Google-BLEU: min(precision, recall) over all 1..max_n grams."""
    c, r = tokenize(candidate), tokenize(reference)
    if not c or not r:
        return 1.0 if not c and not r else 0.0
    cg: Counter = Counter()
    rg: Counter = Counter()
    for n in range(1, max_n + 1):
        cg.update(ngrams(c, n))
        rg.update(ngrams(r, n))
    matches = sum((cg & rg).values())
    if not matches:
        return 0.0
    return min(matches / sum(cg.values()), matches / sum(rg.values()))


def _stem(word: str) -> str:
    for suffix in ("ingly", "edly", "ing", "ies", "ied", "ed", "es", "ly", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def meteor(candidate: str, reference: str) -> float:
    """Approximate METEOR: exact + stem unigram alignment, F-mean, chunk penalty."""
    c, r = tokenize(candidate)[:MAX_LCS_TOKENS], tokenize(reference)[:MAX_LCS_TOKENS]
    if not c or not r:
        return 1.0 if not c and not r else 0.0
    aligned_c: set[int] = set()
    alignment: list[tuple[int, int]] = []
    for stage in (lambda w: w, _stem):
        # Unused reference positions per word form, in order (linear time overall).
        free: dict[str, list[int]] = {}
        used_r = {j for _, j in alignment}
        for j, w in enumerate(r):
            if j not in used_r:
                free.setdefault(stage(w), []).append(j)
        cursor: dict[str, int] = {}
        for i, w in enumerate(c):
            if i in aligned_c:
                continue
            form = stage(w)
            slots = free.get(form)
            k = cursor.get(form, 0)
            if slots and k < len(slots):
                cursor[form] = k + 1
                aligned_c.add(i)
                alignment.append((i, slots[k]))
    m = len(alignment)
    if not m:
        return 0.0
    p, rec = m / len(c), m / len(r)
    fmean = 10 * p * rec / (rec + 9 * p)
    alignment.sort()
    chunks = 1
    for (i1, j1), (i2, j2) in zip(alignment, alignment[1:]):
        if not (i2 == i1 + 1 and j2 == j1 + 1):
            chunks += 1
    penalty = 0.5 * (chunks / m) ** 3
    return max(0.0, fmean * (1 - penalty))


def fuzzy_ratio(a: str, b: str) -> float:
    na, nb = normalize(a)[:4000], normalize(b)[:4000]
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()


def token_f1(candidate: str, reference: str) -> float:
    c, r = Counter(tokenize(candidate)), Counter(tokenize(reference))
    if not c or not r:
        return 1.0 if not c and not r else 0.0
    overlap = sum((c & r).values())
    if not overlap:
        return 0.0
    return _f1(overlap / sum(c.values()), overlap / sum(r.values()))


def cosine_counts(a: dict, b: dict) -> float:
    if not a or not b:
        return 1.0 if not a and not b else 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def bow_cosine(a: str, b: str) -> float:
    return cosine_counts(Counter(tokenize(a)), Counter(tokenize(b)))


def vector_cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


METRICS = {
    "rouge_l": rouge_l,
    "rouge_1": lambda c, r: rouge_n(c, r, 1),
    "rouge_2": lambda c, r: rouge_n(c, r, 2),
    "rouge_3": lambda c, r: rouge_n(c, r, 3),
    "rouge_4": lambda c, r: rouge_n(c, r, 4),
    "rouge_5": lambda c, r: rouge_n(c, r, 5),
    "bleu": bleu,
    "gleu": gleu,
    "meteor": meteor,
    "fuzzy_match": fuzzy_ratio,
    "fuzzy": fuzzy_ratio,
    "token_f1": token_f1,
    "cosine": bow_cosine,
    "bow_cosine": bow_cosine,
}


def similarity(candidate: str, reference: str, metric: str = "rouge_l") -> float:
    fn = METRICS.get(metric)
    if fn is None:
        raise ValueError(f"unknown similarity metric {metric!r}; choose from {', '.join(sorted(METRICS))}")
    return float(fn(candidate or "", reference or ""))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------- diffs

# Words and punctuation carry their trailing whitespace, so whitespace never becomes
# the most common "token" (which made SequenceMatcher quadratic).
_DIFF_TOKEN_RE = re.compile(r"\w+\s*|[^\w\s]+\s*|\s+", re.UNICODE)
DIFF_MAX_TOKENS = 3000


def word_diff(a: str, b: str, max_tokens: int = DIFF_MAX_TOKENS) -> list[tuple[str, str]]:
    """Word-level diff as a list of (op, text) with op in '=', '-', '+'.

    Inputs beyond `max_tokens` tokens are not diffed token by token: the rest of
    each side is emitted as one deleted/inserted block (or kept if identical).
    """
    ta_all = _DIFF_TOKEN_RE.findall(a or "")
    tb_all = _DIFF_TOKEN_RE.findall(b or "")
    ta, tb = ta_all[:max_tokens], tb_all[:max_tokens]
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=len(tb) > 1500)
    out: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.append(("=", "".join(ta[i1:i2])))
        else:
            if i2 > i1:
                out.append(("-", "".join(ta[i1:i2])))
            if j2 > j1:
                out.append(("+", "".join(tb[j1:j2])))
    rest_a, rest_b = "".join(ta_all[max_tokens:]), "".join(tb_all[max_tokens:])
    if rest_a == rest_b:
        if rest_a:
            out.append(("=", rest_a))
    else:
        if rest_a:
            out.append(("-", rest_a))
        if rest_b:
            out.append(("+", rest_b))
    return out


def change_ratio(a: str, b: str) -> float:
    """How much changed between a and b: 0 = same words, 1 = nothing in common.

    Linear time (bag of words over the full texts), so it is safe for any length.
    """
    if (a or "").strip() == (b or "").strip():
        return 0.0
    return round(1.0 - token_f1(a or "", b or ""), 4)


diff_ratio = change_ratio  # backwards-compatible name


def unified_diff(a: str, b: str, from_name: str = "before", to_name: str = "after") -> str:
    lines = difflib.unified_diff(
        (a or "").splitlines(keepends=True),
        (b or "").splitlines(keepends=True),
        fromfile=from_name,
        tofile=to_name,
    )
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines)


# ------------------------------------------------------- language detection

_STOPWORDS = {
    "en": "the and is are was were to of in that it for you with on as this be have not but or they at by from your can will an we our if "
    "about while which their there would should could than then also more into over has had been what when where who how all any "
    "some such only very just a i my me its his her him she he them these those out up get make like good new please thanks thank",
    "es": "el la los las de que y en un una es por para con no se lo su al del como más pero sus le ya este sí porque esta entre cuando muy sin sobre también usted",
    "fr": "le la les de des et est un une que qui dans pour pas sur au avec ce il elle ne se plus par vous nous sont mais ou cette aux du",
    "de": "der die das und ist nicht ein eine zu den mit von sie es im für auf dem des sich auch ich wir ihr sind werden oder aber wie bitte "
    "am ihre ihnen wird hat kann noch nur bei nach zum zur einen einem einer danke vielen",
    "it": "il lo la gli le di che è un una per non con del della sono si ma come anche più questo questa nel alla ho",
    "pt": "o a os as de que e é um uma para com não do da em por mais se como mas foi ao dos das você seu sua",
    "nl": "de het een en van is dat niet ik je op te zijn met voor er maar ook als bij aan wat ze hij u",
    "sv": "och att det som en är på för av med till den har inte jag om ett var men vi ni de",
    "pl": "i w nie się na to że z do jest jak ale o co tak za od po czy jego ich być",
    "tr": "ve bir bu da de için ile çok ne daha gibi ama mi olarak var ben sen değil",
    "id": "yang dan di ke dari ini itu untuk dengan tidak ada akan saya anda adalah juga atau pada",
    "ro": "și în de la nu cu că pe un o este pentru sunt din care mai ce",
}
_STOPSETS = {lang: set(ws.split()) for lang, ws in _STOPWORDS.items()}

_SCRIPTS = [
    ("ja", re.compile(r"[぀-ヿ]")),
    ("ko", re.compile(r"[가-힯]")),
    ("zh", re.compile(r"[一-鿿]")),
    ("ru", re.compile(r"[Ѐ-ӿ]")),
    ("ar", re.compile(r"[\u0600-ۿ]")),
    ("he", re.compile(r"[֐-׿]")),
    ("hi", re.compile(r"[ऀ-ॿ]")),
    ("th", re.compile(r"[฀-๿]")),
    ("el", re.compile(r"[Ͱ-Ͽ]")),
]

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "sv": "Swedish",
    "pl": "Polish",
    "tr": "Turkish",
    "id": "Indonesian",
    "ro": "Romanian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian (Cyrillic)",
    "ar": "Arabic",
    "he": "Hebrew",
    "hi": "Hindi",
    "th": "Thai",
    "el": "Greek",
}


def _strip_code(text: str) -> str:
    text = re.sub(r"```.*?(```|$)", " ", text, flags=re.S)
    return re.sub(r"`[^`]*`", " ", text)


def detect_language(text: str) -> tuple[str | None, float]:
    """Return (language code, confidence 0..1); (None, 0) when unsure."""
    sample = _strip_code(text or "")[:5000]
    letters = [ch for ch in sample if ch.isalpha()]
    if len(letters) < 12:
        return None, 0.0
    # Non-Latin scripts first: they are unambiguous.
    total = len(letters)
    for code, pattern in _SCRIPTS:
        share = len(pattern.findall(sample)) / total
        if share > 0.3:
            if code == "zh" and _SCRIPTS[0][1].search(sample):
                code = "ja"
            return code, min(1.0, share + 0.2)
    toks = [t for t in re.findall(r"[^\W\d_]+", sample.lower())]
    if len(toks) < 6:
        return None, 0.0
    scores = {lang: sum(1 for t in toks if t in words) / len(toks) for lang, words in _STOPSETS.items()}
    distinct = {lang: len({t for t in toks if t in words}) for lang, words in _STOPSETS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, s1), (_, s2) = ranked[0], ranked[1]
    if s1 < 0.12 or s1 - s2 < 0.04 or distinct[best] < 3:
        return None, 0.0
    # Short function words ("a", "o", "e", "de") are shared between languages: only call
    # plausible English something else on clear evidence.
    if best != "en" and scores["en"] >= 0.12 and s1 - scores["en"] < 0.08:
        return None, 0.0
    return best, min(1.0, 0.5 + (s1 - s2) * 3)


def language_name(code: str | None) -> str:
    return LANGUAGE_NAMES.get(code or "", code or "unknown")


def iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_strings(v)
