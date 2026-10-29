"""Redaction of secrets and personal data before logs become committed test files.

Placeholders are consistent within one Redactor: the same e-mail address always
becomes <EMAIL_1>, so a model that echoes the input still matches the redacted
baseline output.
"""

from __future__ import annotations

import functools
import os
import re
import threading
from typing import Any

# Between a setting's name and its value: the name's closing quote (escaped in a JSON
# string), then =, : or =>.
_SEPARATOR = r"\\?[\"']?[ \t]*(?:=>|[:=])[ \t]*"
# The word in a setting's name that says it holds a credential. A rule starts at the word
# itself: a name before it ("[\w.-]{0,40}password") made every position of a text try 40
# starts, and missed the password of a longer name. Only the value is replaced. A pwd
# before a path is the working directory (PWD=/home/bob).
_SECRET_WORD = (
    r"(?:passw(?:or)?d|pwd(?!" + _SEPARATOR + r"[\"']?[/~])|secret(?:[-_]?key)?|api[-_]?key|access[-_]?key|token|credential)"
)
# A value looks like a credential when its first 64 characters hold a digit and a letter.
# Looking no further keeps a text of many names quick (pwd=pwd=pwd=...), and a value may be
# long (an OAuth token). A quoted value is taken whole, symbols and all (xK9#mP2&vL5?qR8(wN3),
# but not a placeholder: ${DB_PASSWORD_1}, {{ ... }}, <SECRET_1>.
_QUOTED_CHAR = r"(?:\\\\|(?!\\?(?P=q))[^\s\\])"
_QUOTED_VALUE = (
    r"\\?(?P<q>[\"'])(?!\$\{|\{\{|<)(?P<v>(?=" + _QUOTED_CHAR + r"{0,63}?\d)(?=" + _QUOTED_CHAR + r"{0,63}?[A-Za-z])"
    + _QUOTED_CHAR + r"{6,4096})\\?(?P=q)"
)
# An unquoted value runs to a space, or to the & ; or , before the next name= or name:, and
# leaves out a sentence's last . , ; : or closing bracket. Code is not a value: tokens[0],
# hashlib.sha256(pw), ${DB_PASSWORD_1}.
_BARE_CHAR = r"(?:[^\s\"'`<>\\&;,]|[&;,](?![\w.-]{1,64}[ \t]*[:=]))"
_BARE_VALUE = (
    r"(?![A-Za-z_][\w.]{0,64}[(\[]|\$\{)(?P<v>(?=" + _BARE_CHAR + r"{0,63}?\d)(?=" + _BARE_CHAR + r"{0,63}?[A-Za-z])"
    + _BARE_CHAR + r"{6,4096}(?<![.,;:)\]}]))"
)
# An AWS secret access key: 40 characters, with upper and lower case letters and digits (a
# SHA-1 is 40 lower-case hex digits).
_AWS_SECRET = r"(?=[A-Za-z0-9/+]{0,39}[A-Z])(?=[A-Za-z0-9/+]{0,39}[a-z])(?=[A-Za-z0-9/+]{0,39}\d)[A-Za-z0-9/+]{40}"  # nosec B105  # a pattern that finds AWS secret keys in order to redact them

# (pattern, what it must contain): a substring test is far cheaper than a regex that tries
# every position of a long text (a document or an image in a request).
_SECRETS = [
    (r"\bsk-ant-[A-Za-z0-9_\-]{20,}", ("sk-ant-",)),  # Anthropic
    (r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}", ("sk-",)),  # OpenAI (and DeepSeek)
    # An AWS secret access key has no prefix, only its 40 characters: it is known by what
    # surrounds it, a "secret access key" label in any spelling (aws_secret_access_key =,
    # SecretAccessKey:, --aws-secret-access-key, => ...) or the access key id just before it.
    (
        r"(?i)secret[ \t_.-]?access[ \t_.-]?key(?:[ \t]+(?:is|was))?[^A-Za-z0-9/+\n]{1,8}(?P<v>[A-Za-z0-9/+]{40})(?![A-Za-z0-9/+=])",
        ("secret",),
    ),
    (r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b[^\n]{0,80}?(?<![A-Za-z0-9/+])(?P<v>[A-Za-z0-9/+]{40})(?![A-Za-z0-9/+=])", ("AKIA", "ASIA")),
    (r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", ("AKIA", "ASIA")),  # AWS access key ids, long-lived and temporary
    # ... or plainly key=<the 40 characters>. Case matters in the value, so only "key" ignores it.
    (r"(?i:key)" + _SEPARATOR + r"\\?[\"']?(?P<v>" + _AWS_SECRET + r")(?![A-Za-z0-9/+=])", ("key", "Key", "KEY")),
    (r"\bgh[pousr]_[A-Za-z0-9]{30,}\b", ("ghp_", "gho_", "ghu_", "ghs_", "ghr_")),
    (r"\bgithub_pat_[A-Za-z0-9_]{40,}\b", ("github_pat_",)),
    (r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b", ("xox",)),  # Slack
    (r"\bAIza[0-9A-Za-z_\-]{35}(?![0-9A-Za-z_\-])", ("AIza",)),  # Google (a key may end in "-")
    (r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b", ("eyJ",)),  # JWTs
    (r"(?i)\b(?:bearer)\s+[A-Za-z0-9_\-.=]{20,}", ()),
    # Each BEGIN looks no further than the next one: a text of BEGINs with no END made every
    # BEGIN rescan the rest (quadratic: a 224 KB message held the GIL for 15 s).
    (r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----(?:(?!-----BEGIN )[\s\S])+?-----END [A-Z ]{0,40}PRIVATE KEY-----", ("PRIVATE KEY",)),
    (r"\bxai-[A-Za-z0-9]{20,}", ("xai-",)),  # xAI
    (r"\bgsk_[A-Za-z0-9]{20,}", ("gsk_",)),  # Groq
    (r"\bpplx-[A-Za-z0-9]{20,}", ("pplx-",)),  # Perplexity
    (r"\bfw_[A-Za-z0-9]{20,}", ("fw_",)),  # Fireworks
    (r"\bhf_[A-Za-z0-9]{20,}", ("hf_",)),  # Hugging Face
    (r"\br8_[A-Za-z0-9]{20,}", ("r8_",)),  # Replicate
    (r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}", ("_live_", "_test_")),  # Stripe
    (r"\bgl(?:pat|dt|rt|ptt|cbt|soat)-[A-Za-z0-9_\-]{20,}", ("gl",)),  # GitLab
    (r"\bnpm_[A-Za-z0-9]{36}\b", ("npm_",)),  # npm
    # Only the value (group v) is replaced. A credential in a URL: postgres://user:PASSWORD@host
    # (a template such as https://{user}:{password}@{host} holds none).
    (r"\b[A-Za-z][A-Za-z0-9+.-]{1,20}://[^\s:/@\"'<>{}$%()]{0,128}:(?P<v>[^\s@/\"'<>{}$%()]{1,256})@", ("://",)),
    (r"(?i)\bauthorization[\"']?[ \t]*[:=][ \t]*[\"']?basic[ \t]+(?P<v>[A-Za-z0-9+/=]{8,})", ("authorization",)),
    # The value of a password, secret or key setting (password=..., "api-key": "...", an Azure
    # key in a config dump, a tool's JSON arguments), when it looks like one.
    (r"(?i)" + _SECRET_WORD + _SEPARATOR + _QUOTED_VALUE, ("pass", "pwd", "secret", "key", "token", "credential")),
    (r"(?i)" + _SECRET_WORD + _SEPARATOR + _BARE_VALUE, ("pass", "pwd", "secret", "key", "token", "credential")),
    # ... and in XML: <password>...</password> (a Maven settings.xml, a .NET config). No > in the
    # value: with it, a run of password>a1password>a1... made every word rescan the rest.
    (
        r"(?i)" + _SECRET_WORD + r">(?P<v>(?=[^<>\s]{0,63}?\d)(?=[^<>\s]{0,63}?[A-Za-z])[^<>\s]{6,4096})</",
        ("pass", "pwd", "secret", "key", "token", "credential"),
    ),
]
SECRET_PATTERNS = [("SECRET", re.compile(p)) for p, _ in _SECRETS]
_MARKERS = {p: markers for p, markers in _SECRETS if markers}

# A JSON key that names a credential: a password or a secret (its value is one, whatever
# it looks like), a token or an API key (its value is one when it looks like one).
_STRONG_KEY_RE = re.compile(r"(?:passw(?:or)?d|pwd|secret|credential)s?$", re.I)
_SECRET_KEY_RE = re.compile(_SECRET_WORD + "$", re.I)
_AWS_SECRET_RE = re.compile(_AWS_SECRET)
# Values that stand for a credential without being one.
_PLACEHOLDER_RE = re.compile(
    r"\$\{.*\}|\$[A-Z_][A-Z0-9_]*|\{\{.*\}\}|<[^<>]*>|%\(?\w*\)?s|[*xX#.\-_]+"
    r"|(?i:none|null|true|false|redacted|" + _SECRET_WORD + ")"
)
_PATH_RE = re.compile(r"[/~]|[A-Za-z]:[\\/]")


@functools.lru_cache(maxsize=1024)
def _key_kind(key: str) -> int:
    """3 for a password or secret key, 2 for a token or API key, 1 for one that ends in
    "key" (an AWS secret access key may be under it), 0 for any other."""
    if _STRONG_KEY_RE.search(key):
        return 3
    if _SECRET_KEY_RE.search(key):
        return 2
    return 1 if key.lower().endswith("key") else 0


def _secret_value(key: str, value: str, item: dict) -> bool:
    """Whether the string `value` under `key` of the JSON object `item` is a credential."""
    kind = _key_kind(key)
    if not kind or len(value) > 4096:
        return False
    v = value.strip()
    if len(v) < 4 or "\n" in v or _PLACEHOLDER_RE.fullmatch(v):
        return False
    if kind == 3:  # a "pwd" may be the working directory: {"pwd": "/workspace/app"}
        return not (key.lower().endswith("pwd") and _PATH_RE.match(v))
    if kind == 2:  # not the text of a logprobs entry: {"token": "Hello", "logprob": -0.1}
        if any(isinstance(k, str) and "logprob" in k.lower().replace("_", "") for k in item):
            return False
        return len(v) >= 8 and " " not in v and any(c.isalpha() for c in v) and not v.isalpha()
    return _AWS_SECRET_RE.fullmatch(v) is not None

PII_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]{1,64}@[\w-]+(?:\.[\w-]+)+\b")),
    ("CARD", re.compile(r"\b\d(?:[ .-]?\d){12,18}\b")),  # ends on a digit: the space after the number stays
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b", re.I)),
    ("PHONE", re.compile(r"(?<![\w-])(?:\+\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\w-])")),
    ("PHONE", re.compile(r"(?<![\w-])\+\d{1,3}(?:[ .-]?\(?\d{1,5}\)?){2,5}(?![\w-])")),
    ("PHONE", re.compile(r"(?<![\w.-])0\d(?:[ .-]\d{2}){4}(?![\w-])")),  # 01 23 45 67 89
    ("PHONE", re.compile(r"(?<![\w.-])0\d{2,4}[ .-]\d{3,4}[ .-]?\d{3,4}(?![\w-])")),  # 020 7946 0958, 030 12345678
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
]

# Environment variables whose values are credentials: their exact values are redacted
# too, which catches keys with no recognisable shape (Azure, Mistral, Cohere...).
_SECRET_ENV_RE = re.compile(r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|AD_TOKEN)$", re.I)


def _env_secrets() -> list[str]:
    values = {v.strip() for k, v in os.environ.items() if _SECRET_ENV_RE.search(k) and v and len(v.strip()) >= 16}
    return sorted(values, key=len, reverse=True)


_IMAGE_DATA_URL_RE = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,")
_BASE64_RE = re.compile(r"[A-Za-z0-9+/=\r\n]*")
_DIGIT_RE = re.compile(r"\d")


def _may_match(pattern: re.Pattern, text: str, lowered: list) -> bool:
    markers = _MARKERS.get(pattern.pattern)
    if markers is not None and pattern.flags & re.I:  # case-insensitive: look in the lowered text
        if not lowered:
            lowered.append(text.lower())
        return any(m in lowered[0] for m in markers)
    if markers is not None:
        return any(m in text for m in markers)
    if "bearer" in pattern.pattern:
        if not lowered:
            lowered.append(text.lower())
        return "bearer" in lowered[0]
    if "@" in pattern.pattern:  # e-mail
        return "@" in text
    return _DIGIT_RE.search(text) is not None  # cards, numbers, IBANs and IPs all hold digits


_PLAIN_KEYS = ("role", "type", "id", "tool_call_id", "name")  # short values here are never secrets


def _identity(label: str, value: str) -> str:
    """The same address or number written differently gets the same placeholder."""
    if label == "EMAIL":
        return value.lower()
    if label in ("PHONE", "CARD", "SSN"):
        return re.sub(r"\D", "", value)
    if label == "IBAN":
        return re.sub(r"\s", "", value).upper()
    return value


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _iban_ok(value: str) -> bool:
    """The IBAN check digits (ISO 13616, mod 97): product codes and the like fail it."""
    text = re.sub(r"\s", "", value).upper()
    digits = "".join(str(int(c, 36)) for c in text[4:] + text[:4])
    return int(digits) % 97 == 1


def _plausible(label: str, value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if label == "CARD":
        return 13 <= len(digits) <= 19 and _luhn_ok(digits)
    if label == "IBAN":
        return _iban_ok(value)
    if label == "PHONE":
        return len(digits) >= 7  # "+10 20 30" is not a phone number
    return True


class Redactor:
    def __init__(self, *, secrets: bool = True, pii: bool = False):
        self.secrets = secrets
        self.patterns = (SECRET_PATTERNS if secrets else []) + (PII_PATTERNS if pii else [])
        self.env_secrets = _env_secrets() if secrets else []
        self.mapping: dict[tuple[str, str], str] = {}
        self.counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def _placeholder(self, label: str, value: str) -> str:
        key = (label, _identity(label, value))
        with self._lock:
            if key not in self.mapping:
                self.counts[label] = self.counts.get(label, 0) + 1
                self.mapping[key] = f"<{label}_{self.counts[label]}>"
            return self.mapping[key]

    def text(self, text: str) -> str:
        if not text or not self.patterns:
            return text
        # Encoded data has nothing readable to redact, but only when that is all the text is:
        # an inline image, or a large raw base64 blob (prose has spaces; base64 does not).
        data_url = _IMAGE_DATA_URL_RE.match(text, 0, 128)
        if data_url and _BASE64_RE.fullmatch(text, data_url.end()):
            return text
        if len(text) > 4096 and _BASE64_RE.fullmatch(text):
            return text
        for secret in self.env_secrets:  # the live value of a key variable, whatever its shape
            if secret in text:
                text = text.replace(secret, self._placeholder("SECRET", secret))
        lowered: list = []
        for label, pattern in self.patterns:
            if not _may_match(pattern, text, lowered):
                continue

            def repl(m: re.Match, label: str = label) -> str:
                whole = m.group(0)
                if "v" in m.re.groupindex:  # only the value part is the secret
                    start, end = m.start("v") - m.start(), m.end("v") - m.start()
                    return whole[:start] + self._placeholder(label, m.group("v")) + whole[end:]
                return self._placeholder(label, whole) if _plausible(label, whole) else whole

            redacted = pattern.sub(repl, text)
            if redacted is not text:
                text, lowered = redacted, []
        return text

    def obj(self, value: Any) -> Any:
        """A copy of JSON-like data with every string redacted. Iterative: data nested
        deeper than Python's recursion limit (a tool's arguments, say) is handled too."""
        root: list = [None]
        stack: list[tuple[Any, Any, Any]] = [(root, 0, value)]
        copies: dict[int, Any] = {}  # a container met again (shared, or containing itself) is copied once
        while stack:
            parent, key, item = stack.pop()
            if isinstance(item, (list, dict)) and id(item) in copies:
                parent[key] = copies[id(item)]
            elif isinstance(item, str):
                parent[key] = self.text(item)
            elif isinstance(item, list):
                parent[key] = copies[id(item)] = copy = [None] * len(item)
                stack.extend((copy, i, v) for i, v in enumerate(item))
            elif isinstance(item, dict):
                parent[key] = copies[id(item)] = out = {}
                for k, v in item.items():
                    if k in _PLAIN_KEYS and isinstance(v, str) and len(v) < 64:
                        out[k] = v
                    elif self.secrets and isinstance(v, str) and isinstance(k, str) and _secret_value(k, v, item):
                        out[k] = self._placeholder("SECRET", v.strip())  # {"api_key": "..."}
                    else:
                        out[k] = None  # keeps the key order; filled in below
                        stack.append((out, k, v))
            else:
                parent[key] = item
        return root[0]

    @property
    def total(self) -> int:
        return sum(self.counts.values())
