"""Minimal JSON-over-HTTP client (urllib) with retries, backoff and error parsing.

Redirects are never followed: provider APIs do not redirect, and following one
would re-send the API key to whatever host the redirect names.
"""

from __future__ import annotations

import codecs
import email.utils
import http.client
import json
import math
import re
import secrets
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from typing import Any, Callable

from ..textutil import clean_surrogates, has_surrogate_escape, json_loads, plural
from .base import ProviderError

RETRY_STATUSES = (408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529)
NON_RETRYABLE_CODES = ("insufficient_quota", "billing_hard_limit_reached", "invalid_api_key")
MAX_DELAY = 60.0

MAX_RESPONSE_BYTES = 64 * 1024 * 1024  # no model answer is this large; a runaway body is cut off
MAX_ERROR_BYTES = 1024 * 1024  # an error page is read this far

# Set by the runner on Ctrl+C: requests stop retrying and new ones are not sent.
CANCEL = threading.Event()
_JITTER = secrets.SystemRandom()  # spreads retries out; the OS's generator, so no scanner asks


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None  # urllib then raises HTTPError for the 3xx response


_OPENER = urllib.request.build_opener(_NoRedirect)


def _user_agent() -> str:
    from .. import __version__

    return f"llmshift/{__version__} (+python-urllib)"


def _host(url: str) -> str:
    """host[:port] of a URL for messages, never with a user:password@ part."""
    netloc = urllib.parse.urlsplit(url).netloc
    return netloc.rsplit("@", 1)[-1] or "the server"


def display_url(url: str) -> str:
    """A URL as it may be shown: no user:password@ part and no query string, also when
    the scheme is missing (admin:secret@box:8000/v1, a typo in base_url)."""
    text = str(url)
    parts = urllib.parse.urlsplit(text)
    if parts.netloc:
        return urllib.parse.urlunsplit((parts.scheme, _host(text), parts.path, "", ""))
    head, sep, tail = text.split("?", 1)[0].split("#", 1)[0].partition("/")
    return head.rsplit("@", 1)[-1] + sep + tail


_SECRET_HEADERS = ("authorization", "x-api-key", "api-key", "x-goog-api-key", "proxy-authorization")


def _check_headers(headers: dict) -> None:
    """HTTP header values are Latin-1 text without line breaks. A key with a stray newline
    (a CRLF .env file) or an invisible character would otherwise fail inside http.client,
    whose error message quotes the whole header, key included."""
    for name, value in headers.items():
        text = str(value)
        try:
            text.encode("latin-1")
            ok = not any(c in text for c in "\r\n\0")
        except UnicodeEncodeError:
            ok = False
        if not ok:
            secret = name.lower() in _SECRET_HEADERS
            raise ProviderError(
                f"the {name} header contains a character HTTP does not allow"
                + (": check the API key for a stray newline or an invisible character (its environment variable or .env file)" if secret else ""),
                code="invalid_api_key" if secret else "invalid_header",
            )


def _decoded(payload: bytes, headers: Any, url: str) -> bytes:
    """The body without its Content-Encoding. llmshift asks for none, but some proxies
    compress anyway; a compressed body is unpacked no further than the size limit."""
    enc = str((headers.get("Content-Encoding") if headers is not None else None) or "").strip().lower()
    if enc in ("", "identity") or not payload:
        return payload
    if enc not in ("gzip", "x-gzip", "deflate"):
        raise ProviderError(f"the response from {_host(url)} is compressed with {enc}, which llmshift cannot read", code="invalid_response")
    # gzip; deflate as the standard says (zlib); deflate as some servers send it (raw)
    for wbits in ((16 + zlib.MAX_WBITS,) if "gzip" in enc else (zlib.MAX_WBITS, -zlib.MAX_WBITS)):
        try:
            return zlib.decompressobj(wbits).decompress(payload, MAX_RESPONSE_BYTES + 1)
        except zlib.error:
            continue
    raise ProviderError(f"the response from {_host(url)} says it is {enc}-compressed but cannot be unpacked", code="invalid_response", retryable=True)


def _text(payload: bytes, headers: Any) -> str:
    """The body as text: UTF-8 (what JSON must be, and what servers that declare
    ISO-8859-1 by default usually send), else the declared charset."""
    charset = None
    try:
        charset = headers.get_content_charset() if headers is not None else None
        if charset:
            codecs.lookup(charset)
    except (LookupError, AttributeError, TypeError, ValueError):
        charset = None
    if payload.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return payload.decode("utf-16", errors="replace")
    if charset and charset.replace("-", "").replace("_", "").startswith(("utf16", "utf32")):
        return payload.decode(charset, errors="replace")
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return payload.decode(charset or "utf-8", errors="replace")


def _parse_error(status: int | None, raw: bytes, url: str, headers: Any = None) -> ProviderError:
    text = _text(raw, headers) if raw else ""
    message, code, param, body = text.strip()[:500] or f"HTTP {status}", None, None, None
    try:
        body = json_loads(text) if text.strip() else None
    except (ValueError, RecursionError):
        body = None
    if body is None and re.match(r"\s*<(?:!doctype|html|head|body|\?xml)", text, re.I):
        # A web server's error page (a proxy, a CDN, a wrong URL): its title, not its markup.
        title = re.search(r"<title[^>]*>(.{0,300}?)</title>", text[:20000], re.I | re.S)
        page = " ".join(title.group(1).split()) if title else ""
        message = f"the server sent a web page ({page}), not an API answer" if page else "the server sent a web page, not an API answer"
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            message = str(err.get("message") or err.get("msg") or message)
            code = err.get("code")
            if isinstance(code, int) or code is None:  # Gemini: {"code": 400, "status": "INVALID_ARGUMENT"}
                code = err.get("type") or err.get("status") or code
            param = err.get("param")
            for detail in err.get("details") or []:
                if isinstance(detail, dict) and detail.get("reason") == "API_KEY_INVALID":
                    code = "invalid_api_key"
        elif isinstance(err, str):
            message = err
        if body.get("type") == "error" and isinstance(body.get("error"), dict):
            code = body["error"].get("type") or code
        errors = body.get("errors")
        if isinstance(errors, list):  # GitHub: "Validation Failed", then what failed
            details = [
                str(e.get("message") or " ".join(str(e[k]) for k in ("resource", "field", "code") if e.get(k))) if isinstance(e, dict) else str(e)
                for e in errors[:5]
            ]
            details = [d for d in details if d and d not in message]
            if details:
                message = f"{message}: {'; '.join(details)}"
    if status is not None and 300 <= status < 400:
        location = (headers.get("Location") if headers is not None else None) or "another URL"
        return ProviderError(
            f"the server redirected to {location}; not followed so credentials are never sent to another host (check the URL, e.g. base_url)",
            status=status,
            code="redirect",
        )
    if status == 404 and not code:
        code = "not_found"
    told_to_wait = headers is not None and bool(headers.get("retry-after") or headers.get("retry-after-ms"))
    # A 403 is final, except when the server says when to retry or it is GitHub's
    # secondary rate limit (GitHub asks for at least a minute's wait).
    secondary = status == 403 and "secondary rate limit" in message.lower()
    retryable = (status in RETRY_STATUSES or (status == 403 and told_to_wait) or secondary) and str(code) not in NON_RETRYABLE_CODES
    wait = _header_delay(headers) if told_to_wait else None  # unreadable: the caller backs off per attempt
    if wait is None and secondary:
        wait = MAX_DELAY
    asked = _asked_wait(headers) if told_to_wait else None
    if retryable and asked is not None and asked > MAX_DELAY:
        # "Retry after 1 day": waiting the maximum and asking again only wastes the run.
        retryable = False
        message += f"; the server asks to wait {_human_wait(asked)} before trying again, longer than llmshift waits ({MAX_DELAY:.0f} s): try again later"
    host = _host(url)
    return ProviderError(f"{message} ({host})", status=status, code=str(code) if code is not None else None, param=param, retryable=retryable, body=body, retry_after=wait)


def _asked_wait(headers: Any) -> float | None:
    """The wait the server asks for in seconds (Retry-After in seconds or as a date,
    retry-after-ms), however long; None when it says nothing readable."""
    if headers is None:
        return None
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            value = float(ms) / 1000.0
            if math.isfinite(value):
                return max(0.0, value)
        except ValueError:
            pass
    ra = headers.get("retry-after")
    if ra:
        try:
            value = float(ra)
            if math.isfinite(value):
                return max(0.0, value)
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(ra)
                return max(0.0, parsed.timestamp() - time.time())
            except (TypeError, ValueError, IndexError, OverflowError):
                pass  # unusual value such as "30s"
    return None


def _header_delay(headers: Any) -> float | None:
    """The wait the server asks for, at most MAX_DELAY; None when it says nothing readable."""
    asked = _asked_wait(headers)
    return None if asked is None else min(MAX_DELAY, asked)


def _human_wait(seconds: float) -> str:
    for unit, size, least in (("day", 86400, 86400), ("h", 3600, 3600), ("min", 60, 600)):
        if seconds >= least:
            n = round(seconds / size)
            return f"{n} {unit}" + ("s" if unit == "day" and n != 1 else "")
    return f"{seconds:.0f} s"


def _retry_delay(headers: Any, attempt: int) -> float:
    """What the server asked for, else exponential backoff for this attempt."""
    delay = _header_delay(headers)
    return delay if delay is not None else min(MAX_DELAY, (0.5 * 2**attempt) + _JITTER.uniform(0, 0.5))


def _via_proxy(url: str) -> str:
    """The words " through the proxy at HOST" when a proxy setting applies to this URL:
    then the proxy, not the API, may be what cannot be reached."""
    parts = urllib.parse.urlsplit(url)
    try:
        proxy = urllib.request.getproxies().get(parts.scheme)
        if not proxy or urllib.request.proxy_bypass(parts.hostname or ""):
            return ""
    except Exception:
        return ""
    return f" through the proxy at {_host(proxy if '://' in proxy else f'http://{proxy}')}"


def _sleep(seconds: float, sleep: Callable[[float], None]) -> None:
    end = time.time() + seconds
    while not CANCEL.is_set():
        remaining = end - time.time()
        if remaining <= 0:
            return
        sleep(min(remaining, 0.25))


def _clean_floats(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_floats(v) for v in value]
    return value


def _encode(body: Any) -> bytes:
    try:
        return json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8", "replace")
    except ValueError:  # NaN/Infinity are not valid JSON: send null instead
        return json.dumps(_clean_floats(body), ensure_ascii=False, allow_nan=False).encode("utf-8", "replace")


def request(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    body: Any = None,
    timeout: float = 120,
    max_retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
    raw: bool = False,
    idempotent: bool = True,
) -> Any:
    """Send a request and return parsed JSON (or bytes when raw=True). With
    idempotent=False (a POST that sends an alert or creates something), a request whose
    answer timed out is not sent again: the server may have received it already."""
    data = _encode(body) if body is not None else None
    hdrs = {"User-Agent": _user_agent(), "Accept": "*/*" if raw else "application/json", "Accept-Encoding": "identity"}
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    hdrs.update({k: v for k, v in (headers or {}).items() if v is not None})
    _check_headers(hdrs)
    try:
        urllib.request.Request(url)
    except ValueError:  # urllib's message quotes the whole URL, which can hold a secret (a webhook token)
        raise ProviderError(f"invalid URL for {_host(url)}: it must start with https:// or http://", code="invalid_url") from None
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):  # urllib would read file:// (a local file) or ftp:// just as happily
        raise ProviderError(f"unsupported URL scheme {scheme}:// - llmshift only talks to https:// and http:// addresses", code="invalid_url")
    if "@" in parts.netloc:  # urllib would look up "user:password@host" as a host name
        raise ProviderError(
            f"the URL for {_host(url)} has a user name or password in it, which is not sent from there: "
            "set credentials as a header instead (headers: {Authorization: ...} under the provider in llmshift.yaml)",
            code="invalid_url",
        )
    attempt = 0
    while True:
        if CANCEL.is_set():
            raise ProviderError("cancelled", code="cancelled")
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method.upper())
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                payload = _decoded(resp.read(MAX_RESPONSE_BYTES + 1), resp.headers, url)
                resp_headers = resp.headers
            if len(payload) > MAX_RESPONSE_BYTES:
                raise ProviderError(f"response from {_host(url)} is larger than {MAX_RESPONSE_BYTES // (1024 * 1024)} MB", code="invalid_response")
            if raw:
                return payload
            text = _text(payload, resp_headers)
            if not text.strip():
                return {}
            try:
                parsed = json_loads(text)
                return clean_surrogates(parsed) if has_surrogate_escape(text) else parsed
            except RecursionError:
                raise ProviderError(f"response from {_host(url)} is nested too deeply to read", code="invalid_response") from None
            except ValueError:
                # JSON cut short on the way (a dropped proxy connection) is worth another try;
                # a web page where the API should be is a wrong URL, and stays one.
                if idempotent and attempt < max_retries and text.lstrip()[:1] in ("{", "["):
                    _sleep(_retry_delay(None, attempt), sleep)
                    attempt += 1
                    continue
                raise ProviderError(f"invalid JSON response from {_host(url)}: {text[:200]!r}", code="invalid_response", retryable=True) from None
        except urllib.error.HTTPError as exc:
            try:
                payload = _decoded(exc.read(MAX_ERROR_BYTES), exc.headers, url)
            except Exception:
                payload = b""
            finally:
                exc.close()  # the error response is a socket too
            err = _parse_error(exc.code, payload, url, exc.headers)
            if err.retryable and attempt < max_retries:
                _sleep(err.retry_after if err.retry_after is not None else _retry_delay(exc.headers, attempt), sleep)
                attempt += 1
                continue
            raise err from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as exc:
            cert = next((e for e in (getattr(exc, "reason", None), exc) if isinstance(e, ssl.SSLCertVerificationError)), None)
            if cert is not None:  # retrying cannot fix a certificate problem
                host = _host(url)
                detail = getattr(cert, "verify_message", None) or cert
                raise ProviderError(
                    f"cannot verify the TLS certificate of {host} ({detail}). With Python from python.org on macOS, run "
                    "'Install Certificates.command' in the Python folder; behind a proxy that inspects TLS, set SSL_CERT_FILE "
                    "to your organisation's CA bundle",
                    code="ssl_error",
                ) from None
            reason = getattr(exc, "reason", exc)
            sent = isinstance(exc, (socket.timeout, TimeoutError))  # urllib wraps timeouts before the request is sent in URLError
            if sent and not idempotent:
                raise ProviderError(f"no answer from {_host(url)} within {timeout:g}s; it may still have received the request", code="timeout") from None
            if attempt < max_retries:
                _sleep(_retry_delay(None, attempt), sleep)
                attempt += 1
                continue
            if sent:  # connected and asked, but no answer in time: slow, not unreachable
                raise ProviderError(
                    f"no answer from {_host(url)}{_via_proxy(url)} within {timeout:g}s ({plural(attempt + 1, 'try', 'tries')}): "
                    "the server was reached but is slow or overloaded; try again later or raise timeout: in llmshift.yaml",
                    code="timeout", retryable=True,
                ) from None
            raise ProviderError(f"cannot reach {_host(url)}{_via_proxy(url)}: {reason}", code="connection_error", retryable=True) from None
        except ValueError:  # http.client refused the request; its message would quote the headers
            raise ProviderError(f"the request to {_host(url)} could not be sent: a header or the URL has a character HTTP does not allow", code="invalid_header") from None


def post_json(url: str, body: Any, **kwargs: Any) -> Any:
    return request("POST", url, body=body, **kwargs)


def get_json(url: str, **kwargs: Any) -> Any:
    return request("GET", url, **kwargs)


def require_object(data: Any, what: str, *keys: str) -> dict:
    """A provider reply must be a JSON object with at least one of `keys`."""
    if not isinstance(data, dict) or (keys and not any(k in data for k in keys)):
        shown = json.dumps(data)[:120] if data not in ({}, None) else "an empty body"
        raise ProviderError(f"unexpected {what} response: {shown}", code="invalid_response", retryable=True)
    return data
