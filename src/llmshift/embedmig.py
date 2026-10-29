"""`llmshift embeddings`: see how search results change before you switch embedding models.

A sample of your documents and queries is embedded with the old and the new model.
For every query the top-k documents under each model are compared (overlap, whether
the old top result survives, rank-biased overlap), and the cost and time of
re-embedding the whole corpus with the new model is estimated.
"""

from __future__ import annotations

import argparse
import heapq
import html
import math
import operator
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from . import console as con
from .errors import EXIT_FAIL, EXIT_OK, UsageError
from .providers.base import ProviderError
from .textutil import credit, json_loads, jsonl_lines, report_json, show_bidi

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".json", ".jsonl", ".ndjson"}
TEXT_FIELDS = ("text", "content", "body", "page_content", "chunk", "document", "passage", "summary")
TITLE_FIELDS = ("title", "name", "heading", "id", "source", "url", "path")
QUERY_FIELDS = ("query", "question", "input", "text", "prompt", "search")
LIST_FIELDS = ("data", "documents", "docs", "records", "items", "chunks", "queries")
SKIP_DIRS = {"node_modules", "__pycache__", "site-packages", "dist", "build", "venv"}
BATCH = 64
RBO_P = 0.9  # rank-biased overlap: how strongly the top ranks dominate
MAX_QUERY_CHARS = 6000

_SCRIPT_RE = re.compile(r"<(script|style|noscript|template|svg)\b.*?</\1\s*>", re.I | re.S)
_BLOCK_END_RE = re.compile(r"<br\s*/?>|</(?:p|div|h[1-6]|li|tr|section|article|pre|blockquote)\s*>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Doc:
    title: str
    text: str


@dataclass
class EmbeddingReport:
    old: str
    new: str
    k: int
    corpus_chunks: int
    sampled_chunks: int
    corpus_tokens: int
    queries: int
    query_source: str  # file | logs | list | pseudo
    mean_overlap: float
    top1_kept: float
    mean_rbo: float
    old_dims: int
    new_dims: int
    min_overlap: float
    reindex_cost: float | None = None
    price_source: str | None = None
    reindex_seconds: float | None = None
    worst: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.mean_overlap + 1e-9 >= self.min_overlap

    def to_dict(self) -> dict:
        return {
            "llmshift_version": __version__,
            "kind": "embeddings",
            "old": self.old,
            "new": self.new,
            "k": self.k,
            "corpus": {"chunks": self.corpus_chunks, "sampled": self.sampled_chunks, "tokens": self.corpus_tokens},
            "queries": {"count": self.queries, "source": self.query_source},
            "mean_overlap": round(self.mean_overlap, 4),
            "top1_kept": round(self.top1_kept, 4),
            "mean_rbo": round(self.mean_rbo, 4),
            "dimensions": {"old": self.old_dims, "new": self.new_dims, "changed": self.old_dims != self.new_dims},
            "reindex": {
                "cost_usd": round(self.reindex_cost, 6) if self.reindex_cost is not None else None,
                "price_source": self.price_source,
                "seconds": round(self.reindex_seconds, 1) if self.reindex_seconds is not None else None,
            },
            "min_overlap": self.min_overlap,
            "passed": self.passed,
            "worst_queries": self.worst,
            "notes": self.notes,
        }


# ------------------------------------------------------------------ corpus


def html_text(raw: str) -> str:
    text = _SCRIPT_RE.sub(" ", raw)
    text = _BLOCK_END_RE.sub("\n\n", text)
    text = html.unescape(_TAG_RE.sub(" ", text))
    lines = [re.sub(r"[ \t\f\v\xa0]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _split_long(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for sentence in _SENTENCE_RE.split(text):
        while len(sentence) > max_chars:  # one huge sentence: cut at a space where possible
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if cur:
                out.append(cur)
                cur = ""
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if cur and len(cur) + 1 + len(sentence) > max_chars:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}" if cur else sentence
    if cur:
        out.append(cur)
    return [x for x in out if x]


def chunk_text(text: str, max_chars: int = 1500) -> list[str]:
    """Paragraphs packed into chunks of at most `max_chars` characters."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    chunks: list[str] = []
    cur = ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        for piece in [para] if len(para) <= max_chars else _split_long(para, max_chars):
            if cur and len(cur) + 2 + len(piece) > max_chars:
                chunks.append(cur)
                cur = piece
            else:
                cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def _first_field(record: dict, names: tuple) -> Any:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and name in ("id",):
            return str(value)
    return None


def _json_records(text: str, suffix: str) -> list:
    if suffix in (".jsonl", ".ndjson"):
        out = []
        for line in jsonl_lines(text):
            line = line.strip()
            if line:
                try:
                    out.append(json_loads(line))
                except (ValueError, RecursionError):
                    continue
        return out
    try:
        data = json_loads(text)
    except (ValueError, RecursionError):
        return []
    if isinstance(data, dict):
        for name in LIST_FIELDS:
            if isinstance(data.get(name), list):
                return data[name]
        return [data]
    return data if isinstance(data, list) else []


def _records(path: Path) -> list[tuple[str, str]]:
    """(title, text) pairs from one corpus file."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    suffix = path.suffix.lower()
    name = path.as_posix()
    if suffix in (".json", ".jsonl", ".ndjson"):
        out = []
        for i, rec in enumerate(_json_records(raw, suffix)):
            if isinstance(rec, str) and rec.strip():
                out.append((f"{name}[{i}]", rec))
            elif isinstance(rec, dict):
                text = _first_field(rec, TEXT_FIELDS)
                if text:
                    out.append((str(_first_field(rec, TITLE_FIELDS) or f"{name}[{i}]"), text))
        return out
    if suffix in (".html", ".htm"):
        raw = html_text(raw)
    return [(name, raw)]


def corpus_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        root = Path(raw)
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            for f in sorted(root.rglob("*")):
                rel = f.relative_to(root).parts
                hidden = any(part.startswith(".") or part in SKIP_DIRS for part in rel)
                if f.is_file() and not hidden and f.suffix.lower() in TEXT_SUFFIXES:
                    files.append(f)
        else:
            raise UsageError(f"corpus path not found: {raw}")
    return files


def load_corpus(paths: list[str], chunk_chars: int = 1500) -> list[Doc]:
    docs: list[Doc] = []
    for f in corpus_files(paths):
        try:
            records = _records(f)
        except OSError as exc:
            con.warn(f"skipped {f}: {exc.strerror or exc}")
            continue
        for title, text in records:
            chunks = chunk_text(text, chunk_chars)
            for i, chunk in enumerate(chunks):
                docs.append(Doc(f"{title} #{i + 1}" if len(chunks) > 1 else title, chunk))
    return docs


# ----------------------------------------------------------------- queries


def _query_strings(record: Any) -> list[str]:
    if isinstance(record, str):
        return [record]
    if isinstance(record, dict):
        for name in QUERY_FIELDS:
            value = record.get(name)
            if isinstance(value, str) and value.strip():
                return [value]
            if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                return list(value)
    return []


def _dedupe(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for text in texts:
        text = " ".join(str(text).split())[:MAX_QUERY_CHARS]
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def load_queries(path: str) -> list[str]:
    """Queries from a text file (one per line) or JSON/JSONL records with a query-like field."""
    p = Path(path)
    if not p.is_file():
        raise UsageError(f"queries file not found: {path}")
    text = p.read_text(encoding="utf-8-sig", errors="replace")
    suffix = p.suffix.lower()
    if suffix in (".json", ".jsonl", ".ndjson"):
        queries = [q for rec in _json_records(text, suffix) for q in _query_strings(rec)]
    else:
        queries = jsonl_lines(text)
    return _dedupe(queries)


def _embedding_inputs(record: Any) -> list[str]:
    """Inputs of an embeddings request found in a log record (llmshift.record logs,
    request/response pairs, OpenAI Batch input lines, flat request bodies)."""
    if not isinstance(record, dict):
        return []
    request = None
    for key in ("request", "body", "request_body", "requestBody", "req"):
        cand = record.get(key)
        if isinstance(cand, str):
            try:
                cand = json_loads(cand)
            except (ValueError, RecursionError):
                cand = None
        if isinstance(cand, dict) and "input" in cand:
            request = cand
            break
    if request is None and "input" in record and "model" in record:
        request = record
    if request is None or "messages" in request or "contents" in request:
        return []
    url = str(record.get("url") or record.get("endpoint") or record.get("path") or "")
    model = str(request.get("model") or "")
    if not ("embedding" in url or re.search(r"embed", model, re.I) or "encoding_format" in request or "dimensions" in request):
        return []
    value = request.get("input")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [x for x in value if isinstance(x, str)]  # token-id inputs cannot be re-embedded as text
    return []


def queries_from_logs(sources: list[str]) -> list[str]:
    from .logparse import LogStats, iter_files, iter_records

    try:
        files = list(iter_files(sources))
    except FileNotFoundError as exc:
        raise UsageError(str(exc)) from None
    found: list[str] = []
    for path in files:
        try:
            for record, _where in iter_records(path, LogStats()):
                found.extend(_embedding_inputs(record))
        except (OSError, EOFError, ValueError) as exc:
            con.warn(f"skipped {path}: {exc}")
    return _dedupe(found)


def pseudo_queries(docs: list[Doc], count: int, seed: int) -> list[str]:
    """First sentences of a sample of documents, when no real queries are available."""
    rng = random.Random(seed)  # nosec B311  # seeded on purpose: the same sample on every run; nothing secret
    picked = rng.sample(docs, min(count, len(docs)))
    out = []
    for doc in picked:
        first = _SENTENCE_RE.split(doc.text.strip(), maxsplit=1)[0].split("\n")[0]
        out.append(first[:200])
    return _dedupe(out)


# ------------------------------------------------------------------ vectors


def _unit(vector: Any, spec: str) -> list[float]:
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ProviderError(f"{spec} returned an empty embedding", code="invalid_response")
    try:
        values = [float(x) for x in vector]
    except (TypeError, ValueError):
        raise ProviderError(f"{spec} returned an embedding with non-numeric values", code="invalid_response") from None
    norm = math.sqrt(math.fsum(x * x for x in values))
    if not math.isfinite(norm):
        raise ProviderError(f"{spec} returned an embedding with non-finite values", code="invalid_response")
    return [x / norm for x in values] if norm else values


def embed_all(provider: Any, texts: list[str]) -> tuple[list[list[float]], float]:
    """Unit vectors for `texts` (in batches) and the seconds it took."""
    from .runner import interruptible_map

    vectors: list[list[float]] = []
    start = time.perf_counter()
    def embed(batch: list[str]) -> Any:
        try:
            return provider.embed(batch)
        except (AttributeError, TypeError, KeyError, IndexError) as exc:  # a reply of an unexpected shape
            raise ProviderError(f"{provider.spec} sent an embeddings reply llmshift cannot read ({exc.__class__.__name__}: {exc})", code="invalid_response") from None

    batches = [texts[i : i + BATCH] for i in range(0, len(texts), BATCH)]
    answers = interruptible_map(embed, batches, 1)  # one at a time, off the main thread: Ctrl+C works
    for batch, got in zip(batches, answers):
        if not isinstance(got, list) or len(got) != len(batch):
            count = len(got) if isinstance(got, list) else "no"
            raise ProviderError(f"{provider.spec} returned {count} embeddings for {len(batch)} texts", code="invalid_response")
        vectors.extend(_unit(v, provider.spec) for v in got)
    elapsed = time.perf_counter() - start
    sizes = {len(v) for v in vectors}
    if len(sizes) > 1:
        raise ProviderError(f"{provider.spec} returned embeddings of different sizes: {sorted(sizes)}", code="invalid_response")
    return vectors, elapsed


def top_k(queries: list[list[float]], docs: list[list[float]], k: int) -> list[list[int]]:
    """Indices of the k most similar documents per query (ties: lower index first)."""
    try:
        import numpy as np  # optional speed-up for large samples
    except ImportError:
        np = None
    if np is not None and queries and docs:
        sims = np.asarray(queries, dtype=float) @ np.asarray(docs, dtype=float).T
        out = []
        for row in sims:
            idx = np.arange(len(row))
            order = np.lexsort((idx, -row))[:k]
            out.append([int(i) for i in order])
        return out
    mul = operator.mul
    out = []
    for qv in queries:
        scores = [sum(map(mul, qv, dv)) for dv in docs]
        out.append(heapq.nsmallest(k, range(len(scores)), key=lambda i: (-scores[i], i)))
    return out


def rank_biased_overlap(a: list[int], b: list[int], p: float = RBO_P) -> float:
    """Top-weighted agreement of two rankings, 1.0 when identical (normalised to their depth)."""
    depth = min(len(a), len(b))
    if depth == 0:
        return 1.0 if not a and not b else 0.0
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    overlap = 0
    total = 0.0
    for d in range(1, depth + 1):
        x, y = a[d - 1], b[d - 1]
        overlap += 1 if x == y else (x in seen_b) + (y in seen_a)
        seen_a.add(x)
        seen_b.add(y)
        total += p ** (d - 1) * overlap / d
    return (1 - p) * total / (1 - p**depth)


# --------------------------------------------------------------------- run


def run_embedding_migration(
    corpus: list[str],
    old: Any,
    new: Any,
    *,
    queries: list[str] | None = None,
    query_source: str = "list",
    k: int = 10,
    max_docs: int = 1000,
    max_queries: int = 200,
    chunk_chars: int = 1500,
    min_overlap: float = 0.6,
    seed: int = 7,
    config: dict | None = None,
    on_start: Any = None,
) -> EmbeddingReport:
    """`on_start()` is called once everything is checked, before the first (paid) request."""
    from . import pricing
    from .providers import resolve

    if k < 1 or max_docs < 1 or max_queries < 1:
        raise UsageError("--k, --max-docs and --max-queries must be at least 1")
    if not 50 <= chunk_chars <= 100_000:
        raise UsageError("--chunk-chars must be between 50 and 100000")
    if not 0.0 <= min_overlap <= 1.0:
        raise UsageError("--min-overlap must be between 0 and 1")
    config = config or {}
    old_p = resolve(old, config) if isinstance(old, str) else old
    new_p = resolve(new, config) if isinstance(new, str) else new
    docs = load_corpus(corpus, chunk_chars)
    if not docs:
        raise UsageError(f"no documents found in {', '.join(corpus)} (looked for {', '.join(sorted(TEXT_SUFFIXES))})")
    rng = random.Random(seed)  # nosec B311  # seeded on purpose: the same sample on every run; nothing secret
    sample = docs if len(docs) <= max_docs else [docs[i] for i in sorted(rng.sample(range(len(docs)), max_docs))]
    notes: list[str] = []
    if queries is None:
        queries, query_source = pseudo_queries(sample, min(100, max_queries), seed), "pseudo"
        notes.append("No queries given: the first sentences of your documents were used as queries. Pass real ones with --queries or --queries-from.")
    queries = _dedupe(queries)
    if not queries:
        raise UsageError("no queries to compare (the queries file or logs contained none)")
    if len(queries) > max_queries:
        queries = [queries[i] for i in sorted(random.Random(seed).sample(range(len(queries)), max_queries))]  # nosec B311  # seeded on purpose: the same sample on every run; nothing secret
    depth = min(k, len(sample))
    if depth < k:
        notes.append(f"Only {len(sample)} document chunks, so the top {depth} were compared instead of the top {k}.")

    if on_start is not None:
        on_start()
    texts = [d.text for d in sample]
    old_docs, _ = embed_all(old_p, texts)
    new_docs, new_seconds = embed_all(new_p, texts)
    old_queries, _ = embed_all(old_p, queries)
    new_queries, _ = embed_all(new_p, queries)
    old_top = top_k(old_queries, old_docs, depth)
    new_top = top_k(new_queries, new_docs, depth)

    rows = []
    for query, a, b in zip(queries, old_top, new_top):
        overlap = len(set(a) & set(b)) / depth
        rows.append({"query": query, "overlap": overlap, "top1_kept": a[0] in b, "rbo": rank_biased_overlap(a, b), "old": a, "new": b})
    n = len(rows)

    def titles(ids: list[int]) -> list[str]:
        return [sample[i].title for i in ids[:3]]

    worst = sorted(rows, key=lambda r: (r["overlap"], r["rbo"], r["query"]))[:5]
    corpus_tokens = sum(pricing.estimate_tokens(d.text) for d in docs)
    price = pricing.price_for(new_p.spec, pricing.overrides_from_config(config))
    cost = price.cost(corpus_tokens, 0) if price is not None else None
    if price is None:
        notes.append(f"No price known for {new_p.spec}: set one under pricing: in llmshift.yaml to estimate the re-index cost.")
    rate = len(texts) / new_seconds if new_seconds > 0 else None
    old_dims, new_dims = len(old_docs[0]), len(new_docs[0])
    if old_dims != new_dims:
        notes.append(f"Vector size changes from {old_dims} to {new_dims}: the vector index needs a new schema, not just new vectors.")
    return EmbeddingReport(
        old=old_p.spec,
        new=new_p.spec,
        k=depth,
        corpus_chunks=len(docs),
        sampled_chunks=len(sample),
        corpus_tokens=corpus_tokens,
        queries=n,
        query_source=query_source,
        mean_overlap=sum(r["overlap"] for r in rows) / n,
        top1_kept=sum(1 for r in rows if r["top1_kept"]) / n,
        mean_rbo=sum(r["rbo"] for r in rows) / n,
        old_dims=old_dims,
        new_dims=new_dims,
        min_overlap=min_overlap,
        reindex_cost=cost,
        price_source=price.source if price is not None else None,
        reindex_seconds=len(docs) / rate if rate else None,
        worst=[
            {"query": r["query"], "overlap": round(r["overlap"], 4), "rbo": round(r["rbo"], 4), "old_top": titles(r["old"]), "new_top": titles(r["new"])}
            for r in worst
            if r["overlap"] < 1.0
        ],
        notes=notes,
    )


# ------------------------------------------------------------------ output


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 90:
        return f"~{max(1, round(seconds))} s"
    if seconds < 5400:
        return f"~{round(seconds / 60)} min"
    return f"~{seconds / 3600:.1f} h"


def report_text(r: EmbeddingReport) -> str:
    from .pricing import money

    sym = con.sym("pass") if r.passed else con.sym("fail")
    lines = [
        con.style("LLMShift embeddings", "bold") + f"  {r.old} -> {r.new}",
        f"  corpus: {r.corpus_chunks:,} chunks (~{r.corpus_tokens:,} tokens); compared {r.sampled_chunks:,} chunks and {r.queries} "
        + {"pseudo": "pseudo-queries", "logs": "queries from your logs", "file": "queries"}.get(r.query_source, "queries"),
        f"  results kept: {r.mean_overlap:.0%} of the top {r.k} on average; the old top result stays in the new top {r.k} for {r.top1_kept:.0%} of queries",
        f"  ranking agreement (rank-biased overlap): {r.mean_rbo:.2f}",
        f"  vector size: {r.old_dims} -> {r.new_dims}" + ("  (the index schema must change)" if r.old_dims != r.new_dims else ""),
        f"  re-index the full corpus: {money(r.reindex_cost)} with {r.new}, {_duration(r.reindex_seconds)}",
        "  " + con.style(f"{sym} {'meets' if r.passed else 'below'} the {r.min_overlap:.0%} minimum overlap", "green" if r.passed else "red"),
    ]
    if r.worst:
        lines.append("")
        lines.append(con.style("Queries whose results change most", "bold"))
        for w in r.worst:
            lines.append(f"  {con.truncate(w['query'], 70)!r}  overlap {w['overlap']:.0%}")
            lines.append(con.style(f"     old: {', '.join(w['old_top'])}", "grey"))
            lines.append(con.style(f"     new: {', '.join(w['new_top'])}", "grey"))
    for note in r.notes:
        lines.append(con.style(f"  note: {note}", "grey"))
    return "\n".join(lines)


def report_markdown(r: EmbeddingReport) -> str:
    from .pricing import money
    from .report import _md

    verdict = "meets" if r.passed else "is below"
    lines = [
        f"## LLMShift embeddings: `{r.old}` -> `{r.new}`",
        "",
        f"Search results kept: **{r.mean_overlap:.0%}** of the top {r.k} on average ({verdict} the {r.min_overlap:.0%} minimum).",
        "",
        "| | |",
        "|---|---|",
        f"| Old top result still in the new top {r.k} | {r.top1_kept:.0%} of queries |",
        f"| Rank-biased overlap | {r.mean_rbo:.2f} |",
        f"| Vector size | {r.old_dims} -> {r.new_dims} |",
        f"| Corpus | {r.corpus_chunks:,} chunks, ~{r.corpus_tokens:,} tokens |",
        f"| Re-index cost | {money(r.reindex_cost)} ({_duration(r.reindex_seconds)}) |",
        f"| Queries compared | {r.queries} ({r.query_source}) |",
    ]
    if r.worst:
        lines += ["", "**Queries whose results change most**", ""]
        for w in r.worst:
            query = _md(str(w["query"])[:100])  # queries come from your users' logs: no live links or images
            lines.append(f"- {query} (overlap {w['overlap']:.0%})")
    for note in r.notes:
        lines += ["", f"> {_md(note)}"]
    lines += ["", credit()]
    return show_bidi("\n".join(lines) + "\n")


def cmd_embeddings(args: argparse.Namespace, cfg: dict) -> int:
    from . import cli

    queries = None
    source = "list"
    if args.queries:
        queries, source = load_queries(args.queries), "file"
        if not queries:
            raise UsageError(f"{args.queries} contains no queries")
    elif args.queries_from:
        queries, source = queries_from_logs(args.queries_from), "logs"
        if not queries:
            raise UsageError(f"no embedding requests with text inputs found in {', '.join(args.queries_from)}")
    report = run_embedding_migration(
        args.corpus,
        args.old,
        args.new,
        queries=queries,
        query_source=source,
        k=args.k,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        chunk_chars=args.chunk_chars,
        min_overlap=args.min_overlap,
        seed=args.seed,
        config=cfg,
        on_start=lambda: con.out(con.style(f"Embedding a sample of your corpus with {args.old} and {args.new}...", "grey"), err=True),
    )
    data = report_json(report.to_dict())
    if args.json:
        cli._write(args.json, data + "\n")
    if args.format == "json":
        con.out(data)
    elif args.format == "markdown":
        con.out(report_markdown(report))
    else:
        con.out(report_text(report))
    cli._step_summary(report_markdown(report))
    return EXIT_OK if report.passed else EXIT_FAIL


def register(sub: Any) -> None:
    from . import cli

    p = sub.add_parser(
        "embeddings",
        help="see how search results change before switching embedding models",
        description="Embed a sample of your documents and queries with both models, compare the top-k results per query, "
        "and estimate the cost and time of re-embedding the whole corpus.",
    )
    p.add_argument("--from", dest="old", required=True, metavar="MODEL", type=cli.model_arg, help="the embedding model you use today")
    p.add_argument("--to", dest="new", required=True, metavar="MODEL", type=cli.model_arg, help="the embedding model you are moving to")
    p.add_argument("--corpus", nargs="+", action="extend", required=True, metavar="PATH", help="documents: files or folders (.txt .md .rst .html .json .jsonl)")
    q = p.add_mutually_exclusive_group()
    q.add_argument("--queries", metavar="FILE", help="real search queries: one per line, or JSON/JSONL records")
    q.add_argument("--queries-from", nargs="+", action="extend", metavar="LOGS", help="take queries from embedding requests in usage logs")
    p.add_argument("--k", type=cli.positive_int, default=10, help="how many top results to compare per query (default 10)")
    p.add_argument("--max-docs", type=int, default=1000, help="document chunks to embed for the comparison (default 1000)")
    p.add_argument("--max-queries", type=int, default=200, help="queries to compare (default 200)")
    p.add_argument("--chunk-chars", type=int, default=1500, help="split documents into chunks of this many characters (default 1500)")
    p.add_argument("--min-overlap", type=float, default=0.6, help="exit 1 when fewer results are kept on average (default 0.6)")
    p.add_argument("--seed", type=int, default=7, help="sampling seed")
    p.add_argument("--format", choices=["table", "json", "markdown"], default="table")
    p.add_argument("--json", metavar="FILE", help="also write the report as JSON")
    p.set_defaults(func=cmd_embeddings)
