"""Rebuild src/llmshift/data/pricing.json from LiteLLM's model price table (MIT licensed).

    python scripts/update_pricing.py                 # download the current table
    python scripts/update_pricing.py litellm.json    # or convert a local copy

Keeps chat, completion, responses and embedding models that have token prices and
stores them as USD per 1M tokens: {"model": [input, output]}, followed for some models by
cache prices and long-context tiers (pricing.convert_litellm). The conversion and its
checks are llmshift's own (pricing.convert_litellm and pricing.prices_problems), so a
table `llmshift update --prices` would refuse is never written here either.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llmshift.pricing import convert_litellm, prices_problems  # noqa: E402

SOURCE = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
OUT = Path(__file__).resolve().parents[1] / "src" / "llmshift" / "data" / "pricing.json"


def main() -> None:
    if len(sys.argv) > 1:
        table = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    else:
        if not SOURCE.startswith("https://"):
            raise SystemExit(f"SOURCE must be an https:// address: {SOURCE}")
        with urllib.request.urlopen(SOURCE, timeout=60) as resp:  # nosec B310  # https only, checked above
            table = json.loads(resp.read())
    if not isinstance(table, dict):
        raise SystemExit("the price table is not a JSON object; the source format may have changed")
    prices = convert_litellm(table)
    if len(prices) < 500:
        raise SystemExit(f"only {len(prices)} priced models found; the source format may have changed")
    data = {
        "about": "Model prices for llmshift pick and cost estimates, in USD per 1M tokens [input, output], "
        "followed where the list has them by the cache read, cache write and 1-hour cache write prices and the "
        "long-context tiers (see pricing.convert_litellm). "
        "Converted from LiteLLM's model_prices_and_context_window.json (MIT License, Copyright (c) 2023 Berri AI; full text in THIRD_PARTY_NOTICES).",
        "source": SOURCE,
        "fetched": _dt.date.today().isoformat(),
        "unit": "USD per 1M tokens",
        "prices": prices,
    }
    problems = prices_problems(data, min_prices=500)
    if problems:
        raise SystemExit("not written: " + "; ".join(problems))
    OUT.write_bytes((json.dumps(data, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8"))
    print(f"wrote {len(prices)} prices to {OUT}")


if __name__ == "__main__":
    main()
