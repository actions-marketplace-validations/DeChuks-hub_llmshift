"""LLMShift: switch AI models without breaking your product.

Command line:  llmshift scan | snapshot | compare | fix | watch | import | demo

Python API:

    import llmshift
    findings = llmshift.scan(".")                      # model calls + retirement dates
    cases = llmshift.snapshot(["logs/app.jsonl"])        # usage logs -> test cases
    result = llmshift.compare(cases, "openai:gpt-5.6-sol")  # what changes on the new model
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.7"

# These submodules share names with the public functions below. Importing them first
# means the functions defined afterwards win; otherwise the first `import
# llmshift.compare` would replace `llmshift.compare` with the module.
from . import compare as _compare_module  # noqa: E402
from . import snapshot as _snapshot_module  # noqa: E402


def scan(paths: Any = ".", **kwargs: Any) -> Any:
    """Find model IDs in code/config and attach lifecycle info. Returns a ScanResult."""
    from .scanner import scan_paths

    return scan_paths(list(paths) if isinstance(paths, (list, tuple)) else [paths], **kwargs)


def snapshot(sources: Any, **kwargs: Any) -> Any:
    """Turn usage logs into test cases (list of TestCase)."""
    return _snapshot_module.build_snapshot(list(sources) if isinstance(sources, (list, tuple)) else [sources], **kwargs).cases


def compare(cases: Any, candidate: Any, **kwargs: Any) -> Any:
    """Run test cases on a candidate model ("provider:model" or a Provider). Returns a CompareResult."""
    return _compare_module.run_compare(cases, candidate, **kwargs)


def load_tests(path: Any) -> Any:
    from .testset import load_tests as _load

    return _load(path)


__all__ = ["__version__", "scan", "snapshot", "compare", "load_tests"]
