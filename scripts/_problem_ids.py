"""Natural-integer parsing for BFCL problem-id filenames.

BFCL per-problem files are named `<category>_<a>-<b>-<c>.json` (live
cats) or `<category>_<n>.json` (curated cats). Python's default
lexicographic sort orders `live_simple_10` before `live_simple_2`,
which silently broke a previous round of cross-model "first-100 slice"
comparisons (Phase H / Round 3). All slicing-by-position must go
through this module instead.

Use `natural_problem_key` as the `sorted(..., key=...)` argument.
For cross-model slicing, prefer intersection-of-IDs in
`scripts/compare_matched_slice.py` — slice-by-position is itself a
sharp edge.
"""

from __future__ import annotations

import re
from pathlib import Path


_NUM_RE = re.compile(r"\d+")


def natural_problem_key(stem: str) -> tuple[int, ...]:
    """Return a sort key derived from the integer fields in a problem stem.

    Examples:
        >>> natural_problem_key("live_simple_2-2-0")
        (2, 2, 0)
        >>> natural_problem_key("live_simple_10-3-6")
        (10, 3, 6)
        >>> natural_problem_key("simple_python_42")
        (42,)
        >>> sorted(["a_10", "a_2", "a_100"], key=natural_problem_key)
        ['a_2', 'a_10', 'a_100']
    """
    nums = _NUM_RE.findall(stem)
    return tuple(int(n) for n in nums) if nums else (0,)


def sorted_problem_files(files: list[Path]) -> list[Path]:
    """Stable natural-integer sort over BFCL per-problem files."""
    return sorted(files, key=lambda p: natural_problem_key(p.stem))
