"""Holm-Bonferroni multiple-testing adjustment.

A pure function of a list of raw p-values. Used by ledger.py (the executor
side, growing the trial ledger) and reimplemented independently in
gate/oracle.py (a second, from-scratch pass over the same ledger rows, see
oracle.py's module docstring for why) so a bug here cannot be confirmed
correct by comparing against itself.
"""
from __future__ import annotations


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Returns adjusted p-values in the same order as the input.

    Standard step-down Holm procedure: sort ascending, multiply the k-th
    smallest (1-indexed) by (n - k + 1), then enforce monotonicity by taking
    a running maximum from the smallest p-value upward, and clip at 1.0.
    """
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted_sorted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        raw_adjusted = pvalues[idx] * (n - rank + 1)
        running_max = max(running_max, raw_adjusted)
        adjusted_sorted[rank - 1] = min(1.0, running_max)
    adjusted = [0.0] * n
    for rank, idx in enumerate(order, start=1):
        adjusted[idx] = adjusted_sorted[rank - 1]
    return adjusted
