"""Hypothesis-test statistics computed from tool output. Kept separate from
tools.py (data retrieval) on purpose: these functions never touch the
database, they only consume the arrays tools.py already pulled out.
"""
from __future__ import annotations

from scipy import stats as _scipy_stats


def two_sample_test(sample_a: list[float], sample_b: list[float]) -> tuple[float, float]:
    """Welch's two-sample t-test (unequal variance). Returns (statistic, p_value)."""
    result = _scipy_stats.ttest_ind(sample_a, sample_b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def pearson_correlation(series_a: list[float], series_b: list[float]) -> tuple[float, float]:
    """Pearson correlation coefficient and its two-sided p-value."""
    result = _scipy_stats.pearsonr(series_a, series_b)
    return float(result.statistic), float(result.pvalue)
