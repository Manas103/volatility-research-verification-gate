"""The 48-hypothesis bank screened against the volatility store
(`vol_gate/volstore.py`). Every hypothesis is a (shape, params) pair fully
determined by this module before any p-value is computed, "pre-registered"
in the sense the no-brainer and breakdown docs mean it: the SQL test a
hypothesis maps to is fixed by its declared parameters, never chosen or
adjusted after seeing the data.

This bank is the bulk-scale, deterministic stand-in for "a language model
proposed this hypothesis", the same precedent `DeterministicRouterClient`
already sets in this repo (used for the 200/60-case and ~184-hypothesis
battery runs, with the real `claude` CLI exercised on a smaller live
sample instead). `scripts/run_volatility_hypothesis_screen.py` runs this
same 48-hypothesis bank for its headline numbers, and separately runs a
real `claude -p` sample proposing a handful of hypotheses in natural
language, parsed into these same five shapes, as live evidence the AI
proposal step genuinely works end to end; see that script and the README
for the honest split between the two.

Five shapes, 48 hypotheses total:
  vol_pair_compare            10  (every unordered pair of the 5 tickers)
  vol_pair_correlation        10  (every unordered pair of the 5 tickers)
  vol_activity_correlation     5  (one per ticker)
  vol_day_trend_correlation    5  (one per ticker)
  vol_regime_compare          18  (5 tickers + pooled "ALL") x 3 split rules
"""
from __future__ import annotations

import itertools

TICKERS = ["SYNA", "SYNB", "SYNC", "SYND", "SYNE"]
SPLIT_RULES = ("msg_count_median", "day_parity", "bin_position")


def build_hypothesis_bank() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for a, b in itertools.combinations(TICKERS, 2):
        out.append(("vol_pair_compare", {"ticker_a": a, "ticker_b": b}))
    for a, b in itertools.combinations(TICKERS, 2):
        out.append(("vol_pair_correlation", {"ticker_a": a, "ticker_b": b}))
    for t in TICKERS:
        out.append(("vol_activity_correlation", {"ticker": t}))
    for t in TICKERS:
        out.append(("vol_day_trend_correlation", {"ticker": t}))
    for t in TICKERS + ["ALL"]:
        for split_by in SPLIT_RULES:
            out.append(("vol_regime_compare", {"ticker": t, "split_by": split_by}))
    assert len(out) == 48, f"expected 48 pre-registered hypotheses, built {len(out)}"
    return out
