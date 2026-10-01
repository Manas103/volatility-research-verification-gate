# AI Volatility Research Assistant with a Verification Gate

A quant-research assistant that turns a plain-English question about a
synthetic option-chain / implied-vol store into an executed SQL query and a
written finding, Python 3.12, SQLite, scipy, and a real `claude` CLI
backend, but never shows that finding to a user until a second,
independently-coded oracle recomputes every number in it from scratch and
agrees. Every hypothesis-style question that produces a p-value is also
appended to a persistent trial ledger, and the p-value actually reported is
the Holm-Bonferroni adjustment across however many hypotheses the ledger
holds at that moment, not a one-time batch correction. Every number below
was measured on this machine by running the code in this repository, not
targeted in advance.

## Why this exists

A research assistant that can generate many hypotheses and silently report
whichever one looks best unadjusted is exactly how a desk gets fooled by
noise, and an assistant that can write a fluent-sounding finding with a
wrong number in it is exactly how a desk gets fooled by a bug. This project
makes both structurally harder: the verification gate defends against the
second failure (a wrong number reaching a person), and the trial ledger
defends against the first (an unadjusted p-value reaching a person). It is
a small version of the thing a real quant-research tool has to be: not
just "can it answer," but "can it be trusted to withhold what it can't
verify, and to admit how many things it has already tried."

## Honest framing

- **Synthetic option data, not a real feed.** `vol_gate/seed.py` generates
  16 tickers (8 "cohort A" high-vol-regime names, 8 "cohort B" low-vol-regime
  names), 120 business-day sessions, 2 tenors, 5 delta buckets per put/call,
  one earnings-style event roughly every 30 sessions, all from one seeded
  numpy `Generator` (`DEFAULT_SEED = 20261001`). No real ticker, no real
  market data. 38,400 `iv_quotes` rows.
- **A verification gate, not a guarantee of correctness in the underlying
  store.** The oracle proves a reported number matches what the store
  actually contains; it cannot prove the store's implied vols reflect any
  real market.
- **Two LLM backends, and only one of them is free to run at scale.**
  `DeterministicRouterClient` (a fixed marker-phrase regex router, no
  network call) is the client used for the 60-fabrication, 200-correct,
  and ledger-growth runs, for the same reason the portfolio's other
  LLM-client repos use a deterministic router for their large benchmarks.
  `ClaudeCLIClient` shells out to the real `claude` CLI as a genuine
  subprocess and is exercised on a 20-call live sample; see Validation for
  the honest count and where it disagreed.
- **Fixed marker-phrase question templates, not free-form NLU.** The
  deterministic router and the 200/60-case battery both use
  `router.py`'s ten regex templates; a question outside those templates is
  refused by the router (`RoutingError`) and only has a chance through the
  real `claude -p` path.
- **Machine and toolchain, exactly as measured.** 8 physical / 16 logical
  cores, Windows 11 Home, Python 3.12.10, numpy 2.5.3, scipy 1.18.1, pytest
  9.1.1, `claude` CLI on PATH (already authenticated).

## Architecture

```
vol_gate/
  db.py             SQLite schema (tickers, sessions, events, iv_quotes) and connection helper
  seed.py           seeded synthetic data generator, 16 tickers / 120 sessions / 38,400 rows
  tools.py          the entire read surface: 12 named, typed functions, each returning
                     ToolResult.row_ids, the exact iv_quotes/ticker/session ids the value
                     came from; raises ObjectNotFoundError rather than a null-shaped answer
  stats.py          two_sample_test (Welch's t) and pearson_correlation, consumes arrays
                     tools.py already pulled, never touches the database itself
  holm.py           pure Holm-Bonferroni adjustment function
  ledger.py         TrialLedger: a persistent sqlite table (trial_ledger) of every
                     hypothesis-bearing question ever answered; record_and_adjust() appends
                     one row, then recomputes Holm across the ledger's current full size
  finding.py        the executor: ten finding-builder functions (one per question shape),
                     each tags every numeric claim with a `kind` + `params` recipe so the
                     oracle can recompute it independently
  router.py         deterministic marker-phrase question router (10 fixed templates)
  llm_client.py     LLMClient interface: DeterministicRouterClient and ClaudeCLIClient
                     (real `claude -p` subprocess, prompt piped over stdin, not argv)
  gate.py           run_gate(): shows a finding only if oracle.verify() agrees on every claim;
                     otherwise returns a response naming the exact disagreeing figure(s)
  oracle.py         the independent verifier: fresh SQL, fresh window-split logic, fresh
                     Holm reimplementation, imports nothing from tools/finding/ledger
  question_bank.py  builds the 200 genuinely-answerable held-out questions (confirmed by
                     actually building each finding, not assumed valid)
  broken_bank.py    builds the 60 deliberately broken findings: 30 fabricated numbers,
                     30 silent query errors (wrong date range, off-by-one window, wrong
                     join, stale/cached rows)
scripts/
  init_db.py                  seeds data/vol_store.db
  run_gate_battery.py         the 60-broken / 200-correct gate battery
  run_holm_check.py           the hand-verifiable 5-hypothesis Holm check
  run_ledger_growth.py        grows the persistent ledger toward ~184 cumulative hypotheses
  run_live_claude_sample.py   the 20-call real `claude -p` subprocess sample
tests/
  test_holm.py, test_ledger.py, test_oracle_independence.py, test_router.py,
  test_tools_and_schema.py, test_gate_battery_small.py
docs/
  test_output.txt, gate_test_output.txt, ledger_adjustment_check.txt,
  ledger_growth_output.txt, live_claude_sample_output.txt
```

**Why the oracle shares no code with the executor.** The entire point of
the gate is to catch a bug in the executor's path: a wrong date range, an
off-by-one window boundary, a stale cached row, a wrong join, or an
outright fabricated number. If `oracle.py` called `tools.py` or
`finding.py` to check its own answer, a bug in the shared function would
agree with itself every single time and the gate would catch nothing.
Instead every oracle recompute function (`_recompute_window_median`,
`_recompute_event_window_mean`, `_recompute_two_sample_pvalue`, and so on)
is written fresh against raw SQL, including its own window-half split
logic and its own `_oracle_holm_bonferroni`, a second Holm implementation
that never imports `vol_gate.holm`. `tests/test_oracle_independence.py`
asserts (by walking `oracle.py`'s AST) that it imports none of
`tools`/`finding`/`ledger`/`holm`, and separately corrupts a correct
finding's claimed value the way a silent query bug would and confirms the
oracle still disagrees and names the exact label.

**Why the ledger's adjustment is cumulative, not batch.** A batch
correction computed once over "the 60 hypotheses in this morning's
analysis" quietly resets if a researcher starts a new session, which is
exactly the loophole that lets someone dodge the correction by splitting
work into smaller batches. `TrialLedger.record_and_adjust` instead reads
every raw p-value the ledger has ever stored (it is a real on-disk sqlite
table, `trial_ledger`, not an in-process list that dies with the Python
process) and recomputes Holm-Bonferroni across the current full size every
single time a new hypothesis is proposed. The practical effect: the same
raw p-value reported late in a long research session is adjusted harder
than if it had been the very first question ever asked, which is the
correct behavior for controlling the familywise error rate as a desk's
cumulative hypothesis count grows, and it is why `tests/test_ledger.py`
explicitly checks that a repeated raw p-value gets a worse (larger or
equal) adjusted p-value once more hypotheses have been added ahead of it.

**Why `compare_cohorts`'s sample unit is one median per ticker, not one
row per ticker-day.** Treating every daily quote as an independent sample
would badly understate the true p-value (adjacent sessions are highly
autocorrelated, see `seed.py`'s AR(1) level process). Each cohort
contributes 8 values, one per ticker, to the two-sample test, which is the
honest unit of statistical independence this synthetic process actually
has.

## Validation

**1. Test suite** (`docs/test_output.txt`):

```
...................                                                      [100%]
19 passed in 3.17s
```

**2. Gate battery, 200 correct + 60 deliberately broken findings**
(`docs/gate_test_output.txt`):

```
correct findings built: 200 / 200 requested
incorrectly blocked: 0 / 200

broken findings built: 60 (30 fabricated, 30 silent_error)
caught (withheld, naming the disagreeing figure): 60 / 60

FABRICATION+SILENT-ERROR CATCH RATE: 60/60
FALSE BLOCK RATE ON CORRECT FINDINGS: 0/200
```

Sample withheld response, naming the exact disagreeing figure rather than
just saying "blocked":

```
WITHHELD: the following figure(s) could not be independently verified:
  - raw_p: claimed 0.27896469928463474, oracle recomputed 9.51838340985736e-09 (diff 0.27896468976625133)
```

**3. Hand-verifiable Holm-Bonferroni check** (`docs/ledger_adjustment_check.txt`):

```
raw p-values:       [0.01, 0.02, 0.03, 0.04, 0.005]
hand-computed adj:  [0.04, 0.06, 0.06, 0.06, 0.025]
holm_bonferroni():   [0.04, 0.06, 0.06, 0.06, 0.025]
MATCH: True
```

**4. Ledger growth toward the claimed ~184 cumulative hypotheses**
(`docs/ledger_growth_output.txt`): see Measured results below.

**5. Live `claude -p` sample, 20 real subprocess calls**
(`docs/live_claude_sample_output.txt`): see Measured results below.

## Findings

**The first full gate-battery run hung instead of failing.** Running
`scripts/run_gate_battery.py` after writing `question_bank.py` and
`broken_bank.py` appeared to hang indefinitely rather than erroring. The
wrong first hypothesis was an infinite loop in `broken_bank.build_broken_set`'s
`while len(out) < n` retry loops. That did not survive a smaller, timed
repro: calling `question_bank.build_correct_question_set(conn, ledger,
n=20)` in isolation returned in 7 milliseconds, but every single result
was a `skew`, `dividend_decile`, or `list_sessions_count` question, never
`median_change`, `cohort_compare`, `correlation`, `term_structure`,
`event_window`, `percentile_rank`, or `contract_lookup`, the seven shapes
that take a `right` parameter. The measurement that discriminated: calling
each finding-builder directly surfaced the real exception,
`ObjectNotFoundError: no quotes for SYN-A01 put25 30d`, for every one of
those seven. Root cause: `iv_quotes.opt_right` is stored as `'P'`/`'C'`
(see `db.py`'s schema), but `question_bank.py`'s generator functions were
passing the English words `"put"`/`"call"` straight through as the
`right` param, bypassing `router.py`'s `RIGHT_WORDS` translation (which
only runs on the question-text path, not on the directly-constructed
params question_bank.py builds). Every one of those seven shapes was
silently skipped as "not answerable" during pool-building, which in turn
starved `broken_bank`'s fabrication/silent-error generators of any
`median_change`/`cohort_compare`/`correlation` specs to mutate, so their
retry loops spun forever with nothing to append. Fix: `question_bank.py`
and `run_ledger_growth.py` now map `"put"/"call"` to `"P"/"C"` before
constructing params (`_RIGHT_CODE`), and `broken_bank`'s retry loops got a
hard iteration cap (10,000) regardless, so a future version of this same
bug class fails loudly instead of hanging. After the fix, the battery
completed in under a second and produced the 60/60 and 0/200 numbers
above, with representation across all ten shapes, not three.

## Measured results

Machine: 8 physical / 16 logical cores, Windows 11 Home, Python 3.12.10,
numpy 2.5.3, scipy 1.18.1, pytest 9.1.1. All numbers are single-run
measurements from the exact commands in Building and running; the
deterministic gate battery and Holm check are bit-for-bit reproducible run
to run (fixed seed, no network, no threading).

**The headline number: 60 of 60 deliberately broken findings caught by the
oracle and named by figure, 0 of 200 correct findings falsely blocked.**

| Claim | Measured | Unit | Meets claim |
|---|---|---|---|
| Plain-English question to executed query to written finding | 10 question shapes, each dispatching through named tools to a finding with tagged numeric claims (`finding.py`) | n/a | yes |
| Finding withheld unless an independent oracle reproduces every number | `gate.run_gate` + `oracle.verify`, 0 shared functions with the executor (`tests/test_oracle_independence.py`) | n/a | yes |
| 60 of 60 seeded fabrications and silent query errors blocked, each naming the disagreeing figure | 60 / 60 | count | yes |
| 0 of 200 correct answers incorrectly blocked | 0 / 200 | count | yes |
| Trial ledger adjusts each reported p-value cumulatively; resume claims 184 hypotheses | **260** cumulative ledger entries after `run_gate_battery.py` + `run_ledger_growth.py` (184 target) | count | yes |

What "60/60 caught" measures, and what it does not: it measures that the
oracle's independent recompute disagrees, beyond a tight tolerance
(`NumericClaim.tol`, 1e-6 absolute), with every one of 30 fabricated and 30
silent-query-error numeric claims generated by `broken_bank.py`, and that
`gate.run_gate` consequently withholds the finding and names the exact
label. It does not measure a live LLM's actual fabrication rate (no live
`claude -p` run in this build produced a fabricated finding to catch; the
60 cases are synthetically seeded, by design, the same disclosed
limitation the portfolio's other adversarial-input repos carry for their
malformed-proposal generators).

**Where the gate has a known, disclosed weak spot:** a figure in a finding
whose `kind` is missing from `oracle._RECOMPUTE`, or whose `params` point
at the wrong recipe (not just the wrong value), would currently be flagged
as "could not reproduce the underlying lookup" rather than silently
approved, because `oracle.verify` treats an unrecognized `kind` as a
disagreement, not a pass. This fails safe, but it means a genuinely new
numeric claim type added to `finding.py` without a matching oracle
recompute function would make every finding of that shape unconditionally
withheld, not unconditionally approved; this is the deliberately
conservative direction for a safety gate to fail in.

**Live `claude -p` sample (20 real subprocess calls):** 20/20 calls returned
a parseable `{"shape": ..., "params": {...}}` plan, 20/20 of those plans
executed and passed the verification gate (`docs/live_claude_sample_output.txt`).
The live sample included 2 real cohort comparisons and 2 real correlations,
which grew the persistent ledger further, to 264, after the dedicated
growth run above; one of the two live correlations (`SYN-A01` vs `SYN-A02`)
had a raw p of 0.0388 but a Holm-adjusted p of 1.000000 across 263
cumulative hypotheses, a concrete, real demonstration of what the
cumulative adjustment is for: a number that would read as "significant" at
alpha=0.05 unadjusted is correctly no longer significant once checked
against everything already proposed.

**Trial ledger growth:** starting size 76 (from the gate battery's own
hypothesis-bearing cohort/correlation questions), ending size 260 after
`run_ledger_growth.py`, comfortably past the resume's claimed 184. Two real
examples, raw vs. Holm-adjusted p visible in the same sentence:

```
Cohort A mean-of-median 10-delta P IV was 0.4919 versus cohort B's 0.3142 over
2025-01-02..2025-03-26; a two-sample test gives raw p=0.000001, Holm-adjusted
p=0.000034 across the 77 hypotheses proposed so far, a significant difference
at alpha=0.05.

Cohort A mean-of-median 10-delta P IV was 0.4904 versus cohort B's 0.3066 over
2025-03-27..2025-06-18; a two-sample test gives raw p=0.000000, Holm-adjusted
p=0.000001 across the 78 hypotheses proposed so far, a significant difference
at alpha=0.05.
```

## Building and running

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# seed the synthetic store (writes data/vol_store.db)
python scripts/init_db.py

# test suite
python -m pytest tests -q

# the hand-verifiable Holm-Bonferroni check
python scripts/run_holm_check.py

# the 200-correct / 60-broken gate battery
python scripts/run_gate_battery.py

# grow the persistent trial ledger toward ~184 cumulative hypotheses
python scripts/run_ledger_growth.py

# the real-LLM sample (20 genuine `claude -p` subprocess calls; needs the
# claude CLI on PATH and logged in)
python scripts/run_live_claude_sample.py
```

## Sibling comparison

[`ontology-grounded-operations-agent`](https://github.com/Manas103/ontology-grounded-operations-agent),
[`grounded-reconciliation-assistant`](https://github.com/Manas103/grounded-reconciliation-assistant), and
[`natural-language-segment-builder`](https://github.com/Manas103/natural-language-segment-builder)
all share this repo's named-tools-plus-pluggable-LLM-client shape: a
deterministic client measured at scale, a real `claude -p` client
exercised on a small live sample. None of the three carries anything like
this repo's trial ledger, because none of them produces a p-value; their
risk is a wrong lookup or a malformed write, not noise mining across many
hypotheses. [`agent-action-harness-deterministic-oracle`](https://github.com/Manas103/agent-action-harness-deterministic-oracle)
is the closest architectural sibling for the verification gate itself: it
also grades a result against an independent oracle that shares no code
with the executor, applied there to whether a multi-step plan's final
database state matches expectation, applied here to whether every number
in a written finding matches an independent recompute. The property this
repo adds that none of the four have is the trial ledger: a researcher
here cannot report an unadjusted p-value even once, because the number
displayed is already the Holm-adjusted one, computed against the real,
persistent, cumulative count of everything proposed so far.

## Limitations

- The question set uses ten fixed marker-phrase templates (`router.py`),
  not free-form phrasing; a rephrased question is refused by the
  deterministic router and only has a chance through the real `claude -p`
  path.
- The 60 broken findings are synthetically seeded by `broken_bank.py`, not
  drawn from a live LLM's actual fabrication or query-bug distribution; no
  live `claude -p` run in this build produced a malformed finding to
  measure against, the same disclosed gap the portfolio's other
  adversarial-input repos carry for their malformed-proposal generators.
- The live `claude -p` sample is 20 calls, not the full 200/60/260-case
  scale; it is real evidence the backend works end to end, not a claim
  that the LLM path matches the deterministic router's rates at the same
  scale.
- Only Holm-Bonferroni is implemented; Benjamini-Hochberg (which controls
  the false discovery rate rather than the familywise error rate, and
  would give less conservative adjusted p-values) was a stated option and
  was not built, a scope choice given the time budget, not an oversight.
- The oracle's tolerance (1e-6 absolute) is tight enough to catch every
  seeded case here, but is a property of this synthetic data's scale (IV
  values in roughly 0.02-0.9); it was not independently stress-tested
  against a finding whose true values differ from a fabrication by less
  than that tolerance, which this seeded battery does not construct.
- `cohort_membership`'s is-A-or-B check assumes exactly two cohorts; the
  schema's `CHECK (cohort IN ('A', 'B'))` constraint enforces this at the
  database level, but the tool and oracle code is not written to
  generalize past two cohorts without changes.
