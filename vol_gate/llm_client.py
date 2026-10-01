"""Pluggable LLM-client interface. Two implementations:

DeterministicRouterClient wraps router.route_question (no network call); it
is the client used for the 60+200 gate battery and the ledger-growth run,
because those need to be exactly reproducible at scale.

ClaudeCLIClient shells out to the real `claude` CLI as a genuine
subprocess, sends it the plain-English question plus a description of the
available shapes/params, and parses a one-line JSON {"shape": ..., "params":
{...}} back out. It is exercised on a small, explicitly-counted live sample
(see scripts/run_live_claude_sample.py), never pretended to run at the
60/200/ledger scale.
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from abc import ABC, abstractmethod

from .finding import Finding, build_finding
from .gate import GateResult, run_gate
from .ledger import TrialLedger
from .router import RoutingError, route_question

SHAPE_CATALOG_DESCRIPTION = """\
Available question shapes and their required JSON params (all dates are
ISO yyyy-mm-dd strings, right is "P" or "C", delta_bucket is one of
10/25/50/75/90, tenor_days is 30 or 60):

median_change: ticker, right, delta_bucket, tenor_days, start_date, end_date
cohort_compare: cohort_a, cohort_b, right, delta_bucket, tenor_days, start_date, end_date
correlation: ticker_a, ticker_b, right, delta_bucket, tenor_days, start_date, end_date
event_window: ticker, event_index, right, delta_bucket, tenor_days, window
term_structure: ticker, right, delta_bucket, session_date
skew: ticker, tenor_days, session_date
percentile_rank: ticker, right, delta_bucket, tenor_days, session_date, lookback_sessions
dividend_decile: ticker
contract_lookup: ticker, right, delta_bucket, tenor_days, session_date
list_sessions_count: ticker, start_date, end_date

Tickers are SYN-A01..SYN-A08 (cohort A) and SYN-B01..SYN-B08 (cohort B).
Reply with exactly one line of JSON: {"shape": "<shape>", "params": {...}}
and nothing else, no explanation, no markdown fences.
"""


class LLMClient(ABC):
    @abstractmethod
    def parse_question(self, question: str) -> tuple[str, dict]:
        """Returns (shape, params). Raises on an unparseable question."""

    def answer(self, conn: sqlite3.Connection, ledger: TrialLedger, question: str) -> GateResult:
        shape, params = self.parse_question(question)
        finding = build_finding(shape, conn, ledger, params)
        return run_gate(conn, finding)


class DeterministicRouterClient(LLMClient):
    def parse_question(self, question: str) -> tuple[str, dict]:
        return route_question(question)


class ClaudeCLIError(Exception):
    pass


class ClaudeCLIClient(LLMClient):
    """A genuine subprocess client. Sends the question over stdin (never as
    a bare command-line argument; see the sibling
    agent-action-harness-deterministic-oracle repo's README for exactly
    the Windows ProcessBuilder quoting bug that motivates piping over
    stdin instead) and parses the single JSON line back out."""

    def __init__(self, claude_path: str = "claude", timeout_s: int = 60):
        self.claude_path = claude_path
        self.timeout_s = timeout_s

    def parse_question(self, question: str) -> tuple[str, dict]:
        prompt = f"{SHAPE_CATALOG_DESCRIPTION}\nQuestion: {question}\n"
        try:
            proc = subprocess.run(
                [self.claude_path, "-p", prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            raise ClaudeCLIError(f"claude CLI not found on PATH: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCLIError(f"claude CLI timed out: {exc}") from exc

        raw = proc.stdout.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ClaudeCLIError(f"no JSON object found in claude CLI output: {raw!r}")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ClaudeCLIError(f"could not parse JSON from claude CLI output: {raw!r}") from exc

        shape = payload.get("shape")
        params = payload.get("params")
        if not isinstance(shape, str) or not isinstance(params, dict):
            raise ClaudeCLIError(f"malformed shape/params in claude CLI output: {payload!r}")
        if "delta_bucket" in params:
            params["delta_bucket"] = int(params["delta_bucket"])
        for key in ("tenor_days", "event_index", "window", "lookback_sessions"):
            if key in params:
                params[key] = int(params[key])
        return shape, params
