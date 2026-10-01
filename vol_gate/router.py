"""Deterministic marker-phrase router: maps a fixed set of plain-English
question templates to a (shape, params) pair the finding builders can run.
No network call, fully reproducible; this is the client used for the
60+200 gate battery and the ledger-growth run, for the same reason the
portfolio's other repos use a deterministic router for their large-scale
benchmarks: a benchmark that size needs to be exactly reproducible to be
trustworthy. Raises RoutingError rather than guessing when nothing matches.
"""
from __future__ import annotations

import re

RIGHT_WORDS = {"put": "P", "call": "C", "P": "P", "C": "C"}


class RoutingError(Exception):
    pass


_PATTERNS = [
    ("median_change", re.compile(
        r"^Did median (?P<delta>\d+)-delta (?P<right>put|call) implied vol for (?P<ticker>[A-Z0-9-]+) "
        r"rise between (?P<start_date>\d{4}-\d{2}-\d{2}) and (?P<end_date>\d{4}-\d{2}-\d{2})\?$"
    )),
    ("cohort_compare", re.compile(
        r"^Is there a significant difference in (?P<delta>\d+)-delta (?P<right>put|call) implied vol "
        r"regime between cohort (?P<cohort_a>[AB]) and cohort (?P<cohort_b>[AB]) between "
        r"(?P<start_date>\d{4}-\d{2}-\d{2}) and (?P<end_date>\d{4}-\d{2}-\d{2})\?$"
    )),
    ("correlation", re.compile(
        r"^What is the correlation between (?P<ticker_a>[A-Z0-9-]+) and (?P<ticker_b>[A-Z0-9-]+) "
        r"(?P<delta>\d+)-delta (?P<right>put|call) implied vol between "
        r"(?P<start_date>\d{4}-\d{2}-\d{2}) and (?P<end_date>\d{4}-\d{2}-\d{2})\?$"
    )),
    ("event_window", re.compile(
        r"^How did (?P<ticker>[A-Z0-9-]+)'s (?P<delta>\d+)-delta (?P<right>put|call) implied vol "
        r"change around event #(?P<event_index>\d+) within (?P<window>\d+) sessions\?$"
    )),
    ("term_structure", re.compile(
        r"^What is (?P<ticker>[A-Z0-9-]+)'s (?P<delta>\d+)-delta (?P<right>put|call) term structure "
        r"slope on (?P<session_date>\d{4}-\d{2}-\d{2})\?$"
    )),
    ("skew", re.compile(
        r"^What is (?P<ticker>[A-Z0-9-]+)'s (?P<tenor_days>\d+)-day put skew on "
        r"(?P<session_date>\d{4}-\d{2}-\d{2})\?$"
    )),
    ("percentile_rank", re.compile(
        r"^Where does (?P<ticker>[A-Z0-9-]+)'s (?P<delta>\d+)-delta (?P<right>put|call) "
        r"(?P<tenor_days>\d+)-day implied vol on (?P<session_date>\d{4}-\d{2}-\d{2}) rank over the "
        r"trailing (?P<lookback_sessions>\d+) sessions\?$"
    )),
    ("dividend_decile", re.compile(
        r"^What dividend decile is (?P<ticker>[A-Z0-9-]+) in\?$"
    )),
    ("contract_lookup", re.compile(
        r"^What is (?P<ticker>[A-Z0-9-]+)'s (?P<delta>\d+)-delta (?P<right>put|call) "
        r"(?P<tenor_days>\d+)-day implied vol on (?P<session_date>\d{4}-\d{2}-\d{2})\?$"
    )),
    ("list_sessions_count", re.compile(
        r"^How many trading sessions does (?P<ticker>[A-Z0-9-]+) have between "
        r"(?P<start_date>\d{4}-\d{2}-\d{2}) and (?P<end_date>\d{4}-\d{2}-\d{2})\?$"
    )),
]


_INT_FIELDS = ("tenor_days", "event_index", "window", "lookback_sessions")


def route_question(question: str) -> tuple[str, dict]:
    for shape, pattern in _PATTERNS:
        m = pattern.match(question.strip())
        if m:
            params = dict(m.groupdict())
            if "right" in params:
                params["right"] = RIGHT_WORDS[params["right"]]
            if "delta" in params:
                params["delta_bucket"] = int(params.pop("delta"))
            for key in _INT_FIELDS:
                if key in params:
                    params[key] = int(params[key])
            return shape, params
    raise RoutingError(f"no template matches: {question!r}")
