"""The verification gate: a finding is shown to the user only if the
independent oracle reproduces every numeric figure in it within tolerance.
If any figure disagrees, the response names the specific disagreeing
figure (claimed value, recomputed value, and the gap), it never just says
"blocked".
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import oracle as oracle_mod
from .finding import Finding


@dataclass
class GateResult:
    approved: bool
    text: str | None
    disagreements: list


def run_gate(conn: sqlite3.Connection, finding: Finding) -> GateResult:
    result = oracle_mod.verify(conn, finding)
    if result.ok:
        return GateResult(approved=True, text=finding.text, disagreements=[])
    lines = ["WITHHELD: the following figure(s) could not be independently verified:"]
    for d in result.disagreements:
        if d.recomputed != d.recomputed:  # NaN marker for a lookup failure
            lines.append(f"  - {d.label}: claimed {d.claimed}, oracle could not reproduce the underlying lookup")
        else:
            lines.append(f"  - {d.label}: claimed {d.claimed}, oracle recomputed {d.recomputed} (diff {d.diff})")
    return GateResult(approved=False, text="\n".join(lines), disagreements=result.disagreements)
