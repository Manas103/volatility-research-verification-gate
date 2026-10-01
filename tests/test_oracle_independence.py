"""Proves the oracle shares no code with the executor, and that breaking
the executor's query intentionally is still caught."""
import ast
import os

from vol_gate import finding, oracle, tools
from vol_gate.gate import run_gate


def test_oracle_module_imports_no_executor_module():
    path = os.path.join(os.path.dirname(oracle.__file__), "oracle.py")
    tree = ast.parse(open(path).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
        if isinstance(node, ast.Import):
            for n in node.names:
                imported.add(n.name.split(".")[-1])
    assert "tools" not in imported
    assert "finding" not in imported
    assert "ledger" not in imported
    assert "holm" not in imported


def test_oracle_and_executor_are_different_function_objects():
    assert oracle._recompute_iv_level is not tools.get_contract
    assert oracle._oracle_holm_bonferroni.__module__ != "vol_gate.holm"


def test_breaking_executor_query_is_still_caught(conn):
    """Builds a correct finding, then corrupts the window-median claim the
    way a wrong-date-range bug would (shifts the value only, leaving the
    recipe pointing at the true window), and confirms the independent
    oracle still disagrees and names the exact figure."""
    f = finding.build_median_change(conn, "SYN-A01", "P", 25, 30, "2025-01-02", "2025-03-27")
    for c in f.claims:
        if c.label == "change":
            c.value = c.value + 0.5  # inject a silent-error-shaped bug: wrong number, same recipe
    result = run_gate(conn, f)
    assert result.approved is False
    assert any(d.label == "change" for d in result.disagreements)
