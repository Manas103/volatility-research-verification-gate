"""Screens the 48-hypothesis bank (vol_gate/hypothesis_bank.py) against the
volatility store imported from order-book-signal-research, through the
same gate/ledger machinery the base repo's option-chain questions use:
every hypothesis is built into a Finding, verified by the independent
oracle (0 disagreements expected, since none of these 48 are deliberately
broken; this is a live sanity check of that, not assumed), recorded into
the persistent trial ledger, and tallied raw-significant vs.
Holm-significant.

Then runs a small, genuinely live `claude -p` sample asking the real
model to propose its own volatility hypotheses (not given in English, the
model picks the shape and params), to demonstrate the "language-model-
proposed" half of the claim beyond the deterministic bank. See the
README for why the 48-hypothesis headline count comes from the
deterministic bank, the same precedent DeterministicRouterClient already
sets in this repo for bulk runs.
"""
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vol_gate import db, gate, volstore
from vol_gate.finding import build_finding
from vol_gate.hypothesis_bank import SPLIT_RULES, TICKERS, build_hypothesis_bank
from vol_gate.ledger import TrialLedger
from vol_gate.llm_client import ClaudeCLIClient, ClaudeCLIError

N_LIVE_CALLS = 10


def screen_bank(conn, ledger) -> dict:
    bank = build_hypothesis_bank()
    n_cleared_raw = 0
    n_survived_holm = 0
    n_gate_withheld = 0
    rows = []
    for shape, params in bank:
        f = build_finding(shape, conn, ledger, params)
        result = gate.run_gate(conn, f)
        if not result.approved:
            n_gate_withheld += 1
        raw_p = next(c.value for c in f.claims if c.label == "raw_p")
        adjusted_p = next(c.value for c in f.claims if c.label == "adjusted_p")
        if raw_p < 0.05:
            n_cleared_raw += 1
        if adjusted_p < 0.05:
            n_survived_holm += 1
        rows.append({
            "shape": shape, "params": params, "raw_p": raw_p, "adjusted_p": adjusted_p,
            "gate_approved": result.approved,
        })
    return {
        "n_hypotheses": len(bank), "n_cleared_raw_p_below_0_05": n_cleared_raw,
        "n_survived_holm_below_0_05": n_survived_holm, "n_gate_withheld": n_gate_withheld,
        "rows": rows,
    }


def run_live_sample(conn, ledger) -> dict:
    client = ClaudeCLIClient()
    proposed: list[tuple[str, dict]] = []
    n_calls = n_parsed = n_valid = n_executed = n_approved = 0
    log = []
    valid_shapes = {"vol_pair_compare", "vol_pair_correlation", "vol_activity_correlation",
                     "vol_day_trend_correlation", "vol_regime_compare"}
    valid_tickers = set(TICKERS) | {"ALL"}

    def _valid(shape: str, params: dict) -> bool:
        if shape not in valid_shapes:
            return False
        if shape in ("vol_pair_compare", "vol_pair_correlation"):
            return (params.get("ticker_a") in TICKERS and params.get("ticker_b") in TICKERS
                    and params.get("ticker_a") != params.get("ticker_b"))
        if shape in ("vol_activity_correlation", "vol_day_trend_correlation"):
            return params.get("ticker") in valid_tickers
        if shape == "vol_regime_compare":
            return params.get("ticker") in valid_tickers and params.get("split_by") in SPLIT_RULES
        return False

    for i in range(N_LIVE_CALLS):
        n_calls += 1
        entry = {"call": n_calls}
        try:
            shape, params = client.propose_vol_hypothesis(avoid=proposed)
            n_parsed += 1
            entry["proposed"] = {"shape": shape, "params": params}
            if not _valid(shape, params):
                entry["outcome"] = "invalid_params"
                log.append(entry)
                continue
            n_valid += 1
            proposed.append((shape, params))
            f = build_finding(shape, conn, ledger, params)
            n_executed += 1
            result = gate.run_gate(conn, f)
            if result.approved:
                n_approved += 1
            entry["outcome"] = "approved" if result.approved else "withheld"
            entry["text"] = f.text
        except ClaudeCLIError as e:
            entry["outcome"] = f"cli_error: {e}"
        except Exception as e:
            entry["outcome"] = f"execution_error: {type(e).__name__}: {e}"
        print(f"[{n_calls}] {entry}")
        log.append(entry)

    return {
        "n_calls": n_calls, "n_parsed_json": n_parsed, "n_valid_params": n_valid,
        "n_executed": n_executed, "n_gate_approved": n_approved, "log": log,
    }


def main() -> None:
    with db.session() as conn:
        n_imported = volstore.import_csv(conn)
        print(f"volatility_bins: {n_imported} rows imported from order-book-signal-research")
        ledger = TrialLedger(conn)
        start_size = ledger.size()
        print(f"ledger size before this run: {start_size}")

        bank_results = screen_bank(conn, ledger)
        print(f"\n48-hypothesis bank: {bank_results['n_hypotheses']} screened")
        print(f"cleared raw p < 0.05: {bank_results['n_cleared_raw_p_below_0_05']} / {bank_results['n_hypotheses']}")
        print(f"survived Holm-adjusted p < 0.05: {bank_results['n_survived_holm_below_0_05']} / {bank_results['n_hypotheses']}")
        print(f"gate withheld (should be 0, none of these 48 are deliberately broken): {bank_results['n_gate_withheld']}")

        print(f"\nledger size after the bank: {ledger.size()}")
        print(f"\nlive claude -p hypothesis-proposal sample ({N_LIVE_CALLS} real calls):")
        live_results = run_live_sample(conn, ledger)
        print(f"\nREAL CLAUDE CLI SUBPROCESS CALLS MADE: {live_results['n_calls']}")
        print(f"parsed into JSON: {live_results['n_parsed_json']} / {live_results['n_calls']}")
        print(f"valid params: {live_results['n_valid_params']} / {live_results['n_calls']}")
        print(f"gate-approved: {live_results['n_gate_approved']} / {live_results['n_executed']}")
        print(f"\nledger size after the live sample: {ledger.size()}")

        out = {
            "bank": {k: v for k, v in bank_results.items() if k != "rows"},
            "bank_rows": bank_results["rows"],
            "live_sample": live_results,
            "ledger_size_before": start_size, "ledger_size_after": ledger.size(),
        }
        with open("docs/volatility_hypothesis_screen_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print("\nResults written to docs/volatility_hypothesis_screen_results.json")


if __name__ == "__main__":
    main()
