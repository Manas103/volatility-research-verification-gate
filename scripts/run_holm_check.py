"""Hand-verifiable Holm-Bonferroni check: 5 hypotheses, raw p-values chosen
so the adjustment can be computed by hand and compared to vol_gate.holm."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vol_gate.holm import holm_bonferroni

raw = [0.01, 0.02, 0.03, 0.04, 0.005]
# hand computation (n=5): sorted ascending with rank*p: 0.005*5=0.025, 0.01*4=0.04,
# 0.02*3=0.06, 0.03*2=0.06, 0.04*1=0.04; running max from smallest rank up:
# 0.025, 0.04, 0.06, 0.06, 0.06 -> mapped back to original order:
expected = [0.04, 0.06, 0.06, 0.06, 0.025]

got = holm_bonferroni(raw)
print("raw p-values:      ", raw)
print("hand-computed adj: ", expected)
print("holm_bonferroni():  ", got)
print("MATCH:", all(abs(a - b) < 1e-12 for a, b in zip(got, expected)))
