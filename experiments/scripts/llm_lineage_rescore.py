"""Re-score only — signatures already on disk, just reload + renormalize."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_lineage as core

core.SIG_DIR = Path("sigs_real_llm")

REF = "llama2-7b-base"
CHAT = "llama2-7b-chat"
INDEP = "deepseek-r1-distill-llama-8b"
LOCAL = ["base+quant-int8", "base+prune-30", "base+noise-1pct"]

phi_ref, s_ref = core.load_sig(REF)
rng = np.random.default_rng(0)
shuffles = []
for _ in range(20):
    perm = rng.permutation(len(phi_ref))
    while np.array_equal(perm, np.arange(len(perm))):
        perm = rng.permutation(len(phi_ref))
    cos = phi_ref @ phi_ref[perm].T
    shuffles.append(float(np.diag(cos).mean()))
mu_null = float(np.mean(shuffles))
sd_null = float(np.std(shuffles, ddof=1))
print(f"[null]  within-model block-shuffle on base (n=20)  mu={mu_null:+.4f}  sigma={sd_null:.4f}\n")

import csv
rows = []
for kind, tag, expected in [
    ("descendant", CHAT, "DESCENDANT"),
    *[("desc-local", t, "DESCENDANT") for t in LOCAL],
    ("independent", INDEP, "NON-DESCENDANT"),
]:
    L, pb = core.lineage(REF, tag)
    z = (L - mu_null) / max(sd_null, 1e-6)
    verdict = ("DESCENDANT" if z > 3.0 else
               "NON-DESCENDANT" if z < 1.645 else "INCONCLUSIVE")
    print(f"[{kind:>12s}]  base vs {tag:32s}  L={L:+.6f}  z={z:+7.1f}  "
          f"{verdict:<15s} blocks=[{pb.min():+.4f}, {pb.max():+.4f}]")
    rows.append((kind, REF, tag, expected, L, z, verdict,
                 float(pb.min()), float(pb.max())))

out = Path("results") / "lineage_real_llm.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind", "model_a", "model_b", "expected",
                "L", "z", "verdict", "min_block", "max_block"])
    w.writerows(rows)

summary = Path("results") / "lineage_real_llm_summary.txt"
summary.write_text(
    f"Null mu={mu_null:.4f} sigma={sd_null:.4f}\n"
    + "\n".join(f"{r[0]:>12s}  base vs {r[2]:32s}  L={r[4]:+.6f}  "
                f"z={r[5]:+7.1f}  {r[6]}  (expected {r[3]})" for r in rows)
)
print(f"\n-> {out}\n-> {summary}")
