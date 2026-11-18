"""Tune the HMM/Viterbi decoder settings on TRAINING trips only.

Probabilities fed to the decoder are out-of-fold: each tuning trip is scored by a model trained on
training trips that exclude it, so the decoder is tuned on the kind of probabilities it will see on new trips.
Grid: beta x gap threshold x emission weight (unreachable penalty fixed at -50). Selection: highest point accuracy, ties -> higher continuity.
The student-testing trips are not used here (they are only used for the final comparison in evaluate_hmm.py).

Outputs: results/hmm_tuning.csv, models/hmm_params.json, reports/figures/hmm_tuning_heatmap.png
"""
import itertools
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_io import list_trips
from hmm import BACKWARD_PEN, Router, viterbi_decode
from metrics import continuity
from pipeline import FEATURES, build_trip_table
from train_models import BAD_LABEL_MEDIAN_M, FIG, MOD, RES, load
from tune_bayes import UNTUNED, make

BETAS = [5, 10, 20, 40]
GAPS = [10, 30, 60]
EMIS_W = [1, 2, 4, 8]
UNREACH = [-50]
if "quick" in sys.argv:  # smoke test only
    BETAS, GAPS, EMIS_W, UNREACH = [10, 25], [30], [1, 4], [-50]
MAX_POINTS = 5000   # keep decoding time reasonable; very long trips are left out of the tuning set
N_FOLDS = 4


def selected_params():
    f = RES / "optuna_best_params.json"
    return json.loads(f.read_text())["best_params"] if f.exists() else UNTUNED


def main():
    tr, _, bad, _ = load()                       # train rows (every 4th point, bad-reference trips removed)
    inv = pd.read_csv(RES / "trip_inventory.csv", dtype={"trip_id": str})
    ok = inv[(inv.partition == "train") & (inv.label_dist_median_m <= BAD_LABEL_MEDIAN_M) & (inv.n_points <= MAX_POINTS)]
    tune_trips = sorted(ok.trip_id)[::3]
    if "quick" in sys.argv:
        tune_trips = tune_trips[:8]
    params = selected_params()
    print(f"{len(tune_trips)} tuning trips, decoder grid {len(BETAS) * len(GAPS) * len(EMIS_W) * len(UNREACH)} settings, model params {params}")

    folds = [tune_trips[k::N_FOLDS] for k in range(N_FOLDS)]
    cases = []
    for fold in folds:
        rows = tr[~tr.trip_id.isin(fold)]
        model = make(params).fit(rows[FEATURES].to_numpy(), rows["y"].to_numpy())
        for tid in fold:
            g, cand, info = build_trip_table(tid, "train")
            p = model.predict_proba(cand[FEATURES].to_numpy())[:, 1]
            c2 = cand.assign(p=p)
            b = c2.loc[c2.groupby("point_idx").p.idxmax()]
            clf = np.full(len(g.P), -1)
            clf[b.point_idx.to_numpy()] = b.arc_id.to_numpy()
            cases.append({"trip": tid, "g": g, "cand": cand, "p": p, "label": info["label_arc"], "clf": clf, "router": Router(g)})
    n_pts = sum(len(c["label"]) for c in cases)
    clf_acc = sum((c["clf"] == c["label"]).sum() for c in cases) / n_pts
    clf_cont = np.mean([continuity(c["clf"], c["g"].arcs) for c in cases])
    print(f"classifier-only on tuning trips: accuracy {clf_acc:.4f}, continuity {clf_cont:.4f} ({n_pts} points)")

    rows = []
    for beta, gap, ew, unr in itertools.product(BETAS, GAPS, EMIS_W, UNREACH):
        t0 = time.time()
        hits, conts = 0, []
        for c in cases:
            pred = viterbi_decode(c["g"], c["cand"], c["p"], beta=beta, gap_s=gap, unreachable=unr, emission_w=ew, router=c["router"])
            hits += (pred == c["label"]).sum()
            conts.append(continuity(pred, c["g"].arcs))
        rows.append({"beta": beta, "gap_s": gap, "emission_w": ew, "unreachable": unr, "accuracy": hits / n_pts,
                     "continuity": float(np.mean(conts)), "seconds": time.time() - t0})
        print(f"beta {beta:3d} gap {gap:3d} w {ew} unreach {unr:4d} -> acc {hits / n_pts:.4f} cont {np.mean(conts):.4f}", flush=True)
    res = pd.DataFrame(rows).sort_values(["accuracy", "continuity"], ascending=False)
    res.to_csv(RES / "hmm_tuning.csv", index=False)
    best = res.iloc[0]
    out = {"beta": float(best.beta), "gap_s": float(best.gap_s), "emission_w": float(best.emission_w), "unreachable": float(best.unreachable),
           "backward_pen": BACKWARD_PEN, "tuning_trips": tune_trips, "tuning_points": int(n_pts),
           "tuned_accuracy": float(best.accuracy), "tuned_continuity": float(best.continuity),
           "classifier_only_accuracy": float(clf_acc), "classifier_only_continuity": float(clf_cont),
           "rule": "max point accuracy on out-of-fold training-trip probabilities, ties -> continuity"}
    (MOD / "hmm_params.json").write_text(json.dumps(out, indent=2))
    print("\nbest:", {k: out[k] for k in ["beta", "gap_s", "emission_w", "tuned_accuracy", "tuned_continuity"]})
    heatmap(res, best.gap_s)


def heatmap(res, gap):
    d = res[res.gap_s == gap].pivot(index="beta", columns="emission_w", values="accuracy").sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(d.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(d.columns))); ax.set_xticklabels([f"w = {c:g}" for c in d.columns])
    ax.set_yticks(range(len(d.index))); ax.set_yticklabels([f"{int(b)} m" for b in d.index])
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, f"{d.values[i, j] * 100:.2f}%", ha="center", va="center", fontsize=8)
    ax.set_xlabel("emission weight"); ax.set_ylabel("beta (transition tolerance)")
    ax.set_title(f"HMM point accuracy on training trips (gap {int(gap)} s)")
    plt.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig(FIG / "hmm_tuning_heatmap.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
