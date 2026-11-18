"""Final model selection + route-reconstruction comparison on the student-testing trips.

Model selection rule (fixed before looking at student-test results): lowest mean cross-validation log loss on the
training trips among all trained models, including the Bayesian-tuned model when tune_bayes.py has been run.
Decoder settings come from models/hmm_params.json (tune_hmm.py, training trips only).

Compared per trip:
  classifier-only          top-scored candidate per point, independently
  distance-only HMM        classic HMM baseline: Gaussian distance emission (sigma 10 m), no classifier
  classifier + HMM default decoder with the hand-set settings (beta 15, gap 30 s, unreachable -50, emission weight 1)
  classifier + HMM tuned   decoder with the tuned settings
Outputs: models/selected_model.json, results/hmm_vs_classifier.csv, reports/figures/hmm_vs_classifier.png,
         Folium maps of a typical and the weakest trip with the predicted route.
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_io import REPO_ROOT, list_trips
from eda import trip_map
from hmm import BACKWARD_PEN, BETA, EMISSION_W, GAP_S, UNREACHABLE, Router, distance_emission, load_params, viterbi_decode
from metrics import continuity
from pipeline import FEATURES, build_trip_table

RES, MOD, FIG = REPO_ROOT / "results", REPO_ROOT / "models", REPO_ROOT / "reports" / "figures"


def select_model():
    cands = pd.read_csv(RES / "model_comparison.csv")[["model", "cv_log_loss"]]
    tuned = RES / "bayes_tuned_vs_untuned.csv"
    if tuned.exists() and (MOD / "LightGBM_tuned.joblib").exists():
        t = pd.read_csv(tuned)
        row = t[t.model == "LightGBM (tuned)"].iloc[0]
        cands = pd.concat([cands, pd.DataFrame([{"model": "LightGBM_tuned", "cv_log_loss": row.cv_log_loss}])])
    cands = cands.sort_values("cv_log_loss")
    best = cands.iloc[0].model
    (MOD / "selected_model.json").write_text(json.dumps(
        {"model_file": f"{best}.joblib", "rule": "min CV log loss (train trips, GroupKFold)",
         "candidates": {r.model: round(float(r.cv_log_loss), 5) for r in cands.itertuples()}}, indent=2))
    print("CV log loss:", {r.model: round(float(r.cv_log_loss), 5) for r in cands.itertuples()})
    print("selected model:", best)
    return best


def main():
    best = select_model()
    model = joblib.load(MOD / f"{best}.joblib")
    hp = load_params(MOD)
    default = {"beta": BETA, "gap_s": GAP_S, "unreachable": UNREACHABLE, "backward_pen": BACKWARD_PEN, "emission_w": EMISSION_W}
    print("HMM settings tuned:", hp, "| default:", default)
    rows, keep = [], {}
    for tid in list_trips("student_test"):
        g, cand, info = build_trip_table(tid, "student_test")
        lab, router = info["label_arc"], Router(g)
        p = model.predict_proba(cand[FEATURES].to_numpy())[:, 1]
        c2 = cand.assign(p=p)
        b = c2.loc[c2.groupby("point_idx").p.idxmax()]
        clf = np.full(len(g.P), -1)
        clf[b.point_idx.to_numpy()] = b.arc_id.to_numpy()
        dist_hmm = viterbi_decode(g, cand, distance_emission(cand), router=router, **hp)
        hmm_def = viterbi_decode(g, cand, p, router=router, **default)
        t0 = time.time()
        hmm = viterbi_decode(g, cand, p, router=router, **hp)
        rows.append({"trip_id": tid, "n_points": len(g.P),
                     "clf_point_acc": (clf == lab).mean(), "dist_hmm_point_acc": (dist_hmm == lab).mean(),
                     "hmm_default_point_acc": (hmm_def == lab).mean(), "hmm_point_acc": (hmm == lab).mean(),
                     "clf_continuity": continuity(clf, g.arcs), "dist_hmm_continuity": continuity(dist_hmm, g.arcs),
                     "hmm_default_continuity": continuity(hmm_def, g.arcs), "hmm_continuity": continuity(hmm, g.arcs),
                     "hmm_seconds": time.time() - t0})
        keep[tid] = hmm
    df = pd.DataFrame(rows)
    w = df.n_points / df.n_points.sum()
    total = {"trip_id": "ALL (point-weighted)", "n_points": int(df.n_points.sum()), "hmm_seconds": df.hmm_seconds.sum()}
    for c in [c for c in df.columns if c.endswith("_acc") or c.endswith("_continuity")]:
        total[c] = (df[c] * w).sum()
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    df.to_csv(RES / "hmm_vs_classifier.csv", index=False)
    print(df.round(4).to_string(index=False))
    plot(df.iloc[:-1])
    trips = df.iloc[:-1].sort_values("n_points")
    typical = trips.iloc[len(trips) // 2].trip_id
    weakest = df.iloc[:-1].sort_values("hmm_point_acc").iloc[0].trip_id
    for tid in dict.fromkeys([typical, weakest]):
        print("map:", trip_map(tid, "student_test", pred_arcs=keep[tid]))


def plot(df):
    FIG.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(df)); w = 0.27
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for ax, suffix, label in [(axes[0], "point_acc", "Point accuracy"), (axes[1], "continuity", "Route continuity")]:
        ax.bar(x - w, df[f"clf_{suffix}"], w, label="Classifier only", color="#bdbdbd")
        ax.bar(x, df[f"dist_hmm_{suffix}"], w, label="Distance-only HMM", color="#90caf9")
        ax.bar(x + w, df[f"hmm_{suffix}"], w, label="Classifier + HMM (tuned)", color="#1565c0")
        ax.set_ylabel(label); ax.set_ylim(0.6, 1.0)
    axes[0].legend(ncol=3, fontsize=8, loc="lower right")
    axes[1].set_xticks(x); axes[1].set_xticklabels([t[-2:] for t in df.trip_id]); axes[1].set_xlabel("student-test trip")
    fig.suptitle("Route reconstruction on the student-testing trips"); plt.tight_layout()
    plt.savefig(FIG / "hmm_vs_classifier.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
