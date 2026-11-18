"""Compare classifier-only vs classifier+HMM on the student-testing trips and pick the served model.

Model selection rule (decided before looking at HMM results): lowest mean cross-validation log loss
on the training trips, because the HMM consumes the classifier's probabilities as emissions.
Writes models/selected_model.json, results/hmm_vs_classifier.csv and a Folium map with the prediction.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from data_io import REPO_ROOT, list_trips
from eda import trip_map
from hmm import viterbi_decode
from metrics import continuity
from pipeline import FEATURES, build_trip_table

RES, MOD = REPO_ROOT / "results", REPO_ROOT / "models"


def main():
    comp = pd.read_csv(RES / "model_comparison.csv")
    best = comp.sort_values("cv_log_loss").iloc[0].model
    (MOD / "selected_model.json").write_text(json.dumps({"model_file": f"{best}.joblib", "rule": "min CV log loss"}, indent=2))
    print("selected model:", best)
    model = joblib.load(MOD / f"{best}.joblib")
    rows, keep = [], {}
    for tid in list_trips("student_test"):
        g, cand, info = build_trip_table(tid, "student_test")
        lab = info["label_arc"]
        p = model.predict_proba(cand[FEATURES].to_numpy())[:, 1]
        c2 = cand.assign(p=p)
        b = c2.loc[c2.groupby("point_idx").p.idxmax()]
        clf = np.full(len(g.P), -1)
        clf[b.point_idx.to_numpy()] = b.arc_id.to_numpy()
        t0 = time.time()
        hmm = viterbi_decode(g, cand, p)
        rows.append({"trip_id": tid, "n_points": len(g.P), "clf_point_acc": (clf == lab).mean(), "hmm_point_acc": (hmm == lab).mean(),
                     "clf_continuity": continuity(clf, g.arcs), "hmm_continuity": continuity(hmm, g.arcs),
                     "hmm_seconds": time.time() - t0})
        keep[tid] = hmm
    df = pd.DataFrame(rows)
    w = df.n_points / df.n_points.sum()
    total = {"trip_id": "ALL (point-weighted)", "n_points": df.n_points.sum()}
    for c in ["clf_point_acc", "hmm_point_acc", "clf_continuity", "hmm_continuity"]:
        total[c] = (df[c] * w).sum()
    total["hmm_seconds"] = df.hmm_seconds.sum()
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    df.to_csv(RES / "hmm_vs_classifier.csv", index=False)
    print(df.round(4).to_string(index=False))
    tid = df.iloc[:-1].sort_values("n_points").iloc[len(df) // 2].trip_id
    print("map:", trip_map(tid, "student_test", pred_arcs=keep[tid]))


if __name__ == "__main__":
    main()
