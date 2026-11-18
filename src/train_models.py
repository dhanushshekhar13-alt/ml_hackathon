"""Train and compare six classifiers on candidate features.

- Group-aware 5-fold CV on the training trips (groups = trip_id, no trip in two folds)
- Refit on all training trips, evaluate on the 10 student-testing trips
- Saves results/model_comparison.csv, results/cv_folds.csv, results/per_trip_student_test.csv,
  models/<name>.joblib and comparison figures.
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # optional: on macOS LightGBM may need `brew install libomp`
    from lightgbm import LGBMClassifier
except Exception as e:  # keep the other models running if it is missing
    LGBMClassifier = None
    print('LightGBM unavailable, skipping it:', e)

from data_io import REPO_ROOT
from metrics import candidate_metrics, sequence_metrics
from pipeline import FEATURES

SEED = 42
POINT_STRIDE = 4          # 1 Hz points are highly autocorrelated: keep every 4th observation for fitting
BAD_LABEL_MEDIAN_M = 50   # trips whose reference route is far from the GPS are excluded from training
RES, MOD, FIG = REPO_ROOT / "results", REPO_ROOT / "models", REPO_ROOT / "reports" / "figures"


def make_models():
    models = {
        "LogisticRegression": ("Linear / probabilistic",
                               make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
                               "StandardScaler, L2, C=1.0, lbfgs"),
        "GaussianNB": ("Linear / probabilistic", GaussianNB(var_smoothing=1e-9), "var_smoothing=1e-9"),
        "KNN": ("Instance-based",
                make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=2)),
                "StandardScaler, k=25, distance-weighted"),
        "RandomForest": ("Ensemble (bagging)",
                         RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5, n_jobs=2, random_state=SEED),
                         "100 trees, max_depth=14, min_leaf=5"),
        "HistGradientBoosting": ("Boosting",
                                 HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED),
                                 "200 iters, lr=0.1, 31 leaves"),
    }
    if LGBMClassifier is not None:
        models["LightGBM"] = ("Boosting (LightGBM)",
                              LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, min_child_samples=20,
                                             random_state=SEED, n_jobs=2, verbose=-1),
                              "100 trees, lr=0.1, 31 leaves, min_child_samples=20")
    return models


def load():
    tr = pd.read_parquet(REPO_ROOT / "data" / "features_train.parquet")
    te = pd.read_parquet(REPO_ROOT / "data" / "features_student_test.parquet")
    inv = pd.read_csv(RES / "trip_inventory.csv", dtype={"trip_id": str})
    bad = inv[(inv.partition == "train") & (inv.label_dist_median_m > BAD_LABEL_MEDIAN_M)].trip_id.tolist()
    tr = tr[~tr.trip_id.isin(bad)]
    tr = tr[tr.point_idx % POINT_STRIDE == 0].reset_index(drop=True)
    npts = inv[inv.partition == "student_test"].set_index("trip_id")["n_points"].to_dict()
    return tr, te, bad, npts


def main():
    tr, te, bad, npts = load()
    print("excluded train trips (bad reference):", bad)
    print("train rows", len(tr), "trips", tr.trip_id.nunique(), "| test rows", len(te))
    Xtr, ytr, gtr = tr[FEATURES].to_numpy(), tr["y"].to_numpy(), tr["trip_id"].to_numpy()
    Xte, yte = te[FEATURES].to_numpy(), te["y"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    rows, fold_rows = [], []
    for name, (family, model, cfg) in make_models().items():
        t0 = time.time()
        fold_f1, fold_ll, fold_top1 = [], [], []
        for k, (a, b) in enumerate(gkf.split(Xtr, ytr, gtr)):
            m = make_models()[name][1]
            m.fit(Xtr[a], ytr[a])
            p = m.predict_proba(Xtr[b])[:, 1]
            cm = candidate_metrics(ytr[b], p)
            sm, _ = sequence_metrics(tr.iloc[b], p, "train")
            fold_f1.append(cm["macro_f1"]); fold_ll.append(cm["log_loss"]); fold_top1.append(sm["top1_accuracy"])
            fold_rows.append({"model": name, "fold": k, "macro_f1": cm["macro_f1"], "log_loss": cm["log_loss"],
                              "top1_accuracy": sm["top1_accuracy"]})
        cv_time = time.time() - t0
        t1 = time.time()
        model.fit(Xtr, ytr)
        train_time = time.time() - t1
        t2 = time.time()
        p = model.predict_proba(Xte)[:, 1]
        inf_time = time.time() - t2
        cm = candidate_metrics(yte, p)
        sm, per_trip = sequence_metrics(te, p, "student_test", npts)
        per_trip.insert(0, "model", name)
        per_trip.to_csv(RES / f"per_trip_student_test_{name}.csv", index=False)
        ptr = model.predict_proba(Xtr)[:, 1]
        train_f1 = candidate_metrics(ytr, ptr)["macro_f1"]
        joblib.dump(model, MOD / f"{name}.joblib")
        row = {"model": name, "family": family, "config": cfg,
               "cv_macro_f1_mean": np.mean(fold_f1), "cv_macro_f1_std": np.std(fold_f1),
               "cv_log_loss": np.mean(fold_ll), "cv_top1": np.mean(fold_top1), "train_macro_f1": train_f1,
               "test_accuracy": cm["accuracy"], "test_precision": cm["precision"], "test_recall": cm["recall"],
               "test_f1": cm["f1"], "test_macro_f1": cm["macro_f1"], "test_log_loss": cm["log_loss"],
               "test_top1": sm["top1_accuracy"], "test_route_continuity": sm["route_continuity"],
               "test_candidate_miss_rate": sm["candidate_miss_rate"], "test_confusion": json.dumps(cm["confusion"]),
               "cv_time_s": cv_time, "train_time_s": train_time,
               "inference_ms_per_1k_rows": 1000 * inf_time / len(Xte) * 1000}
        rows.append(row)
        print(f"{name:22s} CV F1 {row['cv_macro_f1_mean']:.4f}±{row['cv_macro_f1_std']:.4f} | test F1 {cm['macro_f1']:.4f} "
              f"LL {cm['log_loss']:.4f} top1 {sm['top1_accuracy']:.4f} cont {sm['route_continuity']:.4f} | {cv_time:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RES / "cv_folds.csv", index=False)
    plots(res, pd.DataFrame(fold_rows))


def plots(res, folds):
    FIG.mkdir(parents=True, exist_ok=True)
    names = res.model.tolist()
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [folds[folds.model == n].macro_f1 for n in names]
    try:
        ax.boxplot(data, tick_labels=names)
    except TypeError:  # matplotlib < 3.9
        ax.boxplot(data, labels=names)
    ax.set_ylabel("Macro F1 (5-fold GroupKFold)"); ax.set_title("Cross-validation Macro F1 by model")
    plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(FIG / "cv_macro_f1.png", dpi=150); plt.close()

    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, res.test_top1, w, label="Top-1 accuracy")
    ax.bar(x + w / 2, res.test_macro_f1, w, label="Macro F1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Student-testing trips: Top-1 accuracy and Macro F1"); plt.tight_layout()
    plt.savefig(FIG / "student_test_top1_f1.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, res.test_log_loss); ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Student-testing log loss"); plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(FIG / "student_test_log_loss.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3))
    for ax, (_, r) in zip(axes, res.iterrows()):
        cm = np.array(json.loads(r.test_confusion))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8)
        ax.set_title(r.model, fontsize=9); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("pred"); ax.set_ylabel("true")
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrices.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
