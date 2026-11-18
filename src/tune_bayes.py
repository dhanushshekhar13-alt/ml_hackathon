"""Bayesian hyperparameter optimisation (Optuna, TPE sampler) for the chosen model: LightGBM.

Objective  : mean 5-fold GroupKFold log loss on the training trips (groups = trip_id), same rows as train_models.py.
Search     : learning_rate, n_estimators, num_leaves, min_child_samples, reg_lambda, colsample_bytree, subsample.
Trials     : N_TRIALS (default 30), TPE sampler seed 42, median pruning across folds.
Comparison : untuned vs tuned on CV Macro F1, CV log loss, student-test Top-1 / Macro F1 / log loss, inference time.
The student-testing trips are used only for the final comparison, never inside the objective.

Outputs: results/optuna_trials_LightGBM.csv, results/optuna_best_params.json, results/bayes_tuned_vs_untuned.csv,
         models/LightGBM_tuned.joblib, reports/figures/optuna_history.png, optuna_param_importance.png,
         tuned_vs_untuned.png
Usage  : python tune_bayes.py [n_trials]
"""
import json
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from metrics import candidate_metrics, sequence_metrics
from pipeline import FEATURES
from train_models import FIG, MOD, RES, SEED, load

N_TRIALS = 30
UNTUNED = dict(n_estimators=100, learning_rate=0.1, num_leaves=31, min_child_samples=20)
SPACE = {  # documented search space (also written to optuna_best_params.json)
    "learning_rate": "log-uniform [0.01, 0.3]",
    "n_estimators": "int [50, 400]",
    "num_leaves": "int [15, 127]",
    "min_child_samples": "int [5, 200]",
    "reg_lambda": "log-uniform [1e-4, 10]",
    "colsample_bytree": "uniform [0.5, 1.0]",
    "subsample": "uniform [0.5, 1.0] (bagging every iteration)",
}


def make(params):
    p = dict(params)
    if p.get("subsample", 1.0) < 1.0:
        p["subsample_freq"] = 1
    return LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **p)


def cv_scores(params, X, y, groups, trial=None):
    lls, f1s = [], []
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        m = make(params).fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        c = candidate_metrics(y[b], p)
        lls.append(c["log_loss"]); f1s.append(c["macro_f1"])
        if trial is not None:
            trial.report(float(np.mean(lls)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(lls)), float(np.mean(f1s)), float(np.std(f1s))


def evaluate(name, params, Xtr, ytr, gtr, Xte, yte, te, npts):
    cv_ll, cv_f1, cv_f1_std = cv_scores(params, Xtr, ytr, gtr)
    t0 = time.time(); model = make(params).fit(Xtr, ytr); fit_s = time.time() - t0
    t1 = time.time(); p = model.predict_proba(Xte)[:, 1]; inf_s = time.time() - t1
    c = candidate_metrics(yte, p)
    s, _ = sequence_metrics(te, p, "student_test", npts)
    row = {"model": name, "params": json.dumps(params), "cv_macro_f1": cv_f1, "cv_macro_f1_std": cv_f1_std,
           "cv_log_loss": cv_ll, "test_top1": s["top1_accuracy"], "test_macro_f1": c["macro_f1"],
           "test_log_loss": c["log_loss"], "test_route_continuity": s["route_continuity"], "fit_time_s": fit_s,
           "inference_ms_per_1k_rows": 1000 * inf_s / len(Xte) * 1000}
    return model, row


def main():
    global N_TRIALS
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        N_TRIALS = int(sys.argv[1])
    tr, te, bad, npts = load()
    Xtr, ytr, gtr = tr[FEATURES].to_numpy(), tr["y"].to_numpy(), tr["trip_id"].to_numpy()
    Xte, yte = te[FEATURES].to_numpy(), te["y"].to_numpy()
    print(f"Optuna TPE, {N_TRIALS} trials, objective = 5-fold GroupKFold log loss on {tr.trip_id.nunique()} train trips")

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        ll, f1, _ = cv_scores(params, Xtr, ytr, gtr, trial)
        trial.set_user_attr("cv_macro_f1", f1)
        return ll

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    # trial 0 = the untuned configuration, so the search starts from the baseline
    study.enqueue_trial({**UNTUNED, "reg_lambda": 1e-4, "colsample_bytree": 1.0, "subsample": 1.0})
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS,
                   callbacks=[lambda s, t: print(f"trial {t.number:3d} {t.state.name:8s} "
                                                 f"LL {t.value if t.value is not None else float('nan'):.5f} "
                                                 f"best {s.best_value:.5f}", flush=True)])
    search_s = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(RES / "optuna_trials_LightGBM.csv", index=False)
    best = study.best_params
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    (RES / "optuna_best_params.json").write_text(json.dumps(
        {"model": "LightGBM", "objective": "mean 5-fold GroupKFold log loss (train trips)", "sampler": "TPE (seed 42)",
         "pruner": "MedianPruner", "n_trials": N_TRIALS, "n_complete": n_complete, "search_seconds": round(search_s, 1),
         "search_space": SPACE, "best_params": best, "best_cv_log_loss": study.best_value}, indent=2))
    print(f"\nsearch done in {search_s:.0f}s, best CV log loss {study.best_value:.5f}\nbest params {best}")

    _, r_un = evaluate("LightGBM (untuned)", UNTUNED, Xtr, ytr, gtr, Xte, yte, te, npts)
    m_tu, r_tu = evaluate("LightGBM (tuned)", best, Xtr, ytr, gtr, Xte, yte, te, npts)
    joblib.dump(m_tu, MOD / "LightGBM_tuned.joblib")
    comp = pd.DataFrame([r_un, r_tu])
    comp.to_csv(RES / "bayes_tuned_vs_untuned.csv", index=False)
    print(comp.drop(columns=["params"]).round(4).to_string(index=False))
    plots(study, comp)


def plots(study, comp):
    FIG.mkdir(parents=True, exist_ok=True)
    vals = [t.value for t in study.trials if t.value is not None]
    nums = [t.number for t in study.trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(nums, vals, s=18, color="#9e9e9e", label="completed trial")
    ax.plot(nums, np.minimum.accumulate(vals), color="#1565c0", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("CV log loss (lower is better)"); ax.legend()
    ax.set_title("Optuna TPE search for LightGBM"); plt.tight_layout()
    plt.savefig(FIG / "optuna_history.png", dpi=150); plt.close()
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#1565c0")
        ax.set_xlabel("importance for CV log loss (fANOVA)"); ax.set_title("Hyperparameter importance")
        plt.tight_layout(); plt.savefig(FIG / "optuna_param_importance.png", dpi=150); plt.close()
    except Exception as e:  # importance needs enough completed trials
        print("param importance skipped:", e)
    metrics = [("cv_macro_f1", "CV Macro F1"), ("test_top1", "Test Top-1"), ("test_macro_f1", "Test Macro F1")]
    x = np.arange(len(metrics)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (_, r) in enumerate(comp.iterrows()):
        axes[0].bar(x + (i - 0.5) * w, [r[m] for m, _ in metrics], w, label=r.model, color=["#9e9e9e", "#1565c0"][i])
        axes[1].bar(i, r.test_log_loss, color=["#9e9e9e", "#1565c0"][i])
    axes[0].set_xticks(x); axes[0].set_xticklabels([l for _, l in metrics]); axes[0].set_ylim(0.8, 0.95); axes[0].legend()
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(comp.model); axes[1].set_ylabel("Student-test log loss")
    fig.suptitle("Untuned vs tuned LightGBM"); plt.tight_layout()
    plt.savefig(FIG / "tuned_vs_untuned.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
