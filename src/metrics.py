"""Candidate-level and observation/sequence-level metrics."""
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from data_io import load_trip


@lru_cache(maxsize=256)
def arcs_of(trip_id, partition):
    return load_trip(trip_id, partition, with_route=False)["arcs"]


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def continuity(arc_seq, arcs):
    """Fraction of consecutive predicted arcs that are identical or directly connected."""
    a = np.asarray(arc_seq)
    a = a[a >= 0]
    if len(a) < 2:
        return np.nan
    x, y = a[:-1], a[1:]
    ok = (x == y) | (arcs[x, 1] == arcs[y, 0])
    return float(ok.mean())


def top1_predictions(df, p):
    """Pick the highest-scoring candidate per observation."""
    d = df[["trip_id", "point_idx", "arc_id", "y"]].copy()
    d["p"] = p
    idx = d.groupby(["trip_id", "point_idx"])["p"].idxmax()
    return d.loc[idx].sort_values(["trip_id", "point_idx"])


def sequence_metrics(df, p, partition, n_points_by_trip=None):
    """Observation-level Top-1 accuracy, candidate miss rate and route continuity."""
    best = top1_predictions(df, p)
    hit = df.groupby(["trip_id", "point_idx"])["y"].max()
    n_obs = len(hit) if n_points_by_trip is None else int(sum(n_points_by_trip.values()))
    out = {
        "top1_accuracy": float(best["y"].sum() / n_obs),
        "candidate_miss_rate": float(1 - hit.sum() / n_obs),
    }
    conts, per_trip = [], []
    for tid, g in best.groupby("trip_id"):
        c = continuity(g["arc_id"].to_numpy(), arcs_of(tid, partition))
        conts.append(c)
        per_trip.append({"trip_id": tid, "top1": g["y"].mean(), "continuity": c})
    out["route_continuity"] = float(np.nanmean(conts))
    return out, pd.DataFrame(per_trip)
