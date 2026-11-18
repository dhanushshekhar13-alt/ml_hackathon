"""PathSnap FastAPI backend (Laptop 1).

Run from the repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables:
    PATHSNAP_EVAL_DIR  folder whose sub-folders are the trips to serve (default: ../Student folder/student_test)
    PATHSNAP_MODEL     model file name in models/ (default: value in models/selected_model.json)

Reference routes are read only on the server to score predictions; they are never returned to the UI.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_io import load_trip  # noqa: E402
from hmm import viterbi_decode  # noqa: E402
from metrics import continuity  # noqa: E402
from pipeline import FEATURES, TripGeometry, add_features, align_reference, generate_candidates  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT.parent / "Student folder" / "student_test")).resolve()
sel_file = ROOT / "models" / "selected_model.json"
MODEL_NAME = os.environ.get("PATHSNAP_MODEL") or (json.loads(sel_file.read_text())["model_file"] if sel_file.exists() else "HistGradientBoosting.joblib")
MODEL = joblib.load(ROOT / "models" / MODEL_NAME)
log.info("Loaded model %s | serving trips from %s", MODEL_NAME, EVAL_DIR)

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def available_trips():
    if not EVAL_DIR.exists():
        return []
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def arc_polylines(g, arc_seq):
    seq = [int(a) for i, a in enumerate(arc_seq) if a >= 0 and (i == 0 or a != arc_seq[i - 1])]
    lines = []
    for a in seq:
        f, t = g.arcs[a]
        lines.append([[float(g.nodes_ll[f, 1]), float(g.nodes_ll[f, 0])], [float(g.nodes_ll[t, 1]), float(g.nodes_ll[t, 0])]])
    return lines


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "n_trips": len(available_trips())}


@app.get("/trips")
def trips():
    return {"trips": available_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    tid = req.trip_id.strip()
    if tid not in available_trips():
        raise HTTPException(404, f"unknown trip {tid}")
    log.info("REQUEST /predict trip_id=%s", tid)

    # ---------- prediction pipeline (no reference information used) ----------
    trip = load_trip(tid, EVAL_DIR.name, data_dir=EVAL_DIR.parent, with_route=False)
    g = TripGeometry(trip)
    cand = add_features(g, generate_candidates(g))
    n = len(g.P)
    clf_arcs = np.full(n, -1, dtype=np.int64)
    hmm_arcs = np.full(n, -1, dtype=np.int64)
    if not cand.empty:
        p = MODEL.predict_proba(cand[FEATURES].to_numpy())[:, 1]
        best = cand.assign(p=p).groupby("point_idx")["p"].idxmax()
        clf_arcs[cand.loc[best, "point_idx"].to_numpy()] = cand.loc[best, "arc_id"].to_numpy()
        hmm_arcs = viterbi_decode(g, cand, p)
    t_pred = time.time() - t0

    # ---------- server-side evaluation against the reference route ----------
    metrics = {"n_points": n, "points_with_candidates": int(cand["point_idx"].nunique()) if not cand.empty else 0,
               "classifier_continuity": continuity(clf_arcs, g.arcs), "hmm_continuity": continuity(hmm_arcs, g.arcs)}
    ref_route = load_trip(tid, EVAL_DIR.name, data_dir=EVAL_DIR.parent, with_route=True).get("route")
    if ref_route is not None:
        label_arc, _ = align_reference(g, ref_route)
        metrics["classifier_point_accuracy"] = float((clf_arcs == label_arc).mean())
        metrics["hmm_point_accuracy"] = float((hmm_arcs == label_arc).mean())
        pred_set, ref_set = set(hmm_arcs[hmm_arcs >= 0].tolist()), set(ref_route.tolist())
        inter = len(pred_set & ref_set)
        metrics["route_arc_precision"] = inter / max(len(pred_set), 1)
        metrics["route_arc_recall"] = inter / max(len(ref_set), 1)
    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}

    step = max(1, n // 1500)
    resp = {
        "trip_id": tid, "model": MODEL_NAME,
        "gps": [[float(lat), float(lon)] for lon, lat in g.track[::step, :2]],
        "predicted_route": arc_polylines(g, hmm_arcs),
        "classifier_route": arc_polylines(g, clf_arcs),
        "metrics": metrics,
        "timing_ms": {"prediction": round(t_pred * 1000), "total": round((time.time() - t0) * 1000)},
    }
    log.info("RESULT trip_id=%s points=%d hmm_acc=%s clf_acc=%s hmm_cont=%s time_ms=%d", tid, n,
             metrics.get("hmm_point_accuracy"), metrics.get("classifier_point_accuracy"),
             metrics.get("hmm_continuity"), resp["timing_ms"]["total"])
    return resp
