"""Build candidate/feature tables for the train and student_test partitions.

Outputs:
  data/features_train.parquet, data/features_student_test.parquet
  results/trip_inventory.csv  (per-trip counts, gaps, label quality, candidate miss rate)
"""
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data_io import REPO_ROOT, list_trips
from pipeline import build_trip_table

OUT = REPO_ROOT / "data"
RES = REPO_ROOT / "results"


def work(args):
    trip_id, part = args
    t0 = time.time()
    g, cand, info = build_trip_table(trip_id, part)
    dt = np.diff(g.t)
    n = info["n_points"]
    hit = cand.groupby("point_idx")["y"].max() if not cand.empty else pd.Series(dtype=int)
    inv = {
        "trip_id": trip_id, "partition": part, "n_points": n, "n_dropped_rows": info["n_dropped"],
        "n_nodes": len(g.N), "n_arcs": len(g.arcs), "route_len_arcs": len(g.route),
        "duration_s": float(g.t[-1] - g.t[0]), "dt_eq_1s_frac": float(np.mean(dt == 1)) if len(dt) else np.nan,
        "max_gap_s": float(dt.max()) if len(dt) else 0.0, "n_gaps_gt_5s": int((dt > 5).sum()),
        "path_length_km": float(np.hypot(*np.diff(g.P, axis=0).T).sum() / 1000),
        "lon_min": g.track[:, 0].min(), "lon_max": g.track[:, 0].max(),
        "lat_min": g.track[:, 1].min(), "lat_max": g.track[:, 1].max(),
        "label_dist_median_m": float(np.median(info["label_dist"])),
        "label_dist_p95_m": float(np.percentile(info["label_dist"], 95)),
        "points_without_candidates": int(n - cand["point_idx"].nunique()) if not cand.empty else n,
        "candidate_miss_rate": float(1 - hit.sum() / n),
        "mean_candidates": float(cand.groupby("point_idx").size().mean()) if not cand.empty else 0,
        "build_s": round(time.time() - t0, 2),
    }
    return cand, inv


def main():
    OUT.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    invs = []
    for part in ["train", "student_test"]:
        jobs = [(t, part) for t in list_trips(part)]
        with Pool(2) as pool:
            res = pool.map(work, jobs, chunksize=1)
        tables = [c for c, _ in res if not c.empty]
        invs += [i for _, i in res]
        df = pd.concat(tables, ignore_index=True)
        df.to_parquet(OUT / f"features_{part}.parquet", index=False)
        print(part, "rows", len(df), "pos rate", round(df.y.mean(), 4))
    inv = pd.DataFrame(invs)
    inv.to_csv(RES / "trip_inventory.csv", index=False)
    print(inv.groupby("partition")[["n_points", "candidate_miss_rate", "label_dist_median_m", "max_gap_s"]].describe().T)


if __name__ == "__main__":
    main()
