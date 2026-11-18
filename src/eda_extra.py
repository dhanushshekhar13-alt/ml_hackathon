"""Extra EDA for the report: missing-value analysis per file type and static trajectory/road-network plots.

Outputs: results/missing_values.csv, reports/figures/trip_<id>_<partition>_layers.png
Partitions: missing values = train + student_test; plots = one train trip and one student-test trip.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_io import DATA_DIR, REPO_ROOT, list_trips, load_trip
from pipeline import TripGeometry

RES, FIG = REPO_ROOT / "results", REPO_ROOT / "reports" / "figures"
EXT = {"track": 3, "nodes": 2, "arcs": 2, "route": 1}


def missing_values():
    rows = []
    for part in ["train", "student_test"]:
        stats = {e: {"rows": 0, "nan_or_unparsable": 0, "wrong_column_count": 0} for e in EXT}
        extra = {"track_duplicate_time": 0, "track_non_increasing_time": 0, "track_out_of_range_coord": 0,
                 "arcs_bad_node_index": 0, "route_bad_arc_index": 0, "trips": 0}
        for tid in list_trips(part):
            extra["trips"] += 1
            d = DATA_DIR / part / tid
            for e, ncol in EXT.items():
                for line in open(d / f"{tid}.{e}"):
                    parts = line.split()
                    if not parts:
                        continue
                    stats[e]["rows"] += 1
                    if len(parts) != ncol:
                        stats[e]["wrong_column_count"] += 1
                    try:
                        if not np.all(np.isfinite(np.array(parts, dtype=float))):
                            stats[e]["nan_or_unparsable"] += 1
                    except ValueError:
                        stats[e]["nan_or_unparsable"] += 1
            trip = load_trip(tid, part)
            t = trip["track"]
            dt = np.diff(t[:, 2])
            extra["track_duplicate_time"] += int((dt == 0).sum())
            extra["track_non_increasing_time"] += int((dt < 0).sum())
            extra["track_out_of_range_coord"] += int(((np.abs(t[:, 0]) > 180) | (np.abs(t[:, 1]) > 90)).sum())
            extra["arcs_bad_node_index"] += int(((trip["arcs"] < 0) | (trip["arcs"] >= len(trip["nodes"]))).any(axis=1).sum())
            extra["route_bad_arc_index"] += int(((trip["route"] < 0) | (trip["route"] >= len(trip["arcs"]))).sum())
        for e in EXT:
            rows.append({"partition": part, "check": f"{e}: rows", "count": stats[e]["rows"]})
            rows.append({"partition": part, "check": f"{e}: NaN / unparsable values", "count": stats[e]["nan_or_unparsable"]})
            rows.append({"partition": part, "check": f"{e}: wrong column count", "count": stats[e]["wrong_column_count"]})
        for k, v in extra.items():
            rows.append({"partition": part, "check": k.replace("_", " "), "count": v})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "missing_values.csv", index=False)
    print(df.pivot(index="check", columns="partition", values="count").to_string())


def layers_plot(tid, part):
    trip = load_trip(tid, part)
    g = TripGeometry(trip)
    lo, hi = g.P.min(0) - 150, g.P.max(0) + 150
    mid = (g.A + g.B) / 2
    near = np.where((mid[:, 0] > lo[0]) & (mid[:, 0] < hi[0]) & (mid[:, 1] > lo[1]) & (mid[:, 1] < hi[1]))[0]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for a in near:
        ax.plot([g.A[a, 0], g.B[a, 0]], [g.A[a, 1], g.B[a, 1]], color="#bdbdbd", lw=0.6, zorder=1)
    for a in g.route:
        ax.plot([g.A[a, 0], g.B[a, 0]], [g.A[a, 1], g.B[a, 1]], color="#2e7d32", lw=3, alpha=0.8, zorder=2)
    ax.scatter(g.P[:, 0], g.P[:, 1], s=3, color="#1565c0", zorder=3)
    ax.plot([], [], color="#bdbdbd", label=f"road network ({len(near)} arcs shown of {len(g.arcs)})")
    ax.plot([], [], color="#2e7d32", lw=3, label=f"reference route ({len(g.route)} arcs)")
    ax.scatter([], [], s=12, color="#1565c0", label=f"GPS observations ({len(g.P)})")
    ax.set_aspect("equal"); ax.set_xlabel("x (m, local projection)"); ax.set_ylabel("y (m)")
    ax.legend(loc="best", fontsize=8); ax.set_title(f"Trip {tid} ({part}): GPS trajectory over the road network")
    plt.tight_layout()
    out = FIG / f"trip_{tid}_{part}_layers.png"
    plt.savefig(out, dpi=150); plt.close()
    print("plot:", out)


if __name__ == "__main__":
    missing_values()
    layers_plot("00000005", "train")
    layers_plot("00000076", "student_test")
