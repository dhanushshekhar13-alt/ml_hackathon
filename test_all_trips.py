"""Send every available trip to the running PathSnap API and print a summary.

Usage:  python test_all_trips.py                       (API on this laptop)
        python test_all_trips.py http://192.168.x.x:8000
Saves results/api_test_all_trips.csv. Uses only the standard library.
"""
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

API = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


print("health:", call("/health"))
trips = call("/trips")["trips"]
print(f"{len(trips)} trips\n")
hdr = f"{'trip':10s} {'points':>6s} {'clf_acc':>8s} {'hmm_acc':>8s} {'clf_cont':>8s} {'hmm_cont':>8s} {'arc_P':>6s} {'arc_R':>6s} {'ms':>6s}  status"
print(hdr); print("-" * len(hdr))
rows, ok = [], 0
for t in trips:
    t0 = time.time()
    try:
        r = call("/predict", {"trip_id": t})
        m = r["metrics"]
        row = {"trip_id": t, "n_points": m["n_points"], "clf_acc": m.get("classifier_point_accuracy"),
               "hmm_acc": m.get("hmm_point_accuracy"), "clf_cont": m["classifier_continuity"],
               "hmm_cont": m["hmm_continuity"], "arc_precision": m.get("route_arc_precision"),
               "arc_recall": m.get("route_arc_recall"), "ms": r["timing_ms"]["total"], "status": "OK"}
        ok += 1
    except Exception as e:  # keep going so one failure doesn't hide the others
        row = {"trip_id": t, "status": f"FAIL: {e}", "ms": round((time.time() - t0) * 1000)}
    rows.append(row)
    f = lambda k: f"{row[k]:.4f}" if isinstance(row.get(k), float) else "-"
    print(f"{t:10s} {row.get('n_points', '-')!s:>6s} {f('clf_acc'):>8s} {f('hmm_acc'):>8s} {f('clf_cont'):>8s} "
          f"{f('hmm_cont'):>8s} {f('arc_precision'):>6s} {f('arc_recall'):>6s} {row['ms']!s:>6s}  {row['status']}")

good = [r for r in rows if r["status"] == "OK"]
if good:
    n = sum(r["n_points"] for r in good)
    w = lambda k: sum(r[k] * r["n_points"] for r in good if r[k] is not None) / n
    print("-" * len(hdr))
    print(f"{'ALL':10s} {n:>6d} {w('clf_acc'):8.4f} {w('hmm_acc'):8.4f} {w('clf_cont'):8.4f} {w('hmm_cont'):8.4f}")
print(f"\n{ok}/{len(trips)} trips succeeded")
out = Path(__file__).resolve().parent / "results" / "api_test_all_trips.csv"
with open(out, "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
    wr.writeheader(); wr.writerows(rows)
print("saved", out)
