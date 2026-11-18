"""Dataset exploration: summary tables, plots and a Folium map.

Run after build_dataset.py (uses results/trip_inventory.csv).
Outputs to reports/figures/ and reports/maps/.
"""
import sys

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from data_io import DATA_DIR, REPO_ROOT, load_trip
from pipeline import TripGeometry, align_reference

FIG, MAPS, RES = REPO_ROOT / "reports" / "figures", REPO_ROOT / "reports" / "maps", REPO_ROOT / "results"


def network_stats(inv):
    rows = []
    for _, r in inv.iterrows():
        trip = load_trip(r.trip_id, r.partition, with_route=False)
        G = nx.DiGraph()
        G.add_nodes_from(range(len(trip["nodes"])))
        G.add_edges_from(map(tuple, trip["arcs"]))
        wcc = list(nx.weakly_connected_components(G))
        rows.append({"trip_id": r.trip_id, "partition": r.partition, "n_nodes": G.number_of_nodes(),
                     "n_arcs": G.number_of_edges(), "isolated_nodes": nx.number_of_isolates(G),
                     "weak_components": len(wcc), "largest_wcc_frac": max(map(len, wcc)) / G.number_of_nodes(),
                     "strong_components": nx.number_strongly_connected_components(G)})
    return pd.DataFrame(rows)


def plots(inv):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for part, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
        d = inv[inv.partition == part]
        axes[0].hist(d.n_points, bins=30, alpha=0.6, label=part, color=c)
        axes[1].hist(d.label_dist_median_m.clip(upper=30), bins=30, alpha=0.6, label=part, color=c)
        axes[2].hist(np.log10(d.n_arcs), bins=30, alpha=0.6, label=part, color=c)
    axes[0].set_xlabel("GPS observations per trip"); axes[1].set_xlabel("median GPS-to-reference-arc distance (m, clipped 30)")
    axes[2].set_xlabel("log10(#arcs in trip network)")
    for a in axes:
        a.legend(); a.set_ylabel("trips")
    plt.tight_layout(); plt.savefig(FIG / "eda_distributions.png", dpi=150); plt.close()


def sampling_plot(inv):
    gaps = []
    for _, r in inv.iterrows():
        t = np.loadtxt(DATA_DIR / r.partition / r.trip_id / f"{r.trip_id}.track", usecols=2)
        gaps.append(np.diff(np.sort(t)))
    g = np.concatenate(gaps)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(np.clip(g, 0, 15), bins=np.arange(0, 16.5, 1))
    ax.set_yscale("log"); ax.set_xlabel("time between consecutive GPS points (s, clipped 15)"); ax.set_ylabel("count (log)")
    ax.set_title(f"Sampling intervals: {np.mean(g == 1) * 100:.1f}% exactly 1 s")
    plt.tight_layout(); plt.savefig(FIG / "eda_sampling_intervals.png", dpi=150); plt.close()
    return pd.Series(g).describe()


def trip_map(trip_id, partition="train", pred_arcs=None, out=None):
    """GPS points, nearby road network, reference route and (optionally) predicted route."""
    trip = load_trip(trip_id, partition)
    g = TripGeometry(trip)
    ll = g.track[:, :2]
    m = folium.Map(location=[ll[:, 1].mean(), ll[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
    # road network near the track
    lo, hi = g.P.min(0) - 150, g.P.max(0) + 150
    mid = (g.A + g.B) / 2
    near = np.where((mid[:, 0] > lo[0]) & (mid[:, 0] < hi[0]) & (mid[:, 1] > lo[1]) & (mid[:, 1] < hi[1]))[0]
    fg_net = folium.FeatureGroup(name="Road network (near track)")
    for a in near[:6000]:
        f, t = g.arcs[a]
        folium.PolyLine([[g.nodes_ll[f, 1], g.nodes_ll[f, 0]], [g.nodes_ll[t, 1], g.nodes_ll[t, 0]]],
                        color="#9e9e9e", weight=1, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)

    def arcs_line(arc_seq, name, color, weight):
        fg = folium.FeatureGroup(name=name)
        seq = [a for i, a in enumerate(arc_seq) if a >= 0 and (i == 0 or a != arc_seq[i - 1])]
        for a in seq:
            f, t = g.arcs[a]
            folium.PolyLine([[g.nodes_ll[f, 1], g.nodes_ll[f, 0]], [g.nodes_ll[t, 1], g.nodes_ll[t, 0]]],
                            color=color, weight=weight, opacity=0.85).add_to(fg)
        fg.add_to(m)

    if "route" in trip:
        arcs_line(list(trip["route"]), "Reference route", "#2e7d32", 6)
    if pred_arcs is not None:
        arcs_line(list(pred_arcs), "Predicted route", "#d32f2f", 3)
    fg_gps = folium.FeatureGroup(name="Raw GPS observations")
    step = max(1, len(ll) // 1500)
    for lon, lat in ll[::step]:
        folium.CircleMarker([lat, lon], radius=2, color="#1565c0", fill=True, fill_opacity=0.8, weight=0).add_to(fg_gps)
    fg_gps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ('<div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background:white; padding:8px;'
              'border:1px solid #888; font-size:13px">'
              f'<b>Trip {trip_id} ({partition})</b><br><span style="color:#1565c0">&#9679;</span> GPS points<br>'
              '<span style="color:#2e7d32">&#9473;</span> Reference route<br>'
              '<span style="color:#d32f2f">&#9473;</span> Predicted route<br>'
              '<span style="color:#9e9e9e">&#9473;</span> Road network</div>')
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[ll[:, 1].min(), ll[:, 0].min()], [ll[:, 1].max(), ll[:, 0].max()]])
    MAPS.mkdir(parents=True, exist_ok=True)
    out = out or MAPS / f"trip_{trip_id}_{partition}.html"
    m.save(str(out))
    return out


def main():
    inv = pd.read_csv(RES / "trip_inventory.csv", dtype={"trip_id": str})
    inv["trip_id"] = inv.trip_id.str.zfill(8)
    net = network_stats(inv)
    net.to_csv(RES / "network_stats.csv", index=False)
    plots(inv)
    gap_desc = sampling_plot(inv)
    summary = inv.groupby("partition").agg(trips=("trip_id", "count"), obs_total=("n_points", "sum"),
                                           obs_median=("n_points", "median"), dropped_rows=("n_dropped_rows", "sum"),
                                           trips_with_gap_gt_5s=("n_gaps_gt_5s", lambda s: int((s > 0).sum())),
                                           median_label_dist_m=("label_dist_median_m", "median"),
                                           mean_candidate_miss=("candidate_miss_rate", "mean"))
    summary.to_csv(RES / "eda_summary.csv")
    print(summary.to_string())
    print("\nSampling interval stats (s):\n", gap_desc.to_string())
    print("\nNetwork connectivity:\n", net.describe().T[["mean", "min", "max"]].to_string())
    for tid, part in [("00000005", "train"), ("00000007", "student_test")]:
        print("map:", trip_map(tid, part))


if __name__ == "__main__":
    main()
