"""Candidate generation, feature engineering and reference-label alignment.

One row = (GPS observation, candidate road arc). Target y = 1 if the candidate is the
reference arc assigned to that observation, else 0.
"""
import numpy as np
import pandas as pd

from data_io import LocalProjection, clean_track, load_trip

RADIUS_M = 60.0      # candidate search radius
MAX_CANDS = 8        # keep the k nearest arcs inside the radius
HEADING_WIN = 2      # +/- points used to estimate GPS heading
CHUNK = 128

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "heading_diff", "heading_valid",
    "speed", "proj_frac", "arc_len", "n_cands", "reverse_in_cands", "out_deg", "in_deg",
]


class TripGeometry:
    """Projected road network + track for one trip."""

    def __init__(self, trip):
        track, self.n_dropped = clean_track(trip["track"])
        self.trip_id = trip["trip_id"]
        self.track = track
        self.proj = LocalProjection(track[:, 0].mean(), track[:, 1].mean())
        self.P = self.proj.to_xy(track[:, 0], track[:, 1])
        self.t = track[:, 2]
        self.nodes_ll = trip["nodes"]
        self.N = self.proj.to_xy(trip["nodes"][:, 0], trip["nodes"][:, 1])
        self.arcs = trip["arcs"]
        self.A = self.N[self.arcs[:, 0]]
        self.B = self.N[self.arcs[:, 1]]
        self.AB = self.B - self.A
        self.len = np.hypot(self.AB[:, 0], self.AB[:, 1])
        n_nodes = len(self.N)
        self.out_deg = np.bincount(self.arcs[:, 0], minlength=n_nodes)
        self.in_deg = np.bincount(self.arcs[:, 1], minlength=n_nodes)
        # reverse twin lookup
        key = {(int(a), int(b)): i for i, (a, b) in enumerate(self.arcs)}
        self.reverse = np.array([key.get((int(b), int(a)), -1) for a, b in self.arcs])
        self.route = trip.get("route")

    # ---------- geometry helpers ----------
    def point_arc_dist(self, P, idx):
        """Distances from points P (n,2) to arcs idx (m,) -> (n,m) dist, (n,m) projection fraction."""
        A, AB, L2 = self.A[idx], self.AB[idx], np.maximum(self.len[idx] ** 2, 1e-9)
        PA = P[:, None, :] - A[None, :, :]
        t = np.clip((PA * AB[None]).sum(-1) / L2[None], 0.0, 1.0)
        proj = A[None] + t[..., None] * AB[None]
        d = np.hypot(P[:, None, 0] - proj[..., 0], P[:, None, 1] - proj[..., 1])
        return d, t

    def gps_heading(self):
        n = len(self.P)
        i0 = np.clip(np.arange(n) - HEADING_WIN, 0, n - 1)
        i1 = np.clip(np.arange(n) + HEADING_WIN, 0, n - 1)
        v = self.P[i1] - self.P[i0]
        dt = np.maximum(self.t[i1] - self.t[i0], 1e-6)
        disp = np.hypot(v[:, 0], v[:, 1])
        return v, disp / dt, disp


def generate_candidates(g: TripGeometry, radius=RADIUS_M, k=MAX_CANDS):
    """Return DataFrame of (point_idx, arc_id, dist, proj_frac) for k nearest arcs within radius."""
    rows = []
    amin = np.minimum(g.A, g.B)
    amax = np.maximum(g.A, g.B)
    for s in range(0, len(g.P), CHUNK):
        P = g.P[s:s + CHUNK]
        lo, hi = P.min(0) - radius, P.max(0) + radius
        sel = np.where((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) & (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))[0]
        if len(sel) == 0:
            continue
        d, t = g.point_arc_dist(P, sel)
        for r in range(len(P)):
            within = np.where(d[r] <= radius)[0]
            if len(within) == 0:
                continue
            order = within[np.argsort(d[r, within], kind="stable")][:k]
            for rank, c in enumerate(order):
                rows.append((s + r, int(sel[c]), float(d[r, c]), float(t[r, c]), rank))
    return pd.DataFrame(rows, columns=["point_idx", "arc_id", "dist", "proj_frac", "dist_rank"])


def add_features(g: TripGeometry, cand: pd.DataFrame) -> pd.DataFrame:
    if cand.empty:
        return cand
    v, speed, disp = g.gps_heading()
    pi, ai = cand["point_idx"].to_numpy(), cand["arc_id"].to_numpy()
    grp = cand.groupby("point_idx")["dist"]
    cand["n_cands"] = grp.transform("size")
    dmin = grp.transform("min")
    cand["dist_minus_min"] = cand["dist"] - dmin
    cand["dist_ratio"] = cand["dist"] / (dmin + 1.0)
    ab = g.AB[ai]
    vv = v[pi]
    cosang = (ab * vv).sum(1) / (np.hypot(ab[:, 0], ab[:, 1]) * np.hypot(vv[:, 0], vv[:, 1]) + 1e-9)
    hd = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    valid = disp[pi] > 3.0  # heading unreliable when the vehicle is (almost) stationary
    cand["heading_diff"] = np.where(valid, hd, 90.0)
    cand["heading_valid"] = valid.astype(int)
    cand["speed"] = speed[pi]
    cand["arc_len"] = g.len[ai]
    rev = g.reverse[ai]
    pairs = set(zip(pi.tolist(), ai.tolist()))
    cand["reverse_in_cands"] = [int(r >= 0 and (p, int(r)) in pairs) for p, r in zip(pi.tolist(), rev.tolist())]
    cand["out_deg"] = g.out_deg[g.arcs[ai, 1]]
    cand["in_deg"] = g.in_deg[g.arcs[ai, 0]]
    cand["t"] = g.t[pi]
    return cand


def align_reference(g: TripGeometry, route=None):
    """Assign every GPS point to one arc of the reference route, moving forward only.

    Dynamic programming over (point i, route position j) with non-decreasing j minimising the
    summed point-to-arc distance. Returns (label_arc per point, label_dist per point).
    """
    route = g.route if route is None else route
    if route is None or len(route) == 0:
        return None, None
    n, m = len(g.P), len(route)
    dtype = np.int16 if m < 32000 else np.int32
    back = np.empty((n, m), dtype=dtype)
    prev = None
    dists = np.empty((n, m), dtype=np.float32) if n * m < 3e7 else None
    for i in range(n):
        d, _ = g.point_arc_dist(g.P[i:i + 1], route)
        c = np.minimum(d[0], 500.0)
        if dists is not None:
            dists[i] = c
        if prev is None:
            cur = c
            back[i] = np.arange(m)
        else:
            # prefix minimum of prev with argmin
            pm = np.minimum.accumulate(prev)
            idx = np.arange(m)
            is_new = np.r_[True, prev[1:] < pm[:-1]]
            arg = np.maximum.accumulate(np.where(is_new, idx, 0))
            cur = c + pm
            back[i] = arg
        prev = cur
    j = int(np.argmin(prev))
    pos = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j = int(back[i, j])
    label_arc = route[pos]
    if dists is not None:
        label_dist = dists[np.arange(n), pos]
    else:
        label_dist = np.array([g.point_arc_dist(g.P[i:i + 1], route[pos[i]:pos[i] + 1])[0][0, 0] for i in range(n)])
    return label_arc, label_dist


def build_trip_table(trip_id, partition, with_labels=True, data_dir=None):
    kw = {} if data_dir is None else {"data_dir": data_dir}
    trip = load_trip(trip_id, partition, with_route=with_labels, **kw)
    g = TripGeometry(trip)
    cand = add_features(g, generate_candidates(g))
    cand.insert(0, "trip_id", trip_id)
    info = {"trip_id": trip_id, "n_points": len(g.P), "n_dropped": g.n_dropped}
    if with_labels and g.route is not None:
        label_arc, label_dist = align_reference(g)
        if not cand.empty:
            cand["y"] = (cand["arc_id"].to_numpy() == label_arc[cand["point_idx"].to_numpy()]).astype(int)
        info["label_arc"] = label_arc
        info["label_dist"] = label_dist
    return g, cand, info
