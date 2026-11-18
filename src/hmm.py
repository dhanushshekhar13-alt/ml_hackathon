"""HMM + Viterbi route reconstruction on top of classifier candidate probabilities.

Hidden state at observation i = one of its candidate arcs (state <-> arc_id kept explicitly).
Emission   log P(z_i | arc) = log(p_classifier + eps)
Transition log P(arc_b | arc_a) = -|d_route - d_gps| / beta
    d_gps   = straight-line distance between the two GPS observations
    d_route = network distance from the projection on arc_a to the projection on arc_b
              (same arc moving forward -> along-arc distance; otherwise via Dijkstra between nodes)
Unreachable pairs get a large penalty instead of -inf so decoding never dead-ends.
Large time gaps (dt > GAP_S) or observations without candidates split the trajectory into
independently decoded segments.
"""
import heapq

import numpy as np

EPS = 1e-6
BETA = 15.0            # metres; tolerance on route-vs-GPS distance mismatch
UNREACHABLE = -50.0    # log-penalty for disconnected candidate pairs
GAP_S = 30.0           # break the chain when GPS gap is larger than this
BACKWARD_PEN = -8.0    # log-penalty for moving backwards on the same arc


class Router:
    """Cached bounded Dijkstra on the trip's directed road graph."""

    def __init__(self, g):
        self.g = g
        n = len(g.N)
        order = np.argsort(g.arcs[:, 0], kind="stable")
        self.adj_to = g.arcs[order, 1]
        self.adj_w = g.len[order]
        self.start = np.searchsorted(g.arcs[order, 0], np.arange(n + 1))
        self.cache = {}

    def dists_from(self, src, cutoff):
        key = (src, cutoff)
        if key in self.cache:
            return self.cache[key]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for k in range(self.start[u], self.start[u + 1]):
                v, nd = int(self.adj_to[k]), d + float(self.adj_w[k])
                if nd < dist.get(v, np.inf) and nd <= cutoff:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self.cache[key] = dist
        return dist




def viterbi_decode(g, cand, prob, beta=BETA):
    """cand: DataFrame for ONE trip with point_idx, arc_id, proj_frac. Returns array arc per GPS point (-1 = none)."""
    router = Router(g)
    out = np.full(len(g.P), -1, dtype=np.int64)
    cand = cand.assign(p=prob)
    groups = {pi: d for pi, d in cand.groupby("point_idx", sort=True)}
    pts = np.array(sorted(groups))
    if len(pts) == 0:
        return out
    # split where a point has no candidates (gap in pts) AND the time gap is large, or time gap large
    breaks = [0]
    for k in range(1, len(pts)):
        if (g.t[pts[k]] - g.t[pts[k - 1]]) > GAP_S:
            breaks.append(k)
    breaks.append(len(pts))
    for s, e in zip(breaks[:-1], breaks[1:]):
        _decode_segment(g, router, [groups[p] for p in pts[s:e]], pts[s:e], out, beta)
    return out


def _decode_segment(g, router, states, pts, out, beta):
    arcs0 = states[0]["arc_id"].to_numpy()
    score = np.log(states[0]["p"].to_numpy() + EPS)
    backs = []
    for k in range(1, len(states)):
        A, B = states[k - 1], states[k]
        a_arc, a_t = A["arc_id"].to_numpy(), A["proj_frac"].to_numpy()
        b_arc, b_t = B["arc_id"].to_numpy(), B["proj_frac"].to_numpy()
        pa, pb = g.P[pts[k - 1]], g.P[pts[k]]
        d_gps = float(np.hypot(*(pb - pa)))
        dt = max(float(g.t[pts[k]] - g.t[pts[k - 1]]), 1.0)
        cutoff = d_gps * 2 + 100.0 * dt ** 0.5
        bta = beta * dt ** 0.5
        trans = np.full((len(a_arc), len(b_arc)), UNREACHABLE)
        for i, (aa, at) in enumerate(zip(a_arc, a_t)):
            la = g.len[aa]
            rest_a = (1 - at) * la
            reach = router.dists_from(int(g.arcs[aa, 1]), cutoff)
            for j, (bb, bt) in enumerate(zip(b_arc, b_t)):
                if aa == bb:
                    if bt >= at:
                        trans[i, j] = -abs((bt - at) * la - d_gps) / bta
                    else:
                        trans[i, j] = BACKWARD_PEN - abs((at - bt) * la - d_gps) / bta
                    continue
                dn = reach.get(int(g.arcs[bb, 0]))
                if dn is not None:
                    d_route = rest_a + dn + bt * g.len[bb]
                    trans[i, j] = -abs(d_route - d_gps) / bta
        tot = score[:, None] + trans
        bp = tot.argmax(0)
        score = tot[bp, np.arange(len(b_arc))] + np.log(B["p"].to_numpy() + EPS)
        backs.append(bp)
    # backtrack
    j = int(np.argmax(score))
    for k in range(len(states) - 1, -1, -1):
        out[pts[k]] = int(states[k]["arc_id"].to_numpy()[j])
        if k > 0:
            j = int(backs[k - 1][j])
