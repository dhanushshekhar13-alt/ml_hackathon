"""Raw-file loading and local metric projection for PathSnap trips."""
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT.parent / "Student folder"))
EARTH_R = 6371008.8  # mean Earth radius in metres


def list_trips(partition: str, data_dir: Path = DATA_DIR):
    root = Path(data_dir) / partition
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _load(path, dtype=float, ncols=None):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2 if ncols and ncols > 1 else 1)
    return arr


def load_trip(trip_id: str, partition: str, data_dir: Path = DATA_DIR, with_route=True):
    """Return dict with raw arrays. Columns: track = lon, lat, t ; nodes = lon, lat ; arcs = from, to."""
    d = Path(data_dir) / partition / trip_id
    track = np.loadtxt(d / f"{trip_id}.track", ndmin=2)
    nodes = np.loadtxt(d / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(d / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    out = {"trip_id": trip_id, "track": track, "nodes": nodes, "arcs": arcs}
    rp = d / f"{trip_id}.route"
    if with_route and rp.exists():
        out["route"] = np.atleast_1d(np.loadtxt(rp, dtype=np.int64))
    return out


class LocalProjection:
    """Equirectangular projection centred on the trip.

    Trips cover a few km, so the error versus a geodesic distance is far below GPS noise.
    Each trip gets its own origin because trips are in different parts of the world.
    """

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = EARTH_R * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self.ky = EARTH_R * np.pi / 180.0

    def to_xy(self, lon, lat):
        return np.column_stack(((np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky))

    def to_lonlat(self, xy):
        xy = np.asarray(xy)
        return np.column_stack((xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0))


def clean_track(track):
    """Drop non-finite / implausible rows and sort by time; returns (clean_track, n_dropped)."""
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    tr = track[ok]
    order = np.argsort(tr[:, 2], kind="stable")
    tr = tr[order]
    # remove duplicated timestamps (keep first)
    keep = np.r_[True, np.diff(tr[:, 2]) > 0]
    return tr[keep], int(len(track) - keep.sum())
