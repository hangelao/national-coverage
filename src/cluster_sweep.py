"""
cluster_sweep.py
----------------
How far below a state can coordination be pushed before the benefit is lost?

R5 established that national coordination adds almost nothing over intrastate
coordination — the benefit saturates at the state level. That answers the upper
bound but not the actionable one: a planner cannot convene a whole state, but
they can convene a handful of neighbouring markets. This module sweeps the
grouping from standalone (C1) up to whole-state (C2) and measures the cost
saving as a function of cluster size and cluster radius, turning "coordinate
within states" into "coordinate with your nearest N neighbours, within R km".

Clusters are formed *within* states by complete-linkage agglomeration on road
distance, so that:
  • every cluster has diameter <= max_km  (a real convening constraint), and
  • every cluster has at most max_size members.
Complete linkage is the right choice because it bounds the *diameter* of a
cluster rather than its chain length — under single linkage a cluster can span
an arbitrary distance through a chain of close neighbours, which would not be
convenable in practice.

The solve reuses ``budget_frontier.solve_budget_point`` with kind="C2": that
configuration prunes all cross-group links, so passing cluster groups in place
of state groups yields exactly "coordination within clusters only".
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from config import Config
from network_builder import _get_distance

# Grouping limits. ``math.inf`` reproduces the unconstrained end of the sweep:
# max_size=inf, max_km=inf is exactly C2 (whole state), which anchors the top.
DEFAULT_SIZES: tuple[float, ...] = (2, 3, 5, 10, 20, math.inf)
DEFAULT_RADII_KM: tuple[float, ...] = (25.0, 50.0, 100.0, math.inf)


def _condensed_distances(mids: Sequence[str], config: Config) -> np.ndarray:
    """Condensed (scipy-form) road-distance vector for one state's markets."""
    n = len(mids)
    full = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _get_distance(config.road_distances, mids[i], mids[j])
            full[i, j] = full[j, i] = d
    return squareform(full, checks=False)


def _split_oversized(
    labels: np.ndarray, condensed: np.ndarray, max_size: float
) -> np.ndarray:
    """
    Split clusters until none exceeds ``max_size`` members.

    The distance cut bounds diameter but not membership, so a dense
    metropolitan area can produce one very large cluster.

    Splitting must iterate. ``fcluster(criterion="maxclust", t=k)`` bounds the
    NUMBER of sub-clusters, not their sizes, so a single pass readily leaves
    clusters over the cap — an earlier version of this function requested
    ceil(n/max_size) sub-clusters and still returned clusters of 5 under a cap
    of 3. Repeated binary splitting is used instead: it terminates because every
    split strictly shrinks the offending cluster, and the diameter bound is
    preserved because a sub-cluster of a complete-linkage cluster is never wider
    than its parent.
    """
    if not np.isfinite(max_size):
        return labels
    square = squareform(condensed, checks=False)
    out = labels.copy()
    while True:
        oversized = [lab for lab in np.unique(out)
                     if (out == lab).sum() > max_size]
        if not oversized:
            return out
        progressed = False
        next_label = int(out.max()) + 1
        for lab in oversized:
            idx = np.flatnonzero(out == lab)
            if len(idx) < 2:
                continue
            sub = squareform(square[np.ix_(idx, idx)], checks=False)
            if sub.size == 0:
                continue
            sub_labels = fcluster(linkage(sub, method="complete"), t=2,
                                  criterion="maxclust")
            if len(np.unique(sub_labels)) < 2:
                continue  # coincident markets — cannot be separated further
            out[idx[sub_labels == 2]] = next_label
            next_label += 1
            progressed = True
        if not progressed:
            return out  # remaining offenders are coincident points


def build_clusters(
    state_groups: Dict[str, List[str]],
    config: Config,
    max_size: float,
    max_km: float,
) -> Dict[str, List[str]]:
    """
    Partition each state's markets into clusters bounded by size and diameter.

    Returns a mapping usable anywhere ``state_groups`` is accepted. Keys are
    ``"{state}_c{index}"`` so cluster identity stays traceable to its state.
    """
    clusters: Dict[str, List[str]] = {}
    for state, mids in state_groups.items():
        mids = list(mids)
        if len(mids) == 1:
            clusters[f"{state}_c0"] = mids
            continue
        condensed = _condensed_distances(mids, config)
        tree = linkage(condensed, method="complete")
        if np.isfinite(max_km):
            labels = fcluster(tree, t=max_km, criterion="distance")
        else:
            labels = np.ones(len(mids), dtype=int)
        labels = _split_oversized(labels, condensed, max_size)
        for k, lab in enumerate(np.unique(labels)):
            clusters[f"{state}_c{k}"] = [mids[i]
                                         for i in np.flatnonzero(labels == lab)]
    return clusters


def cluster_statistics(
    clusters: Dict[str, List[str]], state_groups: Dict[str, List[str]],
    config: Config,
) -> dict:
    """Descriptive statistics for one clustering, for the results table."""
    sizes = np.array([len(v) for v in clusters.values()], dtype=float)
    diameters = []
    for mids in clusters.values():
        if len(mids) < 2:
            diameters.append(0.0)
            continue
        diameters.append(float(np.max(_condensed_distances(mids, config))))
    diameters = np.array(diameters, dtype=float)
    # Keys are prefixed "realised_" where they could be confused with the sweep
    # *limits* (max_cluster_size / max_cluster_km), which are separate columns.
    return {
        "n_clusters": int(len(sizes)),
        "realised_mean_cluster_size": float(sizes.mean()),
        "realised_max_cluster_size": float(sizes.max()),
        "singleton_clusters": int((sizes == 1).sum()),
        "markets_in_clusters": int(sizes.sum()),
        "mean_cluster_diameter_km": float(diameters.mean()),
        "max_cluster_diameter_km": float(diameters.max()),
    }
