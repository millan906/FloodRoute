"""Stage 4 graph validation.

Validates a road MultiDiGraph against structural and semantic criteria.
Raises GraphValidationFailed if any check fails.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

import networkx as nx

logger = logging.getLogger("floodroute.graph.validate")


class GraphValidationFailed(ValueError):
    """Raised when graph validation fails."""


def compute_graph_stats(G: nx.MultiDiGraph) -> dict[str, Any]:
    """Compute and return graph statistics without raising on issues."""
    n = G.number_of_nodes()
    e = G.number_of_edges()

    # Self-loops
    self_loops = list(nx.selfloop_edges(G))

    # Parallel edges (edges u→v with key > 0, i.e., >1 edge between same pair)
    parallel_pairs: set[tuple[int, int]] = set()
    for u, v, k in G.edges(keys=True):
        if k > 0:
            parallel_pairs.add((u, v))

    # Isolated nodes
    isolated = [nd for nd in G.nodes if G.degree(nd) == 0]

    # Boundary nodes
    boundary = [nd for nd, d in G.nodes(data=True) if d.get("on_boundary", False)]

    # Weakly connected components
    wccs = list(nx.weakly_connected_components(G))
    wccs.sort(key=len, reverse=True)
    largest_wcc_nodes = wccs[0] if wccs else set()
    largest_wcc_n = len(largest_wcc_nodes)
    wcc_coverage = largest_wcc_n / n if n > 0 else 0.0

    # Strongly connected components
    sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
    sccs.sort(key=len, reverse=True)
    largest_scc_n = len(sccs[0]) if sccs else 0
    scc_coverage = largest_scc_n / n if n > 0 else 0.0

    # Edge length statistics — directed and physical
    directed_length = 0.0
    physical_length = 0.0
    wcc_directed_length = 0.0
    wcc_physical_length = 0.0
    non_positive: list[float] = []
    non_finite: list[float] = []

    seen_segs: set[tuple] = set()  # (osm_id, edge_seq) for physical dedup
    seen_segs_wcc: set[tuple] = set()

    for u, v, d in G.edges(data=True):
        lm = d.get("length_m")
        if lm is None or lm == "":
            continue
        try:
            lm = float(lm)
        except (TypeError, ValueError):
            continue

        directed_length += lm
        if not math.isfinite(lm):
            non_finite.append(lm)
        if not (math.isfinite(lm) and lm > 0):
            non_positive.append(lm)

        oid = d.get("osm_id")
        seq = d.get("edge_seq")
        if oid is not None and seq is not None and oid != "" and seq != "":
            seg_key = (oid, seq)
            if seg_key not in seen_segs:
                seen_segs.add(seg_key)
                physical_length += lm
            if u in largest_wcc_nodes and v in largest_wcc_nodes:
                wcc_directed_length += lm
                if seg_key not in seen_segs_wcc:
                    seen_segs_wcc.add(seg_key)
                    wcc_physical_length += lm

    physical_length_coverage = wcc_physical_length / physical_length if physical_length > 0 else 0.0

    # Residual component stats (all WCCs except largest)
    residual_wccs = wccs[1:]
    residual_node_count = sum(len(c) for c in residual_wccs)
    residual_physical_length = physical_length - wcc_physical_length

    # Highway class distribution in residual components
    residual_nodes_set: set = set()
    for c in residual_wccs:
        residual_nodes_set.update(c)
    residual_highway: dict[str, int] = {}
    seen_residual_segs: set[tuple] = set()
    for u, v, d in G.edges(data=True):
        if u not in residual_nodes_set or v not in residual_nodes_set:
            continue
        oid = d.get("osm_id")
        seq = d.get("edge_seq")
        if oid is not None and seq is not None and oid != "" and seq != "":
            seg_key = (oid, seq)
            if seg_key in seen_residual_segs:
                continue
            seen_residual_segs.add(seg_key)
        hw = d.get("highway") or "unknown"
        if hw == "":
            hw = "unknown"
        residual_highway[hw] = residual_highway.get(hw, 0) + 1

    # WCC size distribution
    wcc_size_counter = Counter(len(c) for c in wccs)

    return {
        "node_count": n,
        "edge_count": e,
        "self_loop_count": len(self_loops),
        "parallel_edge_pairs": len(parallel_pairs),
        "isolated_node_count": len(isolated),
        "boundary_node_count": len(boundary),
        "weakly_connected_component_count": len(wccs),
        "largest_wcc_node_count": largest_wcc_n,
        "largest_wcc_node_coverage": round(wcc_coverage, 4),
        "strongly_connected_component_count": len(sccs),
        "largest_scc_node_count": largest_scc_n,
        "largest_scc_node_coverage": round(scc_coverage, 4),
        # Length metrics
        "total_directed_edge_length_m": round(directed_length, 3),
        "total_physical_segment_length_m": round(physical_length, 3),
        "largest_wcc_directed_edge_length_m": round(wcc_directed_length, 3),
        "largest_wcc_physical_segment_length_m": round(wcc_physical_length, 3),
        "largest_wcc_physical_length_coverage": round(physical_length_coverage, 4),
        # Edge length validity
        "non_positive_length_count": len(non_positive),
        "non_finite_length_count": len(non_finite),
        # Residual component details
        "residual_component_count": len(residual_wccs),
        "residual_node_count": residual_node_count,
        "residual_physical_length_m": round(residual_physical_length, 3),
        "residual_highway_classes": residual_highway,
        "wcc_size_distribution": {
            "exactly_2": wcc_size_counter.get(2, 0),
            "3_to_5": sum(v for k, v in wcc_size_counter.items() if 3 <= k <= 5),
            "6_to_20": sum(v for k, v in wcc_size_counter.items() if 6 <= k <= 20),
            "more_than_20": sum(v for k, v in wcc_size_counter.items() if k > 20),
        },
    }


def validate_road_graph(
    G: nx.MultiDiGraph,
    *,
    min_nodes: int = 1,
    min_edges: int = 1,
    min_largest_wcc_node_coverage: float = 0.85,
    min_largest_wcc_physical_length_coverage: float = 0.85,
) -> dict[str, Any]:
    """Validate a road MultiDiGraph.

    Checks:
    - node_count >= min_nodes
    - edge_count >= min_edges
    - all edge lengths are finite and positive
    - graph has a 'crs' attribute
    - largest WCC covers >= min_largest_wcc_node_coverage of all nodes
    - largest WCC covers >= min_largest_wcc_physical_length_coverage of physical km

    The WCC thresholds default to 0.85 and are configurable.  After the
    T-junction fix, the three study municipalities produce 93–98% coverage;
    the 0.85 default is a meaningful gate while allowing for minor genuine
    OSM topology gaps.  Do NOT lower these defaults to paper over a
    construction defect.  Record the thresholds used in the manifest.

    Returns computed stats dict.  Raises GraphValidationFailed on failure.
    """
    stats = compute_graph_stats(G)
    issues: list[str] = []

    if stats["node_count"] < min_nodes:
        issues.append(f"Too few nodes: {stats['node_count']} < {min_nodes}")

    if stats["edge_count"] < min_edges:
        issues.append(f"Too few edges: {stats['edge_count']} < {min_edges}")

    if "crs" not in G.graph:
        issues.append("Graph missing 'crs' attribute")

    if stats["non_positive_length_count"] > 0:
        issues.append(f"{stats['non_positive_length_count']} edges with non-positive length_m")

    if stats["non_finite_length_count"] > 0:
        issues.append(f"{stats['non_finite_length_count']} edges with non-finite length_m")

    if (
        stats["node_count"] > 0
        and stats["largest_wcc_node_coverage"] < min_largest_wcc_node_coverage
    ):
        issues.append(
            f"Largest WCC node coverage {stats['largest_wcc_node_coverage']:.1%} "
            f"< threshold {min_largest_wcc_node_coverage:.1%}"
        )

    if (
        stats["total_physical_segment_length_m"] > 0
        and stats["largest_wcc_physical_length_coverage"] < min_largest_wcc_physical_length_coverage
    ):
        issues.append(
            f"Largest WCC physical-length coverage "
            f"{stats['largest_wcc_physical_length_coverage']:.1%} "
            f"< threshold {min_largest_wcc_physical_length_coverage:.1%}"
        )

    if issues:
        raise GraphValidationFailed(
            "Road graph validation failed:\n" + "\n".join(f"  - {i}" for i in issues)
        )

    logger.info(
        "Graph validation OK: %d nodes, %d edges, WCC-node=%.1f%%, WCC-length=%.1f%%",
        stats["node_count"],
        stats["edge_count"],
        stats["largest_wcc_node_coverage"] * 100,
        stats["largest_wcc_physical_length_coverage"] * 100,
    )
    return stats
