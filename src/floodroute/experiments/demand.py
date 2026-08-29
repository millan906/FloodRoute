"""Stage 8 demand: PSA 2020 barangay population loading and origin snapping.

Design notes
------------
PSA 2020 population counts (``BarangayRecord.population_2020``) are observed
census values and are never modified.  They are stored separately from the
scenario demand units that are derived from them.

Origin snapping
---------------
Each barangay centroid is projected into EPSG:32651 (UTM Zone 51N) and
snapped to the nearest graph node.  Nodes in ``exclude_nodes`` (e.g. shelter
nodes that must not be both supply and demand) are skipped; the next-nearest
eligible node is chosen instead.  All snap distances are recorded for audit.

Demand rounding
---------------
``build_demands`` uses the largest-remainder method (Hamilton method) to
produce integer demands that sum exactly to
``round(total_population × fraction)``.  Each barangay first receives
``floor(population × fraction)`` units; the remaining units are awarded one
at a time to the barangays with the largest fractional remainders, with ties
broken by adm4_pcode (PSGC) ascending.  All arithmetic uses exact rational
numbers (``fractions.Fraction``) to avoid float rounding drift.

Determinism
-----------
Snapping and demand rounding are both deterministic: ties are broken by node
ID (snapping) or adm4_pcode (rounding).  Given identical inputs, identical
outputs are guaranteed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import geopandas as gpd


@dataclass(frozen=True)
class BarangayRecord:
    """One PSA 2020 barangay census record — observed, never modified."""

    psgc: str
    """12-digit PSGC code (e.g. ``'PH0600613001'``)."""

    name: str
    """Barangay name as reported by PSA."""

    population_2020: int
    """PSA 2020 Census count — observed value, never modified."""


@dataclass(frozen=True)
class BarangayOrigin:
    """A barangay record paired with its snapped graph node."""

    psgc: str
    name: str
    population_2020: int
    """PSA observed count — preserved unchanged."""

    origin_node: int
    """Integer node ID in the road graph."""

    snap_distance_m: float
    """Distance from barangay centroid to snapped node (metres, EPSG:32651)."""


def load_psa_population(
    csv_path: Path,
    adm3_filter: str = "PH0600613",
) -> list[BarangayRecord]:
    """Load PSA 2020 barangay populations from CSV.

    Parameters
    ----------
    csv_path:
        Path to a CSV with columns ``adm4_pcode``, ``adm4_name``,
        ``adm3_pcode``, ``population_2020``.
    adm3_filter:
        Only rows whose ``adm3_pcode`` matches this value are loaded.
        Default is San Jose de Buenavista (``'PH0600613'``).

    Returns
    -------
    list[BarangayRecord]
        Sorted by ``psgc`` for determinism.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If any row has ``population_2020 <= 0`` or a duplicate ``adm4_pcode``.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Population CSV not found: {csv_path}")

    records: list[BarangayRecord] = []
    seen: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if adm3_filter and row.get("adm3_pcode") != adm3_filter:
                continue
            psgc = row["adm4_pcode"].strip()
            if psgc in seen:
                raise ValueError(f"Duplicate adm4_pcode: {psgc}")
            seen.add(psgc)
            pop = int(row["population_2020"])
            if pop <= 0:
                raise ValueError(f"Non-positive population for {psgc}: {pop}")
            records.append(
                BarangayRecord(
                    psgc=psgc,
                    name=row["adm4_name"].strip(),
                    population_2020=pop,
                )
            )
    return sorted(records, key=lambda r: r.psgc)


def snap_barangay_origins(
    records: list[BarangayRecord],
    barangay_gpkg: Path,
    nodes_gpkg: Path,
    psgc_column: str = "adm4_pcode",
    municipality_column: str = "adm3_pcode",
    municipality_psgc: str = "PH0600613",
    exclude_nodes: set[int] | None = None,
) -> list[BarangayOrigin]:
    """Snap each barangay centroid to the nearest eligible graph node.

    Parameters
    ----------
    records:
        Barangay population records to snap.
    barangay_gpkg:
        GeoPackage with barangay polygon geometries in EPSG:32651.
    nodes_gpkg:
        GeoPackage with graph node point geometries in EPSG:32651.
        Must have a ``node_id`` column matching the integer node IDs in the
        road graph GraphML.
    psgc_column:
        Column name for the 12-digit PSGC code in *barangay_gpkg*.
    municipality_column:
        Column name for the municipality PSGC in *barangay_gpkg*.
    municipality_psgc:
        Filter barangay GeoPackage to this municipality code.
    exclude_nodes:
        Node IDs that must not be used as origins (e.g. shelter nodes).
        The next-nearest eligible node is chosen instead.

    Returns
    -------
    list[BarangayOrigin]
        One entry per record; sorted by ``psgc``.  Records with no eligible
        node are excluded (edge case: all nodes excluded — should not occur
        in practice).

    Raises
    ------
    KeyError
        If a record's PSGC is not found in the barangay GeoPackage.
    """
    exclude_nodes = exclude_nodes or set()

    bgy_gdf = gpd.read_file(str(barangay_gpkg))
    bgy_gdf = bgy_gdf[bgy_gdf[municipality_column] == municipality_psgc].copy()
    bgy_index: dict[str, object] = {
        row[psgc_column]: row.geometry.centroid for _, row in bgy_gdf.iterrows()
    }

    nodes_gdf = gpd.read_file(str(nodes_gpkg))
    node_ids: list[int] = [int(nid) for nid in nodes_gdf["node_id"]]
    node_geoms: list = nodes_gdf["geometry"].tolist()

    origins: list[BarangayOrigin] = []
    for rec in records:
        if rec.psgc not in bgy_index:
            raise KeyError(f"Barangay PSGC {rec.psgc} not found in {barangay_gpkg}")
        centroid = bgy_index[rec.psgc]
        # Sort by distance, breaking ties by node_id for determinism.
        candidates = sorted(
            [(centroid.distance(g), int(nid)) for g, nid in zip(node_geoms, node_ids, strict=True)],
            key=lambda x: (x[0], x[1]),
        )
        chosen_node: int | None = None
        chosen_dist: float = float("inf")
        for dist, nid in candidates:
            if nid not in exclude_nodes:
                chosen_node = nid
                chosen_dist = dist
                break
        if chosen_node is None:
            continue  # no eligible node — highly unlikely
        origins.append(
            BarangayOrigin(
                psgc=rec.psgc,
                name=rec.name,
                population_2020=rec.population_2020,
                origin_node=chosen_node,
                snap_distance_m=chosen_dist,
            )
        )
    return sorted(origins, key=lambda o: o.psgc)


def build_demands(
    origins: list[BarangayOrigin],
    fraction: float,
) -> tuple[dict[int, int], dict[str, int]]:
    """Convert barangay populations into integer evacuation demand units.

    Uses the **largest-remainder method** (Hamilton method) so that the sum
    of all barangay demands equals ``round(total_population × fraction)``
    exactly, conserving the municipality-level total.

    Algorithm
    ---------
    1. Compute ``target_total = round(sum_of_populations × fraction)``.
    2. Assign ``floor(population_i × fraction)`` to each barangay.
    3. Distribute the remaining units (``target_total − sum_of_floors``) one
       at a time to the barangays with the largest fractional remainder.
       Ties are broken by adm4_pcode (PSGC) ascending for determinism.

    All arithmetic uses exact ``fractions.Fraction`` values to avoid
    floating-point rounding drift.

    If multiple barangays snap to the same origin node their demands are
    summed before being stored in the routing demands dict.

    Parameters
    ----------
    origins:
        Snapped barangay origins.
    fraction:
        Demand fraction in (0, 1] — e.g. ``0.10``, ``0.25``, ``0.50``.

    Returns
    -------
    demands : dict[int, int]
        ``{origin_node: demand_units}`` — aggregated by origin node.
        This is the input dict for the optimizer.
    barangay_demand : dict[str, int]
        ``{adm4_pcode: demand_units}`` — per-barangay audit trail.
        Preserves per-barangay detail even when nodes are shared.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    frac = Fraction(fraction).limit_denominator(10000)
    total_population = sum(o.population_2020 for o in origins)
    target_total = round(Fraction(total_population) * frac)

    # Exact fractional demand per barangay using rational arithmetic.
    exact: dict[str, Fraction] = {
        o.psgc: Fraction(o.population_2020) * frac for o in origins
    }

    # Floor allocation and number of remainder units to distribute.
    floor_alloc: dict[str, int] = {psgc: int(e) for psgc, e in exact.items()}
    remainder_units = target_total - sum(floor_alloc.values())

    # Rank by fractional remainder descending; break ties by PSGC ascending.
    ranked = sorted(
        exact.items(),
        key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]),
    )

    barangay_demand: dict[str, int] = dict(floor_alloc)
    for i in range(remainder_units):
        barangay_demand[ranked[i][0]] += 1

    # Aggregate by origin node.
    psgc_to_node: dict[str, int] = {o.psgc: o.origin_node for o in origins}
    demands: dict[int, int] = {}
    for psgc, d in barangay_demand.items():
        node = psgc_to_node[psgc]
        demands[node] = demands.get(node, 0) + d

    return demands, barangay_demand
