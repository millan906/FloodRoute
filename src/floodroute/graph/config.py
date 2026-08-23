"""Stage 4 road-graph construction configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Oneway tag interpretation
# ---------------------------------------------------------------------------
# Conservative: absent/unknown → bidirectional (never assume restricted access)

#: oneway values that produce a single forward edge (start → end of way geometry)
ONEWAY_FORWARD: frozenset[str] = frozenset({"yes", "1", "true"})

#: oneway values that produce a single reverse edge (end → start of way geometry)
ONEWAY_REVERSE: frozenset[str] = frozenset({"-1", "reverse"})

#: oneway values that explicitly declare bidirectionality
ONEWAY_NO: frozenset[str] = frozenset({"no", "0", "false"})

#: oneway values treated as bidirectional (conservative – access varies by time/event)
ONEWAY_TEMPORAL: frozenset[str] = frozenset({"reversible", "alternating"})

# Any value NOT in ONEWAY_FORWARD or ONEWAY_REVERSE is treated as bidirectional.

# ---------------------------------------------------------------------------
# Grade-separation detection
# ---------------------------------------------------------------------------

#: bridge tag values that indicate the way is elevated (not at ground level).
#: Ways tagged with these values do NOT form at-grade intersections with
#: surface roads at geometric crossings (only at shared endpoints).
BRIDGE_GRADE_SEP: frozenset[str] = frozenset(
    {"yes", "cantilever", "viaduct", "aqueduct", "movable", "trestle"}
)

#: tunnel tag values that indicate the way is underground.
TUNNEL_GRADE_SEP: frozenset[str] = frozenset({"yes", "building_passage", "culvert"})

# ---------------------------------------------------------------------------
# Coordinate precision
# ---------------------------------------------------------------------------

#: Round UTM metre coordinates to this many decimal places when identifying
#: coincident nodes.  3 → millimetre precision, sufficient for UTM at this scale.
COORD_PRECISION: int = 3

# ---------------------------------------------------------------------------
# File / path constants
# ---------------------------------------------------------------------------

OUTPUT_GRAPH_DIR: str = "processed/graph"
GRAPH_MANIFESTS_DIR: str = "processed/graph_manifests"
GRAPH_MANIFEST_SCHEMA_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# GraphConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class GraphConfig:
    """Mutable graph-construction configuration; defaults match module constants."""

    crs: str = "EPSG:32651"
    buffer_metres: float = 500.0
    coord_precision: int = COORD_PRECISION
    oneway_forward: frozenset[str] = field(default_factory=lambda: frozenset(ONEWAY_FORWARD))
    oneway_reverse: frozenset[str] = field(default_factory=lambda: frozenset(ONEWAY_REVERSE))
    bridge_grade_sep: frozenset[str] = field(default_factory=lambda: frozenset(BRIDGE_GRADE_SEP))
    tunnel_grade_sep: frozenset[str] = field(default_factory=lambda: frozenset(TUNNEL_GRADE_SEP))
    output_graph_dir: str = OUTPUT_GRAPH_DIR
    graph_manifests_dir: str = GRAPH_MANIFESTS_DIR

    def resolve(self, data_dir: Path) -> ResolvedGraphConfig:
        return ResolvedGraphConfig(base=self, data_dir=data_dir)


@dataclass
class ResolvedGraphConfig:
    """GraphConfig with all paths resolved to absolute Paths."""

    base: GraphConfig
    data_dir: Path

    @property
    def crs(self) -> str:
        return self.base.crs

    @property
    def buffer_metres(self) -> float:
        return self.base.buffer_metres

    @property
    def coord_precision(self) -> int:
        return self.base.coord_precision

    @property
    def oneway_forward(self) -> frozenset[str]:
        return self.base.oneway_forward

    @property
    def oneway_reverse(self) -> frozenset[str]:
        return self.base.oneway_reverse

    @property
    def bridge_grade_sep(self) -> frozenset[str]:
        return self.base.bridge_grade_sep

    @property
    def tunnel_grade_sep(self) -> frozenset[str]:
        return self.base.tunnel_grade_sep

    @property
    def output_graph_dir(self) -> Path:
        return self.data_dir / self.base.output_graph_dir

    @property
    def graph_manifests_dir(self) -> Path:
        return self.data_dir / self.base.graph_manifests_dir
