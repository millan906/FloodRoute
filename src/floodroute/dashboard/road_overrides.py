"""Road condition overrides for Stage 10.

Overrides are stored separately from the GraphML and never mutate the source graph.
They are applied only during weight-function evaluation inside a single algorithm run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# User-facing condition labels (7 canonical + 3 legacy aliases)
OVERRIDE_STATUS: dict[str, str] = {
    "use_model":              "Use modeled condition",
    "dry_confirmed_passable": "Dry / confirmed passable",
    "flooded_passable":       "Flooded but passable with caution",
    "flooded_unknown":        "Flooded — passability unknown",
    "flooded_impassable":     "Flooded and impassable",
    "road_closed":            "Road closed",
    "unknown":                "Unknown / unverified",
    # Legacy aliases — retained for backward compatibility with serialised session state
    "flood_exposed":          "Flooded but passable with caution (legacy alias)",
    "closed":                 "Road closed (legacy alias)",
    "reported_passable":      "Dry / confirmed passable (legacy alias)",
}

# Evidence types for override provenance
EVIDENCE_TYPES: dict[str, str] = {
    "controlled_assumption":    "Controlled scenario assumption",
    "LGU_report":               "LGU/MDRRMO report",
    "field_observation":        "Field observation",
    "documented_external_source": "Documented external source",
}

# Effective routing behaviour per status
_IMPASSABLE = frozenset({
    "road_closed", "closed",
    "flooded_unknown", "flooded_impassable",
    "unknown",
})
_PASSABLE_OVERRIDE = frozenset({"dry_confirmed_passable", "reported_passable"})
_FLOOD_PENALTY = frozenset({"flooded_passable", "flood_exposed"})


@dataclass
class RoadOverride:
    """One planner-supplied road condition override."""
    u: int                          # source node
    v: int                          # target node
    status: str                     # one of OVERRIDE_STATUS keys
    evidence_type: str = "controlled_assumption"
    source_reference: str = ""
    observation_time: str = ""
    notes: str = ""


@dataclass
class RoadOverrideStore:
    """Accumulates road overrides for one planning session.

    The source graph (GraphML / GeoPackage / JRC attributes) is NEVER modified.
    Overrides affect only the weight function used inside a single algorithm run.
    """
    overrides: dict[tuple[int, int], RoadOverride] = field(default_factory=dict)

    def add(self, override: RoadOverride) -> None:
        self.overrides[(override.u, override.v)] = override

    def remove(self, u: int, v: int) -> None:
        self.overrides.pop((u, v), None)

    def clear(self) -> None:
        self.overrides.clear()

    def __len__(self) -> int:
        return len(self.overrides)

    def to_dict(self) -> dict:
        """Serialise for st.session_state storage."""
        return {
            f"{k[0]},{k[1]}": {
                "u": v.u, "v": v.v, "status": v.status,
                "evidence_type": v.evidence_type,
                "source_reference": v.source_reference,
                "observation_time": v.observation_time,
                "notes": v.notes,
            }
            for k, v in self.overrides.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> RoadOverrideStore:
        store = cls()
        for raw in d.values():
            store.add(RoadOverride(**raw))
        return store


def apply_overrides(
    base_weight_fn,
    store: RoadOverrideStore,
    flood_penalty_fn=None,
) -> object:
    """Return a new weight function that applies overrides before the base function.

    The graph object is NEVER modified. Only the returned weight function is affected.

    Parameters
    ----------
    base_weight_fn:
        Original weight function (signature: (u, v, d) -> float | None).
    store:
        Override store for this session.
    flood_penalty_fn:
        Optional flood-aware weight function applied to flood_exposed overrides.
        If None, flood_exposed edges use the base weight.
    """
    overrides = store.overrides  # read-only reference

    def _weight(u, v, d):
        key = (int(u), int(v))
        override = overrides.get(key)
        if override is None:
            return base_weight_fn(u, v, d)
        status = override.status
        if status in _IMPASSABLE:
            return None  # blocked
        if status in _PASSABLE_OVERRIDE:
            return base_weight_fn(u, v, d)  # treat as passable regardless of model
        if status in _FLOOD_PENALTY and flood_penalty_fn is not None:
            return flood_penalty_fn(u, v, d)
        return base_weight_fn(u, v, d)  # use_model or unknown with None fn

    return _weight


def override_color(status: str) -> tuple[str, str | None]:
    """Return ``(color_hex, dash_array_or_None)`` for an override status.

    Colors follow the UCD map-style specification:
    - confirmed dry/passable: green
    - flooded but passable: amber
    - flooded with unknown passability: orange dashed
    - flooded impassable: red
    - closed: dark red
    - unknown: gray dashed
    - use_model: no visual override (caller should skip drawing)
    """
    _MAP: dict[str, tuple[str, str | None]] = {
        "dry_confirmed_passable": ("#16A34A", None),
        "reported_passable":      ("#16A34A", None),
        "flooded_passable":       ("#F59E0B", None),
        "flood_exposed":          ("#F59E0B", None),
        "flooded_unknown":        ("#F97316", "6 4"),
        "flooded_impassable":     ("#EF4444", None),
        "road_closed":            ("#7F1D1D", None),
        "closed":                 ("#7F1D1D", None),
        "unknown":                ("#9CA3AF", "4 4"),
    }
    return _MAP.get(status, ("#6B7280", "4 4"))
