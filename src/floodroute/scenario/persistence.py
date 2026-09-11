"""JSON scenario preset save/load. Atomic writes. Safe filenames."""
from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = "1.0"


def _scenarios_dir(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent.parent
    d = root / "configs" / "scenario_presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_filename(scenario_id: str) -> str:
    """Generate safe filename from scenario_id (UUID-based, not from name)."""
    return f"scenario_{scenario_id}.json"


def new_scenario_id() -> str:
    return uuid.uuid4().hex[:12]


def save_scenario(
    scenario_config: dict,
    scenarios_dir: Path | None = None,
) -> Path:
    """Atomically save scenario to JSON. Returns path."""
    d = scenarios_dir or _scenarios_dir()
    sid = scenario_config.get("scenario_id") or new_scenario_id()
    scenario_config["scenario_id"] = sid
    scenario_config["schema_version"] = SCHEMA_VERSION
    scenario_config["updated_at_utc"] = datetime.now(UTC).isoformat()
    if "created_at_utc" not in scenario_config:
        scenario_config["created_at_utc"] = scenario_config["updated_at_utc"]
    fname = _safe_filename(sid)
    dest = d / fname
    tmp = d / f"{fname}.tmp_{os.getpid()}"
    tmp.write_text(json.dumps(scenario_config, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, dest)
    return dest


def load_scenario(scenario_id: str, scenarios_dir: Path | None = None) -> dict:
    d = scenarios_dir or _scenarios_dir()
    p = d / _safe_filename(scenario_id)
    if not p.exists():
        raise FileNotFoundError(f"Scenario {scenario_id!r} not found")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {data.get('schema_version')!r}")
    return data


def list_scenarios(scenarios_dir: Path | None = None) -> list[dict]:
    d = scenarios_dir or _scenarios_dir()
    results = []
    for p in sorted(d.glob("scenario_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            results.append(data)
        except Exception:
            pass
    return sorted(results, key=lambda x: x.get("updated_at_utc", ""), reverse=True)


def delete_scenario(scenario_id: str, scenarios_dir: Path | None = None) -> None:
    d = scenarios_dir or _scenarios_dir()
    p = d / _safe_filename(scenario_id)
    if p.exists():
        p.unlink()


def scenario_to_dict(
    scenario: object,
    catalog_fingerprint: str,
    scenario_name: str,
    scenario_id: str | None = None,
) -> dict:
    """Convert ScenarioConfig to saveable dict."""
    sid = scenario_id or new_scenario_id()
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": sid,
        "scenario_name": scenario_name,
        "municipality": scenario.municipality,
        "return_period": scenario.return_period,
        "demand_mode": scenario.demand_mode,
        "demand_fraction": scenario.demand_fraction,
        "demand_exact": scenario.demand_exact,
        "selected_facility_ids": list(scenario.selected_facility_ids),
        "facility_capacities": dict(scenario.facility_capacities),
        "algorithms": list(scenario.algorithms),
        "capacity_multipliers": list(scenario.capacity_multipliers),
        "flood_penalties": [str(p) for p in scenario.flood_penalties],
        "catalog_fingerprint": catalog_fingerprint,
        "road_conditions": list(getattr(scenario, "road_conditions", [])),
    }


def scenario_from_dict(data: dict, catalog: object = None) -> tuple:
    """Return (ScenarioConfig, unresolved_fids: list[str]).

    Parameters
    ----------
    data:
        Parsed scenario JSON dict.
    catalog:
        Optional FacilityCatalog. When provided, facility IDs not present in
        the catalog are collected into ``unresolved_fids``.  When ``None``,
        ``unresolved_fids`` is always empty (caller must check separately).
    """
    from floodroute.scenario.config import ScenarioConfig
    penalties = []
    for p in data.get("flood_penalties", [10.0]):
        penalties.append("prohibited" if str(p) == "prohibited" else float(p))
    sc = ScenarioConfig(
        municipality=data.get("municipality", "PH0600613"),
        return_period=data.get("return_period", "RP100"),
        demand_mode=data.get("demand_mode", "fraction"),
        demand_fraction=float(data.get("demand_fraction", 0.25)),
        demand_exact=int(data.get("demand_exact", 0)),
        selected_facility_ids=list(data.get("selected_facility_ids", [])),
        facility_capacities={k: int(v) for k, v in data.get("facility_capacities", {}).items()},
        algorithms=tuple(data.get("algorithms", ("A", "B", "B+", "C"))),
        capacity_multipliers=tuple(
            float(v) for v in data.get("capacity_multipliers", [0.50, 0.75, 1.00, 1.25])
        ),
        flood_penalties=tuple(penalties),
        catalog_fingerprint=data.get("catalog_fingerprint", ""),
        road_conditions=list(data.get("road_conditions", [])),
    )
    unresolved: list[str] = []
    if catalog is not None:
        for fid in sc.selected_facility_ids:
            if catalog.get(fid) is None:  # type: ignore[union-attr]
                unresolved.append(fid)
    return sc, unresolved
