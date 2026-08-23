"""Stage 4 graph-construction manifest generation."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from floodroute.graph.config import GRAPH_MANIFEST_SCHEMA_VERSION


def compute_file_sha256(path: Path) -> str:
    """Return lower-hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sw_versions() -> dict[str, str]:
    pkgs = ["networkx", "geopandas", "shapely", "pyproj", "fiona", "numpy"]
    sw: dict[str, str] = {
        "python": platform.python_version(),
        "floodroute": importlib.metadata.version("floodroute"),
    }
    for pkg in pkgs:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            sw[pkg] = importlib.metadata.version(pkg)
    return sw


def build_graph_manifest(
    *,
    municipality_code: str,
    municipality_name: str,
    source_roads_path: Path,
    source_roads_sha256: str,
    source_admin_path: Path,
    parameters: dict[str, Any],
    build_stats: dict[str, Any],
    graph_stats: dict[str, Any],
    validation_thresholds: dict[str, Any],
    output_graphml_path: Path,
    output_nodes_gpkg_path: Path,
    output_edges_gpkg_path: Path,
    validation_status: str,
) -> dict[str, Any]:
    """Build a graph construction manifest dict."""
    return {
        "schema_version": GRAPH_MANIFEST_SCHEMA_VERSION,
        "manifest_type": "graph_construction",
        "municipality_code": municipality_code,
        "municipality_name": municipality_name,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source": {
            "roads_gpkg": str(source_roads_path),
            "roads_sha256": source_roads_sha256,
            "admin_gpkg": str(source_admin_path),
        },
        "parameters": parameters,
        "software": _sw_versions(),
        "build_stats": build_stats,
        "graph_stats": graph_stats,
        "validation_thresholds": validation_thresholds,
        "outputs": {
            "graphml": {
                "path": str(output_graphml_path),
                "sha256": compute_file_sha256(output_graphml_path),
            },
            "nodes_gpkg": {
                "path": str(output_nodes_gpkg_path),
                "sha256": compute_file_sha256(output_nodes_gpkg_path),
            },
            "edges_gpkg": {
                "path": str(output_edges_gpkg_path),
                "sha256": compute_file_sha256(output_edges_gpkg_path),
            },
        },
        "validation_status": validation_status,
    }


def write_graph_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write manifest dict to JSON at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
