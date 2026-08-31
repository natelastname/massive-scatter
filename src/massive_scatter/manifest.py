from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .spec import AggregateRequest, PlotManifest

SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2, 3})
MAX_SAFE_VIEWER_EXTENT = 2**53 - 1


@dataclass(frozen=True, slots=True)
class LevelManifest:
    level: int
    cell_size: int
    height: int
    width: int
    occupied_chunks: int = 0
    occupied_cells: int = 0

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "level": self.level,
            "cell_size": self.cell_size,
            "shape": [self.height, self.width],
        }
        if self.occupied_chunks:
            result["occupied_chunks"] = self.occupied_chunks
        if self.occupied_cells:
            result["occupied_cells"] = self.occupied_cells
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LevelManifest:
        height, width = value["shape"]
        return cls(
            level=int(value["level"]),
            cell_size=int(value["cell_size"]),
            height=int(height),
            width=int(width),
            occupied_chunks=int(value.get("occupied_chunks", 0)),
            occupied_cells=int(value.get("occupied_cells", 0)),
        )


@dataclass(frozen=True, slots=True)
class Manifest:
    """Portable description of a ``.msplot`` dataset."""

    point_count: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    tile_size: int
    base_cell_size: int
    color_field: str | None
    levels: tuple[LevelManifest, ...]
    exact_fields: dict[str, str] = field(default_factory=dict)
    aggregates: tuple[AggregateRequest, ...] = ()
    plot: PlotManifest | None = None
    lod_storage: str | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def max_level(self) -> int:
        return self.levels[-1].level

    @property
    def uses_sparse_lod(self) -> bool:
        return self.schema_version >= 3 and self.lod_storage == "sparse_parquet"

    def validate(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(
                str(value) for value in sorted(SUPPORTED_SCHEMA_VERSIONS)
            )
            raise ValueError(
                f"Unsupported manifest schema {self.schema_version}; "
                f"supported versions are {supported}."
            )
        if self.schema_version >= 3 and self.lod_storage != "sparse_parquet":
            raise ValueError("Schema v3 datasets must use sparse_parquet LOD storage.")
        if self.point_count <= 0:
            raise ValueError("A plot must contain at least one point.")
        if self.max_x < self.min_x or self.max_y < self.min_y:
            raise ValueError("Manifest bounds are inverted.")
        if self.width > MAX_SAFE_VIEWER_EXTENT or self.height > MAX_SAFE_VIEWER_EXTENT:
            raise ValueError(
                "The coordinate span exceeds 2^53-1. Absolute int64 origins are "
                "supported, but the viewer requires each axis span to fit in "
                "JavaScript's exact integer range."
            )
        if self.tile_size < 2 or self.tile_size & (self.tile_size - 1):
            raise ValueError("tile_size must be a power of two greater than one.")
        if self.base_cell_size < 1 or self.base_cell_size & (self.base_cell_size - 1):
            raise ValueError("base_cell_size must be a positive power of two.")
        if not self.levels:
            raise ValueError("At least one LOD level is required.")

        previous: LevelManifest | None = None
        for expected_level, level in enumerate(self.levels):
            if level.level != expected_level:
                raise ValueError("LOD levels must be contiguous and start at zero.")
            if level.cell_size != self.base_cell_size * 2**level.level:
                raise ValueError("LOD cell sizes must double at every level.")
            if level.height <= 0 or level.width <= 0:
                raise ValueError("LOD shapes must be positive.")
            if self.uses_sparse_lod and level.occupied_cells <= 0:
                raise ValueError("Sparse LOD levels must contain occupied cells.")
            if previous is not None:
                if level.height != (previous.height + 1) // 2:
                    raise ValueError("LOD heights do not form a factor-two pyramid.")
                if level.width != (previous.width + 1) // 2:
                    raise ValueError("LOD widths do not form a factor-two pyramid.")
            previous = level

        keys = [request.key for request in self.aggregates]
        if len(keys) != len(set(keys)):
            raise ValueError("Aggregate request keys must be unique.")
        for request in self.aggregates:
            if request.source not in self.exact_fields:
                raise ValueError(
                    f"Aggregate {request.key!r} refers to unknown exact field "
                    f"{request.source!r}."
                )
            if self.exact_fields[request.source] != request.storage:
                raise ValueError(
                    f"Aggregate {request.key!r} storage disagrees with exact "
                    "field mapping."
                )
        if self.plot is not None and self.plot.exact_fields != self.exact_fields:
            raise ValueError("Plot and manifest exact-field mappings disagree.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "point_count": self.point_count,
            "coordinate_dtype": "int64",
            "bounds": {
                "min_x": str(self.min_x),
                "max_x": str(self.max_x),
                "min_y": str(self.min_y),
                "max_y": str(self.max_y),
            },
            "origin": {"x": str(self.min_x), "y": str(self.min_y)},
            "extent": {"width": self.width, "height": self.height},
            "tile_size": self.tile_size,
            "base_cell_size": self.base_cell_size,
            "color_field": self.color_field,
            "levels": [level.to_dict() for level in self.levels],
        }
        if self.lod_storage is not None:
            result["lod_storage"] = self.lod_storage
        if self.exact_fields:
            result["exact_fields"] = dict(self.exact_fields)
        if self.aggregates:
            result["aggregates"] = [request.to_dict() for request in self.aggregates]
        if self.plot is not None:
            result["plot"] = self.plot.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Manifest:
        bounds = value["bounds"]
        manifest = cls(
            schema_version=int(value["schema_version"]),
            point_count=int(value["point_count"]),
            min_x=int(bounds["min_x"]),
            max_x=int(bounds["max_x"]),
            min_y=int(bounds["min_y"]),
            max_y=int(bounds["max_y"]),
            tile_size=int(value["tile_size"]),
            base_cell_size=int(value["base_cell_size"]),
            color_field=value.get("color_field"),
            levels=tuple(LevelManifest.from_dict(item) for item in value["levels"]),
            exact_fields={
                str(source): str(storage)
                for source, storage in value.get("exact_fields", {}).items()
            },
            aggregates=tuple(
                AggregateRequest.from_dict(item) for item in value.get("aggregates", [])
            ),
            plot=(
                PlotManifest.from_dict(value["plot"])
                if value.get("plot") is not None
                else None
            ),
            lod_storage=(
                str(value["lod_storage"])
                if value.get("lod_storage") is not None
                else None
            ),
        )
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, dataset_path: str | Path) -> Manifest:
        path = Path(dataset_path) / "manifest.json"
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, dataset_path: str | Path) -> None:
        path = Path(dataset_path) / "manifest.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
