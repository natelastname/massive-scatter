from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .spec import AggregateRequest, AxesManifest, PlotManifest

SCHEMA_VERSION = 4
LOD_STORAGE = "layered_sparse_parquet"
MAX_SAFE_VIEWER_EXTENT = 2**53 - 1


@dataclass(frozen=True, slots=True)
class LevelManifest:
    level: int
    cell_size: int
    height: int
    width: int
    occupied_cells: int

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "cell_size": self.cell_size,
            "shape": [self.height, self.width],
            "occupied_cells": self.occupied_cells,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LevelManifest:
        height, width = value["shape"]
        return cls(
            level=int(value["level"]),
            cell_size=int(value["cell_size"]),
            height=int(height),
            width=int(width),
            occupied_cells=int(value["occupied_cells"]),
        )


@dataclass(frozen=True, slots=True)
class LayerManifest:
    """Storage and rendering metadata for one independently queryable layer."""

    id: str
    path: str
    zorder: float
    point_count: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    base_cell_size: int
    color_field: str | None
    levels: tuple[LevelManifest, ...]
    exact_fields: dict[str, str] = field(default_factory=dict)
    aggregates: tuple[AggregateRequest, ...] = ()
    plot: PlotManifest | None = None

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def max_level(self) -> int:
        return self.levels[-1].level

    def validate(self) -> None:
        if not self.id:
            raise ValueError("Layer ids may not be empty.")
        pure_path = PurePosixPath(self.path)
        if not self.path or pure_path.is_absolute() or ".." in pure_path.parts:
            raise ValueError("Layer paths must be safe relative paths.")
        if self.point_count <= 0:
            raise ValueError("A scatter layer must contain at least one point.")
        if self.max_x < self.min_x or self.max_y < self.min_y:
            raise ValueError(f"Layer {self.id!r} bounds are inverted.")
        if self.width > MAX_SAFE_VIEWER_EXTENT or self.height > MAX_SAFE_VIEWER_EXTENT:
            raise ValueError(
                f"Layer {self.id!r} coordinate span exceeds 2^53-1; the viewer "
                "requires each axis span to fit JavaScript's exact integer range."
            )
        if self.base_cell_size < 1 or self.base_cell_size & (self.base_cell_size - 1):
            raise ValueError("base_cell_size must be a positive power of two.")
        if not self.levels:
            raise ValueError("Every scatter layer requires at least one LOD level.")

        previous: LevelManifest | None = None
        for expected_level, level in enumerate(self.levels):
            if level.level != expected_level:
                raise ValueError("LOD levels must be contiguous and start at zero.")
            if level.cell_size != self.base_cell_size * 2**level.level:
                raise ValueError("LOD cell sizes must double at every level.")
            if level.height <= 0 or level.width <= 0:
                raise ValueError("LOD shapes must be positive.")
            if level.occupied_cells <= 0:
                raise ValueError("LOD levels must contain occupied cells.")
            if previous is not None:
                if level.height != (previous.height + 1) // 2:
                    raise ValueError("LOD heights do not form a factor-two pyramid.")
                if level.width != (previous.width + 1) // 2:
                    raise ValueError("LOD widths do not form a factor-two pyramid.")
            previous = level

        keys = [request.key for request in self.aggregates]
        if len(keys) != len(set(keys)):
            raise ValueError("Aggregate request keys must be unique within a layer.")
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
            raise ValueError("Layer plot and exact-field mappings disagree.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result: dict[str, Any] = {
            "id": self.id,
            "path": self.path,
            "zorder": self.zorder,
            "point_count": self.point_count,
            "bounds": {
                "min_x": str(self.min_x),
                "max_x": str(self.max_x),
                "min_y": str(self.min_y),
                "max_y": str(self.max_y),
            },
            "origin": {"x": str(self.min_x), "y": str(self.min_y)},
            "extent": {"width": self.width, "height": self.height},
            "base_cell_size": self.base_cell_size,
            "color_field": self.color_field,
            "levels": [level.to_dict() for level in self.levels],
        }
        if self.exact_fields:
            result["exact_fields"] = dict(self.exact_fields)
        if self.aggregates:
            result["aggregates"] = [request.to_dict() for request in self.aggregates]
        if self.plot is not None:
            result["plot"] = self.plot.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LayerManifest:
        bounds = value["bounds"]
        layer = cls(
            id=str(value["id"]),
            path=str(value["path"]),
            zorder=float(value.get("zorder", 0.0)),
            point_count=int(value["point_count"]),
            min_x=int(bounds["min_x"]),
            max_x=int(bounds["max_x"]),
            min_y=int(bounds["min_y"]),
            max_y=int(bounds["max_y"]),
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
        )
        layer.validate()
        return layer


@dataclass(frozen=True, slots=True)
class Manifest:
    """Portable description of a layered ``.msplot`` figure."""

    min_x: int
    max_x: int
    min_y: int
    max_y: int
    axes: AxesManifest
    layers: tuple[LayerManifest, ...]

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def lod_storage(self) -> str:
        return LOD_STORAGE

    @property
    def point_count(self) -> int:
        return sum(layer.point_count for layer in self.layers)

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1

    def validate(self) -> None:
        if not self.layers:
            raise ValueError("A figure must contain at least one scatter layer.")
        if self.max_x < self.min_x or self.max_y < self.min_y:
            raise ValueError("Manifest bounds are inverted.")
        if self.width > MAX_SAFE_VIEWER_EXTENT or self.height > MAX_SAFE_VIEWER_EXTENT:
            raise ValueError(
                "The figure coordinate span exceeds 2^53-1. Absolute int64 origins "
                "are supported, but the viewer requires each axis span to fit in "
                "JavaScript's exact integer range."
            )

        ids = [layer.id for layer in self.layers]
        paths = [layer.path for layer in self.layers]
        if len(ids) != len(set(ids)):
            raise ValueError("Layer ids must be unique.")
        if len(paths) != len(set(paths)):
            raise ValueError("Layer paths must be unique.")

        for layer in self.layers:
            layer.validate()
            if layer.plot is not None and layer.plot.axes != self.axes:
                raise ValueError(
                    "All scatter layers in an axes must share axes metadata."
                )

        expected = (
            min(layer.min_x for layer in self.layers),
            max(layer.max_x for layer in self.layers),
            min(layer.min_y for layer in self.layers),
            max(layer.max_y for layer in self.layers),
        )
        if expected != (self.min_x, self.max_x, self.min_y, self.max_y):
            raise ValueError("Figure bounds must be the union of layer bounds.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": SCHEMA_VERSION,
            "lod_storage": LOD_STORAGE,
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
            "axes": self.axes.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Manifest:
        schema_version = int(value.get("schema_version", -1))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported .msplot schema {schema_version}; only schema "
                f"{SCHEMA_VERSION} is supported. Rebuild the dataset."
            )
        lod_storage = value.get("lod_storage")
        if lod_storage != LOD_STORAGE:
            raise ValueError(
                f"Unsupported LOD storage {lod_storage!r}; only {LOD_STORAGE!r} "
                "is supported. Rebuild the dataset."
            )

        bounds = value["bounds"]
        manifest = cls(
            min_x=int(bounds["min_x"]),
            max_x=int(bounds["max_x"]),
            min_y=int(bounds["min_y"]),
            max_y=int(bounds["max_y"]),
            axes=AxesManifest.from_dict(value.get("axes", {})),
            layers=tuple(LayerManifest.from_dict(item) for item in value["layers"]),
        )
        manifest.validate()
        serialized_count = int(value.get("point_count", manifest.point_count))
        if serialized_count != manifest.point_count:
            raise ValueError(
                "Figure point_count does not equal the sum of layer counts."
            )
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
