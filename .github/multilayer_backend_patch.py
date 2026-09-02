from pathlib import Path


def write(path: str, content: str) -> None:
    Path(path).write_text(content)


write(
    "src/massive_scatter/manifest.py",
    r'''from __future__ import annotations

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
                raise ValueError("All scatter layers in an axes must share axes metadata.")

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
            raise ValueError("Figure point_count does not equal the sum of layer counts.")
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
''',
)

write(
    "src/massive_scatter/builder.py",
    r'''from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .manifest import MAX_SAFE_VIEWER_EXTENT, LayerManifest, Manifest
from .sparse_lod import build_sparse_lod_pyramid
from .spec import AggregateRequest, AxesManifest, CompiledPlot, PlotManifest

Progress = Callable[[str], None]
MAX_CATEGORIES = 32


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """Bounded-memory build settings."""

    base_cell_size: int = 64
    part_rows: int = 1_000_000
    batch_size: int = 131_072
    overwrite: bool = False

    def validate(self) -> None:
        if self.base_cell_size < 1 or self.base_cell_size & (self.base_cell_size - 1):
            raise ValueError("base_cell_size must be a positive power of two.")
        if self.part_rows < 1:
            raise ValueError("part_rows must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")


@dataclass(slots=True)
class LayerBuild:
    """One independently stored scatter layer to include in a figure build."""

    batches: Iterable[pa.RecordBatch | pa.Table]
    x: str = "x"
    y: str = "y"
    color: str | None = None
    plot: CompiledPlot | None = None
    zorder: float = 0.0


@dataclass(frozen=True, slots=True)
class _IngestResult:
    point_count: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    point_files: tuple[Path, ...]
    numeric_ranges: dict[str, tuple[float, float]]
    categorical_fields: dict[str, tuple[str, ...]]


def _canonical_table(
    value: pa.RecordBatch | pa.Table,
    *,
    x_field: str,
    y_field: str,
    exact_fields: Mapping[str, str],
    field_kinds: Mapping[str, str],
) -> pa.Table:
    table = (
        pa.Table.from_batches([value]) if isinstance(value, pa.RecordBatch) else value
    )
    required = list(dict.fromkeys([x_field, y_field, *exact_fields]))
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"Input batch is missing columns: {', '.join(missing)}")

    x = table[x_field]
    y = table[y_field]
    if not pa.types.is_integer(x.type) or not pa.types.is_integer(y.type):
        raise TypeError("x and y columns must use an integer Arrow type.")
    if x.null_count or y.null_count:
        raise ValueError("x and y columns may not contain null values.")

    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        "x": pc.cast(x, pa.int64(), safe=True),
        "y": pc.cast(y, pa.int64(), safe=True),
    }
    for source, storage in exact_fields.items():
        source_field = table[source]
        if source_field.null_count:
            raise ValueError(f"Field {source!r} may not contain null values.")
        kind = field_kinds[source]
        if kind == "numeric":
            if not (
                pa.types.is_integer(source_field.type)
                or pa.types.is_floating(source_field.type)
                or pa.types.is_decimal(source_field.type)
            ):
                raise TypeError(f"Field {source!r} must be numeric.")
            source_field = pc.cast(source_field, pa.float64(), safe=True)
            values = np.asarray(source_field.combine_chunks(), dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Field {source!r} may not contain NaN or infinity.")
        elif kind == "categorical":
            if not (
                pa.types.is_string(source_field.type)
                or pa.types.is_large_string(source_field.type)
                or pa.types.is_integer(source_field.type)
                or pa.types.is_floating(source_field.type)
                or pa.types.is_boolean(source_field.type)
                or pa.types.is_dictionary(source_field.type)
            ):
                raise TypeError(
                    f"Categorical field {source!r} must contain scalar strings, "
                    "numbers, or booleans."
                )
            source_field = pc.cast(source_field, pa.string(), safe=False)
        else:
            raise ValueError(f"Unknown field kind {kind!r} for {source!r}.")
        columns[storage] = source_field

    return pa.table(columns)


def _write_point_parts(
    dataset_path: Path,
    batches: Iterable[pa.RecordBatch | pa.Table],
    *,
    x_field: str,
    y_field: str,
    exact_fields: Mapping[str, str],
    field_kinds: Mapping[str, str],
    part_rows: int,
    progress: Progress,
) -> _IngestResult:
    points_path = dataset_path / "points"
    points_path.mkdir(parents=True)

    pending: list[pa.Table] = []
    pending_rows = 0
    part_index = 0
    point_count = 0
    min_x: int | None = None
    max_x: int | None = None
    min_y: int | None = None
    max_y: int | None = None
    point_files: list[Path] = []
    index_rows: list[dict[str, int | str]] = []
    numeric_ranges: dict[str, tuple[float, float]] = {}
    category_values: dict[str, dict[str, None]] = {
        source: {} for source, kind in field_kinds.items() if kind == "categorical"
    }

    def update_field_metadata(table: pa.Table) -> None:
        for source, storage in exact_fields.items():
            kind = field_kinds[source]
            column = table[storage].combine_chunks()
            if kind == "numeric":
                values = np.asarray(column, dtype=np.float64)
                current = (float(values.min()), float(values.max()))
                previous = numeric_ranges.get(source)
                numeric_ranges[source] = (
                    current[0] if previous is None else min(previous[0], current[0]),
                    current[1] if previous is None else max(previous[1], current[1]),
                )
            else:
                values = category_values[source]
                for item in column.unique().to_pylist():
                    values[str(item)] = None
                    if len(values) > MAX_CATEGORIES:
                        raise ValueError(
                            f"Categorical field {source!r} has more than "
                            f"{MAX_CATEGORIES} values; high-cardinality categorical "
                            "summaries are not supported yet."
                        )

    def flush() -> None:
        nonlocal pending, pending_rows, part_index, point_count
        nonlocal min_x, max_x, min_y, max_y
        if not pending:
            return

        table = pa.concat_tables(pending).combine_chunks()
        update_field_metadata(table)
        path = points_path / f"part-{part_index:06d}.parquet"
        pq.write_table(
            table,
            path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            row_group_size=min(part_rows, 131_072),
        )

        x_values = table["x"].combine_chunks().to_numpy(zero_copy_only=False)
        y_values = table["y"].combine_chunks().to_numpy(zero_copy_only=False)
        part_min_x = int(x_values.min())
        part_max_x = int(x_values.max())
        part_min_y = int(y_values.min())
        part_max_y = int(y_values.max())
        rows = table.num_rows
        index_rows.append(
            {
                "path": path.relative_to(dataset_path).as_posix(),
                "count": rows,
                "min_x": part_min_x,
                "max_x": part_max_x,
                "min_y": part_min_y,
                "max_y": part_max_y,
            }
        )
        point_files.append(path)
        point_count += rows
        min_x = part_min_x if min_x is None else min(min_x, part_min_x)
        max_x = part_max_x if max_x is None else max(max_x, part_max_x)
        min_y = part_min_y if min_y is None else min(min_y, part_min_y)
        max_y = part_max_y if max_y is None else max(max_y, part_max_y)
        progress(f"wrote {path.name}: {rows:,} points ({point_count:,} total)")

        part_index += 1
        pending = []
        pending_rows = 0

    for value in batches:
        table = _canonical_table(
            value,
            x_field=x_field,
            y_field=y_field,
            exact_fields=exact_fields,
            field_kinds=field_kinds,
        )
        if table.num_rows == 0:
            continue
        pending.append(table)
        pending_rows += table.num_rows
        if pending_rows >= part_rows:
            flush()
    flush()

    if point_count == 0 or None in {min_x, max_x, min_y, max_y}:
        raise ValueError("The input did not contain any points.")

    index_table = pa.table(
        {
            "path": pa.array([row["path"] for row in index_rows], pa.string()),
            "count": pa.array([row["count"] for row in index_rows], pa.uint64()),
            "min_x": pa.array([row["min_x"] for row in index_rows], pa.int64()),
            "max_x": pa.array([row["max_x"] for row in index_rows], pa.int64()),
            "min_y": pa.array([row["min_y"] for row in index_rows], pa.int64()),
            "max_y": pa.array([row["max_y"] for row in index_rows], pa.int64()),
        }
    )
    pq.write_table(index_table, dataset_path / "index.parquet", compression="zstd")

    assert min_x is not None and max_x is not None
    assert min_y is not None and max_y is not None
    return _IngestResult(
        point_count=point_count,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        point_files=tuple(point_files),
        numeric_ranges=numeric_ranges,
        categorical_fields={
            source: tuple(values) for source, values in category_values.items()
        },
    )


def _direct_contract(
    color: str | None,
) -> tuple[dict[str, str], dict[str, str], tuple[AggregateRequest, ...]]:
    if color is None:
        return {}, {}, ()
    return (
        {color: "color"},
        {color: "numeric"},
        (AggregateRequest("direct_color", color, "color", "max"),),
    )


def _build_layer(
    root: Path,
    relative_path: Path,
    *,
    layer_id: str,
    layer: LayerBuild,
    settings: BuildConfig,
    progress: Progress,
) -> LayerManifest:
    if layer.plot is not None and layer.color is not None:
        raise ValueError("color= and plot= may not be combined")
    if layer.plot is None:
        exact_fields, field_kinds, aggregates = _direct_contract(layer.color)
    else:
        if layer.plot.x != layer.x or layer.plot.y != layer.y:
            raise ValueError("Compiled plot x/y fields disagree with layer x/y fields.")
        exact_fields = layer.plot.exact_fields
        field_kinds = dict(layer.plot.field_kinds)
        aggregates = layer.plot.aggregates

    layer_path = root / relative_path
    layer_path.mkdir(parents=True)
    progress("streaming exact points to Parquet")
    ingest = _write_point_parts(
        layer_path,
        layer.batches,
        x_field=layer.x,
        y_field=layer.y,
        exact_fields=exact_fields,
        field_kinds=field_kinds,
        part_rows=settings.part_rows,
        progress=progress,
    )

    width = ingest.max_x - ingest.min_x + 1
    height = ingest.max_y - ingest.min_y + 1
    if width > MAX_SAFE_VIEWER_EXTENT or height > MAX_SAFE_VIEWER_EXTENT:
        raise ValueError(
            "The viewer preserves arbitrary int64 origins, but each layer axis span "
            "must be at most 2^53-1 so unit offsets remain exact in JavaScript."
        )

    levels = build_sparse_lod_pyramid(
        layer_path,
        point_files=list(ingest.point_files),
        point_count=ingest.point_count,
        min_x=ingest.min_x,
        max_x=ingest.max_x,
        min_y=ingest.min_y,
        max_y=ingest.max_y,
        base_cell_size=settings.base_cell_size,
        batch_size=settings.batch_size,
        part_rows=settings.part_rows,
        aggregates=aggregates,
        progress=progress,
    )
    plot_manifest = (
        PlotManifest(
            scatter=layer.plot.scatter,
            axes=layer.plot.axes,
            exact_fields=dict(layer.plot.exact_fields),
            categorical_fields=ingest.categorical_fields,
            numeric_ranges=ingest.numeric_ranges,
        )
        if layer.plot is not None
        else None
    )
    return LayerManifest(
        id=layer_id,
        path=relative_path.as_posix(),
        zorder=float(layer.zorder),
        point_count=ingest.point_count,
        min_x=ingest.min_x,
        max_x=ingest.max_x,
        min_y=ingest.min_y,
        max_y=ingest.max_y,
        base_cell_size=settings.base_cell_size,
        color_field=layer.color,
        levels=levels,
        exact_fields=dict(exact_fields),
        aggregates=aggregates,
        plot=plot_manifest,
    )


def build_figure_dataset(
    output: str | Path,
    layers: Sequence[LayerBuild],
    *,
    axes: AxesManifest,
    config: BuildConfig | None = None,
    progress: Progress | None = None,
) -> Manifest:
    """Build one figure containing independently queryable scatter layers."""

    if not layers:
        raise ValueError("A figure must contain at least one scatter layer.")
    settings = config or BuildConfig()
    settings.validate()
    report = progress or (lambda _message: None)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not settings.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass overwrite=True to replace it."
        )

    temporary_path = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    temporary_path.mkdir()
    try:
        built_layers: list[LayerManifest] = []
        for index, layer in enumerate(layers):
            layer_id = f"layer-{index:03d}"
            relative_path = Path("layers") / layer_id

            def layer_report(message: str, *, prefix: str = layer_id) -> None:
                report(f"[{prefix}] {message}")

            built_layers.append(
                _build_layer(
                    temporary_path,
                    relative_path,
                    layer_id=layer_id,
                    layer=layer,
                    settings=settings,
                    progress=layer_report,
                )
            )

        min_x = min(layer.min_x for layer in built_layers)
        max_x = max(layer.max_x for layer in built_layers)
        min_y = min(layer.min_y for layer in built_layers)
        max_y = max(layer.max_y for layer in built_layers)
        if max_x - min_x + 1 > MAX_SAFE_VIEWER_EXTENT or max_y - min_y + 1 > MAX_SAFE_VIEWER_EXTENT:
            raise ValueError(
                "The union of layer coordinates exceeds 2^53-1 on an axis; the "
                "viewer requires the shared figure span to remain exactly representable."
            )

        manifest = Manifest(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            axes=axes,
            layers=tuple(built_layers),
        )
        manifest.save(temporary_path)

        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        temporary_path.replace(output_path)
        report(
            f"built {output_path} with {manifest.point_count:,} points across "
            f"{len(manifest.layers)} layer(s)"
        )
        return manifest
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


def build_dataset(
    output: str | Path,
    batches: Iterable[pa.RecordBatch | pa.Table],
    *,
    x: str = "x",
    y: str = "y",
    color: str | None = None,
    config: BuildConfig | None = None,
    progress: Progress | None = None,
    plot: CompiledPlot | None = None,
) -> Manifest:
    """Build a one-layer figure using the same layered format as the plot API."""

    if plot is not None and color is not None:
        raise ValueError("color= and plot= may not be combined")
    axes = plot.axes if plot is not None else AxesManifest()
    return build_figure_dataset(
        output,
        [LayerBuild(batches=batches, x=x, y=y, color=color, plot=plot)],
        axes=axes,
        config=config,
        progress=progress,
    )
''',
)

write(
    "src/massive_scatter/dataset.py",
    r'''from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from .manifest import LayerManifest, Manifest
from .sparse_dataset import SparseLodReader


class _LayerDataset:
    """Query one layer in its own origin-relative coordinate system."""

    def __init__(self, path: Path, manifest: LayerManifest) -> None:
        self.path = path
        self.manifest = manifest
        self._parts = pq.read_table(self.path / "index.parquet").to_pylist()
        self._lod = SparseLodReader(self.path, self.manifest)

    def _candidate_parts(
        self, min_x: int, max_x: int, min_y: int, max_y: int
    ) -> tuple[list[Path], int]:
        paths: list[Path] = []
        upper_bound = 0
        for part in self._parts:
            if (
                int(part["max_x"]) < min_x
                or int(part["min_x"]) > max_x
                or int(part["max_y"]) < min_y
                or int(part["min_y"]) > max_y
            ):
                continue
            paths.append(self.path / str(part["path"]))
            upper_bound += int(part["count"])
        return paths, upper_bound

    @staticmethod
    def _empty_response(origin_x: int, origin_y: int) -> dict[str, Any]:
        return {
            "mode": "exact",
            "origin": [origin_x, origin_y],
            "x": [],
            "y": [],
            "color": None,
            "fields": {},
            "point_count": 0,
        }

    def _exact_view(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        candidate_paths: list[Path],
        max_points: int,
        origin_x: int,
        origin_y: int,
    ) -> dict[str, Any] | None:
        if not candidate_paths:
            return self._empty_response(origin_x, origin_y)

        absolute_min_x = self.manifest.min_x + math.ceil(min_x)
        absolute_max_x = self.manifest.min_x + math.floor(max_x)
        absolute_min_y = self.manifest.min_y + math.ceil(min_y)
        absolute_max_y = self.manifest.min_y + math.floor(max_y)
        if absolute_min_x > absolute_max_x or absolute_min_y > absolute_max_y:
            return self._empty_response(origin_x, origin_y)

        dataset = pads.dataset(
            [str(path) for path in candidate_paths], format="parquet"
        )
        predicate = (
            (pads.field("x") >= absolute_min_x)
            & (pads.field("x") <= absolute_max_x)
            & (pads.field("y") >= absolute_min_y)
            & (pads.field("y") <= absolute_max_y)
        )
        storage_fields = self.manifest.exact_fields
        storage_columns = list(dict.fromkeys(storage_fields.values()))
        columns = ["x", "y", *storage_columns]

        x_values: list[int] = []
        y_values: list[int] = []
        field_values: dict[str, list[Any]] = {source: [] for source in storage_fields}
        count = 0
        for batch in dataset.to_batches(
            columns=columns,
            filter=predicate,
            batch_size=min(131_072, max_points + 1),
            batch_readahead=1,
            fragment_readahead=1,
        ):
            count += batch.num_rows
            if count > max_points:
                return None

            absolute_x = np.asarray(batch.column(0), dtype=np.int64)
            absolute_y = np.asarray(batch.column(1), dtype=np.int64)
            relative_x = absolute_x - self.manifest.min_x - origin_x
            relative_y = absolute_y - self.manifest.min_y - origin_y
            x_values.extend(int(value) for value in relative_x)
            y_values.extend(int(value) for value in relative_y)

            for source, storage in storage_fields.items():
                index = columns.index(storage)
                field_values[source].extend(batch.column(index).to_pylist())

        direct_color = (
            field_values.get(self.manifest.color_field)
            if self.manifest.color_field is not None
            else None
        )
        return {
            "mode": "exact",
            "origin": [origin_x, origin_y],
            "x": x_values,
            "y": y_values,
            "color": direct_color,
            "fields": field_values,
            "point_count": len(x_values),
        }

    def view(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        pixel_width: int,
        pixel_height: int,
        max_points: int,
        max_cells: int,
    ) -> dict[str, Any]:
        clipped_min_x = max(0.0, min_x)
        clipped_max_x = min(float(self.manifest.width - 1), max_x)
        clipped_min_y = max(0.0, min_y)
        clipped_max_y = min(float(self.manifest.height - 1), max_y)
        origin_x = math.floor(clipped_min_x)
        origin_y = math.floor(clipped_min_y)
        if clipped_min_x > clipped_max_x or clipped_min_y > clipped_max_y:
            return self._empty_response(origin_x, origin_y)

        absolute_min_x = self.manifest.min_x + math.ceil(clipped_min_x)
        absolute_max_x = self.manifest.min_x + math.floor(clipped_max_x)
        absolute_min_y = self.manifest.min_y + math.ceil(clipped_min_y)
        absolute_max_y = self.manifest.min_y + math.floor(clipped_max_y)
        candidates, candidate_upper_bound = self._candidate_parts(
            absolute_min_x, absolute_max_x, absolute_min_y, absolute_max_y
        )

        units_per_pixel = max(
            (clipped_max_x - clipped_min_x) / pixel_width,
            (clipped_max_y - clipped_min_y) / pixel_height,
            0.0,
        )
        should_probe_exact = (
            units_per_pixel <= self.manifest.base_cell_size
            or candidate_upper_bound <= max_points * 4
        )
        if should_probe_exact:
            exact = self._exact_view(
                min_x=clipped_min_x,
                max_x=clipped_max_x,
                min_y=clipped_min_y,
                max_y=clipped_max_y,
                candidate_paths=candidates,
                max_points=max_points,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            if exact is not None:
                return exact

        return self._lod.aggregate_view(
            min_x=clipped_min_x,
            max_x=clipped_max_x,
            min_y=clipped_min_y,
            max_y=clipped_max_y,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            max_cells=max_cells,
        )

    def check(self) -> list[str]:
        problems: list[str] = []
        indexed_count = sum(int(part["count"]) for part in self._parts)
        if indexed_count != self.manifest.point_count:
            problems.append(
                "index count "
                f"{indexed_count} != manifest count {self.manifest.point_count}"
            )
        for part in self._parts:
            path = self.path / str(part["path"])
            if not path.is_file():
                problems.append(f"missing point part: {part['path']}")
        problems.extend(self._lod.check())
        return problems


class MassiveScatterDataset:
    """Query all figure layers against one shared viewport and camera origin."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.manifest = Manifest.load(self.path)
        indexed = [
            (index, _LayerDataset(self.path / layer.path, layer))
            for index, layer in enumerate(self.manifest.layers)
        ]
        self._layers = [
            dataset
            for _, dataset in sorted(
                indexed, key=lambda item: (item[1].manifest.zorder, item[0])
            )
        ]

    def view(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        pixel_width: int,
        pixel_height: int,
        max_points: int = 200_000,
        max_cells: int = 200_000,
    ) -> dict[str, Any]:
        values = (min_x, max_x, min_y, max_y)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Viewport bounds must be finite.")
        if min_x > max_x or min_y > max_y:
            raise ValueError("Viewport bounds are inverted.")
        if pixel_width < 1 or pixel_height < 1:
            raise ValueError("Viewport pixel dimensions must be positive.")
        if max_points < 1 or max_cells < 1:
            raise ValueError("max_points and max_cells must be positive.")

        clipped_min_x = max(0.0, min_x)
        clipped_max_x = min(float(self.manifest.width - 1), max_x)
        clipped_min_y = max(0.0, min_y)
        clipped_max_y = min(float(self.manifest.height - 1), max_y)
        origin_x = math.floor(clipped_min_x)
        origin_y = math.floor(clipped_min_y)
        if clipped_min_x > clipped_max_x or clipped_min_y > clipped_max_y:
            return {"origin": [origin_x, origin_y], "layers": []}

        layer_responses: list[dict[str, Any]] = []
        for dataset in self._layers:
            layer = dataset.manifest
            layer_offset_x = layer.min_x - self.manifest.min_x
            layer_offset_y = layer.min_y - self.manifest.min_y
            response = dataset.view(
                min_x=clipped_min_x - layer_offset_x,
                max_x=clipped_max_x - layer_offset_x,
                min_y=clipped_min_y - layer_offset_y,
                max_y=clipped_max_y - layer_offset_y,
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                max_points=max_points,
                max_cells=max_cells,
            )
            child_origin_x, child_origin_y = response.pop("origin")
            shift_x = child_origin_x + layer_offset_x - origin_x
            shift_y = child_origin_y + layer_offset_y - origin_y
            response["x"] = [value + shift_x for value in response["x"]]
            response["y"] = [value + shift_y for value in response["y"]]
            response["id"] = layer.id
            response["zorder"] = layer.zorder
            layer_responses.append(response)

        return {"origin": [origin_x, origin_y], "layers": layer_responses}

    def check(self) -> list[str]:
        problems: list[str] = []
        for dataset in self._layers:
            for problem in dataset.check():
                problems.append(f"{dataset.manifest.id}: {problem}")
        return problems
''',
)

write(
    "src/massive_scatter/plot.py",
    r'''from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from .builder import BuildConfig, LayerBuild, build_figure_dataset
from .source import input_batches
from .spec import (
    AxesManifest,
    CompiledPlot,
    CountValue,
    EncodingManifest,
    FieldValue,
    compile_encodings,
)

Source = str | Path | Iterable[pa.RecordBatch | pa.Table]


@dataclass(slots=True)
class _ScatterCall:
    source: Source
    x: str
    y: str
    c: str | FieldValue | CountValue | None
    color: str | None
    cmap: str
    marker: str | FieldValue
    size: float | str | FieldValue
    alpha: float | str | FieldValue | CountValue
    label: str | None
    zorder: float


@dataclass(frozen=True, slots=True)
class ScatterLayer:
    """Lightweight handle identifying one scatter call in an axes."""

    index: int
    zorder: float

    @property
    def id(self) -> str:
        return f"layer-{self.index:03d}"


class Axes:
    """A deliberately small Matplotlib-like single-axes plotting surface."""

    def __init__(self) -> None:
        self._scatters: list[_ScatterCall] = []
        self._title: str | None = None
        self._xlabel: str | None = None
        self._ylabel: str | None = None
        self._legend = False

    def scatter(
        self,
        source: Source,
        *,
        x: str,
        y: str,
        c: str | FieldValue | CountValue | None = None,
        color: str | None = None,
        cmap: str = "viridis",
        marker: str | FieldValue = "o",
        s: float | str | FieldValue = 3.0,
        alpha: float | str | FieldValue | CountValue = 0.92,
        label: str | None = None,
        zorder: float | None = None,
    ) -> ScatterLayer:
        """Add an independently stored/queryable scatter layer to this axes."""

        if c is not None and color is not None:
            raise ValueError(
                "Pass either c= for a data mapping or color= for a constant."
            )
        index = len(self._scatters)
        selected_zorder = float(index) if zorder is None else float(zorder)
        if not math.isfinite(selected_zorder):
            raise ValueError("zorder must be finite")
        self._scatters.append(
            _ScatterCall(
                source=source,
                x=x,
                y=y,
                c=c,
                color=color,
                cmap=cmap,
                marker=marker,
                size=s,
                alpha=alpha,
                label=label,
                zorder=selected_zorder,
            )
        )
        return ScatterLayer(index=index, zorder=selected_zorder)

    def set_title(self, value: str) -> None:
        self._title = value

    def set_xlabel(self, value: str) -> None:
        self._xlabel = value

    def set_ylabel(self, value: str) -> None:
        self._ylabel = value

    def set(
        self,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> None:
        if title is not None:
            self._title = title
        if xlabel is not None:
            self._xlabel = xlabel
        if ylabel is not None:
            self._ylabel = ylabel

    def legend(self, visible: bool = True) -> None:
        self._legend = visible

    def _axes_manifest(self) -> AxesManifest:
        return AxesManifest(
            title=self._title,
            xlabel=self._xlabel,
            ylabel=self._ylabel,
            legend=self._legend,
        )


class Figure:
    """A single interactive figure containing one axes and many scatter layers."""

    def __init__(self, axes: Axes) -> None:
        self.axes = axes

    @staticmethod
    def _compile(call: _ScatterCall, axes: AxesManifest) -> CompiledPlot:
        if call.color is not None:
            compiled = compile_encodings(
                x=call.x,
                y=call.y,
                color=CountValue(),
                marker=call.marker,
                size=call.size,
                alpha=call.alpha,
                cmap=call.cmap,
                label=call.label,
                axes=axes,
            )
            return CompiledPlot(
                x=compiled.x,
                y=compiled.y,
                exact_fields=compiled.exact_fields,
                field_kinds=compiled.field_kinds,
                aggregate_plan=compiled.aggregate_plan,
                scatter=compiled.scatter.__class__(
                    color=EncodingManifest("constant", value=call.color),
                    marker=compiled.scatter.marker,
                    size=compiled.scatter.size,
                    alpha=compiled.scatter.alpha,
                    cmap=compiled.scatter.cmap,
                    label=compiled.scatter.label,
                ),
                axes=compiled.axes,
            )

        color = call.c if call.c is not None else CountValue()
        return compile_encodings(
            x=call.x,
            y=call.y,
            color=color,
            marker=call.marker,
            size=call.size,
            alpha=call.alpha,
            cmap=call.cmap,
            label=call.label,
            axes=axes,
        )

    def write(
        self,
        output: str | Path,
        *,
        config: BuildConfig | None = None,
        progress=None,
    ):
        if not self.axes._scatters:
            raise ValueError("The figure has no scatter layers.")
        settings = config or BuildConfig()
        axes_manifest = self.axes._axes_manifest()
        builds: list[LayerBuild] = []
        for call in self.axes._scatters:
            compiled = self._compile(call, axes_manifest)
            if isinstance(call.source, (str, Path)):
                batches = input_batches(
                    call.source,
                    columns=list(compiled.required_columns),
                    batch_size=settings.batch_size,
                )
            else:
                batches = call.source
            builds.append(
                LayerBuild(
                    batches=batches,
                    x=call.x,
                    y=call.y,
                    plot=compiled,
                    zorder=call.zorder,
                )
            )

        return build_figure_dataset(
            output,
            builds,
            axes=axes_manifest,
            config=settings,
            progress=progress,
        )


def subplots() -> tuple[Figure, Axes]:
    """Create the single figure/axes pair supported by the current grammar."""

    axes = Axes()
    return Figure(axes), axes
''',
)

# Sparse LOD reader now consumes per-layer storage metadata.
sparse = Path("src/massive_scatter/sparse_dataset.py")
text = sparse.read_text()
text = text.replace("from .manifest import LevelManifest, Manifest", "from .manifest import LayerManifest, LevelManifest")
text = text.replace("def __init__(self, path: Path, manifest: Manifest) -> None:", "def __init__(self, path: Path, manifest: LayerManifest) -> None:")
sparse.write_text(text)

# Export the layer handle.
init = Path("src/massive_scatter/__init__.py")
text = init.read_text()
text = text.replace("from .plot import Axes, Figure, subplots", "from .plot import Axes, Figure, ScatterLayer, subplots")
text = text.replace('    "Figure",\n', '    "Figure",\n    "ScatterLayer",\n')
init.write_text(text)

write(
    "tests/test_manifest.py",
    r'''import pytest

from massive_scatter.manifest import (
    LOD_STORAGE,
    SCHEMA_VERSION,
    LayerManifest,
    LevelManifest,
    Manifest,
)
from massive_scatter.spec import AxesManifest


def _layer(origin: int = 0) -> LayerManifest:
    return LayerManifest(
        id="layer-000",
        path="layers/layer-000",
        zorder=0.0,
        point_count=2,
        min_x=origin,
        max_x=origin + 1,
        min_y=-origin,
        max_y=-origin + 1,
        base_cell_size=1,
        color_field=None,
        levels=(
            LevelManifest(0, 1, 2, 2, 2),
            LevelManifest(1, 2, 1, 1, 1),
        ),
    )


def _manifest(origin: int = 0) -> Manifest:
    layer = _layer(origin)
    return Manifest(
        min_x=layer.min_x,
        max_x=layer.max_x,
        min_y=layer.min_y,
        max_y=layer.max_y,
        axes=AxesManifest(),
        layers=(layer,),
    )


def test_manifest_round_trip_preserves_large_int64_origin(tmp_path):
    origin = 9_100_000_000_000_000
    manifest = _manifest(origin)
    manifest.save(tmp_path)

    raw = (tmp_path / "manifest.json").read_text()
    assert f'"min_x": "{origin}"' in raw
    assert f'"schema_version": {SCHEMA_VERSION}' in raw
    assert f'"lod_storage": "{LOD_STORAGE}"' in raw
    assert Manifest.load(tmp_path) == manifest


def test_manifest_rejects_old_schema_versions():
    payload = _manifest().to_dict()
    for version in (1, 2, 3):
        old = dict(payload)
        old["schema_version"] = version
        with pytest.raises(ValueError, match="only schema 4 is supported"):
            Manifest.from_dict(old)


def test_manifest_rejects_non_layered_storage():
    payload = _manifest().to_dict()
    payload["lod_storage"] = "sparse_parquet"
    with pytest.raises(ValueError, match="layered_sparse_parquet"):
        Manifest.from_dict(payload)


def test_manifest_requires_unique_layer_ids_and_union_bounds():
    layer = _layer()
    duplicate = LayerManifest.from_dict(layer.to_dict() | {"path": "layers/other"})
    with pytest.raises(ValueError, match="Layer ids must be unique"):
        Manifest(0, 1, 0, 1, AxesManifest(), (layer, duplicate)).validate()
''',
)

write(
    "tests/test_build_query.py",
    r'''import numpy as np
import pyarrow as pa

from massive_scatter import BuildConfig, MassiveScatterDataset, build_dataset


def make_batch(origin: int, size: int = 1024) -> pa.RecordBatch:
    x = np.arange(size, dtype=np.int64) + origin
    y = ((np.arange(size, dtype=np.int64) * 37) % size) + origin
    color = (np.arange(size, dtype=np.int64) % 11).astype(np.float64)
    return pa.record_batch([x, y, color], names=["x", "y", "weight"])


def test_build_exact_and_aggregate_views(tmp_path):
    origin = 9_100_000_000_000_000
    output = tmp_path / "example.msplot"
    manifest = build_dataset(
        output,
        [make_batch(origin)],
        color="weight",
        config=BuildConfig(base_cell_size=4, part_rows=128, batch_size=64),
    )

    assert manifest.point_count == 1024
    assert manifest.min_x == origin
    assert manifest.width == 1024
    assert len(manifest.layers) == 1
    dataset = MassiveScatterDataset(output)
    assert dataset.check() == []

    exact = dataset.view(
        min_x=100,
        max_x=110,
        min_y=0,
        max_y=1023,
        pixel_width=1000,
        pixel_height=1000,
        max_points=100,
    )
    assert exact["origin"] == [100, 0]
    layer = exact["layers"][0]
    assert layer["mode"] == "exact"
    assert layer["x"] == list(range(11))
    assert all(isinstance(value, int) for value in layer["x"])

    aggregate = dataset.view(
        min_x=0,
        max_x=1023,
        min_y=0,
        max_y=1023,
        pixel_width=8,
        pixel_height=8,
        max_points=10,
        max_cells=64,
    )
    layer = aggregate["layers"][0]
    assert layer["mode"] == "aggregate"
    assert sum(layer["count"]) == 1024
    assert layer["cell_count"] <= 64


def test_unit_separation_survives_shared_origin_rebasing(tmp_path):
    origin = 9_100_000_000_000_000
    batch = pa.record_batch(
        [
            pa.array([origin, origin + 1], type=pa.int64()),
            pa.array([origin, origin + 1], type=pa.int64()),
        ],
        names=["x", "y"],
    )
    output = tmp_path / "precision.msplot"
    build_dataset(
        output,
        [batch],
        config=BuildConfig(base_cell_size=1, part_rows=8),
    )

    response = MassiveScatterDataset(output).view(
        min_x=0,
        max_x=1,
        min_y=0,
        max_y=1,
        pixel_width=100,
        pixel_height=100,
        max_points=10,
    )
    assert response["origin"] == [0, 0]
    layer = response["layers"][0]
    assert layer["mode"] == "exact"
    assert layer["x"] == [0, 1]
    assert layer["y"] == [0, 1]
''',
)

write(
    "tests/test_lod.py",
    r'''import pyarrow as pa

from massive_scatter import BuildConfig, MassiveScatterDataset, build_dataset


def test_parent_levels_preserve_counts_and_max_color(tmp_path):
    x = [0, 1, 4, 5, 8, 9, 12, 13]
    y = [0, 4, 1, 5, 8, 12, 9, 13]
    color = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0]
    batch = pa.record_batch([x, y, color], names=["x", "y", "weight"])
    output = tmp_path / "lod.msplot"
    manifest = build_dataset(
        output,
        [batch],
        color="weight",
        config=BuildConfig(base_cell_size=1, part_rows=4),
    )

    assert len(manifest.layers[0].levels) >= 2
    dataset = MassiveScatterDataset(output)
    assert dataset.check() == []
    response = dataset.view(
        min_x=0,
        max_x=13,
        min_y=0,
        max_y=13,
        pixel_width=1,
        pixel_height=1,
        max_points=1,
        max_cells=1,
    )
    layer = response["layers"][0]
    assert layer["mode"] == "aggregate"
    assert sum(layer["count"]) == len(x)
    assert max(layer["color"]) == 9.0
''',
)

write(
    "tests/test_sparse_lod.py",
    r'''import math

import numpy as np
import pyarrow as pa

from massive_scatter import BuildConfig, MassiveScatterDataset, build_dataset


def test_extreme_aspect_ratio_uses_sparse_cell_rows(tmp_path):
    point_count = 20_000
    x = np.arange(point_count, dtype=np.int64)
    y = np.arange(point_count, dtype=np.int64) * 100_000
    batch = pa.record_batch([x, y], names=["x", "y"])
    output = tmp_path / "sparse.msplot"

    manifest = build_dataset(
        output,
        [batch],
        config=BuildConfig(base_cell_size=64, part_rows=2048, batch_size=1024),
    )

    assert manifest.schema_version == 4
    assert manifest.lod_storage == "layered_sparse_parquet"
    layer = manifest.layers[0]
    assert layer.levels[0].occupied_cells == point_count
    layer_path = output / layer.path
    assert (layer_path / "lod" / "0" / "index.parquet").is_file()

    parts = sorted((layer_path / "lod" / "0").glob("part-*.parquet"))
    assert len(parts) == math.ceil(point_count / 1024)
    assert len(parts) < point_count // 100

    dataset = MassiveScatterDataset(output)
    assert dataset.check() == []
    aggregate = dataset.view(
        min_x=0,
        max_x=point_count - 1,
        min_y=0,
        max_y=(point_count - 1) * 100_000,
        pixel_width=1,
        pixel_height=1,
        max_points=1,
        max_cells=4,
    )["layers"][0]
    assert aggregate["mode"] == "aggregate"
    assert aggregate["cell_count"] <= 4
    assert sum(aggregate["count"]) == point_count
''',
)

write(
    "tests/test_server.py",
    r'''import pyarrow as pa
from fastapi.testclient import TestClient

from massive_scatter import BuildConfig, build_dataset
from massive_scatter.server import create_app


def test_api_serves_layered_manifest_and_view(tmp_path):
    output = tmp_path / "api.msplot"
    build_dataset(
        output,
        [pa.record_batch([[0, 1, 2], [2, 1, 0]], names=["x", "y"])],
        config=BuildConfig(base_cell_size=1, part_rows=4),
    )
    client = TestClient(create_app(output))

    manifest = client.get("/api/manifest")
    assert manifest.status_code == 200
    payload = manifest.json()
    assert payload["point_count"] == 3
    assert [layer["id"] for layer in payload["layers"]] == ["layer-000"]

    view = client.post(
        "/api/view",
        json={
            "xmin": 0,
            "xmax": 2,
            "ymin": 0,
            "ymax": 2,
            "width": 100,
            "height": 100,
        },
    )
    assert view.status_code == 200
    response = view.json()
    assert response["origin"] == [0, 0]
    assert response["layers"][0]["mode"] == "exact"
    assert response["layers"][0]["point_count"] == 3
    assert client.get("/api/view").status_code == 405
''',
)

write(
    "tests/test_plot_api.py",
    r'''import math

import pyarrow as pa

import massive_scatter as ms


def test_plot_api_compiles_fields_reducers_and_axes(tmp_path):
    batch = pa.record_batch(
        [
            [0, 0, 1, 2],
            [0, 0, 0, 0],
            [1.0, 3.0, 9.0, 17.0],
            ["E3", "E3", "P3", "other"],
            [3.0, 4.0, 5.0, 6.0],
        ],
        names=["n", "value", "weight", "kind", "point_size"],
    )

    fig, ax = ms.subplots()
    handle = ax.scatter(
        [batch],
        x="n",
        y="value",
        c=ms.mean("weight"),
        cmap="plasma",
        marker=ms.field("kind"),
        s="point_size",
        alpha=ms.max("weight"),
        label="episodes",
    )
    assert handle.id == "layer-000"
    ax.set(title="Episode geometry", xlabel="n", ylabel="a(n)")
    ax.legend()

    output = tmp_path / "plot.msplot"
    manifest = fig.write(output, config=ms.BuildConfig(base_cell_size=1, part_rows=2))

    assert manifest.axes.title == "Episode geometry"
    assert manifest.axes.legend is True
    layer = manifest.layers[0]
    assert layer.plot is not None
    assert layer.plot.scatter.cmap == "plasma"
    assert layer.plot.categorical_fields["kind"] == ("E3", "P3", "other")
    assert layer.plot.numeric_ranges["weight"] == (1.0, 17.0)
    assert layer.plot.numeric_ranges["point_size"] == (3.0, 6.0)
    assert {(item.source, item.reducer) for item in layer.aggregates} == {
        ("weight", "mean"),
        ("weight", "max"),
    }
    assert all(item.source != "point_size" for item in layer.aggregates)

    dataset = ms.MassiveScatterDataset(output)
    exact = dataset.view(
        min_x=0,
        max_x=2,
        min_y=0,
        max_y=0,
        pixel_width=100,
        pixel_height=100,
        max_points=10,
    )["layers"][0]
    assert exact["mode"] == "exact"
    assert exact["fields"]["kind"] == ["E3", "E3", "P3", "other"]
    assert exact["fields"]["weight"] == [1.0, 3.0, 9.0, 17.0]

    aggregate = dataset.view(
        min_x=0,
        max_x=2,
        min_y=0,
        max_y=0,
        pixel_width=1,
        pixel_height=1,
        max_points=1,
        max_cells=2,
    )["layers"][0]
    assert aggregate["mode"] == "aggregate"
    by_request = {
        item.reducer: aggregate["aggregates"][item.key] for item in layer.aggregates
    }
    assert math.isclose(by_request["mean"][0], 13.0 / 3.0)
    assert by_request["max"][0] == 9.0
    assert by_request["mean"][1] == 17.0
    assert by_request["max"][1] == 17.0


def test_constant_color_and_count_require_no_source_field(tmp_path):
    batch = pa.record_batch([[0, 1], [0, 1]], names=["x", "y"])
    fig, ax = ms.subplots()
    ax.scatter([batch], x="x", y="y", color="#ff0080", alpha=0.5)
    output = tmp_path / "constant.msplot"
    manifest = fig.write(output, config=ms.BuildConfig(base_cell_size=1))
    layer = manifest.layers[0]
    assert layer.plot is not None
    assert layer.plot.scatter.color.value == "#ff0080"
    assert layer.exact_fields == {}
    assert layer.aggregates == ()


def test_multiple_scatter_layers_keep_independent_lod_and_zorder(tmp_path):
    dense = pa.record_batch([list(range(20)), [0] * 20], names=["x", "y"])
    sparse = pa.record_batch([[0, 10], [1, 1]], names=["x", "y"])

    fig, ax = ms.subplots()
    dense_handle = ax.scatter(
        [dense], x="x", y="y", color="red", label="dense"
    )
    sparse_handle = ax.scatter(
        [sparse], x="x", y="y", color="blue", label="sparse", zorder=-1
    )
    ax.legend()
    assert dense_handle.id == "layer-000"
    assert sparse_handle.id == "layer-001"
    assert sparse_handle.zorder == -1

    output = tmp_path / "layers.msplot"
    manifest = fig.write(
        output,
        config=ms.BuildConfig(base_cell_size=1, part_rows=8),
    )
    assert [layer.id for layer in manifest.layers] == ["layer-000", "layer-001"]
    assert [layer.zorder for layer in manifest.layers] == [0.0, -1.0]
    assert manifest.point_count == 22
    assert manifest.min_y == 0 and manifest.max_y == 1

    response = ms.MassiveScatterDataset(output).view(
        min_x=0,
        max_x=19,
        min_y=0,
        max_y=1,
        pixel_width=100,
        pixel_height=100,
        max_points=5,
        max_cells=10,
    )
    assert response["origin"] == [0, 0]
    assert [layer["id"] for layer in response["layers"]] == [
        "layer-001",
        "layer-000",
    ]
    by_id = {layer["id"]: layer for layer in response["layers"]}
    assert by_id["layer-001"]["mode"] == "exact"
    assert by_id["layer-001"]["point_count"] == 2
    assert by_id["layer-000"]["mode"] == "aggregate"
    assert by_id["layer-000"]["cell_count"] <= 10
''',
)
