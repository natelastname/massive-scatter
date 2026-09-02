from __future__ import annotations

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
        if (
            max_x - min_x + 1 > MAX_SAFE_VIEWER_EXTENT
            or max_y - min_y + 1 > MAX_SAFE_VIEWER_EXTENT
        ):
            raise ValueError(
                "The union of layer coordinates exceeds 2^53-1 on an axis; the "
                "viewer requires the shared figure span to remain exactly "
                "representable."
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
