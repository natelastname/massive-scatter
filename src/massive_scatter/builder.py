from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .lod import build_lod_pyramid
from .manifest import MAX_SAFE_VIEWER_EXTENT, Manifest

Progress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """Bounded-memory build settings."""

    tile_size: int = 256
    base_cell_size: int = 64
    part_rows: int = 1_000_000
    batch_size: int = 131_072
    overwrite: bool = False

    def validate(self) -> None:
        if self.tile_size < 2 or self.tile_size & (self.tile_size - 1):
            raise ValueError("tile_size must be a power of two greater than one.")
        if self.base_cell_size < 1 or self.base_cell_size & (self.base_cell_size - 1):
            raise ValueError("base_cell_size must be a positive power of two.")
        if self.part_rows < 1:
            raise ValueError("part_rows must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")


@dataclass(frozen=True, slots=True)
class _IngestResult:
    point_count: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    point_files: tuple[Path, ...]


def _canonical_table(
    value: pa.RecordBatch | pa.Table,
    *,
    x_field: str,
    y_field: str,
    color_field: str | None,
) -> pa.Table:
    table = (
        pa.Table.from_batches([value]) if isinstance(value, pa.RecordBatch) else value
    )
    required = [x_field, y_field] + ([color_field] if color_field else [])
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"Input batch is missing columns: {', '.join(missing)}")

    x = table[x_field]
    y = table[y_field]
    if not pa.types.is_integer(x.type) or not pa.types.is_integer(y.type):
        raise TypeError("x and y columns must use an integer Arrow type.")
    if x.null_count or y.null_count:
        raise ValueError("x and y columns may not contain null values.")

    x = pc.cast(x, pa.int64(), safe=True)
    y = pc.cast(y, pa.int64(), safe=True)
    columns: dict[str, pa.Array | pa.ChunkedArray] = {"x": x, "y": y}

    if color_field is not None:
        color = table[color_field]
        if not (
            pa.types.is_integer(color.type)
            or pa.types.is_floating(color.type)
            or pa.types.is_decimal(color.type)
        ):
            raise TypeError("The color column must be numeric.")
        if color.null_count:
            raise ValueError("The color column may not contain null values.")
        color = pc.cast(color, pa.float64(), safe=True)
        color_values = np.asarray(color.combine_chunks(), dtype=np.float64)
        if not np.isfinite(color_values).all():
            raise ValueError("The color column may not contain NaN or infinite values.")
        columns["color"] = color

    return pa.table(columns)


def _write_point_parts(
    dataset_path: Path,
    batches: Iterable[pa.RecordBatch | pa.Table],
    *,
    x_field: str,
    y_field: str,
    color_field: str | None,
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

    def flush() -> None:
        nonlocal pending, pending_rows, part_index, point_count
        nonlocal min_x, max_x, min_y, max_y
        if not pending:
            return

        table = pa.concat_tables(pending).combine_chunks()
        path = points_path / f"part-{part_index:06d}.parquet"
        pq.write_table(
            table,
            path,
            compression="zstd",
            use_dictionary=False,
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
            color_field=color_field,
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
    )


def build_dataset(
    output: str | Path,
    batches: Iterable[pa.RecordBatch | pa.Table],
    *,
    x: str = "x",
    y: str = "y",
    color: str | None = None,
    config: BuildConfig | None = None,
    progress: Progress | None = None,
) -> Manifest:
    """Build an exact-point store and sparse aggregate pyramid.

    Input is consumed as Arrow batches. Memory use is bounded by the caller's
    batch size, one Parquet output part, and a small number of numerical Zarr
    chunks; it is independent of the full point count and rectangular extent.
    """

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
        report("streaming exact points to Parquet")
        ingest = _write_point_parts(
            temporary_path,
            batches,
            x_field=x,
            y_field=y,
            color_field=color,
            part_rows=settings.part_rows,
            progress=report,
        )

        width = ingest.max_x - ingest.min_x + 1
        height = ingest.max_y - ingest.min_y + 1
        if width > MAX_SAFE_VIEWER_EXTENT or height > MAX_SAFE_VIEWER_EXTENT:
            raise ValueError(
                "The MVP preserves arbitrary int64 origins, but each axis span must "
                "be at most 2^53-1 so unit offsets remain exact in JavaScript."
            )

        levels = build_lod_pyramid(
            temporary_path,
            point_files=list(ingest.point_files),
            point_count=ingest.point_count,
            min_x=ingest.min_x,
            max_x=ingest.max_x,
            min_y=ingest.min_y,
            max_y=ingest.max_y,
            tile_size=settings.tile_size,
            base_cell_size=settings.base_cell_size,
            batch_size=settings.batch_size,
            has_color=color is not None,
            progress=report,
        )
        manifest = Manifest(
            point_count=ingest.point_count,
            min_x=ingest.min_x,
            max_x=ingest.max_x,
            min_y=ingest.min_y,
            max_y=ingest.max_y,
            tile_size=settings.tile_size,
            base_cell_size=settings.base_cell_size,
            color_field=color,
            levels=levels,
        )
        manifest.save(temporary_path)

        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        temporary_path.replace(output_path)
        report(f"built {output_path} with {manifest.point_count:,} points")
        return manifest
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
