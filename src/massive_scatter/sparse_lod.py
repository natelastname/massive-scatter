from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .manifest import LevelManifest
from .spec import AggregateRequest

Progress = Callable[[str], None]


def _state_names(request: AggregateRequest) -> tuple[str, ...]:
    if request.reducer == "mean":
        return (f"{request.key}_sum", f"{request.key}_count")
    if request.reducer == "sum":
        return (f"{request.key}_sum",)
    return (f"{request.key}_value",)


def sparse_level_columns(
    aggregates: tuple[AggregateRequest, ...],
) -> tuple[str, ...]:
    names = ["cell_x", "cell_y", "count"]
    for request in aggregates:
        names.extend(_state_names(request))
    return tuple(names)


def sparse_state_columns(request: AggregateRequest) -> tuple[str, ...]:
    return _state_names(request)


def _table_name(level: int) -> str:
    return f"cells_{level}"


def _create_cell_table(
    connection: sqlite3.Connection,
    table: str,
    aggregates: tuple[AggregateRequest, ...],
) -> None:
    columns = [
        "cell_y INTEGER NOT NULL",
        "cell_x INTEGER NOT NULL",
        "count INTEGER NOT NULL",
    ]
    for request in aggregates:
        if request.reducer in {"sum", "mean"}:
            columns.append(f'"{request.key}_sum" REAL NOT NULL')
        if request.reducer == "mean":
            columns.append(f'"{request.key}_count" INTEGER NOT NULL')
        if request.reducer in {"min", "max"}:
            columns.append(f'"{request.key}_value" REAL NOT NULL')
    columns.append("PRIMARY KEY (cell_y, cell_x)")
    connection.execute(
        f'CREATE TABLE "{table}" ({", ".join(columns)}) WITHOUT ROWID'
    )


def _upsert_assignments(aggregates: tuple[AggregateRequest, ...]) -> list[str]:
    assignments = ['"count" = "count" + excluded."count"']
    for request in aggregates:
        if request.reducer in {"sum", "mean"}:
            name = f"{request.key}_sum"
            assignments.append(f'"{name}" = "{name}" + excluded."{name}"')
        if request.reducer == "mean":
            name = f"{request.key}_count"
            assignments.append(f'"{name}" = "{name}" + excluded."{name}"')
        if request.reducer in {"min", "max"}:
            name = f"{request.key}_value"
            function = "MIN" if request.reducer == "min" else "MAX"
            assignments.append(
                f'"{name}" = {function}("{name}", excluded."{name}")'
            )
    return assignments


def _upsert_sql(table: str, aggregates: tuple[AggregateRequest, ...]) -> str:
    names = ["cell_y", "cell_x", "count"]
    for request in aggregates:
        names.extend(_state_names(request))
    quoted = [f'"{name}"' for name in names]
    placeholders = ", ".join("?" for _ in names)
    assignments = ", ".join(_upsert_assignments(aggregates))
    return (
        f'INSERT INTO "{table}" ({", ".join(quoted)}) VALUES ({placeholders}) '
        f"ON CONFLICT(cell_y, cell_x) DO UPDATE SET {assignments}"
    )


def _parent_sql(
    child: str,
    parent: str,
    aggregates: tuple[AggregateRequest, ...],
) -> str:
    names = ["cell_y", "cell_x", "count"]
    for request in aggregates:
        names.extend(_state_names(request))
    quoted = [f'"{name}"' for name in names]
    selected = ['"cell_y" >> 1', '"cell_x" >> 1', '"count"']
    selected.extend(f'"{name}"' for name in names[3:])
    assignments = ", ".join(_upsert_assignments(aggregates))
    # WHERE 1 avoids SQLite's INSERT ... SELECT ... ON CONFLICT parsing
    # ambiguity and keeps the operation as one streaming child-table pass.
    return (
        f'INSERT INTO "{parent}" ({", ".join(quoted)}) '
        f'SELECT {", ".join(selected)} FROM "{child}" WHERE 1 '
        f"ON CONFLICT(cell_y, cell_x) DO UPDATE SET {assignments}"
    )


def _batch_rows(
    batch: pa.RecordBatch,
    *,
    min_x: int,
    min_y: int,
    base_cell_size: int,
    aggregates: tuple[AggregateRequest, ...],
    storage_index: dict[str, int],
) -> tuple[Iterator[tuple[object, ...]], int]:
    x = np.asarray(batch.column(0), dtype=np.int64)
    y = np.asarray(batch.column(1), dtype=np.int64)
    if len(x) == 0:
        return iter(()), 0

    cell_x = (x - min_x) // base_cell_size
    cell_y = (y - min_y) // base_cell_size
    order = np.lexsort((cell_x, cell_y))
    cell_x = cell_x[order]
    cell_y = cell_y[order]
    boundaries = (
        np.flatnonzero((cell_x[1:] != cell_x[:-1]) | (cell_y[1:] != cell_y[:-1]))
        + 1
    )
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(order)]))
    counts = (stops - starts).astype(np.int64, copy=False)

    columns: list[list[object]] = [
        [int(value) for value in cell_y[starts]],
        [int(value) for value in cell_x[starts]],
        [int(value) for value in counts],
    ]
    value_cache: dict[str, np.ndarray] = {}
    for request in aggregates:
        values = value_cache.get(request.storage)
        if values is None:
            values = np.asarray(
                batch.column(storage_index[request.storage]), dtype=np.float64
            )[order]
            value_cache[request.storage] = values
        if request.reducer in {"sum", "mean"}:
            reduced = np.add.reduceat(values, starts)
            columns.append([float(value) for value in reduced])
        if request.reducer == "mean":
            columns.append([int(value) for value in counts])
        elif request.reducer == "min":
            reduced = np.minimum.reduceat(values, starts)
            columns.append([float(value) for value in reduced])
        elif request.reducer == "max":
            reduced = np.maximum.reduceat(values, starts)
            columns.append([float(value) for value in reduced])

    return (tuple(values) for values in zip(*columns, strict=True)), len(starts)


def _populate_base_level(
    connection: sqlite3.Connection,
    *,
    table: str,
    point_files: list[Path],
    point_count: int,
    min_x: int,
    min_y: int,
    base_cell_size: int,
    batch_size: int,
    aggregates: tuple[AggregateRequest, ...],
    progress: Progress,
) -> None:
    storage_columns = list(dict.fromkeys(request.storage for request in aggregates))
    columns = ["x", "y", *storage_columns]
    storage_index = {name: index + 2 for index, name in enumerate(storage_columns)}
    statement = _upsert_sql(table, aggregates)
    processed = 0
    next_report = 1_000_000

    for point_file in point_files:
        parquet = pq.ParquetFile(point_file)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            rows, _ = _batch_rows(
                batch,
                min_x=min_x,
                min_y=min_y,
                base_cell_size=base_cell_size,
                aggregates=aggregates,
                storage_index=storage_index,
            )
            connection.executemany(statement, rows)
            processed += batch.num_rows
            if processed >= next_report or processed == point_count:
                progress(
                    f"LOD 0 aggregated {processed:,}/{point_count:,} points into "
                    "sparse cells"
                )
                next_report = ((processed // 1_000_000) + 1) * 1_000_000
        connection.commit()


def _arrow_schema(aggregates: tuple[AggregateRequest, ...]) -> pa.Schema:
    fields = [
        pa.field("cell_x", pa.int64(), nullable=False),
        pa.field("cell_y", pa.int64(), nullable=False),
        pa.field("count", pa.uint64(), nullable=False),
    ]
    for request in aggregates:
        if request.reducer in {"sum", "mean"}:
            fields.append(pa.field(f"{request.key}_sum", pa.float64(), nullable=False))
        if request.reducer == "mean":
            fields.append(pa.field(f"{request.key}_count", pa.uint64(), nullable=False))
        if request.reducer in {"min", "max"}:
            fields.append(pa.field(f"{request.key}_value", pa.float64(), nullable=False))
    return pa.schema(fields)


def _rows_to_table(
    rows: list[tuple[object, ...]],
    aggregates: tuple[AggregateRequest, ...],
) -> pa.Table:
    schema = _arrow_schema(aggregates)
    columns = list(zip(*rows, strict=True))
    arrays = [pa.array(values, type=field.type) for values, field in zip(columns, schema)]
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_level_index(level_path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.table(
        {
            "path": pa.array([row["path"] for row in rows], pa.string()),
            "count": pa.array([row["count"] for row in rows], pa.uint64()),
            "min_x": pa.array([row["min_x"] for row in rows], pa.int64()),
            "max_x": pa.array([row["max_x"] for row in rows], pa.int64()),
            "min_y": pa.array([row["min_y"] for row in rows], pa.int64()),
            "max_y": pa.array([row["max_y"] for row in rows], pa.int64()),
        }
    )
    pq.write_table(table, level_path / "index.parquet", compression="zstd")


def _export_level(
    connection: sqlite3.Connection,
    *,
    dataset_path: Path,
    level: int,
    table: str,
    cell_size: int,
    shape: tuple[int, int],
    part_rows: int,
    batch_size: int,
    aggregates: tuple[AggregateRequest, ...],
    progress: Progress,
) -> LevelManifest:
    level_path = dataset_path / "lod" / str(level)
    level_path.mkdir(parents=True, exist_ok=True)
    names = sparse_level_columns(aggregates)
    selected = ", ".join(f'"{name}"' for name in names)
    cursor = connection.execute(
        f'SELECT {selected} FROM "{table}" ORDER BY cell_y, cell_x'
    )
    rows_per_part = max(1, min(part_rows, batch_size))
    index_rows: list[dict[str, object]] = []
    occupied_cells = 0
    part_index = 0

    while True:
        rows = cursor.fetchmany(rows_per_part)
        if not rows:
            break
        typed_rows = [tuple(row) for row in rows]
        table_value = _rows_to_table(typed_rows, aggregates)
        path = level_path / f"part-{part_index:06d}.parquet"
        pq.write_table(
            table_value,
            path,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            row_group_size=min(rows_per_part, 131_072),
        )
        cell_x = table_value["cell_x"].combine_chunks().to_numpy(zero_copy_only=False)
        cell_y = table_value["cell_y"].combine_chunks().to_numpy(zero_copy_only=False)
        count = table_value.num_rows
        index_rows.append(
            {
                "path": path.relative_to(dataset_path).as_posix(),
                "count": count,
                "min_x": int(cell_x.min()),
                "max_x": int(cell_x.max()),
                "min_y": int(cell_y.min()),
                "max_y": int(cell_y.max()),
            }
        )
        occupied_cells += count
        part_index += 1

    _write_level_index(level_path, index_rows)
    progress(
        f"LOD {level}: {occupied_cells:,} occupied cells in {part_index:,} "
        "Parquet parts"
    )
    return LevelManifest(
        level=level,
        cell_size=cell_size,
        height=shape[0],
        width=shape[1],
        occupied_cells=occupied_cells,
    )


def _configure_sqlite(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA locking_mode = EXCLUSIVE")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -131072")


def build_sparse_lod_pyramid(
    dataset_path: Path,
    *,
    point_files: list[Path],
    point_count: int,
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
    base_cell_size: int,
    batch_size: int,
    part_rows: int,
    aggregates: tuple[AggregateRequest, ...] = (),
    progress: Progress | None = None,
) -> tuple[LevelManifest, ...]:
    """Build an occupied-cell-only mergeable LOD pyramid.

    The temporary SQLite tables are an external-memory aggregation index. The
    portable artifact contains only sparse Parquet rows and small per-level
    part indexes; empty logical cells are never materialized.
    """

    report = progress or (lambda _message: None)
    base_shape = (
        math.ceil((max_y - min_y + 1) / base_cell_size),
        math.ceil((max_x - min_x + 1) / base_cell_size),
    )
    database_path = dataset_path / ".lod-build.sqlite3"
    connection = sqlite3.connect(database_path)
    _configure_sqlite(connection)
    levels: list[LevelManifest] = []

    try:
        table = _table_name(0)
        _create_cell_table(connection, table, aggregates)
        report(
            f"building sparse LOD 0: {base_shape[1]} × {base_shape[0]} logical "
            f"cells at {base_cell_size} units/cell"
        )
        _populate_base_level(
            connection,
            table=table,
            point_files=point_files,
            point_count=point_count,
            min_x=min_x,
            min_y=min_y,
            base_cell_size=base_cell_size,
            batch_size=batch_size,
            aggregates=aggregates,
            progress=report,
        )

        level = 0
        shape = base_shape
        while True:
            levels.append(
                _export_level(
                    connection,
                    dataset_path=dataset_path,
                    level=level,
                    table=table,
                    cell_size=base_cell_size * 2**level,
                    shape=shape,
                    part_rows=part_rows,
                    batch_size=batch_size,
                    aggregates=aggregates,
                    progress=report,
                )
            )
            if shape == (1, 1):
                break

            parent_level = level + 1
            parent_shape = ((shape[0] + 1) // 2, (shape[1] + 1) // 2)
            parent = _table_name(parent_level)
            report(
                f"building sparse LOD {parent_level}: {parent_shape[1]} × "
                f"{parent_shape[0]} logical cells"
            )
            _create_cell_table(connection, parent, aggregates)
            connection.execute(_parent_sql(table, parent, aggregates))
            connection.commit()
            connection.execute(f'DROP TABLE "{table}"')
            connection.commit()
            table = parent
            level = parent_level
            shape = parent_shape

        return tuple(levels)
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
