from __future__ import annotations

import itertools
import math
import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from .manifest import LevelManifest
from .spec import AggregateRequest

Progress = Callable[[str], None]


def _array(group: zarr.Group, path: str) -> zarr.Array:
    value = group[path]
    if not isinstance(value, zarr.Array):
        raise TypeError(f"Expected Zarr array at {path}, found {type(value).__name__}.")
    return value


def _shape_2d(array: zarr.Array) -> tuple[int, int]:
    if len(array.shape) != 2:
        raise ValueError(f"Expected a two-dimensional array, found {array.shape}.")
    return int(array.shape[0]), int(array.shape[1])


class _OccupiedChunks:
    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path)
        self._connection.execute("""
            CREATE TABLE occupied (
                level INTEGER NOT NULL,
                chunk_y INTEGER NOT NULL,
                chunk_x INTEGER NOT NULL,
                PRIMARY KEY (level, chunk_y, chunk_x)
            ) WITHOUT ROWID
            """)

    def add(self, level: int, chunk_y: int, chunk_x: int) -> None:
        self._connection.execute(
            "INSERT OR IGNORE INTO occupied VALUES (?, ?, ?)",
            (level, chunk_y, chunk_x),
        )

    def commit(self) -> None:
        self._connection.commit()

    def count(self, level: int) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM occupied WHERE level = ?", (level,)
        ).fetchone()
        assert row is not None
        return int(row[0])

    def coordinates(self, level: int) -> Iterable[tuple[int, int]]:
        cursor = self._connection.execute(
            """
            SELECT chunk_y, chunk_x
            FROM occupied
            WHERE level = ?
            ORDER BY (chunk_y >> 1), (chunk_x >> 1), chunk_y, chunk_x
            """,
            (level,),
        )
        yield from ((int(y), int(x)) for y, x in cursor)

    def close(self) -> None:
        self._connection.close()


def _create_numeric_array(
    group: zarr.Group,
    name: str,
    *,
    shape: tuple[int, int],
    tile_size: int,
    dtype: str,
    fill_value,
) -> zarr.Array:
    return group.create_array(
        name=name,
        shape=shape,
        chunks=(tile_size, tile_size),
        dtype=dtype,
        fill_value=fill_value,
        config={"write_empty_chunks": False},
    )


def _create_level_arrays(
    root: zarr.Group,
    *,
    level: int,
    shape: tuple[int, int],
    tile_size: int,
    aggregates: tuple[AggregateRequest, ...],
) -> zarr.Array:
    levels = root.require_group("levels")
    level_group = levels.create_group(str(level))
    count = _create_numeric_array(
        level_group,
        "count",
        shape=shape,
        tile_size=tile_size,
        dtype="uint64",
        fill_value=0,
    )
    aggregate_group = level_group.create_group("aggregates")
    for request in aggregates:
        request_group = aggregate_group.create_group(request.key)
        if request.reducer in {"sum", "mean"}:
            _create_numeric_array(
                request_group,
                "sum",
                shape=shape,
                tile_size=tile_size,
                dtype="float64",
                fill_value=0.0,
            )
        if request.reducer == "mean":
            _create_numeric_array(
                request_group,
                "count",
                shape=shape,
                tile_size=tile_size,
                dtype="uint64",
                fill_value=0,
            )
        if request.reducer in {"min", "max"}:
            _create_numeric_array(
                request_group,
                "value",
                shape=shape,
                tile_size=tile_size,
                dtype="float64",
                fill_value=np.nan,
            )
    return count


def _chunk_slices(
    shape: tuple[int, int], tile_size: int, chunk_y: int, chunk_x: int
) -> tuple[slice, slice]:
    y0 = chunk_y * tile_size
    x0 = chunk_x * tile_size
    return (
        slice(y0, min(shape[0], y0 + tile_size)),
        slice(x0, min(shape[1], x0 + tile_size)),
    )


def _update_aggregate_chunk(
    root: zarr.Group,
    *,
    request: AggregateRequest,
    level: int,
    ys: slice,
    xs: slice,
    local_y: np.ndarray,
    local_x: np.ndarray,
    values: np.ndarray,
) -> None:
    prefix = f"levels/{level}/aggregates/{request.key}"
    if request.reducer in {"sum", "mean"}:
        sums_array = _array(root, f"{prefix}/sum")
        sums = np.asarray(sums_array[ys, xs], dtype=np.float64)
        np.add.at(sums, (local_y, local_x), values)
        sums_array[ys, xs] = sums
    if request.reducer == "mean":
        n_array = _array(root, f"{prefix}/count")
        n = np.asarray(n_array[ys, xs], dtype=np.uint64)
        np.add.at(n, (local_y, local_x), 1)
        n_array[ys, xs] = n
    elif request.reducer in {"min", "max"}:
        value_array = _array(root, f"{prefix}/value")
        current = np.asarray(value_array[ys, xs], dtype=np.float64)
        if request.reducer == "min":
            work = np.where(np.isnan(current), np.inf, current)
            np.minimum.at(work, (local_y, local_x), values)
        else:
            work = np.where(np.isnan(current), -np.inf, current)
            np.maximum.at(work, (local_y, local_x), values)
        value_array[ys, xs] = work


def _build_base_level(
    *,
    root: zarr.Group,
    count_array: zarr.Array,
    point_files: list[Path],
    min_x: int,
    min_y: int,
    base_cell_size: int,
    tile_size: int,
    batch_size: int,
    aggregates: tuple[AggregateRequest, ...],
    occupied: _OccupiedChunks,
) -> None:
    storage_columns = list(dict.fromkeys(request.storage for request in aggregates))
    columns = ["x", "y", *storage_columns]
    storage_index = {name: index + 2 for index, name in enumerate(storage_columns)}

    for point_file in point_files:
        parquet = pq.ParquetFile(point_file)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            x = np.asarray(batch.column(0), dtype=np.int64)
            y = np.asarray(batch.column(1), dtype=np.int64)
            cell_x = (x - min_x) // base_cell_size
            cell_y = (y - min_y) // base_cell_size
            chunk_x = cell_x // tile_size
            chunk_y = cell_y // tile_size
            local_x = cell_x % tile_size
            local_y = cell_y % tile_size

            order = np.lexsort((chunk_x, chunk_y))
            chunk_x = chunk_x[order]
            chunk_y = chunk_y[order]
            local_x = local_x[order]
            local_y = local_y[order]
            aggregate_values = {
                request.key: np.asarray(
                    batch.column(storage_index[request.storage]), dtype=np.float64
                )[order]
                for request in aggregates
            }

            if len(order) == 0:
                continue
            boundaries = (
                np.flatnonzero(
                    (chunk_x[1:] != chunk_x[:-1]) | (chunk_y[1:] != chunk_y[:-1])
                )
                + 1
            )
            starts = np.concatenate(([0], boundaries))
            stops = np.concatenate((boundaries, [len(order)]))

            for start, stop in zip(starts, stops, strict=True):
                cx = int(chunk_x[start])
                cy = int(chunk_y[start])
                ys, xs = _chunk_slices(_shape_2d(count_array), tile_size, cy, cx)
                counts = np.asarray(count_array[ys, xs], dtype=np.uint64)
                ly = local_y[start:stop]
                lx = local_x[start:stop]
                np.add.at(counts, (ly, lx), 1)
                count_array[ys, xs] = counts

                for request in aggregates:
                    _update_aggregate_chunk(
                        root,
                        request=request,
                        level=0,
                        ys=ys,
                        xs=xs,
                        local_y=ly,
                        local_x=lx,
                        values=aggregate_values[request.key][start:stop],
                    )

                occupied.add(0, cy, cx)
        occupied.commit()


def _downsample_sum(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    padded = np.zeros((height + height % 2, width + width % 2), dtype=values.dtype)
    padded[:height, :width] = values
    return padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2).sum(
        axis=(1, 3), dtype=values.dtype
    )


def _downsample_extreme(values: np.ndarray, *, reducer: str) -> np.ndarray:
    height, width = values.shape
    fill = np.inf if reducer == "min" else -np.inf
    padded = np.full(
        (height + height % 2, width + width % 2), fill, dtype=np.float64
    )
    padded[:height, :width] = np.where(np.isnan(values), fill, values)
    reshaped = padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2)
    result = (
        reshaped.min(axis=(1, 3))
        if reducer == "min"
        else reshaped.max(axis=(1, 3))
    )
    result[~np.isfinite(result)] = np.nan
    return result


def _merge_parent_state(
    root: zarr.Group,
    *,
    request: AggregateRequest,
    child_level: int,
    parent_level: int,
    child_ys: slice,
    child_xs: slice,
    target_ys: slice,
    target_xs: slice,
    source_height: int,
    source_width: int,
) -> None:
    child_prefix = f"levels/{child_level}/aggregates/{request.key}"
    parent_prefix = f"levels/{parent_level}/aggregates/{request.key}"
    if request.reducer in {"sum", "mean"}:
        child = np.asarray(
            _array(root, f"{child_prefix}/sum")[child_ys, child_xs],
            dtype=np.float64,
        )
        reduced = _downsample_sum(child)
        parent_array = _array(root, f"{parent_prefix}/sum")
        target = np.asarray(parent_array[target_ys, target_xs], dtype=np.float64)
        target[:source_height, :source_width] += reduced[:source_height, :source_width]
        parent_array[target_ys, target_xs] = target
    if request.reducer == "mean":
        child_n = np.asarray(
            _array(root, f"{child_prefix}/count")[child_ys, child_xs],
            dtype=np.uint64,
        )
        reduced_n = _downsample_sum(child_n)
        parent_n_array = _array(root, f"{parent_prefix}/count")
        target_n = np.asarray(parent_n_array[target_ys, target_xs], dtype=np.uint64)
        target_n[:source_height, :source_width] += reduced_n[
            :source_height, :source_width
        ]
        parent_n_array[target_ys, target_xs] = target_n
    elif request.reducer in {"min", "max"}:
        child_values = np.asarray(
            _array(root, f"{child_prefix}/value")[child_ys, child_xs],
            dtype=np.float64,
        )
        reduced_values = _downsample_extreme(child_values, reducer=request.reducer)
        parent_array = _array(root, f"{parent_prefix}/value")
        target = np.asarray(parent_array[target_ys, target_xs], dtype=np.float64)
        source = reduced_values[:source_height, :source_width]
        target_slice = target[:source_height, :source_width]
        if request.reducer == "min":
            merged = np.minimum(
                np.where(np.isnan(target_slice), np.inf, target_slice),
                np.where(np.isnan(source), np.inf, source),
            )
        else:
            merged = np.maximum(
                np.where(np.isnan(target_slice), -np.inf, target_slice),
                np.where(np.isnan(source), -np.inf, source),
            )
        merged[~np.isfinite(merged)] = np.nan
        target[:source_height, :source_width] = merged
        parent_array[target_ys, target_xs] = target


def _build_parent_level(
    *,
    root: zarr.Group,
    child_count: zarr.Array,
    parent_count: zarr.Array,
    child_level: int,
    parent_level: int,
    tile_size: int,
    aggregates: tuple[AggregateRequest, ...],
    occupied: _OccupiedChunks,
) -> None:
    half_tile = tile_size // 2
    coordinates = occupied.coordinates(child_level)
    grouped = itertools.groupby(
        coordinates, key=lambda item: (item[0] // 2, item[1] // 2)
    )

    for (parent_chunk_y, parent_chunk_x), children in grouped:
        parent_ys, parent_xs = _chunk_slices(
            _shape_2d(parent_count), tile_size, parent_chunk_y, parent_chunk_x
        )
        output_shape = (
            parent_ys.stop - parent_ys.start,
            parent_xs.stop - parent_xs.start,
        )
        counts = np.zeros(output_shape, dtype=np.uint64)

        for child_chunk_y, child_chunk_x in children:
            child_ys, child_xs = _chunk_slices(
                _shape_2d(child_count), tile_size, child_chunk_y, child_chunk_x
            )
            child_counts = np.asarray(child_count[child_ys, child_xs], dtype=np.uint64)
            reduced_counts = _downsample_sum(child_counts)
            offset_y = (child_chunk_y % 2) * half_tile
            offset_x = (child_chunk_x % 2) * half_tile
            end_y = min(output_shape[0], offset_y + reduced_counts.shape[0])
            end_x = min(output_shape[1], offset_x + reduced_counts.shape[1])
            height = end_y - offset_y
            width = end_x - offset_x
            counts[offset_y:end_y, offset_x:end_x] += reduced_counts[:height, :width]

            target_ys = slice(parent_ys.start + offset_y, parent_ys.start + end_y)
            target_xs = slice(parent_xs.start + offset_x, parent_xs.start + end_x)
            for request in aggregates:
                _merge_parent_state(
                    root,
                    request=request,
                    child_level=child_level,
                    parent_level=parent_level,
                    child_ys=child_ys,
                    child_xs=child_xs,
                    target_ys=target_ys,
                    target_xs=target_xs,
                    source_height=height,
                    source_width=width,
                )

        parent_count[parent_ys, parent_xs] = counts
        occupied.add(parent_level, parent_chunk_y, parent_chunk_x)
    occupied.commit()


def build_lod_pyramid(
    dataset_path: Path,
    *,
    point_files: list[Path],
    point_count: int,
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
    tile_size: int,
    base_cell_size: int,
    batch_size: int,
    aggregates: tuple[AggregateRequest, ...] = (),
    progress: Progress | None = None,
) -> tuple[LevelManifest, ...]:
    """Build sparse numerical LOD arrays from mergeable aggregate states."""

    report = progress or (lambda _message: None)
    lod_path = dataset_path / "lod.zarr"
    root = zarr.open_group(lod_path, mode="w")
    root.attrs.update(
        {
            "format": "massive-scatter-lod",
            "tile_size": tile_size,
            "base_cell_size": base_cell_size,
        }
    )

    base_shape = (
        math.ceil((max_y - min_y + 1) / base_cell_size),
        math.ceil((max_x - min_x + 1) / base_cell_size),
    )
    index_path = dataset_path / ".lod-build-index.sqlite3"
    occupied = _OccupiedChunks(index_path)
    levels: list[LevelManifest] = []

    try:
        report(
            f"building LOD 0: {base_shape[1]} × {base_shape[0]} logical cells "
            f"at {base_cell_size} units/cell"
        )
        count = _create_level_arrays(
            root,
            level=0,
            shape=base_shape,
            tile_size=tile_size,
            aggregates=aggregates,
        )
        _build_base_level(
            root=root,
            count_array=count,
            point_files=point_files,
            min_x=min_x,
            min_y=min_y,
            base_cell_size=base_cell_size,
            tile_size=tile_size,
            batch_size=batch_size,
            aggregates=aggregates,
            occupied=occupied,
        )
        levels.append(
            LevelManifest(
                level=0,
                cell_size=base_cell_size,
                height=base_shape[0],
                width=base_shape[1],
                occupied_chunks=occupied.count(0),
            )
        )

        level = 0
        shape = base_shape
        while shape[0] > tile_size or shape[1] > tile_size:
            parent_level = level + 1
            parent_shape = ((shape[0] + 1) // 2, (shape[1] + 1) // 2)
            report(
                f"building LOD {parent_level}: {parent_shape[1]} × "
                f"{parent_shape[0]} logical cells"
            )
            parent_count = _create_level_arrays(
                root,
                level=parent_level,
                shape=parent_shape,
                tile_size=tile_size,
                aggregates=aggregates,
            )
            child_count = _array(root, f"levels/{level}/count")
            _build_parent_level(
                root=root,
                child_count=child_count,
                parent_count=parent_count,
                child_level=level,
                parent_level=parent_level,
                tile_size=tile_size,
                aggregates=aggregates,
                occupied=occupied,
            )
            levels.append(
                LevelManifest(
                    level=parent_level,
                    cell_size=base_cell_size * 2**parent_level,
                    height=parent_shape[0],
                    width=parent_shape[1],
                    occupied_chunks=occupied.count(parent_level),
                )
            )
            level = parent_level
            shape = parent_shape

        top_count_array = _array(root, f"levels/{level}/count")
        top_count = np.asarray(top_count_array[:], dtype=np.uint64)
        if int(top_count.sum(dtype=np.uint64)) != point_count:
            raise RuntimeError(
                "LOD count invariant failed: top-level count does not match input."
            )
        return tuple(levels)
    finally:
        occupied.close()
        index_path.unlink(missing_ok=True)
