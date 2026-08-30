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
    """Disk-backed set of occupied Zarr chunks used while building parents."""

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


def _create_level_arrays(
    root: zarr.Group,
    *,
    level: int,
    shape: tuple[int, int],
    tile_size: int,
    has_color: bool,
) -> tuple[zarr.Array, zarr.Array | None]:
    levels = root.require_group("levels")
    level_group = levels.create_group(str(level))
    count = level_group.create_array(
        name="count",
        shape=shape,
        chunks=(tile_size, tile_size),
        dtype="uint64",
        fill_value=0,
        config={"write_empty_chunks": False},
    )
    color_max: zarr.Array | None = None
    if has_color:
        color_max = level_group.create_array(
            name="color_max",
            shape=shape,
            chunks=(tile_size, tile_size),
            dtype="float64",
            fill_value=np.nan,
            config={"write_empty_chunks": False},
        )
    return count, color_max


def _chunk_slices(
    shape: tuple[int, int], tile_size: int, chunk_y: int, chunk_x: int
) -> tuple[slice, slice]:
    y0 = chunk_y * tile_size
    x0 = chunk_x * tile_size
    return (
        slice(y0, min(shape[0], y0 + tile_size)),
        slice(x0, min(shape[1], x0 + tile_size)),
    )


def _build_base_level(
    *,
    count_array: zarr.Array,
    color_array: zarr.Array | None,
    point_files: list[Path],
    min_x: int,
    min_y: int,
    base_cell_size: int,
    tile_size: int,
    batch_size: int,
    occupied: _OccupiedChunks,
) -> None:
    for point_file in point_files:
        columns = ["x", "y"] + (["color"] if color_array is not None else [])
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
            color = (
                np.asarray(batch.column(2), dtype=np.float64)[order]
                if color_array is not None
                else None
            )

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

                if color_array is not None and color is not None:
                    colors = np.asarray(color_array[ys, xs], dtype=np.float64)
                    work = np.where(np.isnan(colors), -np.inf, colors)
                    np.maximum.at(work, (ly, lx), color[start:stop])
                    work[counts == 0] = np.nan
                    color_array[ys, xs] = work

                occupied.add(0, cy, cx)
        occupied.commit()


def _downsample_sum(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    padded = np.zeros((height + height % 2, width + width % 2), dtype=np.uint64)
    padded[:height, :width] = values
    return padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2).sum(
        axis=(1, 3), dtype=np.uint64
    )


def _downsample_max(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    padded = np.full(
        (height + height % 2, width + width % 2), -np.inf, dtype=np.float64
    )
    padded[:height, :width] = np.where(np.isnan(values), -np.inf, values)
    return padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2).max(
        axis=(1, 3)
    )


def _build_parent_level(
    *,
    child_count: zarr.Array,
    child_color: zarr.Array | None,
    parent_count: zarr.Array,
    parent_color: zarr.Array | None,
    child_level: int,
    parent_level: int,
    tile_size: int,
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
        colors = (
            np.full(output_shape, -np.inf, dtype=np.float64)
            if parent_color is not None
            else None
        )

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
            counts[offset_y:end_y, offset_x:end_x] += reduced_counts[
                : end_y - offset_y, : end_x - offset_x
            ]

            if child_color is not None and colors is not None:
                child_colors = np.asarray(
                    child_color[child_ys, child_xs], dtype=np.float64
                )
                reduced_colors = _downsample_max(child_colors)
                target = colors[offset_y:end_y, offset_x:end_x]
                np.maximum(
                    target,
                    reduced_colors[: end_y - offset_y, : end_x - offset_x],
                    out=target,
                )

        parent_count[parent_ys, parent_xs] = counts
        if parent_color is not None and colors is not None:
            colors[counts == 0] = np.nan
            parent_color[parent_ys, parent_xs] = colors
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
    has_color: bool,
    progress: Progress | None = None,
) -> tuple[LevelManifest, ...]:
    """Build sparse numerical LOD arrays without creating raster image tiles."""

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
        count, color = _create_level_arrays(
            root,
            level=0,
            shape=base_shape,
            tile_size=tile_size,
            has_color=has_color,
        )
        _build_base_level(
            count_array=count,
            color_array=color,
            point_files=point_files,
            min_x=min_x,
            min_y=min_y,
            base_cell_size=base_cell_size,
            tile_size=tile_size,
            batch_size=batch_size,
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
            parent_count, parent_color = _create_level_arrays(
                root,
                level=parent_level,
                shape=parent_shape,
                tile_size=tile_size,
                has_color=has_color,
            )
            child_count = _array(root, f"levels/{level}/count")
            child_color = (
                _array(root, f"levels/{level}/color_max") if has_color else None
            )
            _build_parent_level(
                child_count=child_count,
                child_color=child_color,
                parent_count=parent_count,
                parent_color=parent_color,
                child_level=level,
                parent_level=parent_level,
                tile_size=tile_size,
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
