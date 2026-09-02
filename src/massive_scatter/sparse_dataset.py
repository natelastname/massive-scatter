from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from .manifest import LayerManifest, LevelManifest
from .sparse_lod import sparse_level_columns, sparse_state_columns
from .spec import AggregateRequest


class _IndexRow(TypedDict):
    path: str
    count: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int


def _index_row(value: dict[str, object]) -> _IndexRow:
    path = value.get("path")
    count = value.get("count")
    min_x = value.get("min_x")
    max_x = value.get("max_x")
    min_y = value.get("min_y")
    max_y = value.get("max_y")
    if not isinstance(path, str):
        raise TypeError("Sparse LOD index path must be a string.")
    integers = (count, min_x, max_x, min_y, max_y)
    if not all(isinstance(item, int) for item in integers):
        raise TypeError("Sparse LOD index bounds/count must be integers.")
    assert isinstance(count, int)
    assert isinstance(min_x, int)
    assert isinstance(max_x, int)
    assert isinstance(min_y, int)
    assert isinstance(max_y, int)
    return {
        "path": path,
        "count": count,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
    }


class SparseLodReader:
    """Read occupied-cell Parquet LOD levels with coarse part pruning."""

    def __init__(self, path: Path, manifest: LayerManifest) -> None:
        self.path = path
        self.manifest = manifest
        self._indexes: dict[int, list[_IndexRow]] = {}

    def _index(self, level: int) -> list[_IndexRow]:
        cached = self._indexes.get(level)
        if cached is not None:
            return cached
        path = self.path / "lod" / str(level) / "index.parquet"
        rows = [_index_row(row) for row in pq.read_table(path).to_pylist()]
        self._indexes[level] = rows
        return rows

    def _candidate_parts(
        self,
        level: int,
        *,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> tuple[list[Path], int]:
        paths: list[Path] = []
        upper_bound = 0
        for part in self._index(level):
            if (
                part["max_x"] < x0
                or part["min_x"] >= x1
                or part["max_y"] < y0
                or part["min_y"] >= y1
            ):
                continue
            paths.append(self.path / part["path"])
            upper_bound += part["count"]
        return paths, upper_bound

    @staticmethod
    def _cell_bounds(
        level: LevelManifest,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> tuple[int, int, int, int]:
        x0 = max(0, math.floor(min_x / level.cell_size))
        x1 = min(level.width, math.floor(max_x / level.cell_size) + 1)
        y0 = max(0, math.floor(min_y / level.cell_size))
        y1 = min(level.height, math.floor(max_y / level.cell_size) + 1)
        return x0, x1, y0, y1

    def choose_level(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        pixel_width: int,
        pixel_height: int,
        max_cells: int,
    ) -> LevelManifest:
        units_per_pixel = max(
            (max_x - min_x) / max(1, pixel_width),
            (max_y - min_y) / max(1, pixel_height),
            1.0,
        )
        level = self.manifest.levels[-1]
        for candidate in self.manifest.levels:
            if candidate.cell_size >= units_per_pixel:
                level = candidate
                break

        while True:
            x0, x1, y0, y1 = self._cell_bounds(
                level,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
            )
            _, upper_bound = self._candidate_parts(
                level.level, x0=x0, x1=x1, y0=y0, y1=y1
            )
            if upper_bound <= max_cells or level.level >= self.manifest.max_level:
                return level
            level = self.manifest.levels[level.level + 1]

    @staticmethod
    def _finalized_aggregate(request: AggregateRequest, table: pa.Table) -> list[float]:
        state = sparse_state_columns(request)
        if request.reducer == "sum":
            values = np.asarray(table[state[0]].combine_chunks(), dtype=np.float64)
        elif request.reducer == "mean":
            sums = np.asarray(table[state[0]].combine_chunks(), dtype=np.float64)
            counts = np.asarray(table[state[1]].combine_chunks(), dtype=np.uint64)
            values = sums / counts
        else:
            values = np.asarray(table[state[0]].combine_chunks(), dtype=np.float64)
        return [float(value) for value in values]

    def aggregate_view(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        pixel_width: int,
        pixel_height: int,
        max_cells: int,
    ) -> dict[str, Any]:
        level = self.choose_level(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            max_cells=max_cells,
        )
        x0, x1, y0, y1 = self._cell_bounds(
            level,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )
        cell_size = level.cell_size
        origin_x = x0 * cell_size
        origin_y = y0 * cell_size

        def empty() -> dict[str, Any]:
            return {
                "mode": "aggregate",
                "level": level.level,
                "cell_size": cell_size,
                "origin": [origin_x, origin_y],
                "x": [],
                "y": [],
                "count": [],
                "color": None,
                "aggregates": {},
                "cell_count": 0,
            }

        if x0 >= x1 or y0 >= y1:
            return empty()
        paths, _ = self._candidate_parts(level.level, x0=x0, x1=x1, y0=y0, y1=y1)
        if not paths:
            return empty()

        dataset = pads.dataset([str(path) for path in paths], format="parquet")
        predicate = (
            (pads.field("cell_x") >= x0)
            & (pads.field("cell_x") < x1)
            & (pads.field("cell_y") >= y0)
            & (pads.field("cell_y") < y1)
        )
        table = dataset.to_table(
            columns=list(sparse_level_columns(self.manifest.aggregates)),
            filter=predicate,
        )
        if table.num_rows == 0:
            return empty()
        if table.num_rows > max_cells:
            # The part index is deliberately an upper bound, so reaching this
            # branch should only be possible at the final one-cell level.
            raise RuntimeError(
                f"Sparse LOD query returned {table.num_rows:,} cells, exceeding "
                f"the {max_cells:,} cell budget."
            )

        cell_x = np.asarray(table["cell_x"].combine_chunks(), dtype=np.int64)
        cell_y = np.asarray(table["cell_y"].combine_chunks(), dtype=np.int64)
        counts = np.asarray(table["count"].combine_chunks(), dtype=np.uint64)
        half = cell_size / 2
        x_values = ((cell_x - x0).astype(np.float64) * cell_size + half).tolist()
        y_values = ((cell_y - y0).astype(np.float64) * cell_size + half).tolist()
        aggregate_values = {
            request.key: self._finalized_aggregate(request, table)
            for request in self.manifest.aggregates
        }

        legacy_color: list[float] | None = None
        if self.manifest.color_field:
            request = next(
                (
                    item
                    for item in self.manifest.aggregates
                    if item.source == self.manifest.color_field
                ),
                None,
            )
            if request is not None:
                legacy_color = aggregate_values[request.key]

        return {
            "mode": "aggregate",
            "level": level.level,
            "cell_size": cell_size,
            "origin": [origin_x, origin_y],
            "x": x_values,
            "y": y_values,
            "count": [int(value) for value in counts],
            "color": legacy_color,
            "aggregates": aggregate_values,
            "cell_count": table.num_rows,
        }

    def check(self) -> list[str]:
        problems: list[str] = []
        expected_columns = set(sparse_level_columns(self.manifest.aggregates))
        for level in self.manifest.levels:
            index_path = self.path / "lod" / str(level.level) / "index.parquet"
            if not index_path.is_file():
                problems.append(f"missing LOD {level.level} index")
                continue
            parts = self._index(level.level)
            indexed_cells = sum(part["count"] for part in parts)
            if indexed_cells != level.occupied_cells:
                problems.append(
                    f"LOD {level.level} index count {indexed_cells} != manifest "
                    f"occupied-cell count {level.occupied_cells}"
                )
            for part in parts:
                path = self.path / part["path"]
                if not path.is_file():
                    problems.append(f"missing LOD part: {part['path']}")
                    continue
                names = set(pq.ParquetFile(path).schema_arrow.names)
                if names != expected_columns:
                    problems.append(f"LOD part schema differs: {part['path']}")

        top = self.manifest.levels[-1]
        top_parts = [self.path / part["path"] for part in self._index(top.level)]
        if top_parts:
            total = 0
            for path in top_parts:
                values = pq.read_table(path, columns=["count"])["count"]
                total += int(np.asarray(values.combine_chunks(), dtype=np.uint64).sum())
            if total != self.manifest.point_count:
                problems.append(
                    f"top-level count sum {total} != manifest point count "
                    f"{self.manifest.point_count}"
                )
        return problems
