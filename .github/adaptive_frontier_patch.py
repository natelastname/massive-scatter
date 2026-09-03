from pathlib import Path
import re

Path("src/massive_scatter/sparse_dataset.py").write_text('''from __future__ import annotations

import math
from pathlib import Path
from typing import TypedDict

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
    """Read occupied-cell Parquet levels as an implicit refinement tree."""

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

    def _empty_level_table(self, level: int) -> pa.Table:
        parts = self._index(level)
        if not parts:
            raise RuntimeError(f"LOD {level} has no indexed parts.")
        schema = pq.read_schema(self.path / parts[0]["path"])
        return pa.Table.from_batches([], schema=schema)

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

    def _read_bounds(
        self,
        level: int,
        *,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> pa.Table:
        if x0 >= x1 or y0 >= y1:
            return self._empty_level_table(level)
        paths, _ = self._candidate_parts(level, x0=x0, x1=x1, y0=y0, y1=y1)
        if not paths:
            return self._empty_level_table(level)
        dataset = pads.dataset([str(path) for path in paths], format="parquet")
        predicate = (
            (pads.field("cell_x") >= x0)
            & (pads.field("cell_x") < x1)
            & (pads.field("cell_y") >= y0)
            & (pads.field("cell_y") < y1)
        )
        return dataset.to_table(
            columns=list(sparse_level_columns(self.manifest.aggregates)),
            filter=predicate,
        )

    def choose_seed_level(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        pixel_width: int,
        pixel_height: int,
        max_primitives: int,
    ) -> LevelManifest:
        """Choose the finest uniform starting level whose coarse index fits budget."""

        units_per_pixel = max(
            (max_x - min_x) / max(1, pixel_width),
            (max_y - min_y) / max(1, pixel_height),
            1e-12,
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
                level.level,
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
            )
            if upper_bound <= max_primitives or level.level >= self.manifest.max_level:
                return level
            level = self.manifest.levels[level.level + 1]

    def view_table(
        self,
        level: LevelManifest,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> pa.Table:
        x0, x1, y0, y1 = self._cell_bounds(
            level,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )
        return self._read_bounds(level.level, x0=x0, x1=x1, y0=y0, y1=y1)

    def children(
        self,
        parent_level: int,
        parents: set[tuple[int, int]],
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> pa.Table:
        """Return visible occupied children of the selected parent cells."""

        if parent_level <= 0:
            raise ValueError("Level zero cells have exact points, not LOD children.")
        child = self.manifest.levels[parent_level - 1]
        if not parents:
            return self._empty_level_table(child.level)

        vx0, vx1, vy0, vy1 = self._cell_bounds(
            child,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )
        px = [coord[0] for coord in parents]
        py = [coord[1] for coord in parents]
        x0 = max(vx0, min(px) * 2)
        x1 = min(vx1, (max(px) + 1) * 2)
        y0 = max(vy0, min(py) * 2)
        y1 = min(vy1, (max(py) + 1) * 2)
        table = self._read_bounds(child.level, x0=x0, x1=x1, y0=y0, y1=y1)
        if table.num_rows == 0:
            return table

        cell_x = np.asarray(table["cell_x"].combine_chunks(), dtype=np.int64)
        cell_y = np.asarray(table["cell_y"].combine_chunks(), dtype=np.int64)
        keep = np.fromiter(
            (
                (int(x) // 2, int(y) // 2) in parents
                for x, y in zip(cell_x, cell_y, strict=True)
            ),
            dtype=np.bool_,
            count=table.num_rows,
        )
        return table.filter(pa.array(keep))

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

    def response_batch(
        self,
        level: int,
        table: pa.Table,
        *,
        origin_x: int,
        origin_y: int,
    ) -> dict[str, object]:
        """Finalize one selected frontier level for transport to the viewer."""

        level_manifest = self.manifest.levels[level]
        cell_size = level_manifest.cell_size
        if table.num_rows == 0:
            return {
                "level": level,
                "cell_size": cell_size,
                "x": [],
                "y": [],
                "count": [],
                "color": None,
                "aggregates": {},
                "cell_count": 0,
            }

        cell_x = np.asarray(table["cell_x"].combine_chunks(), dtype=np.int64)
        cell_y = np.asarray(table["cell_y"].combine_chunks(), dtype=np.int64)
        counts = np.asarray(table["count"].combine_chunks(), dtype=np.uint64)
        half = cell_size / 2
        x_values = (cell_x.astype(np.float64) * cell_size + half - origin_x).tolist()
        y_values = (cell_y.astype(np.float64) * cell_size + half - origin_y).tolist()
        aggregate_values = {
            request.key: self._finalized_aggregate(request, table)
            for request in self.manifest.aggregates
        }

        direct_color: list[float] | None = None
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
                direct_color = aggregate_values[request.key]

        return {
            "level": level,
            "cell_size": cell_size,
            "x": x_values,
            "y": y_values,
            "count": [int(value) for value in counts],
            "color": direct_color,
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
''')
Path("src/massive_scatter/dataset.py").write_text('''from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from .manifest import LayerManifest, Manifest
from .sparse_dataset import SparseLodReader


class _LayerDataset:
    """Query one layer by selecting an adaptive frontier through its LOD tree."""

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
    def _empty_points() -> dict[str, Any]:
        return {
            "x": [],
            "y": [],
            "color": None,
            "fields": {},
            "point_count": 0,
        }

    def _empty_response(self, origin_x: int, origin_y: int) -> dict[str, Any]:
        return {
            "origin": [origin_x, origin_y],
            "points": self._empty_points(),
            "cells": [],
            "primitive_count": 0,
        }

    @staticmethod
    def _table_coords(table: pa.Table) -> list[tuple[int, int]]:
        cell_x = np.asarray(table["cell_x"].combine_chunks(), dtype=np.int64)
        cell_y = np.asarray(table["cell_y"].combine_chunks(), dtype=np.int64)
        return [(int(x), int(y)) for x, y in zip(cell_x, cell_y, strict=True)]

    @staticmethod
    def _filter_cells(
        table: pa.Table,
        selected: set[tuple[int, int]],
        *,
        keep_selected: bool,
        child_parent: bool = False,
    ) -> pa.Table:
        if table.num_rows == 0:
            return table
        cell_x = np.asarray(table["cell_x"].combine_chunks(), dtype=np.int64)
        cell_y = np.asarray(table["cell_y"].combine_chunks(), dtype=np.int64)
        if child_parent:
            memberships = (
                (int(x) // 2, int(y) // 2) in selected
                for x, y in zip(cell_x, cell_y, strict=True)
            )
        else:
            memberships = (
                (int(x), int(y)) in selected
                for x, y in zip(cell_x, cell_y, strict=True)
            )
        mask = np.fromiter(
            (
                membership if keep_selected else not membership
                for membership in memberships
            ),
            dtype=np.bool_,
            count=table.num_rows,
        )
        return table.filter(pa.array(mask))

    @staticmethod
    def _select_refinements(
        table: pa.Table,
        child_counts: Counter[tuple[int, int]],
        *,
        primitive_count: int,
        max_primitives: int,
    ) -> tuple[set[tuple[int, int]], int]:
        coords = _LayerDataset._table_coords(table)
        counts = np.asarray(table["count"].combine_chunks(), dtype=np.uint64)
        candidates: list[tuple[int, int, int, tuple[int, int]]] = []
        for index, coord in enumerate(coords):
            delta = child_counts.get(coord, 0) - 1
            candidates.append((delta, int(counts[index]), index, coord))
        candidates.sort()

        selected: set[tuple[int, int]] = set()
        updated_count = primitive_count
        for delta, _count, _index, coord in candidates:
            if delta <= 0 or updated_count + delta <= max_primitives:
                selected.add(coord)
                updated_count += delta
        return selected, updated_count

    @staticmethod
    def _select_exact_cells(
        table: pa.Table,
        *,
        primitive_count: int,
        max_primitives: int,
    ) -> tuple[set[tuple[int, int]], int]:
        coords = _LayerDataset._table_coords(table)
        counts = np.asarray(table["count"].combine_chunks(), dtype=np.uint64)
        candidates = sorted(
            (
                (int(count) - 1, int(count), index, coord)
                for index, (count, coord) in enumerate(zip(counts, coords, strict=True))
            )
        )

        selected: set[tuple[int, int]] = set()
        conservative_count = primitive_count
        for delta, _count, _index, coord in candidates:
            if delta <= 0 or conservative_count + delta <= max_primitives:
                selected.add(coord)
                conservative_count += delta
        return selected, conservative_count

    def _exact_cells_view(
        self,
        selected_cells: set[tuple[int, int]],
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        origin_x: int,
        origin_y: int,
        expected_points: int,
    ) -> dict[str, Any]:
        if not selected_cells:
            return self._empty_points()

        base = self.manifest.base_cell_size
        min_cell_x = min(coord[0] for coord in selected_cells)
        max_cell_x = max(coord[0] for coord in selected_cells)
        min_cell_y = min(coord[1] for coord in selected_cells)
        max_cell_y = max(coord[1] for coord in selected_cells)

        viewport_min_x = self.manifest.min_x + math.ceil(min_x)
        viewport_max_x = self.manifest.min_x + math.floor(max_x)
        viewport_min_y = self.manifest.min_y + math.ceil(min_y)
        viewport_max_y = self.manifest.min_y + math.floor(max_y)
        cell_min_x = self.manifest.min_x + min_cell_x * base
        cell_max_x = self.manifest.min_x + (max_cell_x + 1) * base - 1
        cell_min_y = self.manifest.min_y + min_cell_y * base
        cell_max_y = self.manifest.min_y + (max_cell_y + 1) * base - 1

        absolute_min_x = max(viewport_min_x, cell_min_x)
        absolute_max_x = min(viewport_max_x, cell_max_x)
        absolute_min_y = max(viewport_min_y, cell_min_y)
        absolute_max_y = min(viewport_max_y, cell_max_y)
        if absolute_min_x > absolute_max_x or absolute_min_y > absolute_max_y:
            return self._empty_points()

        candidate_paths, _ = self._candidate_parts(
            absolute_min_x,
            absolute_max_x,
            absolute_min_y,
            absolute_max_y,
        )
        if not candidate_paths:
            return self._empty_points()

        dataset = pads.dataset([str(path) for path in candidate_paths], format="parquet")
        predicate = (
            (pads.field("x") >= absolute_min_x)
            & (pads.field("x") <= absolute_max_x)
            & (pads.field("y") >= absolute_min_y)
            & (pads.field("y") <= absolute_max_y)
        )
        storage_fields = self.manifest.exact_fields
        storage_columns = list(dict.fromkeys(storage_fields.values()))
        columns = ["x", "y", *storage_columns]
        storage_indexes = {
            source: columns.index(storage) for source, storage in storage_fields.items()
        }

        x_values: list[int] = []
        y_values: list[int] = []
        field_values: dict[str, list[Any]] = {source: [] for source in storage_fields}

        for batch in dataset.to_batches(
            columns=columns,
            filter=predicate,
            batch_size=131_072,
            batch_readahead=1,
            fragment_readahead=1,
        ):
            absolute_x = np.asarray(batch.column(0), dtype=np.int64)
            absolute_y = np.asarray(batch.column(1), dtype=np.int64)
            local_x = absolute_x - self.manifest.min_x
            local_y = absolute_y - self.manifest.min_y
            cell_x = local_x // base
            cell_y = local_y // base
            keep = np.fromiter(
                (
                    (int(x), int(y)) in selected_cells
                    for x, y in zip(cell_x, cell_y, strict=True)
                ),
                dtype=np.bool_,
                count=batch.num_rows,
            )
            if not keep.any():
                continue

            selected_indexes = np.flatnonzero(keep)
            selected_x = absolute_x[keep] - self.manifest.min_x - origin_x
            selected_y = absolute_y[keep] - self.manifest.min_y - origin_y
            x_values.extend(int(value) for value in selected_x)
            y_values.extend(int(value) for value in selected_y)

            arrow_indexes = pa.array(selected_indexes, type=pa.int64())
            for source, index in storage_indexes.items():
                field_values[source].extend(batch.column(index).take(arrow_indexes).to_pylist())

        if len(x_values) > expected_points:
            raise RuntimeError(
                "Exact leaf query returned more points than the selected level-zero "
                "cell counts permit."
            )

        direct_color = (
            field_values.get(self.manifest.color_field)
            if self.manifest.color_field is not None
            else None
        )
        return {
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
        max_primitives: int,
        target_cell_pixels: float,
    ) -> dict[str, Any]:
        clipped_min_x = max(0.0, min_x)
        clipped_max_x = min(float(self.manifest.width - 1), max_x)
        clipped_min_y = max(0.0, min_y)
        clipped_max_y = min(float(self.manifest.height - 1), max_y)
        origin_x = math.floor(clipped_min_x)
        origin_y = math.floor(clipped_min_y)
        if clipped_min_x > clipped_max_x or clipped_min_y > clipped_max_y:
            return self._empty_response(origin_x, origin_y)

        units_per_pixel = max(
            (clipped_max_x - clipped_min_x) / max(1, pixel_width),
            (clipped_max_y - clipped_min_y) / max(1, pixel_height),
            1e-12,
        )
        target_units = units_per_pixel * target_cell_pixels
        seed = self._lod.choose_seed_level(
            min_x=clipped_min_x,
            max_x=clipped_max_x,
            min_y=clipped_min_y,
            max_y=clipped_max_y,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            max_primitives=max_primitives,
        )
        seed_table = self._lod.view_table(
            seed,
            min_x=clipped_min_x,
            max_x=clipped_max_x,
            min_y=clipped_min_y,
            max_y=clipped_max_y,
        )
        if seed_table.num_rows > max_primitives:
            raise RuntimeError(
                "Seed LOD exceeded the frontier primitive budget despite coarse "
                "index selection."
            )

        level_tables: dict[int, pa.Table] = {seed.level: seed_table}
        primitive_count = seed_table.num_rows

        for level in range(seed.level, 0, -1):
            table = level_tables.get(level)
            if table is None or table.num_rows == 0:
                continue
            if self.manifest.levels[level].cell_size <= target_units:
                break

            parent_coords = set(self._table_coords(table))
            children = self._lod.children(
                level,
                parent_coords,
                min_x=clipped_min_x,
                max_x=clipped_max_x,
                min_y=clipped_min_y,
                max_y=clipped_max_y,
            )
            child_counts = Counter(
                (int(x) // 2, int(y) // 2)
                for x, y in zip(
                    np.asarray(children["cell_x"].combine_chunks(), dtype=np.int64),
                    np.asarray(children["cell_y"].combine_chunks(), dtype=np.int64),
                    strict=True,
                )
            )
            selected, conservative_count = self._select_refinements(
                table,
                child_counts,
                primitive_count=primitive_count,
                max_primitives=max_primitives,
            )
            if not selected:
                break

            level_tables[level] = self._filter_cells(
                table,
                selected,
                keep_selected=False,
            )
            selected_children = self._filter_cells(
                children,
                selected,
                keep_selected=True,
                child_parent=True,
            )
            level_tables[level - 1] = selected_children
            primitive_count = conservative_count

        points = self._empty_points()
        level_zero = level_tables.get(0)
        if (
            level_zero is not None
            and level_zero.num_rows
            and self.manifest.base_cell_size > target_units
        ):
            selected_exact, conservative_count = self._select_exact_cells(
                level_zero,
                primitive_count=primitive_count,
                max_primitives=max_primitives,
            )
            if selected_exact:
                counts = np.asarray(level_zero["count"].combine_chunks(), dtype=np.uint64)
                coords = self._table_coords(level_zero)
                count_by_coord = {
                    coord: int(count) for coord, count in zip(coords, counts, strict=True)
                }
                expected_points = sum(count_by_coord[coord] for coord in selected_exact)
                points = self._exact_cells_view(
                    selected_exact,
                    min_x=clipped_min_x,
                    max_x=clipped_max_x,
                    min_y=clipped_min_y,
                    max_y=clipped_max_y,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    expected_points=expected_points,
                )
                level_tables[0] = self._filter_cells(
                    level_zero,
                    selected_exact,
                    keep_selected=False,
                )
                primitive_count = (
                    conservative_count - expected_points + int(points["point_count"])
                )

        cells = [
            self._lod.response_batch(
                level,
                table,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            for level, table in sorted(level_tables.items(), reverse=True)
            if table.num_rows
        ]
        actual_primitive_count = int(points["point_count"]) + sum(
            int(batch["cell_count"]) for batch in cells
        )
        if actual_primitive_count > max_primitives:
            raise RuntimeError("Adaptive frontier exceeded its primitive budget.")

        return {
            "origin": [origin_x, origin_y],
            "points": points,
            "cells": cells,
            "primitive_count": actual_primitive_count,
        }

    def check(self) -> list[str]:
        problems: list[str] = []
        indexed_count = sum(int(part["count"]) for part in self._parts)
        if indexed_count != self.manifest.point_count:
            problems.append(
                "index count " f"{indexed_count} != manifest count {self.manifest.point_count}"
            )
        for part in self._parts:
            path = self.path / str(part["path"])
            if not path.is_file():
                problems.append(f"missing point part: {part['path']}")
        problems.extend(self._lod.check())
        return problems


class MassiveScatterDataset:
    """Query all figure layers against one shared adaptive-frontier budget."""

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
                indexed,
                key=lambda item: (item[1].manifest.zorder, item[0]),
            )
        ]

    def _visible_layers(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> list[_LayerDataset]:
        visible: list[_LayerDataset] = []
        for dataset in self._layers:
            layer = dataset.manifest
            offset_x = layer.min_x - self.manifest.min_x
            offset_y = layer.min_y - self.manifest.min_y
            local_min_x = min_x - offset_x
            local_max_x = max_x - offset_x
            local_min_y = min_y - offset_y
            local_max_y = max_y - offset_y
            if (
                local_max_x < 0
                or local_min_x > layer.width - 1
                or local_max_y < 0
                or local_min_y > layer.height - 1
            ):
                continue
            visible.append(dataset)
        return visible

    def view(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        pixel_width: int,
        pixel_height: int,
        max_primitives: int = 200_000,
        target_cell_pixels: float = 2.0,
    ) -> dict[str, Any]:
        values = (min_x, max_x, min_y, max_y, target_cell_pixels)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Viewport bounds and target_cell_pixels must be finite.")
        if min_x > max_x or min_y > max_y:
            raise ValueError("Viewport bounds are inverted.")
        if pixel_width < 1 or pixel_height < 1:
            raise ValueError("Viewport pixel dimensions must be positive.")
        if max_primitives < 1:
            raise ValueError("max_primitives must be positive.")
        if target_cell_pixels <= 0:
            raise ValueError("target_cell_pixels must be positive.")

        clipped_min_x = max(0.0, min_x)
        clipped_max_x = min(float(self.manifest.width - 1), max_x)
        clipped_min_y = max(0.0, min_y)
        clipped_max_y = min(float(self.manifest.height - 1), max_y)
        origin_x = math.floor(clipped_min_x)
        origin_y = math.floor(clipped_min_y)
        if clipped_min_x > clipped_max_x or clipped_min_y > clipped_max_y:
            return {"origin": [origin_x, origin_y], "layers": []}

        visible = self._visible_layers(
            min_x=clipped_min_x,
            max_x=clipped_max_x,
            min_y=clipped_min_y,
            max_y=clipped_max_y,
        )
        if not visible:
            return {"origin": [origin_x, origin_y], "layers": []}
        if max_primitives < len(visible):
            raise ValueError(
                "max_primitives must be at least the number of visible layers."
            )

        quotient, remainder = divmod(max_primitives, len(visible))
        layer_responses: list[dict[str, Any]] = []
        for index, dataset in enumerate(visible):
            budget = quotient + (1 if index < remainder else 0)
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
                max_primitives=budget,
                target_cell_pixels=target_cell_pixels,
            )
            child_origin_x, child_origin_y = response.pop("origin")
            shift_x = child_origin_x + layer_offset_x - origin_x
            shift_y = child_origin_y + layer_offset_y - origin_y

            points = response["points"]
            points["x"] = [value + shift_x for value in points["x"]]
            points["y"] = [value + shift_y for value in points["y"]]
            for batch in response["cells"]:
                batch["x"] = [value + shift_x for value in batch["x"]]
                batch["y"] = [value + shift_y for value in batch["y"]]

            response["id"] = layer.id
            response["zorder"] = layer.zorder
            response["budget"] = budget
            layer_responses.append(response)

        total_primitives = sum(
            int(response["primitive_count"]) for response in layer_responses
        )
        if total_primitives > max_primitives:
            raise RuntimeError("Figure frontier exceeded the global primitive budget.")
        return {
            "origin": [origin_x, origin_y],
            "layers": layer_responses,
            "primitive_count": total_primitives,
        }

    def check(self) -> list[str]:
        problems: list[str] = []
        for dataset in self._layers:
            for problem in dataset.check():
                problems.append(f"{dataset.manifest.id}: {problem}")
        return problems
''')
Path("src/massive_scatter/server.py").write_text('''from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .dataset import MassiveScatterDataset


class ViewRequest(BaseModel):
    """Viewport request carried in the POST body rather than the URL."""

    model_config = ConfigDict(extra="forbid")

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    width: int = Field(default=1_024, ge=1, le=16_384)
    height: int = Field(default=768, ge=1, le=16_384)
    max_primitives: int = Field(default=200_000, ge=1, le=1_000_000)
    target_cell_pixels: float = Field(default=2.0, gt=0.0, le=64.0)


def _find_viewer(viewer_dir: str | Path | None) -> Path | None:
    if viewer_dir is not None:
        candidate = Path(viewer_dir).expanduser().resolve()
        if not (candidate / "index.html").is_file():
            raise FileNotFoundError(f"Viewer index not found in {candidate}")
        return candidate

    packaged = Path(__file__).with_name("_viewer")
    if (packaged / "index.html").is_file():
        return packaged

    repository_build = Path(__file__).resolve().parents[2] / "viewer" / "dist"
    if (repository_build / "index.html").is_file():
        return repository_build
    return None


def create_app(
    dataset_path: str | Path,
    *,
    viewer_dir: str | Path | None = None,
) -> FastAPI:
    dataset = MassiveScatterDataset(dataset_path)
    app = FastAPI(title="massive-scatter", version="0.1.0")

    @app.get("/api/manifest")
    def manifest() -> dict[str, Any]:
        return dataset.manifest.to_dict()

    @app.post("/api/view")
    def view(request: ViewRequest) -> dict[str, Any]:
        try:
            return dataset.view(
                min_x=request.xmin,
                max_x=request.xmax,
                min_y=request.ymin,
                max_y=request.ymax,
                pixel_width=request.width,
                pixel_height=request.height,
                max_primitives=request.max_primitives,
                target_cell_pixels=request.target_cell_pixels,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/view", include_in_schema=False)
    def view_requires_post() -> None:
        raise HTTPException(
            status_code=405,
            detail="Method Not Allowed",
            headers={"Allow": "POST"},
        )

    static_dir = _find_viewer(viewer_dir)
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="viewer")
    else:

        @app.get("/", response_class=HTMLResponse)
        def viewer_not_built() -> str:
            return """
            <!doctype html>
            <html lang="en"><head><meta charset="utf-8">
            <title>massive-scatter</title></head>
            <body style="font-family: sans-serif; max-width: 50rem; margin: 4rem auto">
              <h1>Viewer assets have not been built</h1>
              <p>The data API is running. Build the TypeScript viewer with:</p>
              <pre>cd viewer\nnpm install\nnpm run build</pre>
              <p>Then restart <code>massive-scatter serve</code>.</p>
              <p><a href="/docs">Open the API documentation</a></p>
            </body></html>
            """

    return app
''')
Path("tests/test_build_query.py").write_text('''import numpy as np
import pyarrow as pa

from massive_scatter import BuildConfig, MassiveScatterDataset, build_dataset


def make_batch(origin: int, size: int = 1024) -> pa.RecordBatch:
    x = np.arange(size, dtype=np.int64) + origin
    y = ((np.arange(size, dtype=np.int64) * 37) % size) + origin
    color = (np.arange(size, dtype=np.int64) % 11).astype(np.float64)
    return pa.record_batch([x, y, color], names=["x", "y", "weight"])


def _represented_points(layer: dict[str, object]) -> int:
    points = layer["points"]
    assert isinstance(points, dict)
    cells = layer["cells"]
    assert isinstance(cells, list)
    return int(points["point_count"]) + sum(sum(batch["count"]) for batch in cells)


def test_build_adaptive_frontier_views(tmp_path):
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
        max_primitives=100,
    )
    assert exact["origin"] == [100, 0]
    layer = exact["layers"][0]
    assert layer["points"]["x"] == list(range(11))
    assert all(isinstance(value, int) for value in layer["points"]["x"])
    assert layer["primitive_count"] <= 100

    coarse = dataset.view(
        min_x=0,
        max_x=1023,
        min_y=0,
        max_y=1023,
        pixel_width=8,
        pixel_height=8,
        max_primitives=64,
    )
    layer = coarse["layers"][0]
    assert layer["primitive_count"] <= 64
    assert _represented_points(layer) == 1024


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
        max_primitives=10,
    )
    assert response["origin"] == [0, 0]
    layer = response["layers"][0]
    assert layer["points"]["x"] == [0, 1]
    assert layer["points"]["y"] == [0, 1]
    assert layer["cells"] == []
''')
Path("tests/test_lod.py").write_text('''import pyarrow as pa

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
    layer = dataset.view(
        min_x=0,
        max_x=13,
        min_y=0,
        max_y=13,
        pixel_width=1,
        pixel_height=1,
        max_primitives=1,
    )["layers"][0]

    assert layer["points"]["point_count"] == 0
    assert layer["primitive_count"] == 1
    assert sum(sum(batch["count"]) for batch in layer["cells"]) == len(x)
    colors = [
        value
        for batch in layer["cells"]
        for value in (batch["color"] or [])
    ]
    assert max(colors) == 9.0
''')
Path("tests/test_sparse_lod.py").write_text('''import math

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
    layer_manifest = manifest.layers[0]
    assert layer_manifest.levels[0].occupied_cells == point_count
    layer_path = output / layer_manifest.path
    assert (layer_path / "lod" / "0" / "index.parquet").is_file()

    parts = sorted((layer_path / "lod" / "0").glob("part-*.parquet"))
    assert len(parts) == math.ceil(point_count / 1024)
    assert len(parts) < point_count // 100

    dataset = MassiveScatterDataset(output)
    assert dataset.check() == []
    layer = dataset.view(
        min_x=0,
        max_x=point_count - 1,
        min_y=0,
        max_y=(point_count - 1) * 100_000,
        pixel_width=1,
        pixel_height=1,
        max_primitives=4,
    )["layers"][0]
    assert layer["primitive_count"] <= 4
    assert layer["points"]["point_count"] == 0
    assert sum(sum(batch["count"]) for batch in layer["cells"]) == point_count
''')
Path("tests/test_server.py").write_text('''import pyarrow as pa
from fastapi.testclient import TestClient

from massive_scatter import BuildConfig, build_dataset
from massive_scatter.server import create_app


def test_api_serves_adaptive_frontier(tmp_path):
    output = tmp_path / "api.msplot"
    build_dataset(
        output,
        [pa.record_batch([[0, 1, 2], [2, 1, 0]], names=["x", "y"])],
        config=BuildConfig(base_cell_size=1, part_rows=4),
    )

    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<!doctype html><title>test</title>")
    client = TestClient(create_app(output, viewer_dir=viewer))

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
            "max_primitives": 3,
        },
    )
    assert view.status_code == 200
    response = view.json()
    assert response["origin"] == [0, 0]
    assert response["primitive_count"] == 3
    assert response["layers"][0]["points"]["point_count"] == 3
    assert response["layers"][0]["cells"] == []

    wrong_method = client.get("/api/view")
    assert wrong_method.status_code == 405
    assert wrong_method.headers["allow"] == "POST"
''')
Path("tests/test_plot_api.py").write_text('''import math

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
        max_primitives=10,
    )["layers"][0]
    assert exact["points"]["fields"]["kind"] == ["E3", "E3", "P3", "other"]
    assert exact["points"]["fields"]["weight"] == [1.0, 3.0, 9.0, 17.0]
    assert exact["cells"] == []

    coarse = dataset.view(
        min_x=0,
        max_x=2,
        min_y=0,
        max_y=0,
        pixel_width=1,
        pixel_height=1,
        max_primitives=2,
    )["layers"][0]
    assert coarse["points"]["point_count"] == 0
    cells = coarse["cells"]
    assert sum(batch["cell_count"] for batch in cells) == 2
    by_request = {
        item.reducer: [
            value
            for cell_batch in cells
            for value in cell_batch["aggregates"][item.key]
        ]
        for item in layer.aggregates
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


def test_multiple_scatter_layers_share_global_frontier_budget_and_zorder(tmp_path):
    dense = pa.record_batch([list(range(20)), [0] * 20], names=["x", "y"])
    sparse = pa.record_batch([[0, 10], [1, 1]], names=["x", "y"])

    fig, ax = ms.subplots()
    dense_handle = ax.scatter([dense], x="x", y="y", color="red", label="dense")
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

    response = ms.MassiveScatterDataset(output).view(
        min_x=0,
        max_x=19,
        min_y=0,
        max_y=1,
        pixel_width=100,
        pixel_height=100,
        max_primitives=10,
    )
    assert response["origin"] == [0, 0]
    assert response["primitive_count"] <= 10
    assert [layer["id"] for layer in response["layers"]] == [
        "layer-001",
        "layer-000",
    ]
    by_id = {layer["id"]: layer for layer in response["layers"]}
    assert by_id["layer-001"]["points"]["point_count"] == 2
    assert by_id["layer-001"]["cells"] == []
    assert by_id["layer-000"]["primitive_count"] <= 5
''')
Path("tests/test_adaptive_frontier.py").write_text('''import pyarrow as pa

from massive_scatter import BuildConfig, MassiveScatterDataset, build_dataset


def _represented_points(layer: dict[str, object]) -> int:
    points = layer["points"]
    assert isinstance(points, dict)
    cells = layer["cells"]
    assert isinstance(cells, list)
    return int(points["point_count"]) + sum(sum(batch["count"]) for batch in cells)


def test_sparse_branches_refine_to_exact_inside_coarse_frontier(tmp_path):
    batch = pa.record_batch(
        [[0, 1, 2, 3, 15], [0, 0, 0, 0, 0]],
        names=["x", "y"],
    )
    output = tmp_path / "adaptive.msplot"
    build_dataset(
        output,
        [batch],
        config=BuildConfig(base_cell_size=1, part_rows=32),
    )

    layer = MassiveScatterDataset(output).view(
        min_x=0,
        max_x=15,
        min_y=0,
        max_y=0,
        pixel_width=160,
        pixel_height=10,
        max_primitives=3,
        target_cell_pixels=2.0,
    )["layers"][0]

    assert layer["primitive_count"] == 3
    assert layer["points"]["point_count"] == 1
    assert layer["points"]["x"] == [15]
    assert [(batch["level"], batch["cell_count"]) for batch in layer["cells"]] == [
        (1, 2)
    ]
    assert _represented_points(layer) == 5


def test_frontier_budget_and_full_refinement_to_leaves(tmp_path):
    batch = pa.record_batch(
        [[0, 1, 2, 3, 15], [0, 0, 0, 0, 0]],
        names=["x", "y"],
    )
    output = tmp_path / "adaptive-exact.msplot"
    build_dataset(
        output,
        [batch],
        config=BuildConfig(base_cell_size=1, part_rows=32),
    )

    response = MassiveScatterDataset(output).view(
        min_x=0,
        max_x=15,
        min_y=0,
        max_y=0,
        pixel_width=160,
        pixel_height=10,
        max_primitives=5,
        target_cell_pixels=2.0,
    )
    layer = response["layers"][0]
    assert response["primitive_count"] == 5
    assert layer["primitive_count"] == 5
    assert layer["points"]["point_count"] == 5
    assert layer["cells"] == []
    assert sorted(layer["points"]["x"]) == [0, 1, 2, 3, 15]
    assert _represented_points(layer) == 5
''')

index = Path("viewer/index.html")
text = index.read_text()
text = text.replace(
    '          exact limit\n          <input id="max-points" type="number" min="1000" max="1000000" step="1000" value="200000" />',
    '          primitive budget\n          <input id="max-primitives" type="number" min="1000" max="1000000" step="1000" value="200000" />',
)
index.write_text(text)

main = Path("viewer/src/main.ts")
text = main.read_text()
text = text.replace(
    "  Deck,\n  type PickingInfo,",
    "  Deck,\n  type Layer,\n  type PickingInfo,",
)
text, count = re.subn(
    r"interface LayerViewResponse \{.*?interface ViewResponse \{",
    '''interface PrimitiveViewResponse {
  x: number[];
  y: number[];
  color: number[] | null;
  fields?: Record<string, Scalar[]>;
  aggregates?: Record<string, number[]>;
  count?: number[];
}

interface PointViewResponse extends PrimitiveViewResponse {
  point_count: number;
}

interface CellViewResponse extends PrimitiveViewResponse {
  level: number;
  cell_size: number;
  count: number[];
  aggregates: Record<string, number[]>;
  cell_count: number;
}

interface LayerViewResponse {
  id: string;
  zorder: number;
  points: PointViewResponse;
  cells: CellViewResponse[];
  primitive_count: number;
  budget: number;
}

interface ViewResponse {''',
    text,
    flags=re.S,
)
if count != 1:
    raise SystemExit(f"LayerViewResponse replacement count={count}")
text = text.replace(
    "interface PlotDatum {\n  layerId: string;",
    "interface PlotDatum {\n  layerId: string;\n  kind: 'point' | 'cell';\n  level?: number;\n  cellSize?: number;",
)
text = text.replace(
    "const maxPointsInput = requiredElement<HTMLInputElement>('max-points');",
    "const maxPrimitivesInput = requiredElement<HTMLInputElement>('max-primitives');",
)
text = text.replace(
    "    max_points: Math.max(1, Number(maxPointsInput.value) || 200_000),\n    max_cells: 200_000,",
    "    max_primitives: Math.max(1, Number(maxPrimitivesInput.value) || 200_000),",
)
text, count = re.subn(
    r"function responseData\(response: LayerViewResponse\): PlotDatum\[\] \{.*?\n\}\n\nfunction encodingValue",
    '''function responseData(
  layerId: string,
  response: PrimitiveViewResponse,
  kind: 'point' | 'cell',
  level?: number,
  cellSize?: number,
): PlotDatum[] {
  const counts = response.count;
  const legacyValues = response.color ?? counts ?? new Array(response.x.length).fill(1);
  const fieldArrays = response.fields ?? {};
  const aggregateArrays = response.aggregates ?? {};
  return response.x.map((x, index) => {
    const fields: Record<string, Scalar> = {};
    for (const [name, values] of Object.entries(fieldArrays)) {
      fields[name] = values[index] ?? null;
    }
    const aggregateValues: Record<string, number> = {};
    for (const [name, values] of Object.entries(aggregateArrays)) {
      aggregateValues[name] = values[index] ?? 0;
    }
    return {
      layerId,
      kind,
      level,
      cellSize,
      position: [x, response.y[index] ?? 0],
      legacyValue: legacyValues[index] ?? 0,
      count: counts?.[index] ?? 1,
      fields,
      aggregates: aggregateValues,
    };
  });
}

function encodingValue''',
    text,
    flags=re.S,
)
if count != 1:
    raise SystemExit(f"responseData replacement count={count}")
text = text.replace(
    "function normalized(value: number, range: NumericRange, low = 0, high = 1): number {",
    '''function mergeNumericRange(
  left: NumericRange | null,
  right: NumericRange,
): NumericRange {
  return left
    ? [Math.min(left[0], right[0]), Math.max(left[1], right[1])]
    : right;
}

interface RangeBatch {
  data: PlotDatum[];
  aggregate: boolean;
}

function combinedLegacyRange(batches: RangeBatch[]): NumericRange {
  let result: NumericRange | null = null;
  for (const batch of batches) {
    if (batch.data.length === 0) continue;
    result = mergeNumericRange(
      result,
      finiteRangeBy(batch.data, datum => datum.legacyValue),
    );
  }
  return result ?? [0, 1];
}

function combinedEncodingRange(
  layer: LayerManifest,
  encoding: EncodingManifest,
  batches: RangeBatch[],
): NumericRange {
  let result: NumericRange | null = null;
  for (const batch of batches) {
    if (batch.data.length === 0) continue;
    result = mergeNumericRange(
      result,
      encodingRange(layer, encoding, batch.data, batch.aggregate),
    );
  }
  return result ?? [0, 1];
}

function normalized(value: number, range: NumericRange, low = 0, high = 1): number {''',
)
text, count = re.subn(
    r"function makeDeckLayer\(response: LayerViewResponse\) \{.*?\n\}\n\nfunction renderLayers",
    '''function makeDeckLayers(response: LayerViewResponse): Layer[] {
  const layer = layerManifest(response.id);
  const pointData = responseData(layer.id, response.points, 'point');
  const cellGroups = response.cells.map(batch => ({
    batch,
    data: responseData(layer.id, batch, 'cell', batch.level, batch.cell_size),
  }));
  const rangeBatches: RangeBatch[] = [
    {data: pointData, aggregate: false},
    ...cellGroups.map(group => ({data: group.data, aggregate: true})),
  ];

  let colorRange: NumericRange | null;
  let alphaRange: NumericRange | null = null;
  if (!layer.plot) {
    colorRange = combinedLegacyRange(rangeBatches);
  } else {
    const scatter = layer.plot.scatter;
    colorRange =
      scatter.color.kind === 'constant'
        ? null
        : combinedEncodingRange(layer, scatter.color, rangeBatches);
    if (scatter.alpha.kind !== 'constant') {
      alphaRange = combinedEncodingRange(layer, scatter.alpha, rangeBatches);
    }
  }
  currentColorRanges.set(layer.id, colorRange);

  const result: Layer[] = [];
  for (const {batch, data} of cellGroups) {
    if (data.length === 0) continue;
    result.push(
      new GridCellLayer<PlotDatum>({
        id: `cells-${layer.id}-lod-${batch.level}`,
        data,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        cellSize: batch.cell_size,
        coverage: 1,
        extruded: false,
        getPosition: datum => aggregateCellCorner(datum.position, batch.cell_size),
        getFillColor: datum =>
          datumColor(layer, datum, true, colorRange, alphaRange),
        opacity: 1,
        pickable: true,
      }),
    );
  }

  if (pointData.length === 0) return result;
  if (layer.plot) {
    result.push(
      new IconLayer<PlotDatum>({
        id: `points-styled-${layer.id}`,
        data: pointData,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        getPosition: datum => datum.position,
        getIcon: datum => markerIcon(datumMarker(layer, datum)),
        getSize: datum => datumSize(layer, datum),
        sizeUnits: 'pixels',
        sizeMinPixels: 2,
        sizeMaxPixels: 24,
        getColor: datum =>
          datumColor(layer, datum, false, colorRange, alphaRange),
        pickable: true,
      }),
    );
  } else {
    result.push(
      new ScatterplotLayer<PlotDatum>({
        id: `points-native-${layer.id}`,
        data: pointData,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        getPosition: datum => datum.position,
        getRadius: 0.42,
        radiusUnits: 'common',
        radiusMinPixels: 1.35,
        radiusMaxPixels: 5,
        getFillColor: datum =>
          datumColor(layer, datum, false, colorRange, alphaRange),
        opacity: 0.92,
        stroked: false,
        pickable: true,
      }),
    );
  }
  return result;
}

function renderLayers''',
    text,
    flags=re.S,
)
if count != 1:
    raise SystemExit(f"makeDeckLayer replacement count={count}")
text, count = re.subn(
    r"function renderLayers\(response: ViewResponse\) \{.*?\n\}\n\nfunction orderedManifestLayers",
    '''function renderLayers(response: ViewResponse) {
  currentColorRanges.clear();
  renderOrigin = response.origin;
  const renderViewState = toRenderViewState(worldViewState, renderOrigin);
  const deckLayers = response.layers.flatMap(makeDeckLayers);
  deck.setProps({viewState: renderViewState, layers: deckLayers});
  renderLegend();

  const exactPoints = response.layers.reduce(
    (total, layer) => total + layer.points.point_count,
    0,
  );
  const aggregateCells = response.layers.reduce(
    (total, layer) =>
      total + layer.cells.reduce((subtotal, batch) => subtotal + batch.cell_count, 0),
    0,
  );
  status.textContent =
    `adaptive frontier · ${integerFormat(response.primitive_count)} primitives · ` +
    `${integerFormat(exactPoints)} exact points + ` +
    `${integerFormat(aggregateCells)} aggregate cells`;
}

function orderedManifestLayers''',
    text,
    flags=re.S,
)
if count != 1:
    raise SystemExit(f"renderLayers replacement count={count}")
text, count = re.subn(
    r"function tooltip\(info: PickingInfo<PlotDatum>\) \{.*?\n\}\n\nfunction escapeHtml",
    '''function tooltip(info: PickingInfo<PlotDatum>) {
  if (!manifest || !currentResponse || !info.object) return null;
  const layer = layerManifest(info.object.layerId);
  const [relativeX, relativeY] = localToWorld(renderOrigin, info.object.position);
  const absoluteX = addIntegerOffset(manifest.origin.x, relativeX);
  const absoluteY = addIntegerOffset(manifest.origin.y, relativeY);
  const lines: string[] = [];
  const label = layer.plot?.scatter.label;
  if (label) lines.push(escapeHtml(label));
  lines.push(`x: ${escapeHtml(absoluteX)}`, `y: ${escapeHtml(absoluteY)}`);

  if (info.object.kind === 'cell') {
    if (info.object.level !== undefined) {
      lines.push(`LOD: ${info.object.level}`);
    }
    lines.push(`count: ${integerFormat(info.object.count)}`);
    for (const definition of layer.aggregates ?? []) {
      const value = info.object.aggregates[definition.key];
      if (value !== undefined) {
        lines.push(
          `${escapeHtml(definition.reducer)}(${escapeHtml(definition.source)}): ` +
          escapeHtml(compactFormat(value)),
        );
      }
    }
  } else if (layer.plot) {
    for (const [name, value] of Object.entries(info.object.fields)) {
      lines.push(`${escapeHtml(name)}: ${escapeHtml(String(value))}`);
    }
  } else {
    lines.push(`value: ${escapeHtml(String(info.object.legacyValue))}`);
  }
  return {html: lines.join('<br/>')};
}

function escapeHtml''',
    text,
    flags=re.S,
)
if count != 1:
    raise SystemExit(f"tooltip replacement count={count}")
text = text.replace(
    "document.activeElement !== maxPointsInput",
    "document.activeElement !== maxPrimitivesInput",
)
text = text.replace(
    "maxPointsInput.addEventListener('change', () => scheduleViewRequest(0));",
    "maxPrimitivesInput.addEventListener('change', () => scheduleViewRequest(0));",
)
main.write_text(text)

readme = Path("README.md")
text = readme.read_text()
text = text.replace(
    "- exact-point responses at high zoom and aggregate square-cell responses at low\n  zoom;",
    "- adaptive mixed-level frontiers that refine sparse regions all the way to exact\n  points while dense regions remain aggregate cells;",
)
text = re.sub(
    r"### Exact mode versus aggregate mode\n.*?### Titles, labels, and legends",
    '''### Adaptive frontier selection

Viewport rendering no longer switches an entire layer between exact and aggregate
mode. The sparse factor-two LOD pyramid is treated as an implicit tree. A query
starts from a budget-fitting coarse level and selectively replaces a visible cell
with its occupied children whenever that refinement is visually useful and fits
the primitive budget. Sparse one-child branches can therefore descend much farther
than dense branches. Level-zero cells refine to exact source points under the same
rule.

The selected frontier is disjoint: every represented region is covered by either
one aggregate cell or descendants of that cell, never both. A single layer may
therefore return coarse cells, finer cells, and exact points at the same time.

`max_primitives` is one GPU-facing budget shared across the visible figure. It
replaces the old separate exact-point and aggregate-cell budgets.
`target_cell_pixels` controls the desired maximum projected width of an aggregate
cell before the selector tries to refine it.

### Titles, labels, and legends''',
    text,
    flags=re.S,
)
text = text.replace(
    '  "max_points": 200000,\n  "max_cells": 200000',
    '  "max_primitives": 200000,\n  "target_cell_pixels": 2.0',
)
text = text.replace(
    "At high zoom the response contains exact points and requested exact style fields\nas offsets from a local origin. If the exact result exceeds the point budget,\nthe server selects a sparse Parquet LOD and returns finalized aggregate values\nplus cell counts. No raster image tiles are generated or transferred.",
    "The response contains one adaptive frontier per visible layer: an exact-point batch\nplus zero or more aggregate-cell batches at different LOD levels, all expressed as\noffsets from one shared viewport origin. No raster image tiles are generated or\ntransferred.",
)
text = text.replace(
    "Likely future extensions include multiple layers, categorical aggregate",
    "Likely future extensions include categorical aggregate",
)
readme.write_text(text)
