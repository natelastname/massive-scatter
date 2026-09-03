from __future__ import annotations

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
                field_values[source].extend(
                    batch.column(index).take(arrow_indexes).to_pylist()
                )

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
                counts = np.asarray(
                    level_zero["count"].combine_chunks(), dtype=np.uint64
                )
                coords = self._table_coords(level_zero)
                count_by_coord = {
                    coord: int(count)
                    for coord, count in zip(coords, counts, strict=True)
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
