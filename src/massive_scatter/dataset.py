from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import zarr

from .manifest import LevelManifest, Manifest
from .sparse_dataset import SparseLodReader
from .spec import AggregateRequest


def _array(group: zarr.Group, path: str) -> zarr.Array:
    value = group[path]
    if not isinstance(value, zarr.Array):
        raise TypeError(f"Expected Zarr array at {path}, found {type(value).__name__}.")
    return value


class MassiveScatterDataset:
    """Query exact points or bounded numerical LOD summaries from a dataset."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.manifest = Manifest.load(self.path)
        self._parts = pq.read_table(self.path / "index.parquet").to_pylist()
        self._sparse_lod = (
            SparseLodReader(self.path, self.manifest)
            if self.manifest.uses_sparse_lod
            else None
        )
        self._lod = (
            None
            if self._sparse_lod is not None
            else zarr.open_group(self.path / "lod.zarr", mode="r")
        )

    def _legacy_lod(self) -> zarr.Group:
        if self._lod is None:
            raise RuntimeError("Legacy Zarr LOD storage is not open.")
        return self._lod

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

    def _exact_storage_fields(self) -> dict[str, str]:
        if self.manifest.exact_fields:
            return self.manifest.exact_fields
        # Backwards-compatible reader for pre-grammar .msplot datasets.
        if self.manifest.color_field:
            return {self.manifest.color_field: "color"}
        return {}

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
        storage_fields = self._exact_storage_fields()
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

        legacy_color = (
            field_values.get(self.manifest.color_field)
            if self.manifest.color_field is not None
            else None
        )
        return {
            "mode": "exact",
            "origin": [origin_x, origin_y],
            "x": x_values,
            "y": y_values,
            "color": legacy_color,
            "fields": field_values,
            "point_count": len(x_values),
        }

    def _choose_level(
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

        while level.level < self.manifest.max_level:
            x0 = max(0, math.floor(min_x / level.cell_size))
            x1 = min(level.width, math.floor(max_x / level.cell_size) + 1)
            y0 = max(0, math.floor(min_y / level.cell_size))
            y1 = min(level.height, math.floor(max_y / level.cell_size) + 1)
            if max(0, x1 - x0) * max(0, y1 - y0) <= max_cells:
                break
            level = self.manifest.levels[level.level + 1]
        return level

    def _finalized_aggregate(
        self,
        request: AggregateRequest,
        *,
        level: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        local_y: np.ndarray,
        local_x: np.ndarray,
    ) -> list[float]:
        prefix = f"levels/{level}/aggregates/{request.key}"
        if request.reducer == "sum":
            state = np.asarray(
                _array(self._legacy_lod(), f"{prefix}/sum")[y0:y1, x0:x1],
                dtype=np.float64,
            )
            selected = state[local_y, local_x]
        elif request.reducer == "mean":
            sums = np.asarray(
                _array(self._legacy_lod(), f"{prefix}/sum")[y0:y1, x0:x1],
                dtype=np.float64,
            )
            counts = np.asarray(
                _array(self._legacy_lod(), f"{prefix}/count")[y0:y1, x0:x1],
                dtype=np.uint64,
            )
            selected_sums = sums[local_y, local_x]
            selected_counts = counts[local_y, local_x]
            selected = selected_sums / selected_counts
        else:
            state = np.asarray(
                _array(self._legacy_lod(), f"{prefix}/value")[y0:y1, x0:x1],
                dtype=np.float64,
            )
            selected = state[local_y, local_x]
        return [float(value) for value in selected]

    def _aggregate_view(
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
        if self._sparse_lod is not None:
            return self._sparse_lod.aggregate_view(
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                max_cells=max_cells,
            )
        level = self._choose_level(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            max_cells=max_cells,
        )
        cell_size = level.cell_size
        x0 = max(0, math.floor(min_x / cell_size))
        x1 = min(level.width, math.floor(max_x / cell_size) + 1)
        y0 = max(0, math.floor(min_y / cell_size))
        y1 = min(level.height, math.floor(max_y / cell_size) + 1)
        origin_x = x0 * cell_size
        origin_y = y0 * cell_size

        if x0 >= x1 or y0 >= y1:
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

        count_array = _array(self._legacy_lod(), f"levels/{level.level}/count")
        counts = np.asarray(count_array[y0:y1, x0:x1], dtype=np.uint64)
        local_y, local_x = np.nonzero(counts)
        selected_counts = counts[local_y, local_x]
        half = cell_size / 2
        x_values = (local_x.astype(np.float64) * cell_size + half).tolist()
        y_values = (local_y.astype(np.float64) * cell_size + half).tolist()

        aggregate_values = {
            request.key: self._finalized_aggregate(
                request,
                level=level.level,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                local_y=local_y,
                local_x=local_x,
            )
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
            elif not self.manifest.aggregates:
                # Pre-grammar dataset layout.
                color_array = _array(self._legacy_lod(), f"levels/{level.level}/color_max")
                colors = np.asarray(color_array[y0:y1, x0:x1], dtype=np.float64)
                legacy_color = colors[local_y, local_x].tolist()

        return {
            "mode": "aggregate",
            "level": level.level,
            "cell_size": cell_size,
            "origin": [origin_x, origin_y],
            "x": x_values,
            "y": y_values,
            "count": [int(value) for value in selected_counts],
            "color": legacy_color,
            "aggregates": aggregate_values,
            "cell_count": len(x_values),
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

        return self._aggregate_view(
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

        if self._sparse_lod is not None:
            problems.extend(self._sparse_lod.check())
            return problems

        for level in self.manifest.levels:
            count = _array(self._legacy_lod(), f"levels/{level.level}/count")
            if tuple(count.shape) != (level.height, level.width):
                problems.append(f"LOD {level.level} shape differs from manifest")
            for request in self.manifest.aggregates:
                prefix = f"levels/{level.level}/aggregates/{request.key}"
                paths = (
                    [f"{prefix}/sum", f"{prefix}/count"]
                    if request.reducer == "mean"
                    else (
                        [f"{prefix}/sum"]
                        if request.reducer == "sum"
                        else [f"{prefix}/value"]
                    )
                )
                for state_path in paths:
                    state = _array(self._legacy_lod(), state_path)
                    if tuple(state.shape) != (level.height, level.width):
                        problems.append(
                            f"LOD {level.level} aggregate {request.key} shape "
                            "differs from manifest"
                        )

        top = self.manifest.levels[-1]
        top_count_array = _array(self._legacy_lod(), f"levels/{top.level}/count")
        top_count = np.asarray(top_count_array[:], dtype=np.uint64)
        if int(top_count.sum(dtype=np.uint64)) != self.manifest.point_count:
            problems.append("top-level LOD count does not equal point count")
        return problems
