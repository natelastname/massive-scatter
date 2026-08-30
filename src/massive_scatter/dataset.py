from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import zarr

from .manifest import LevelManifest, Manifest


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
        self._lod = zarr.open_group(self.path / "lod.zarr", mode="r")

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
        columns = ["x", "y"] + (["color"] if self.manifest.color_field else [])

        x_values: list[int] = []
        y_values: list[int] = []
        color_values: list[float] | None = [] if self.manifest.color_field else None
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
            # Subtract int64 dataset origins on the CPU. The values sent to the
            # GPU are small offsets from a viewport-local origin.
            relative_x = absolute_x - self.manifest.min_x - origin_x
            relative_y = absolute_y - self.manifest.min_y - origin_y
            x_values.extend(int(value) for value in relative_x)
            y_values.extend(int(value) for value in relative_y)
            if color_values is not None:
                color_values.extend(
                    float(value)
                    for value in np.asarray(batch.column(2), dtype=np.float64)
                )

        return {
            "mode": "exact",
            "origin": [origin_x, origin_y],
            "x": x_values,
            "y": y_values,
            "color": color_values,
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
                "cell_count": 0,
            }

        count_array = _array(self._lod, f"levels/{level.level}/count")
        counts = np.asarray(count_array[y0:y1, x0:x1], dtype=np.uint64)
        local_y, local_x = np.nonzero(counts)
        selected_counts = counts[local_y, local_x]
        # A cell is rendered at its center. base_cell_size is normally even;
        # half-unit centers are still exactly representable in binary.
        half = cell_size / 2
        x_values = (local_x.astype(np.float64) * cell_size + half).tolist()
        y_values = (local_y.astype(np.float64) * cell_size + half).tolist()

        color_values: list[float] | None = None
        if self.manifest.color_field:
            color_array = _array(self._lod, f"levels/{level.level}/color_max")
            colors = np.asarray(color_array[y0:y1, x0:x1], dtype=np.float64)
            color_values = colors[local_y, local_x].tolist()

        return {
            "mode": "aggregate",
            "level": level.level,
            "cell_size": cell_size,
            "origin": [origin_x, origin_y],
            "x": x_values,
            "y": y_values,
            "count": [int(value) for value in selected_counts],
            "color": color_values,
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
        """Return exact local-offset points when feasible, otherwise an LOD summary."""

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
        """Return integrity problems; an empty list means the dataset is consistent."""

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

        for level in self.manifest.levels:
            count = _array(self._lod, f"levels/{level.level}/count")
            if tuple(count.shape) != (level.height, level.width):
                problems.append(f"LOD {level.level} shape differs from manifest")

        top = self.manifest.levels[-1]
        top_count_array = _array(self._lod, f"levels/{top.level}/count")
        top_count = np.asarray(top_count_array[:], dtype=np.uint64)
        if int(top_count.sum(dtype=np.uint64)) != self.manifest.point_count:
            problems.append("top-level LOD count does not equal point count")
        return problems
