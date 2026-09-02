from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from .manifest import LayerManifest, Manifest
from .sparse_dataset import SparseLodReader


class _LayerDataset:
    """Query one layer in its own origin-relative coordinate system."""

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
        storage_fields = self.manifest.exact_fields
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

        direct_color = (
            field_values.get(self.manifest.color_field)
            if self.manifest.color_field is not None
            else None
        )
        return {
            "mode": "exact",
            "origin": [origin_x, origin_y],
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
        max_points: int,
        max_cells: int,
    ) -> dict[str, Any]:
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

        return self._lod.aggregate_view(
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
        problems.extend(self._lod.check())
        return problems


class MassiveScatterDataset:
    """Query all figure layers against one shared viewport and camera origin."""

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
                indexed, key=lambda item: (item[1].manifest.zorder, item[0])
            )
        ]

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
            return {"origin": [origin_x, origin_y], "layers": []}

        layer_responses: list[dict[str, Any]] = []
        for dataset in self._layers:
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
                max_points=max_points,
                max_cells=max_cells,
            )
            child_origin_x, child_origin_y = response.pop("origin")
            shift_x = child_origin_x + layer_offset_x - origin_x
            shift_y = child_origin_y + layer_offset_y - origin_y
            response["x"] = [value + shift_x for value in response["x"]]
            response["y"] = [value + shift_y for value in response["y"]]
            response["id"] = layer.id
            response["zorder"] = layer.zorder
            layer_responses.append(response)

        return {"origin": [origin_x, origin_y], "layers": layer_responses}

    def check(self) -> list[str]:
        problems: list[str] = []
        for dataset in self._layers:
            for problem in dataset.check():
                problems.append(f"{dataset.manifest.id}: {problem}")
        return problems
