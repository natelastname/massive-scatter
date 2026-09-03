import numpy as np
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
