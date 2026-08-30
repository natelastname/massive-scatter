import numpy as np
import pyarrow as pa

from massive_scatter import BuildConfig, MassiveScatterDataset, build_dataset


def make_batch(origin: int, size: int = 1024) -> pa.RecordBatch:
    x = np.arange(size, dtype=np.int64) + origin
    y = ((np.arange(size, dtype=np.int64) * 37) % size) + origin
    color = (np.arange(size, dtype=np.int64) % 11).astype(np.float64)
    return pa.record_batch([x, y, color], names=["x", "y", "weight"])


def test_build_exact_and_aggregate_views(tmp_path):
    origin = 9_100_000_000_000_000
    output = tmp_path / "example.msplot"
    manifest = build_dataset(
        output,
        [make_batch(origin)],
        color="weight",
        config=BuildConfig(
            tile_size=8,
            base_cell_size=4,
            part_rows=128,
            batch_size=64,
        ),
    )

    assert manifest.point_count == 1024
    assert manifest.min_x == origin
    assert manifest.width == 1024
    dataset = MassiveScatterDataset(output)
    assert dataset.check() == []

    exact = dataset.view(
        min_x=100,
        max_x=110,
        min_y=0,
        max_y=1023,
        pixel_width=1000,
        pixel_height=1000,
        max_points=100,
    )
    assert exact["mode"] == "exact"
    assert exact["origin"] == [100, 0]
    assert exact["x"] == list(range(11))
    assert all(isinstance(value, int) for value in exact["x"])

    aggregate = dataset.view(
        min_x=0,
        max_x=1023,
        min_y=0,
        max_y=1023,
        pixel_width=8,
        pixel_height=8,
        max_points=10,
        max_cells=64,
    )
    assert aggregate["mode"] == "aggregate"
    assert sum(aggregate["count"]) == 1024
    assert aggregate["cell_count"] <= 64


def test_unit_separation_survives_origin_subtraction(tmp_path):
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
        config=BuildConfig(tile_size=8, base_cell_size=1, part_rows=8),
    )

    response = MassiveScatterDataset(output).view(
        min_x=0,
        max_x=1,
        min_y=0,
        max_y=1,
        pixel_width=100,
        pixel_height=100,
        max_points=10,
    )
    assert response["mode"] == "exact"
    assert response["x"] == [0, 1]
    assert response["y"] == [0, 1]
