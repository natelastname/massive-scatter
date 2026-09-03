import math

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
