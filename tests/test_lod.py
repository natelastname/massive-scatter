import pyarrow as pa

from massive_scatter import BuildConfig, MassiveScatterDataset, build_dataset


def test_parent_levels_preserve_counts_and_max_color(tmp_path):
    # Occupied child chunks with both even and odd chunk_y exercise grouping by
    # parent chunk rather than input order.
    x = [0, 1, 4, 5, 8, 9, 12, 13]
    y = [0, 4, 1, 5, 8, 12, 9, 13]
    color = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0]
    batch = pa.record_batch([x, y, color], names=["x", "y", "weight"])
    output = tmp_path / "lod.msplot"
    manifest = build_dataset(
        output,
        [batch],
        color="weight",
        config=BuildConfig(tile_size=4, base_cell_size=1, part_rows=4),
    )

    assert len(manifest.levels) >= 2
    dataset = MassiveScatterDataset(output)
    assert dataset.check() == []
    response = dataset.view(
        min_x=0,
        max_x=13,
        min_y=0,
        max_y=13,
        pixel_width=1,
        pixel_height=1,
        max_points=1,
        max_cells=1,
    )
    assert response["mode"] == "aggregate"
    assert sum(response["count"]) == len(x)
    assert max(response["color"]) == 9.0
