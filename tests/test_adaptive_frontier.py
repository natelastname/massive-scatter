import pyarrow as pa

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
