import math

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
