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
    ax.scatter(
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
    ax.set(title="Episode geometry", xlabel="n", ylabel="a(n)")
    ax.legend()

    output = tmp_path / "plot.msplot"
    manifest = fig.write(
        output,
        config=ms.BuildConfig(base_cell_size=1, part_rows=2),
    )

    assert manifest.plot is not None
    assert manifest.plot.axes.title == "Episode geometry"
    assert manifest.plot.axes.legend is True
    assert manifest.plot.scatter.cmap == "plasma"
    assert manifest.plot.categorical_fields["kind"] == ("E3", "P3", "other")
    assert manifest.plot.numeric_ranges["weight"] == (1.0, 17.0)
    assert manifest.plot.numeric_ranges["point_size"] == (3.0, 6.0)
    assert {(item.source, item.reducer) for item in manifest.aggregates} == {
        ("weight", "mean"),
        ("weight", "max"),
    }
    assert all(item.source != "point_size" for item in manifest.aggregates)

    dataset = ms.MassiveScatterDataset(output)
    exact = dataset.view(
        min_x=0,
        max_x=2,
        min_y=0,
        max_y=0,
        pixel_width=100,
        pixel_height=100,
        max_points=10,
    )
    assert exact["mode"] == "exact"
    assert exact["fields"]["kind"] == ["E3", "E3", "P3", "other"]
    assert exact["fields"]["weight"] == [1.0, 3.0, 9.0, 17.0]

    aggregate = dataset.view(
        min_x=0,
        max_x=2,
        min_y=0,
        max_y=0,
        pixel_width=1,
        pixel_height=1,
        max_points=1,
        max_cells=2,
    )
    assert aggregate["mode"] == "aggregate"
    by_request = {
        item.reducer: aggregate["aggregates"][item.key] for item in manifest.aggregates
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
    manifest = fig.write(
        output,
        config=ms.BuildConfig(base_cell_size=1),
    )
    assert manifest.plot is not None
    assert manifest.plot.scatter.color.value == "#ff0080"
    assert manifest.exact_fields == {}
    assert manifest.aggregates == ()


def test_second_scatter_layer_is_rejected_for_first_grammar():
    fig, ax = ms.subplots()
    batch = pa.record_batch([[0], [0]], names=["x", "y"])
    ax.scatter([batch], x="x", y="y")
    try:
        ax.scatter([batch], x="x", y="y")
    except NotImplementedError as exc:
        assert "one scatter layer" in str(exc)
    else:
        raise AssertionError("expected second scatter call to fail")
