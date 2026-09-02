import pyarrow as pa
from fastapi.testclient import TestClient

from massive_scatter import BuildConfig, build_dataset
from massive_scatter.server import create_app


def test_api_serves_layered_manifest_and_view(tmp_path):
    output = tmp_path / "api.msplot"
    build_dataset(
        output,
        [pa.record_batch([[0, 1, 2], [2, 1, 0]], names=["x", "y"])],
        config=BuildConfig(base_cell_size=1, part_rows=4),
    )
    client = TestClient(create_app(output))

    manifest = client.get("/api/manifest")
    assert manifest.status_code == 200
    payload = manifest.json()
    assert payload["point_count"] == 3
    assert [layer["id"] for layer in payload["layers"]] == ["layer-000"]

    view = client.post(
        "/api/view",
        json={
            "xmin": 0,
            "xmax": 2,
            "ymin": 0,
            "ymax": 2,
            "width": 100,
            "height": 100,
        },
    )
    assert view.status_code == 200
    response = view.json()
    assert response["origin"] == [0, 0]
    assert response["layers"][0]["mode"] == "exact"
    assert response["layers"][0]["point_count"] == 3
    assert client.get("/api/view").status_code == 405
