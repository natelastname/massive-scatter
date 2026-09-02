from dataclasses import replace

import pytest

from massive_scatter.manifest import (
    LOD_STORAGE,
    SCHEMA_VERSION,
    LayerManifest,
    LevelManifest,
    Manifest,
)
from massive_scatter.spec import AxesManifest


def _layer(origin: int = 0) -> LayerManifest:
    return LayerManifest(
        id="layer-000",
        path="layers/layer-000",
        zorder=0.0,
        point_count=2,
        min_x=origin,
        max_x=origin + 1,
        min_y=-origin,
        max_y=-origin + 1,
        base_cell_size=1,
        color_field=None,
        levels=(
            LevelManifest(0, 1, 2, 2, 2),
            LevelManifest(1, 2, 1, 1, 1),
        ),
    )


def _manifest(origin: int = 0) -> Manifest:
    layer = _layer(origin)
    return Manifest(
        min_x=layer.min_x,
        max_x=layer.max_x,
        min_y=layer.min_y,
        max_y=layer.max_y,
        axes=AxesManifest(),
        layers=(layer,),
    )


def test_manifest_round_trip_preserves_large_int64_origin(tmp_path):
    origin = 9_100_000_000_000_000
    manifest = _manifest(origin)
    manifest.save(tmp_path)

    raw = (tmp_path / "manifest.json").read_text()
    assert f'"min_x": "{origin}"' in raw
    assert f'"schema_version": {SCHEMA_VERSION}' in raw
    assert f'"lod_storage": "{LOD_STORAGE}"' in raw
    assert Manifest.load(tmp_path) == manifest


def test_manifest_rejects_old_schema_versions():
    payload = _manifest().to_dict()
    for version in (1, 2, 3):
        old = dict(payload)
        old["schema_version"] = version
        with pytest.raises(ValueError, match="only schema 4 is supported"):
            Manifest.from_dict(old)


def test_manifest_rejects_non_layered_storage():
    payload = _manifest().to_dict()
    payload["lod_storage"] = "sparse_parquet"
    with pytest.raises(ValueError, match="layered_sparse_parquet"):
        Manifest.from_dict(payload)


def test_layer_manifest_rejects_nonfinite_zorder():
    with pytest.raises(ValueError, match="zorder must be finite"):
        replace(_layer(), zorder=float("nan")).validate()


def test_manifest_requires_unique_layer_ids_and_union_bounds():
    layer = _layer()
    duplicate = LayerManifest.from_dict(layer.to_dict() | {"path": "layers/other"})
    with pytest.raises(ValueError, match="Layer ids must be unique"):
        Manifest(0, 1, 0, 1, AxesManifest(), (layer, duplicate)).validate()
