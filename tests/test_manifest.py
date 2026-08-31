import pytest

from massive_scatter.manifest import SCHEMA_VERSION, LevelManifest, Manifest


def _manifest(origin: int = 0) -> Manifest:
    return Manifest(
        point_count=2,
        min_x=origin,
        max_x=origin + 1,
        min_y=-origin,
        max_y=-origin + 1,
        base_cell_size=1,
        color_field=None,
        levels=(
            LevelManifest(
                level=0,
                cell_size=1,
                height=2,
                width=2,
                occupied_cells=2,
            ),
            LevelManifest(
                level=1,
                cell_size=2,
                height=1,
                width=1,
                occupied_cells=1,
            ),
        ),
    )


def test_manifest_round_trip_preserves_large_int64_origin(tmp_path):
    origin = 9_100_000_000_000_000
    manifest = _manifest(origin)
    manifest.save(tmp_path)

    raw = (tmp_path / "manifest.json").read_text()
    assert f'"min_x": "{origin}"' in raw
    assert f'"schema_version": {SCHEMA_VERSION}' in raw
    assert '"lod_storage": "sparse_parquet"' in raw
    assert Manifest.load(tmp_path) == manifest


def test_manifest_rejects_old_schema_versions():
    payload = _manifest().to_dict()
    for version in (1, 2):
        old = dict(payload)
        old["schema_version"] = version
        with pytest.raises(ValueError, match="only schema 3 is supported"):
            Manifest.from_dict(old)


def test_manifest_rejects_non_sparse_storage():
    payload = _manifest().to_dict()
    payload["lod_storage"] = "other"
    with pytest.raises(ValueError, match="only 'sparse_parquet' is supported"):
        Manifest.from_dict(payload)
