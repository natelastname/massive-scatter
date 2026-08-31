from massive_scatter.manifest import SCHEMA_VERSION, LevelManifest, Manifest


def _legacy_manifest(origin: int, *, schema_version: int = 2) -> Manifest:
    return Manifest(
        point_count=2,
        min_x=origin,
        max_x=origin + 1,
        min_y=-origin,
        max_y=-origin + 1,
        tile_size=8,
        base_cell_size=1,
        color_field=None,
        levels=(
            LevelManifest(
                level=0,
                cell_size=1,
                height=2,
                width=2,
                occupied_chunks=1,
            ),
        ),
        schema_version=schema_version,
    )


def test_manifest_round_trip_preserves_large_int64_origin(tmp_path):
    origin = 9_100_000_000_000_000
    manifest = _legacy_manifest(origin)
    manifest.save(tmp_path)

    raw = (tmp_path / "manifest.json").read_text()
    assert f'"min_x": "{origin}"' in raw
    assert '"schema_version": 2' in raw
    assert Manifest.load(tmp_path) == manifest


def test_manifest_reader_accepts_legacy_schema_v1():
    origin = 9_100_000_000_000_000
    legacy = _legacy_manifest(origin, schema_version=1)
    loaded = Manifest.from_dict(legacy.to_dict())
    assert loaded.schema_version == 1
    assert loaded.min_x == origin


def test_sparse_manifest_round_trip_uses_current_schema(tmp_path):
    manifest = Manifest(
        point_count=2,
        min_x=0,
        max_x=1,
        min_y=0,
        max_y=100_000,
        tile_size=8,
        base_cell_size=64,
        color_field=None,
        levels=(
            LevelManifest(
                level=0,
                cell_size=64,
                height=1563,
                width=1,
                occupied_cells=2,
            ),
        ),
        lod_storage="sparse_parquet",
    )
    manifest.save(tmp_path)

    loaded = Manifest.load(tmp_path)
    assert loaded == manifest
    assert loaded.schema_version == SCHEMA_VERSION == 3
    assert loaded.uses_sparse_lod
