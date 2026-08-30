from massive_scatter.manifest import LevelManifest, Manifest


def test_manifest_round_trip_preserves_large_int64_origin(tmp_path):
    origin = 9_100_000_000_000_000
    manifest = Manifest(
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
    )
    manifest.save(tmp_path)

    raw = (tmp_path / "manifest.json").read_text()
    assert f'"min_x": "{origin}"' in raw
    assert Manifest.load(tmp_path) == manifest
