from pathlib import Path


def replace(path: str, old: str, new: str, *, count: int = -1) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old!r}")
    file.write_text(text.replace(old, new, count))


# Builder: remove the unused Zarr-era tile-size knob and legacy naming.
replace("src/massive_scatter/builder.py", "    tile_size: int = 256\n", "")
replace(
    "src/massive_scatter/builder.py",
    "        if self.tile_size < 2 or self.tile_size & (self.tile_size - 1):\n"
    "            raise ValueError(\"tile_size must be a power of two greater than one.\")\n",
    "",
)
replace(
    "src/massive_scatter/builder.py",
    "    ``plot`` is the compiled contract used by the higher-level Matplotlib-like\n"
    "    API. The historical x/y/color arguments remain supported and are compiled\n"
    "    to the same generic max-reducer machinery for backwards compatibility.\n",
    "    ``plot`` is the compiled contract used by the higher-level Matplotlib-like\n"
    "    API. The x/y/color arguments form the lower-level direct builder API and\n"
    "    compile to the same generic reducer machinery.\n",
)
replace("src/massive_scatter/builder.py", "_legacy_contract", "_direct_contract")
replace(
    "src/massive_scatter/builder.py",
    "            tile_size=settings.tile_size,\n",
    "",
)
replace(
    "src/massive_scatter/builder.py",
    "            lod_storage=\"sparse_parquet\",\n",
    "",
)

# CLI: remove --tile-size entirely.
replace("src/massive_scatter/cli.py", "    tile_size: int = 256,\n", "")
replace("src/massive_scatter/cli.py", "            tile_size=tile_size,\n", "")

# Tests: stop constructing the deleted setting or referring to the deleted backend.
for path in [
    "tests/test_build_query.py",
    "tests/test_lod.py",
    "tests/test_plot_api.py",
    "tests/test_sparse_lod.py",
]:
    file = Path(path)
    text = file.read_text()
    text = text.replace("tile_size=8,\n            ", "")
    text = text.replace("tile_size=8, ", "")
    text = text.replace("tile_size=4, ", "")
    text = text.replace("tile_size=2, ", "")
    text = text.replace('    assert not (output / "lod.zarr").exists()\n', "")
    file.write_text(text)

Path("tests/test_manifest.py").write_text(
    '''import pytest

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
'''
)

# Documentation: schema v3 is the only format, not one branch of a compatibility matrix.
replace(
    "README.md",
    "The legacy `--color` column is\nnumeric and uses a `max` reducer at aggregate LOD.",
    "The direct `--color` column is numeric and uses a `max` reducer at aggregate\nLOD.",
)
old_schema = '''## Compatibility and `.msplot` schema

New datasets are written as `.msplot` schema v3 and use
`lod_storage="sparse_parquet"`. Schema v3 stores one row per occupied LOD cell
rather than a logically dense Zarr array split into sparse chunks.

The reader remains backward-compatible with schema-v1 and schema-v2 Zarr
artifacts, including the former hard-coded `color_max` layout. Existing
`.msplot` files therefore remain readable; rebuilding them is only necessary if
you want the new sparse-Parquet storage behavior.

The lower-level historical API also remains supported:
'''
new_schema = '''## `.msplot` schema

The only supported dataset format is schema v3 with
`lod_storage="sparse_parquet"`. Each LOD stores one row per occupied cell. The
reader intentionally rejects every other schema version or LOD storage type;
rebuild source data instead of carrying format-conversion or compatibility code.

The lower-level direct API is:
'''
replace("README.md", old_schema, new_schema)
replace(
    "README.md",
    "That legacy `color=` field is implemented through the same generalized reducer\n"
    "machinery and retains its historical aggregate `max` semantics.",
    "That `color=` field is implemented through the same generalized reducer machinery\n"
    "and uses aggregate `max` semantics.",
)
replace(
    "README.md",
    "--tile-size        legacy Zarr chunk setting; schema-v3 sparse LOD does not use it\n",
    "",
)

# Remove the obsolete dependency; uv lock is regenerated by the workflow.
replace("pyproject.toml", '  "zarr>=3.1,<4",\n', "")

# The old dense/chunked LOD implementation is not part of the codebase anymore.
Path("src/massive_scatter/lod.py").unlink()
