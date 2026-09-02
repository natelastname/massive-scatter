from pathlib import Path


def replace(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old[:80]!r}")
    file.write_text(text.replace(old, new))


replace(
    "src/massive_scatter/manifest.py",
    "import json\n",
    "import json\nimport math\n",
)
replace(
    "src/massive_scatter/manifest.py",
    '        if not self.id:\n            raise ValueError("Layer ids may not be empty.")\n',
    '        if not self.id:\n'
    '            raise ValueError("Layer ids may not be empty.")\n'
    '        if not math.isfinite(self.zorder):\n'
    '            raise ValueError("Layer zorder must be finite.")\n',
)

replace(
    "tests/test_manifest.py",
    "import pytest\n",
    "from dataclasses import replace\n\nimport pytest\n",
)
replace(
    "tests/test_manifest.py",
    "\ndef test_manifest_requires_unique_layer_ids_and_union_bounds():\n",
    "\ndef test_layer_manifest_rejects_nonfinite_zorder():\n"
    "    with pytest.raises(ValueError, match=\"zorder must be finite\"):\n"
    "        replace(_layer(), zorder=float(\"nan\")).validate()\n\n\n"
    "def test_manifest_requires_unique_layer_ids_and_union_bounds():\n",
)

replace(
    "README.md",
    '''```text
example.msplot/
├── manifest.json          # bounds, int64 origin, plot grammar, LOD metadata
├── index.parquet          # bounding box and count for each exact-point part
├── points/
│   └── part-*.parquet     # exact points and exact-only style fields
└── lod/
    ├── 0/
    │   ├── index.parquet  # coarse bounding boxes for LOD parts
    │   └── part-*.parquet # one row per occupied cell
    ├── 1/
    │   └── ...
    └── ...
```
''',
    '''```text
example.msplot/
├── manifest.json
└── layers/
    ├── layer-000/
    │   ├── index.parquet
    │   ├── points/part-*.parquet
    │   └── lod/
    │       ├── 0/index.parquet
    │       ├── 0/part-*.parquet
    │       └── ...
    ├── layer-001/
    │   └── ...
    └── ...
```

Every scatter call owns one layer directory. Its exact points, spatial index,
and sparse LOD pyramid are independent of every other layer; only the figure
bounds, axes, camera, and legend are shared.
''',
)
replace(
    "README.md",
    "The x and y columns must be integer-valued. The direct `--color` column is numeric and uses a `max` reducer at aggregate\nLOD.\n",
    "The x and y columns must be integer-valued. The direct `--color` column is\n"
    "numeric and uses a `max` reducer at aggregate LOD.\n",
)
replace(
    "README.md",
    '    label=None,\n)\n',
    '    label=None,\n    zorder=None,\n)\n',
)
replace(
    "README.md",
    '| `label` | layer/legend label | persisted in plot metadata |\n',
    '| `label` | layer/legend label | persisted in plot metadata |\n'
    '| `zorder` | layer draw order | same layer draw order |\n',
)
replace(
    "README.md",
    "current categorical domain is bounded to at most 32 global values. This keeps\n",
    "current categorical domain is bounded to at most 32 values per layer. This keeps\n",
)
replace(
    "README.md",
    "points into partitioned Parquet, builds the sparse mergeable LOD pyramid, writes\n"
    "the plot metadata to the manifest, and returns the resulting `Manifest`.\n",
    "points into per-layer partitioned Parquet, builds each layer's sparse mergeable\n"
    "LOD pyramid, writes the figure/layer metadata, and returns the resulting\n"
    "`Manifest`.\n",
)
replace(
    "README.md",
    "- categorical exact fields with at most 32 global values.\n",
    "- categorical exact fields with at most 32 values per layer.\n",
)
replace(
    "README.md",
    '''At high zoom the response contains exact points and requested exact style fields
as offsets from a local origin. If the exact result exceeds the point budget,
the server selects a sparse Parquet LOD and returns finalized aggregate values
plus cell counts. No raster image tiles are generated or transferred.
''',
    '''The response has one shared viewport-local origin plus a `layers` array. Each
layer independently returns either exact points or sparse aggregate cells, so a
sparse layer can remain exact while a denser layer in the same viewport moves to
a coarser LOD. All layer coordinates are rebased into the shared response origin
before transfer. No raster image tiles are generated or transferred.
''',
)
replace(
    "README.md",
    "Likely future extensions include multiple layers, categorical aggregate\n",
    "Likely future extensions include layer visibility controls, shared color\n"
    "normalization, categorical aggregate\n",
)
