from pathlib import Path

path = Path("README.md")
text = path.read_text()
text = text.replace(
    "zoomed-out views use sparse numerical level-of-detail (LOD) arrays rather than a\n"
    "pyramid of pre-rendered PNG tiles.",
    "zoomed-out views use occupied-cell-only level-of-detail (LOD) tables rather than a\n"
    "dense grid or a pyramid of pre-rendered PNG tiles.",
)
text = text.replace(
    "- sparse Zarr v3 numerical LOD arrays;",
    "- occupied-cell-only Parquet LOD levels with mergeable reducer state;",
)
old_tree = "\n".join(
    [
        "example.msplot/",
        "├── manifest.json          # bounds, int64 origin, plot grammar, LOD metadata",
        "├── index.parquet          # bounding box and count for each point part",
        "├── points/",
        "│   └── part-*.parquet     # exact points and exact-only style fields",
        "└── lod.zarr/",
        "    └── levels/*/",
        "        ├── count",
        "        └── aggregates/",
        "            └── */         # mergeable reducer state",
    ]
)
new_tree = "\n".join(
    [
        "example.msplot/",
        "├── manifest.json          # bounds, int64 origin, plot grammar, LOD metadata",
        "├── index.parquet          # bounding box and count for each exact-point part",
        "├── points/",
        "│   └── part-*.parquet     # exact points and exact-only style fields",
        "└── lod/",
        "    ├── 0/",
        "    │   ├── index.parquet  # coarse bounding boxes for LOD parts",
        "    │   └── part-*.parquet # one row per occupied cell",
        "    ├── 1/",
        "    │   └── ...",
        "    └── ...",
    ]
)
if old_tree not in text:
    raise SystemExit("artifact tree not found")
text = text.replace(old_tree, new_tree)
marker = (
    "The finest numerical LOD begins at `base_cell_size` units per cell (64 by\n"
    "default). Below that scale the viewer asks for exact points. Aggregate cells are\n"
    "rendered as the square spatial bins they represent.\n"
)
addition = (
    marker
    + "\nOnly occupied cells are stored. Empty cells in the logical rectangle consume no\n"
    "LOD rows, files, or chunks. During construction a temporary on-disk SQLite\n"
    "B-tree merges duplicate occupied cells. Parent levels are formed by shifting the\n"
    "cell coordinates by one binary digit and merging the same sufficient statistics,\n"
    "then each level is streamed to sorted Parquet parts. The temporary SQLite build\n"
    "index is deleted when the dataset is complete.\n\n"
    "This matters for extreme-aspect-ratio data. A plot can have millions of points\n"
    "spread across billions of logical cells without degenerating into millions of\n"
    "small dense storage chunks; an isolated occupied cell is just one compact table\n"
    "row.\n"
)
if marker not in text:
    raise SystemExit("LOD explanation marker not found")
text = text.replace(marker, addition, 1)
old_compat = (
    "New generalized plot-grammar datasets are written as `.msplot` schema v2.\n\n"
    "The current reader remains backward-compatible with legacy schema-v1 datasets,\n"
    "including the former hard-coded `color_max` LOD layout. Existing `.msplot` files\n"
    "therefore do not need to be rebuilt merely to use the new reader."
)
new_compat = (
    "New datasets are written as `.msplot` schema v3 and use\n"
    "`lod_storage=\"sparse_parquet\"`. Schema v3 stores one row per occupied LOD cell\n"
    "rather than a logically dense Zarr array split into sparse chunks.\n\n"
    "The reader remains backward-compatible with schema-v1 and schema-v2 Zarr\n"
    "artifacts, including the former hard-coded `color_max` layout. Existing\n"
    "`.msplot` files therefore remain readable; rebuilding them is only necessary if\n"
    "you want the new sparse-Parquet storage behavior."
)
if old_compat not in text:
    raise SystemExit("compatibility paragraph not found")
text = text.replace(old_compat, new_compat)
text = text.replace(
    "Peak build memory is governed by the caller's batch, one output Parquet part,\n"
    "and a small number of fixed-size Zarr chunks—not by total point count or\n"
    "rectangular extent.",
    "Peak build memory is governed by the caller's batch and bounded Parquet/SQLite\n"
    "working buffers—not by total point count or rectangular extent. The temporary\n"
    "SQLite aggregation index is allowed to spill to disk and is removed after the\n"
    "portable Parquet LOD hierarchy has been written.",
)
text = text.replace(
    "--tile-size        numerical Zarr chunk width/height (default 256)",
    "--tile-size        legacy Zarr chunk setting; schema-v3 sparse LOD does not use it",
)
text = text.replace(
    "the server selects a Zarr LOD and returns finalized aggregate values plus cell\n"
    "counts.",
    "the server selects a sparse Parquet LOD and returns finalized aggregate values\n"
    "plus cell counts.",
)
path.write_text(text)
