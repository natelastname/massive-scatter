# massive-scatter

`massive-scatter` builds zoomable scatter plots whose logical dimensions can be
far larger than an in-memory raster. Exact points remain the source of truth;
zoomed-out views use sparse numerical level-of-detail (LOD) arrays rather than a
pyramid of pre-rendered PNG tiles.

The MVP is aimed at integer-coordinate scientific sequences and similarly sparse
point sets:

- exact `int64` coordinates stored in partitioned Parquet;
- bounded-memory Arrow batch ingestion;
- sparse Zarr v3 arrays containing `count` and optional `color_max` reductions;
- exact-point responses at high zoom and aggregate responses at low zoom;
- a FastAPI viewport service;
- a deck.gl orthographic viewer with dots, pan/zoom, axes, and hover values.

## Why this does not allocate the rectangular figure

A dataset may occupy a logical rectangle billions of units wide and high while
containing only millions of points. `massive-scatter` never creates that dense
image. Its canonical artifact is:

```text
example.msplot/
├── manifest.json          # bounds, int64 origin, LOD metadata
├── index.parquet          # bounding box and count for each point part
├── points/
│   └── part-*.parquet     # exact points
└── lod.zarr/
    └── levels/*/
        ├── count          # sparse numerical chunks
        └── color_max      # optional sparse numerical chunks
```

The finest numerical LOD intentionally begins at `base_cell_size` units per
cell (64 by default). Below that scale the viewer asks for exact points. This
avoids generating high-resolution image tiles that contain only a handful of
points and could have been rendered directly.

## Installation

The repository uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
```

Build the TypeScript viewer once:

```bash
cd viewer
npm install
npm run build
cd ..
```

The Vite build is written to `src/massive_scatter/_viewer`, where the Python
server can find it.

## Quick start

Generate a deterministic million-point, roughly square test plot without ever
holding all points in memory:

```bash
uv run massive-scatter generate demo.msplot --points 1000000
uv run massive-scatter check demo.msplot
uv run massive-scatter serve demo.msplot
```

Open `http://127.0.0.1:8000`.

Build from Parquet or CSV/TSV:

```bash
uv run massive-scatter build points.parquet figure.msplot \
  --x n \
  --y value \
  --color support-weight
```

The x and y columns must be integer-valued. The optional color column must be
numeric.

Useful commands:

```bash
uv run massive-scatter info figure.msplot
uv run massive-scatter check figure.msplot
uv run massive-scatter serve figure.msplot --port 8080
```

Important build controls:

```text
--batch-size       Arrow/Parquet scan batch size (default 131072)
--part-rows        target rows per exact-point Parquet part (default 1000000)
--tile-size        numerical Zarr chunk width/height (default 256)
--base-cell-size   first aggregate cell width/height in native units (default 64)
--overwrite        replace an existing output dataset
```

## Python API

The core API consumes an iterable of Arrow batches:

```python
import pyarrow as pa

from massive_scatter import BuildConfig, build_dataset


def batches():
    for start in range(0, 10_000_000, 100_000):
        stop = min(10_000_000, start + 100_000)
        yield pa.record_batch(
            [range(start, stop), range(start, stop)],
            names=["x", "y"],
        )


build_dataset(
    "figure.msplot",
    batches(),
    config=BuildConfig(base_cell_size=64),
)
```

Peak build memory is governed by the caller's batch, one output Parquet part,
and a few fixed-size Zarr chunks—not by the total point count or viewport area.

## Precision model

Absolute source coordinates are retained as signed `int64`. The manifest writes
absolute origins as decimal strings, so a JavaScript parser cannot round them.
The viewer works in offsets from the dataset origin and each API response uses a
second viewport-local origin. Both the deck.gl camera and layer positions are
rebased into that same response-local frame whenever the origin changes. The GPU
therefore never has to reconcile small local point offsets with a large global
camera target.

The tested invariant is:

> Two source points one unit apart remain one unit apart in an exact viewport,
> even when their absolute coordinates exceed JavaScript's `2^53` integer limit.

For the MVP, each axis *span* must be at most `2^53 - 1`; the absolute origin may
use the full signed `int64` range. Supporting spans larger than that requires a
segmented/BigInt camera state, not merely a different storage type.

## View API

`POST /api/view` accepts dataset-relative viewport data as JSON in the request
body rather than encoding the camera state in the URL:

```json
{
  "xmin": 0,
  "xmax": 1000000,
  "ymin": 0,
  "ymax": 1000000,
  "width": 1200,
  "height": 800,
  "max_points": 200000,
  "max_cells": 200000
}
```

At high zoom the response contains exact points as offsets from a local origin.
If the exact result would exceed the point budget, the server selects a Zarr LOD
whose numerical grid fits the requested display budget. No raster image tiles
are generated or transferred.

## Development

```bash
uv run ruff check .
uv run black --check .
uv run basedpyright
uv run pytest

cd viewer
npm test
npm run build
```

The current MVP deliberately leaves several extensions for later work: Arrow IPC
or binary HTTP responses, pluggable reductions, appendable datasets, parallel LOD
construction, sharded Zarr storage, and a general spatial index for point clouds
whose Parquet part bounding boxes overlap heavily.
