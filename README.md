# massive-scatter

`massive-scatter` builds zoomable scatter plots whose logical dimensions can be
far larger than an in-memory raster. Exact points remain the source of truth;
zoomed-out views use sparse numerical level-of-detail (LOD) arrays rather than a
pyramid of pre-rendered PNG tiles.

The project is aimed at integer-coordinate scientific sequences and similarly
sparse point sets:

- exact `int64` coordinates stored in partitioned Parquet;
- bounded-memory Arrow batch ingestion;
- sparse Zarr v3 numerical LOD arrays;
- exact-point responses at high zoom and aggregate square-cell responses at low
  zoom;
- mergeable field reductions (`sum`, `mean`, `min`, `max`) compiled from plot
  encodings;
- a FastAPI viewport service;
- a deck.gl orthographic viewer with pan/zoom, axes, hover values, per-point
  styling, and generated legends.

## Why this does not allocate the rectangular figure

A dataset may occupy a logical rectangle billions of units wide and high while
containing only millions of points. `massive-scatter` never creates that dense
image. Its canonical artifact is approximately:

```text
example.msplot/
├── manifest.json          # bounds, int64 origin, plot grammar, LOD metadata
├── index.parquet          # bounding box and count for each point part
├── points/
│   └── part-*.parquet     # exact points and exact-only style fields
└── lod.zarr/
    └── levels/*/
        ├── count
        └── aggregates/
            └── */         # mergeable reducer state
```

The finest numerical LOD begins at `base_cell_size` units per cell (64 by
default). Below that scale the viewer asks for exact points. Aggregate cells are
rendered as the square spatial bins they represent.

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

Generate a deterministic million-point test plot:

```bash
uv run massive-scatter generate demo.msplot --points 1000000
uv run massive-scatter check demo.msplot
uv run massive-scatter serve demo.msplot
```

Open `http://127.0.0.1:8000`.

The original direct builder remains supported:

```bash
uv run massive-scatter build points.parquet figure.msplot \
  --x n \
  --y value \
  --color support-weight
```

The x and y columns must be integer-valued. The legacy `--color` column is
numeric and uses a `max` reducer at aggregate LOD.

Useful commands:

```bash
uv run massive-scatter info figure.msplot
uv run massive-scatter check figure.msplot
uv run massive-scatter serve figure.msplot --port 8080
```

## Matplotlib-like plot grammar

For richer figures, use the Python plotting surface. It copies familiar
Matplotlib concepts while compiling them to a declarative, out-of-core plot
specification:

```python
import massive_scatter as ms

fig, ax = ms.subplots()

ax.scatter(
    "points.parquet",
    x="n",
    y="value",
    c=ms.mean("omega"),
    cmap="viridis",
    marker=ms.field("event_type"),
    s="importance",
    alpha=0.9,
    label="Enots-Wolley",
)

ax.set(
    title="Enots-Wolley sequence",
    xlabel="n",
    ylabel="a(n)",
)
ax.legend()

fig.write("ew.msplot")
```

`source` may also be an iterable of Arrow `RecordBatch` or `Table` objects, so
large generated data does not need to be materialized in Python memory.

### Encodings

The first grammar supports:

```text
x, y       source coordinate fields (integer)
c           numeric field or mergeable reducer expression
color       constant CSS color (mutually exclusive with c)
cmap        viridis, plasma, or magma
marker      constant marker or categorical field
s           constant size or numeric exact-point field
alpha       constant, count(), or numeric mergeable field
label       legend label
```

Examples:

```python
# Constant styling.
ax.scatter(source, x="x", y="y", color="#ff0080", marker="^", s=4)

# Default mean reduction for a numeric color field.
ax.scatter(source, x="x", y="y", c="score")

# Explicit aggregate semantics.
ax.scatter(source, x="x", y="y", c=ms.max("score"))
ax.scatter(source, x="x", y="y", c=ms.mean("score"), alpha=ms.count())
```

Available reducer constructors are:

```python
ms.sum("field")
ms.mean("field")
ms.min("field")
ms.max("field")
ms.count()          # implicit spatial-cell population
ms.field("field")  # field reference; channel supplies its default reducer
```

### LOD contract

Every aggregate reducer stores mergeable sufficient state. Higher LOD levels
are built solely by merging child state; raw points are not reopened.

| reducer | persisted state | parent merge | finalized value |
| --- | --- | --- | --- |
| `count()` | `n` | sum | `n` |
| `sum(x)` | `sum` | sum | `sum` |
| `mean(x)` | `sum`, `n` | componentwise sum | `sum / n` |
| `min(x)` | `min` | min | `min` |
| `max(x)` | `max` | max | `max` |

In particular, `mean` is never implemented as a mean of child means.

`x` and `y` are not reducers: they determine the spatial bin. At aggregate LOD
the rendered object is that square bin.

Marker and size fields are deliberately **exact-point-only**. Once points are
aggregated, the viewer renders square spatial cells rather than inventing an
arbitrary aggregate marker or size. Numeric color and alpha encodings use their
finalized reducer values.

Categorical exact-point fields currently support at most 32 global values. This
keeps marker domains and generated legends bounded; high-cardinality categorical
sketches are future work.

### Current grammar scope

The current API intentionally supports one figure, one axes, and one massive
scatter layer. Multiple subplots and multiple layers are not yet implemented.
This is a scope boundary, not an attempt to emulate unsupported Matplotlib
behavior.

Axes metadata currently includes:

```python
ax.set_title("title")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set(title="title", xlabel="x", ylabel="y")
ax.legend()
```

The viewer generates categorical marker keys and continuous color bars from the
compiled manifest.

## Direct Python builder

The lower-level API still consumes an iterable of Arrow batches:

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
and a small number of fixed-size Zarr chunks—not by total point count or
rectangular extent.

Important build controls:

```text
--batch-size       Arrow/Parquet scan batch size (default 131072)
--part-rows        target rows per exact-point Parquet part (default 1000000)
--tile-size        numerical Zarr chunk width/height (default 256)
--base-cell-size   first aggregate cell width/height in native units (default 64)
--overwrite        replace an existing output dataset
```

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

For now, each axis *span* must be at most `2^53 - 1`; the absolute origin may use
the full signed `int64` range. Supporting spans larger than that requires a
segmented/BigInt camera state, not merely a different storage type.

## View API

`POST /api/view` accepts dataset-relative viewport data as JSON:

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

At high zoom the response contains exact points and requested exact style fields
as offsets from a local origin. If the exact result exceeds the point budget,
the server selects a Zarr LOD and returns finalized aggregate values plus cell
counts. No raster image tiles are generated or transferred.

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

Likely future extensions include multiple layers, categorical aggregate
histograms/top-k summaries, more visual scales, Arrow IPC or binary HTTP
responses, approximate mergeable reducers (quantiles/heavy hitters), appendable
datasets, and parallel LOD construction.
