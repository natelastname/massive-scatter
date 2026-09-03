# massive-scatter

`massive-scatter` builds zoomable scatter plots whose logical dimensions can be
far larger than an in-memory raster. Exact points remain the source of truth;
zoomed-out views use occupied-cell-only level-of-detail (LOD) tables rather than a
dense grid or a pyramid of pre-rendered PNG tiles.

The project is aimed at integer-coordinate scientific sequences and similarly
sparse point sets:

- exact `int64` coordinates stored in partitioned Parquet;
- bounded-memory Arrow batch ingestion;
- occupied-cell-only Parquet LOD levels with mergeable reducer state;
- adaptive mixed-level frontiers that refine sparse regions all the way to exact
  points while dense regions remain aggregate cells;
- mergeable field reductions (`sum`, `mean`, `min`, `max`) compiled from plot
  encodings;
- a FastAPI viewport service;
- a deck.gl orthographic viewer with pan/zoom, axes, hover values, generated
  legends, and typed/binary GPU attributes rather than object-per-primitive input.

## Why this does not allocate the rectangular figure

A dataset may occupy a logical rectangle billions of units wide and high while
containing only millions of points. `massive-scatter` never creates that dense
image. Its canonical artifact is approximately:

```text
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

The finest numerical LOD begins at `base_cell_size` units per cell (64 by
default). It is the last aggregate level of the same implicit tree: selected
level-zero cells can refine further to their exact source-point leaves. Aggregate
cells are rendered as the square spatial bins they represent.

Only occupied cells are stored. Empty cells in the logical rectangle consume no
LOD rows, files, or chunks. During construction a temporary on-disk SQLite
B-tree merges duplicate occupied cells. Parent levels are formed by shifting the
cell coordinates by one binary digit and merging the same sufficient statistics,
then each level is streamed to sorted Parquet parts. The temporary SQLite build
index is deleted when the dataset is complete.

This matters for extreme-aspect-ratio data. A plot can have millions of points
spread across billions of logical cells without degenerating into millions of
small dense storage chunks; an isolated occupied cell is just one compact table
row.

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

The x and y columns must be integer-valued. The direct `--color` column is
numeric and uses a `max` reducer at aggregate LOD.

Useful commands:

```bash
uv run massive-scatter info figure.msplot
uv run massive-scatter check figure.msplot
uv run massive-scatter serve figure.msplot --port 8080
```

## Matplotlib-like plot grammar

For richer figures, use the Python plotting API. It deliberately copies a small,
familiar part of Matplotlib's grammar, but compiles the figure into a declarative
out-of-core plot specification rather than keeping all plotted data in memory.

The public plotting surface consists of:

```python
ms.subplots()
ms.Figure
ms.Axes
ms.field(...)
ms.sum(...)
ms.mean(...)
ms.min(...)
ms.max(...)
ms.count()
```

A representative multi-layer figure is:

```python
import massive_scatter as ms

fig, ax = ms.subplots()

ax.scatter(
    "ew.parquet",
    x="n",
    y="value",
    color="black",
    label="EW",
)
ax.scatter(
    "toy.parquet",
    x="n",
    y="value",
    color="red",
    label="Toy EW",
    zorder=2,
)

ax.set(
    title="Enots-Wolley",
    xlabel="n",
    ylabel="a(n)",
)
ax.legend()

fig.write("ew.msplot")
```

### `subplots()`

```python
fig, ax = ms.subplots()
```

Returns the single `Figure` / `Axes` pair supported by the first grammar
version. Multiple subplots are not yet implemented.

### `Axes.scatter(...)`

The current signature is conceptually:

```python
ax.scatter(
    source,
    *,
    x,
    y,
    c=None,
    color=None,
    cmap="viridis",
    marker="o",
    s=3.0,
    alpha=0.92,
    label=None,
    zorder=None,
)
```

`source` may be either:

- a file-backed source accepted by the normal input layer, such as Parquet or
  CSV/TSV; or
- an iterable of Arrow `RecordBatch` / `Table` objects.

The iterable form lets generated or streamed data remain bounded-memory instead
of first materializing the full dataset in Python.

Repeated `scatter()` calls create independent layers on the same axes. Each layer
keeps its own exact-point store, sparse LOD pyramid, reducers, styling, and adaptive
frontier while sharing the figure camera, axes, and legend. Call order is the
default z-order; pass `zorder=` to override it.

`c=` and `color=` are mutually exclusive. `c=` means a data mapping; `color=`
means a constant CSS color.

### Scatter channels

| argument | exact-point meaning | aggregate-LOD meaning |
| --- | --- | --- |
| `x` | integer source x coordinate | determines the spatial bin |
| `y` | integer source y coordinate | determines the spatial bin |
| `c` | numeric per-point field | finalized numeric reducer value |
| `color` | constant CSS color | same constant color |
| `cmap` | continuous color map | same map applied to aggregate value |
| `marker` | constant marker or categorical field | deliberately discarded; aggregate geometry is the square bin |
| `s` | constant or numeric per-point size | deliberately discarded; aggregate geometry is the square bin |
| `alpha` | constant, numeric field, or `count()` | constant or finalized reducer/count |
| `label` | layer/legend label | persisted in plot metadata |
| `zorder` | layer draw order | same layer draw order |

Supported continuous color maps are currently:

```text
viridis
plasma
magma
```

#### Color

Use `c=` for numeric data-driven color:

```python
ax.scatter(source, x="x", y="y", c="score")
ax.scatter(source, x="x", y="y", c=ms.max("score"))
ax.scatter(source, x="x", y="y", c=ms.mean("score"))
```

A bare numeric field uses `mean` as its default aggregate reducer. An explicit
reducer expression overrides that default.

Use `color=` for a constant CSS color:

```python
ax.scatter(source, x="x", y="y", color="#ff0080")
```

If neither `c=` nor `color=` is supplied, the plot uses aggregate cell count as
the color value.

#### Marker

A known marker string is interpreted as a constant marker:

```python
ax.scatter(source, x="x", y="y", marker="^")
```

Currently recognized constants include circles, squares, triangles, diamonds,
`x`, and `+` forms (`"o"`, `"circle"`, `"s"`, `"square"`, `"^"`,
`"triangle"`, `"D"`, `"diamond"`, `"x"`, `"+"`).

A field reference is categorical and applies per exact point:

```python
ax.scatter(source, x="x", y="y", marker=ms.field("event_type"))
```

A non-marker string is also interpreted as a categorical source field. The
current categorical domain is bounded to at most 32 values per layer. This keeps
exact-point marker domains and generated legend entries bounded.

Marker fields do **not** receive an aggregate reducer. Once multiple points have
collapsed into one LOD cell, the visible object is the square spatial cell, not
a fabricated representative marker.

#### Size

A numeric constant sets a fixed exact-point marker size:

```python
ax.scatter(source, x="x", y="y", s=4)
```

A string or `ms.field(...)` maps an exact numeric field to per-point size:

```python
ax.scatter(source, x="x", y="y", s="importance")
```

Like marker, size is intentionally exact-point-only. Aggregate cells keep their
true square spatial geometry instead of deriving a synthetic marker size.

#### Alpha

Alpha may be a constant:

```python
ax.scatter(source, x="x", y="y", alpha=0.8)
```

or a numeric field:

```python
ax.scatter(source, x="x", y="y", alpha="confidence")
```

A bare numeric alpha field uses `mean` at aggregate LOD. It may also use an
explicit reducer expression or the implicit cell population:

```python
ax.scatter(source, x="x", y="y", alpha=ms.max("confidence"))
ax.scatter(source, x="x", y="y", alpha=ms.count())
```

### Field and reducer expressions

`ms.field(...)` represents a source column, optionally with an explicit reducer:

```python
ms.field("score")
ms.field("score", reduce="max")
```

Convenience reducer constructors are:

```python
ms.sum("field")
ms.mean("field")
ms.min("field")
ms.max("field")
ms.count()
```

`ms.count()` is not backed by a source column. It means the implicit number of
exact points represented by an aggregate spatial cell.

Reducer requests are deduplicated during compilation. If multiple visual
channels request the same `(field, reducer)` pair, the LOD pyramid stores one
mergeable reducer state and both channels reference it.

### Mergeable LOD contract

Every aggregate reducer stores mergeable sufficient state. Higher LOD levels
are built solely by merging child state; raw points are not reopened.

| reducer | persisted state | parent merge | finalized value |
| --- | --- | --- | --- |
| `count()` | `n` | sum | `n` |
| `sum(x)` | `sum` | sum | `sum` |
| `mean(x)` | `sum`, `n` | componentwise sum | `sum / n` |
| `min(x)` | `min` | min | `min` |
| `max(x)` | `max` | max | `max` |

In particular, `mean` is **not** implemented by averaging child means. Each cell
persists `(sum, count)`, and a parent is formed by summing those sufficient
statistics before finalizing `sum / count`. The test suite contains a regression
test specifically for this weighted-parent-mean invariant.

`x` and `y` are not reducers. They define the spatial bin itself.

### Adaptive frontier selection

Viewport rendering no longer switches an entire layer between exact and aggregate
mode. The sparse factor-two LOD pyramid is treated as an implicit tree. A query
starts from a budget-fitting coarse level and selectively replaces a visible cell
with its occupied children whenever that refinement is visually useful and fits
the primitive budget. Sparse one-child branches can therefore descend much farther
than dense branches. Level-zero cells refine to exact source points under the same
rule.

The selected frontier is disjoint: every represented region is covered by either
one aggregate cell or descendants of that cell, never both. A single layer may
therefore return coarse cells, finer cells, and exact points at the same time.

`max_primitives` is one GPU-facing budget shared across the visible figure. It
replaces the old separate exact-point and aggregate-cell budgets.
`target_cell_pixels` controls the desired maximum projected width of an aggregate
cell before the selector tries to refine it.

The GPU is only the terminal rasterizer. After the frontier is selected, the viewer
packs positions, colors, and sizes into typed deck.gl binary attributes and avoids
creating one JavaScript object per visible primitive. Picking uses compact response
arrays as sidecars indexed by deck.gl's picking index. The HTTP response is still
JSON today, so JSON parsing and the subsequent typed-array packing remain a future
transport optimization target (for example Arrow IPC).

### Titles, labels, and legends

Axes metadata can be set using either dedicated methods:

```python
ax.set_title("title")
ax.set_xlabel("x")
ax.set_ylabel("y")
```

or the combined form:

```python
ax.set(title="title", xlabel="x", ylabel="y")
```

Enable generated legend UI with:

```python
ax.legend()
```

The viewer generates legend content from the compiled manifest rather than from
hand-authored entries:

- categorical marker mappings produce marker/category keys;
- continuous numeric color mappings produce color bars;
- the scatter `label=` is persisted as layer metadata.

### `Figure.write(...)`

After defining the axes, compile and build the plot with:

```python
manifest = fig.write("figure.msplot")
```

The method also accepts the normal build configuration and progress callback:

```python
manifest = fig.write(
    "figure.msplot",
    config=ms.BuildConfig(base_cell_size=64),
    progress=print,
)
```

`Figure.write()` compiles the visual grammar into an explicit plot/aggregate
contract, requests only the source columns required by the figure, streams exact
points into per-layer partitioned Parquet, builds each layer's sparse mergeable
LOD pyramid, writes the figure/layer metadata, and returns the resulting
`Manifest`.

### Current API scope and limitations

The first plot grammar intentionally supports:

- one figure;
- one axes;
- multiple independently queryable scatter layers on one axes;
- numeric data-mapped color;
- constant or categorical exact-point markers;
- constant or numeric exact-point sizes;
- numeric/constant/count alpha;
- `viridis`, `plasma`, and `magma` continuous color maps;
- categorical exact fields with at most 32 values per layer.

Not yet implemented:

- multiple subplots;
- categorical color aggregation;
- aggregate categorical markers;
- aggregate marker-size semantics;
- high-cardinality categorical sketches/histograms.

These are explicit scope boundaries. In particular, aggregate marker and size are
not approximated with arbitrary rules simply to mimic a Matplotlib call shape.

## `.msplot` schema

The only supported dataset format is schema v4 with
`lod_storage="layered_sparse_parquet"`. Every figure stores one or more layer
directories under `layers/`; each layer has its own exact Parquet parts and sparse
LOD pyramid. The reader intentionally rejects every other schema version or LOD
storage type; rebuild source data instead of carrying compatibility code.

The lower-level direct API is:

```python
build_dataset(
    output,
    batches,
    x="x",
    y="y",
    color="score",
)
```

That `color=` field is implemented through the same generalized reducer machinery
and uses aggregate `max` semantics.

## Direct Python builder

The lower-level API consumes an iterable of Arrow batches:

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

Peak build memory is governed by the caller's batch and bounded Parquet/SQLite
working buffers—not by total point count or rectangular extent. The temporary
SQLite aggregation index is allowed to spill to disk and is removed after the
portable Parquet LOD hierarchy has been written.

Important build controls:

```text
--batch-size       Arrow/Parquet scan batch size (default 131072)
--part-rows        target rows per exact-point Parquet part (default 1000000)
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
  "max_primitives": 200000,
  "target_cell_pixels": 2.0
}
```

The response has one shared viewport-local origin plus a `layers` array. Each
layer returns an adaptive frontier containing an exact-point batch plus zero or more
aggregate-cell batches at potentially different LOD levels. Sparse branches can
therefore reach exact leaves while dense branches in the same layer remain coarse.
All coordinates are rebased into the shared response origin before transfer. No
raster image tiles are generated or transferred.

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

The reducer tests explicitly verify that aggregate means merge `(sum, count)`
rather than averaging child means.

Likely future extensions include layer visibility controls, shared color
normalization, categorical aggregate
histograms/top-k summaries, more visual scales, Arrow IPC or binary HTTP
responses, approximate mergeable reducers (quantiles/heavy hitters), appendable
datasets, and parallel LOD construction.
