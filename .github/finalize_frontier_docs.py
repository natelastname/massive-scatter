from pathlib import Path

path = Path("README.md")
text = path.read_text()
text = text.replace(
    "- a deck.gl orthographic viewer with pan/zoom, axes, hover values, per-point\n  styling, and generated legends.",
    "- a deck.gl orthographic viewer with pan/zoom, axes, hover values, generated\n  legends, and typed/binary GPU attributes rather than object-per-primitive input.",
)
text = text.replace(
    "The finest numerical LOD begins at `base_cell_size` units per cell (64 by\ndefault). Below that scale the viewer asks for exact points. Aggregate cells are\nrendered as the square spatial bins they represent.",
    "The finest numerical LOD begins at `base_cell_size` units per cell (64 by\ndefault). It is the last aggregate level of the same implicit tree: selected\nlevel-zero cells can refine further to their exact source-point leaves. Aggregate\ncells are rendered as the square spatial bins they represent.",
)
text = text.replace(
    "Repeated `scatter()` calls create independent layers on the same axes. Each layer\nkeeps its own exact-point store, sparse LOD pyramid, reducers, styling, and exact-vs-\naggregate decision while sharing the figure camera, axes, and legend. Call order is\nthe default z-order; pass `zorder=` to override it.",
    "Repeated `scatter()` calls create independent layers on the same axes. Each layer\nkeeps its own exact-point store, sparse LOD pyramid, reducers, styling, and adaptive\nfrontier while sharing the figure camera, axes, and legend. Call order is the\ndefault z-order; pass `zorder=` to override it.",
)
anchor = """`target_cell_pixels` controls the desired maximum projected width of an aggregate
cell before the selector tries to refine it.
"""
addition = anchor + """
The GPU is only the terminal rasterizer. After the frontier is selected, the viewer
packs positions, colors, and sizes into typed deck.gl binary attributes and avoids
creating one JavaScript object per visible primitive. Picking uses compact response
arrays as sidecars indexed by deck.gl's picking index. The HTTP response is still
JSON today, so JSON parsing and the subsequent typed-array packing remain a future
transport optimization target (for example Arrow IPC).
"""
if anchor not in text:
    raise SystemExit("adaptive frontier documentation anchor not found")
text = text.replace(anchor, addition)
old = """The response has one shared viewport-local origin plus a `layers` array. Each
layer independently returns either exact points or sparse aggregate cells, so a
sparse layer can remain exact while a denser layer in the same viewport moves to
a coarser LOD. All layer coordinates are rebased into the shared response origin
before transfer. No raster image tiles are generated or transferred.
"""
new = """The response has one shared viewport-local origin plus a `layers` array. Each
layer returns an adaptive frontier containing an exact-point batch plus zero or more
aggregate-cell batches at potentially different LOD levels. Sparse branches can
therefore reach exact leaves while dense branches in the same layer remain coarse.
All coordinates are rebased into the shared response origin before transfer. No
raster image tiles are generated or transferred.
"""
if old not in text:
    raise SystemExit("view API documentation anchor not found")
path.write_text(text.replace(old, new))
