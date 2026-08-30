# Project instructions

- Preserve exact source coordinates. Raster or aggregate output is never the canonical data.
- Use bounded-memory batches; do not introduce code paths that materialize the full dataset.
- High zoom renders points as dots. Do not connect sequence values with lines.
- Keep GPU positions local to a dataset or viewport origin; never cast large absolute int64 coordinates directly to float32.
- LOD reductions must be mergeable and must preserve the total-count invariant at every level.
- Add a regression test for every storage-format, precision, or LOD change.
- Run Python checks and the TypeScript viewer build before merging.
