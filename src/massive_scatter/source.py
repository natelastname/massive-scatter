from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.dataset as pads


def input_batches(
    path: str | Path,
    *,
    columns: list[str],
    batch_size: int = 131_072,
) -> Iterator[pa.RecordBatch]:
    """Stream selected columns from a Parquet dataset or CSV file."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    suffix = source.suffix.lower()
    if source.is_dir() or suffix in {".parquet", ".pq"}:
        dataset = pads.dataset(source, format="parquet")
        missing = sorted(set(columns) - set(dataset.schema.names))
        if missing:
            raise ValueError(f"Input is missing columns: {', '.join(missing)}")
        yield from dataset.to_batches(
            columns=columns,
            batch_size=batch_size,
            batch_readahead=2,
            fragment_readahead=1,
        )
        return

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        reader = pacsv.open_csv(
            source,
            read_options=pacsv.ReadOptions(block_size=max(1 << 20, batch_size * 32)),
            parse_options=pacsv.ParseOptions(delimiter=delimiter),
            convert_options=pacsv.ConvertOptions(include_columns=columns),
        )
        missing = sorted(set(columns) - set(reader.schema.names))
        if missing:
            raise ValueError(f"Input is missing columns: {', '.join(missing)}")
        yield from reader
        return

    raise ValueError(
        f"Unsupported input {source}. The MVP accepts Parquet datasets "
        "and CSV/TSV files."
    )


def synthetic_batches(
    point_count: int,
    *,
    batch_size: int = 131_072,
    origin_x: int = 0,
    origin_y: int = 0,
) -> Iterator[pa.RecordBatch]:
    """Generate a deterministic, square, permutation-like scatter plot.

    The coordinates are suitable for precision tests: x values are consecutive,
    y values occupy a comparable range, and the ``color`` field is deterministic.
    """

    if point_count < 1:
        raise ValueError("point_count must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    multiplier = max(3, math.isqrt(point_count) | 1)
    while math.gcd(multiplier, point_count) != 1:
        multiplier += 2

    for start in range(0, point_count, batch_size):
        stop = min(point_count, start + batch_size)
        relative_x = np.arange(start, stop, dtype=np.int64)
        # multiplier is O(sqrt(N)), keeping the product in int64 for every
        # practical MVP dataset while producing a permutation modulo N.
        relative_y = (relative_x * multiplier + 17) % point_count
        color = ((relative_y ^ relative_x) % 23).astype(np.float64)
        yield pa.record_batch(
            [
                pa.array(relative_x + np.int64(origin_x), type=pa.int64()),
                pa.array(relative_y + np.int64(origin_y), type=pa.int64()),
                pa.array(color, type=pa.float64()),
            ],
            names=["x", "y", "color"],
        )
