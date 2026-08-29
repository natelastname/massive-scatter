from __future__ import annotations

import json
from pathlib import Path

import uvicorn
from cyclopts import App

from .builder import BuildConfig, build_dataset
from .dataset import MassiveScatterDataset
from .manifest import Manifest
from .server import create_app
from .source import input_batches, synthetic_batches

app = App(
    name="massive-scatter",
    help="Build and explore precision-preserving, out-of-core scatter plots.",
)


def _progress(message: str) -> None:
    print(message, flush=True)


@app.command
def build(
    input_path: Path,
    output: Path,
    *,
    x: str = "x",
    y: str = "y",
    color: str | None = None,
    tile_size: int = 256,
    base_cell_size: int = 64,
    part_rows: int = 1_000_000,
    batch_size: int = 131_072,
    overwrite: bool = False,
) -> None:
    """Build a .msplot directory from Parquet or CSV/TSV points."""

    columns = list(dict.fromkeys([x, y] + ([color] if color else [])))
    batches = input_batches(input_path, columns=columns, batch_size=batch_size)
    build_dataset(
        output,
        batches,
        x=x,
        y=y,
        color=color,
        config=BuildConfig(
            tile_size=tile_size,
            base_cell_size=base_cell_size,
            part_rows=part_rows,
            batch_size=batch_size,
            overwrite=overwrite,
        ),
        progress=_progress,
    )


@app.command
def generate(
    output: Path,
    *,
    points: int = 1_000_000,
    batch_size: int = 131_072,
    tile_size: int = 256,
    base_cell_size: int = 64,
    origin_x: int = 0,
    origin_y: int = 0,
    overwrite: bool = False,
) -> None:
    """Generate a deterministic square test dataset without holding it in RAM."""

    build_dataset(
        output,
        synthetic_batches(
            points,
            batch_size=batch_size,
            origin_x=origin_x,
            origin_y=origin_y,
        ),
        color="color",
        config=BuildConfig(
            tile_size=tile_size,
            base_cell_size=base_cell_size,
            part_rows=max(batch_size, 1_000_000),
            batch_size=batch_size,
            overwrite=overwrite,
        ),
        progress=_progress,
    )


@app.command
def info(dataset: Path) -> None:
    """Print the portable dataset manifest."""

    print(json.dumps(Manifest.load(dataset).to_dict(), indent=2, sort_keys=True))


@app.command
def check(dataset: Path) -> None:
    """Check point-index and LOD count invariants."""

    problems = MassiveScatterDataset(dataset).check()
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        raise SystemExit(1)
    print("OK")


@app.command
def serve(
    dataset: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    viewer_dir: Path | None = None,
) -> None:
    """Serve the query API and the built deck.gl viewer."""

    uvicorn.run(create_app(dataset, viewer_dir=viewer_dir), host=host, port=port)


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
