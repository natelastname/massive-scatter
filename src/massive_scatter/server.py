from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .dataset import MassiveScatterDataset


def _find_viewer(viewer_dir: str | Path | None) -> Path | None:
    if viewer_dir is not None:
        candidate = Path(viewer_dir).expanduser().resolve()
        if not (candidate / "index.html").is_file():
            raise FileNotFoundError(f"Viewer index not found in {candidate}")
        return candidate

    packaged = Path(__file__).with_name("_viewer")
    if (packaged / "index.html").is_file():
        return packaged

    repository_build = Path(__file__).resolve().parents[2] / "viewer" / "dist"
    if (repository_build / "index.html").is_file():
        return repository_build
    return None


def create_app(
    dataset_path: str | Path,
    *,
    viewer_dir: str | Path | None = None,
) -> FastAPI:
    dataset = MassiveScatterDataset(dataset_path)
    app = FastAPI(title="massive-scatter", version="0.1.0")

    @app.get("/api/manifest")
    def manifest() -> dict[str, Any]:
        return dataset.manifest.to_dict()

    @app.get("/api/view")
    def view(
        xmin: Annotated[float, Query()],
        xmax: Annotated[float, Query()],
        ymin: Annotated[float, Query()],
        ymax: Annotated[float, Query()],
        width: Annotated[int, Query(ge=1, le=16_384)] = 1_024,
        height: Annotated[int, Query(ge=1, le=16_384)] = 768,
        max_points: Annotated[int, Query(ge=1, le=1_000_000)] = 200_000,
        max_cells: Annotated[int, Query(ge=1, le=1_000_000)] = 200_000,
    ) -> dict[str, Any]:
        try:
            return dataset.view(
                min_x=xmin,
                max_x=xmax,
                min_y=ymin,
                max_y=ymax,
                pixel_width=width,
                pixel_height=height,
                max_points=max_points,
                max_cells=max_cells,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    static_dir = _find_viewer(viewer_dir)
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="viewer")
    else:

        @app.get("/", response_class=HTMLResponse)
        def viewer_not_built() -> str:
            return """
            <!doctype html>
            <html lang="en"><head><meta charset="utf-8">
            <title>massive-scatter</title></head>
            <body style="font-family: sans-serif; max-width: 50rem; margin: 4rem auto">
              <h1>Viewer assets have not been built</h1>
              <p>The data API is running. Build the TypeScript viewer with:</p>
              <pre>cd viewer\nnpm install\nnpm run build</pre>
              <p>Then restart <code>massive-scatter serve</code>.</p>
              <p><a href="/docs">Open the API documentation</a></p>
            </body></html>
            """

    return app
