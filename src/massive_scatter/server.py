from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .dataset import MassiveScatterDataset


class ViewRequest(BaseModel):
    """Viewport request carried in the POST body rather than the URL."""

    model_config = ConfigDict(extra="forbid")

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    width: int = Field(default=1_024, ge=1, le=16_384)
    height: int = Field(default=768, ge=1, le=16_384)
    max_points: int = Field(default=200_000, ge=1, le=1_000_000)
    max_cells: int = Field(default=200_000, ge=1, le=1_000_000)


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

    @app.post("/api/view")
    def view(request: ViewRequest) -> dict[str, Any]:
        try:
            return dataset.view(
                min_x=request.xmin,
                max_x=request.xmax,
                min_y=request.ymin,
                max_y=request.ymax,
                pixel_width=request.width,
                pixel_height=request.height,
                max_points=request.max_points,
                max_cells=request.max_cells,
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
