from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from .builder import BuildConfig, LayerBuild, build_figure_dataset
from .source import input_batches
from .spec import (
    AxesManifest,
    CompiledPlot,
    CountValue,
    EncodingManifest,
    FieldValue,
    compile_encodings,
)

Source = str | Path | Iterable[pa.RecordBatch | pa.Table]


@dataclass(slots=True)
class _ScatterCall:
    source: Source
    x: str
    y: str
    c: str | FieldValue | CountValue | None
    color: str | None
    cmap: str
    marker: str | FieldValue
    size: float | str | FieldValue
    alpha: float | str | FieldValue | CountValue
    label: str | None
    zorder: float


@dataclass(frozen=True, slots=True)
class ScatterLayer:
    """Lightweight handle identifying one scatter call in an axes."""

    index: int
    zorder: float

    @property
    def id(self) -> str:
        return f"layer-{self.index:03d}"


class Axes:
    """A deliberately small Matplotlib-like single-axes plotting surface."""

    def __init__(self) -> None:
        self._scatters: list[_ScatterCall] = []
        self._title: str | None = None
        self._xlabel: str | None = None
        self._ylabel: str | None = None
        self._legend = False

    def scatter(
        self,
        source: Source,
        *,
        x: str,
        y: str,
        c: str | FieldValue | CountValue | None = None,
        color: str | None = None,
        cmap: str = "viridis",
        marker: str | FieldValue = "o",
        s: float | str | FieldValue = 3.0,
        alpha: float | str | FieldValue | CountValue = 0.92,
        label: str | None = None,
        zorder: float | None = None,
    ) -> ScatterLayer:
        """Add an independently stored/queryable scatter layer to this axes."""

        if c is not None and color is not None:
            raise ValueError(
                "Pass either c= for a data mapping or color= for a constant."
            )
        index = len(self._scatters)
        selected_zorder = float(index) if zorder is None else float(zorder)
        if not math.isfinite(selected_zorder):
            raise ValueError("zorder must be finite")
        self._scatters.append(
            _ScatterCall(
                source=source,
                x=x,
                y=y,
                c=c,
                color=color,
                cmap=cmap,
                marker=marker,
                size=s,
                alpha=alpha,
                label=label,
                zorder=selected_zorder,
            )
        )
        return ScatterLayer(index=index, zorder=selected_zorder)

    def set_title(self, value: str) -> None:
        self._title = value

    def set_xlabel(self, value: str) -> None:
        self._xlabel = value

    def set_ylabel(self, value: str) -> None:
        self._ylabel = value

    def set(
        self,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> None:
        if title is not None:
            self._title = title
        if xlabel is not None:
            self._xlabel = xlabel
        if ylabel is not None:
            self._ylabel = ylabel

    def legend(self, visible: bool = True) -> None:
        self._legend = visible

    def _axes_manifest(self) -> AxesManifest:
        return AxesManifest(
            title=self._title,
            xlabel=self._xlabel,
            ylabel=self._ylabel,
            legend=self._legend,
        )


class Figure:
    """A single interactive figure containing one axes and many scatter layers."""

    def __init__(self, axes: Axes) -> None:
        self.axes = axes

    @staticmethod
    def _compile(call: _ScatterCall, axes: AxesManifest) -> CompiledPlot:
        if call.color is not None:
            compiled = compile_encodings(
                x=call.x,
                y=call.y,
                color=CountValue(),
                marker=call.marker,
                size=call.size,
                alpha=call.alpha,
                cmap=call.cmap,
                label=call.label,
                axes=axes,
            )
            return CompiledPlot(
                x=compiled.x,
                y=compiled.y,
                exact_fields=compiled.exact_fields,
                field_kinds=compiled.field_kinds,
                aggregate_plan=compiled.aggregate_plan,
                scatter=compiled.scatter.__class__(
                    color=EncodingManifest("constant", value=call.color),
                    marker=compiled.scatter.marker,
                    size=compiled.scatter.size,
                    alpha=compiled.scatter.alpha,
                    cmap=compiled.scatter.cmap,
                    label=compiled.scatter.label,
                ),
                axes=compiled.axes,
            )

        color = call.c if call.c is not None else CountValue()
        return compile_encodings(
            x=call.x,
            y=call.y,
            color=color,
            marker=call.marker,
            size=call.size,
            alpha=call.alpha,
            cmap=call.cmap,
            label=call.label,
            axes=axes,
        )

    def write(
        self,
        output: str | Path,
        *,
        config: BuildConfig | None = None,
        progress=None,
    ):
        if not self.axes._scatters:
            raise ValueError("The figure has no scatter layers.")
        settings = config or BuildConfig()
        axes_manifest = self.axes._axes_manifest()
        builds: list[LayerBuild] = []
        for call in self.axes._scatters:
            compiled = self._compile(call, axes_manifest)
            if isinstance(call.source, (str, Path)):
                batches = input_batches(
                    call.source,
                    columns=list(compiled.required_columns),
                    batch_size=settings.batch_size,
                )
            else:
                batches = call.source
            builds.append(
                LayerBuild(
                    batches=batches,
                    x=call.x,
                    y=call.y,
                    plot=compiled,
                    zorder=call.zorder,
                )
            )

        return build_figure_dataset(
            output,
            builds,
            axes=axes_manifest,
            config=settings,
            progress=progress,
        )


def subplots() -> tuple[Figure, Axes]:
    """Create the single figure/axes pair supported by the current grammar."""

    axes = Axes()
    return Figure(axes), axes
