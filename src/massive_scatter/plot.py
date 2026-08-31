from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from .builder import BuildConfig, build_dataset
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


class Axes:
    """A deliberately small Matplotlib-like single-axes plotting surface."""

    def __init__(self) -> None:
        self._scatter: _ScatterCall | None = None
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
    ) -> None:
        """Add the scatter layer for this figure.

        The first grammar version intentionally supports one massive scatter
        layer. ``c`` maps a numeric source field (or mergeable field expression)
        to color; use ``color=`` for a constant CSS color. Field-valued marker
        and size channels apply to exact-point mode. Aggregate LOD always uses
        spatial square cells.
        """

        if self._scatter is not None:
            raise NotImplementedError(
                "The first plot grammar supports one scatter layer per axes."
            )
        if c is not None and color is not None:
            raise ValueError(
                "Pass either c= for a data mapping or color= for a constant."
            )
        self._scatter = _ScatterCall(
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
        )

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
    """A single interactive massive-scatter figure."""

    def __init__(self, axes: Axes) -> None:
        self.axes = axes

    @staticmethod
    def _compile(call: _ScatterCall, axes: AxesManifest) -> CompiledPlot:
        if call.color is not None:
            # compile_encodings treats a string color as a field mapping, so use
            # a harmless placeholder and replace the resulting color manifest.
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
        call = self.axes._scatter
        if call is None:
            raise ValueError("The figure has no scatter layer.")
        compiled = self._compile(call, self.axes._axes_manifest())
        settings = config or BuildConfig()

        if isinstance(call.source, (str, Path)):
            batches = input_batches(
                call.source,
                columns=list(compiled.required_columns),
                batch_size=settings.batch_size,
            )
        else:
            batches = call.source

        return build_dataset(
            output,
            batches,
            x=call.x,
            y=call.y,
            config=settings,
            progress=progress,
            plot=compiled,
        )


def subplots() -> tuple[Figure, Axes]:
    """Create the single figure/axes pair supported by the first plot grammar."""

    axes = Axes()
    return Figure(axes), axes
