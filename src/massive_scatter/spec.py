from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ReducerName = Literal["sum", "mean", "min", "max"]


@dataclass(frozen=True, slots=True)
class FieldValue:
    """A source column together with its aggregate-LOD reduction."""

    source: str
    reducer: ReducerName | None = None

    def with_default_reducer(self, reducer: ReducerName) -> FieldValue:
        return self if self.reducer is not None else FieldValue(self.source, reducer)


@dataclass(frozen=True, slots=True)
class CountValue:
    """The implicit number of exact points represented by an aggregate cell."""


COUNT = CountValue()


def field(source: str, *, reduce: ReducerName | None = None) -> FieldValue:
    return FieldValue(source, reduce)


def sum(source: str) -> FieldValue:  # noqa: A001
    return FieldValue(source, "sum")


def mean(source: str) -> FieldValue:
    return FieldValue(source, "mean")


def min(source: str) -> FieldValue:  # noqa: A001
    return FieldValue(source, "min")


def max(source: str) -> FieldValue:  # noqa: A001
    return FieldValue(source, "max")


def count() -> CountValue:
    return COUNT


@dataclass(frozen=True, slots=True)
class AggregateRequest:
    """One mergeable numerical summary required by the plot at aggregate LOD."""

    key: str
    source: str
    storage: str
    reducer: ReducerName

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "source": self.source,
            "storage": self.storage,
            "reducer": self.reducer,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AggregateRequest:
        reducer = str(value["reducer"])
        if reducer not in {"sum", "mean", "min", "max"}:
            raise ValueError(f"Unsupported aggregate reducer: {reducer}")
        return cls(
            key=str(value["key"]),
            source=str(value["source"]),
            storage=str(value["storage"]),
            reducer=reducer,  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AggregatePlan:
    """Deduplicated mergeable reducer state required by aggregate LOD."""

    requests: tuple[AggregateRequest, ...] = ()

    def by_key(self) -> dict[str, AggregateRequest]:
        return {request.key: request for request in self.requests}


@dataclass(frozen=True, slots=True)
class EncodingManifest:
    kind: Literal["constant", "field", "count"]
    value: str | float | None = None
    source: str | None = None
    aggregate: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"kind": self.kind}
        if self.value is not None:
            result["value"] = self.value
        if self.source is not None:
            result["source"] = self.source
        if self.aggregate is not None:
            result["aggregate"] = self.aggregate
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EncodingManifest:
        kind = str(value["kind"])
        if kind not in {"constant", "field", "count"}:
            raise ValueError(f"Unsupported encoding kind: {kind}")
        raw_value = value.get("value")
        if raw_value is not None and not isinstance(raw_value, (str, int, float)):
            raise TypeError("Encoding constants must be strings or numbers.")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            value=raw_value,
            source=value.get("source"),
            aggregate=value.get("aggregate"),
        )


@dataclass(frozen=True, slots=True)
class ScatterManifest:
    color: EncodingManifest
    marker: EncodingManifest
    size: EncodingManifest
    alpha: EncodingManifest
    cmap: str = "viridis"
    label: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "scatter",
            "color": self.color.to_dict(),
            "marker": self.marker.to_dict(),
            "size": self.size.to_dict(),
            "alpha": self.alpha.to_dict(),
            "cmap": self.cmap,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ScatterManifest:
        if value.get("type") != "scatter":
            raise ValueError("Only scatter plot manifests are supported.")
        return cls(
            color=EncodingManifest.from_dict(value["color"]),
            marker=EncodingManifest.from_dict(value["marker"]),
            size=EncodingManifest.from_dict(value["size"]),
            alpha=EncodingManifest.from_dict(value["alpha"]),
            cmap=str(value.get("cmap", "viridis")),
            label=value.get("label"),
        )


@dataclass(frozen=True, slots=True)
class AxesManifest:
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    legend: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "legend": self.legend,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AxesManifest:
        return cls(
            title=value.get("title"),
            xlabel=value.get("xlabel"),
            ylabel=value.get("ylabel"),
            legend=bool(value.get("legend", False)),
        )


@dataclass(frozen=True, slots=True)
class PlotManifest:
    scatter: ScatterManifest
    axes: AxesManifest
    exact_fields: dict[str, str]
    categorical_fields: dict[str, tuple[str, ...]]
    numeric_ranges: dict[str, tuple[float, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "scatter": self.scatter.to_dict(),
            "axes": self.axes.to_dict(),
            "exact_fields": dict(self.exact_fields),
            "categorical_fields": {
                key: list(values) for key, values in self.categorical_fields.items()
            },
            "numeric_ranges": {
                key: [minimum, maximum]
                for key, (minimum, maximum) in self.numeric_ranges.items()
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PlotManifest:
        return cls(
            scatter=ScatterManifest.from_dict(value["scatter"]),
            axes=AxesManifest.from_dict(value.get("axes", {})),
            exact_fields={
                str(k): str(v) for k, v in value.get("exact_fields", {}).items()
            },
            categorical_fields={
                str(k): tuple(str(item) for item in items)
                for k, items in value.get("categorical_fields", {}).items()
            },
            numeric_ranges={
                str(k): (float(items[0]), float(items[1]))
                for k, items in value.get("numeric_ranges", {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class CompiledPlot:
    """Build-time plot contract produced by the Matplotlib-like API."""

    x: str
    y: str
    exact_fields: dict[str, str]
    field_kinds: dict[str, Literal["numeric", "categorical"]]
    aggregate_plan: AggregatePlan
    scatter: ScatterManifest
    axes: AxesManifest

    @property
    def aggregates(self) -> tuple[AggregateRequest, ...]:
        return self.aggregate_plan.requests

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.x, self.y, *self.exact_fields.keys())))


def _aggregate_key(index: int) -> str:
    return f"agg_{index:03d}"


def compile_encodings(
    *,
    x: str,
    y: str,
    color: str | FieldValue | CountValue,
    marker: str | FieldValue,
    size: float | str | FieldValue,
    alpha: float | str | FieldValue | CountValue,
    cmap: str,
    label: str | None,
    axes: AxesManifest,
) -> CompiledPlot:
    if cmap not in {"viridis", "plasma", "magma"}:
        raise ValueError("cmap must be one of: viridis, plasma, magma")

    exact_sources: list[str] = []
    field_kinds: dict[str, Literal["numeric", "categorical"]] = {}
    aggregate_requests: list[tuple[str, ReducerName]] = []

    def register_field(source: str, kind: Literal["numeric", "categorical"]) -> None:
        if source not in exact_sources:
            exact_sources.append(source)
        previous = field_kinds.get(source)
        if previous is None:
            field_kinds[source] = kind
        elif previous != kind:
            raise ValueError(
                f"Field {source!r} is used as both numeric and categorical; "
                "the first plot grammar requires one interpretation per source field."
            )

    def register_aggregate(value: FieldValue, default: ReducerName) -> str:
        selected = value.with_default_reducer(default)
        assert selected.reducer is not None
        pair = (selected.source, selected.reducer)
        try:
            index = aggregate_requests.index(pair)
        except ValueError:
            aggregate_requests.append(pair)
            index = len(aggregate_requests) - 1
        return _aggregate_key(index)

    def numeric_encoding(
        value: float | str | FieldValue | CountValue,
        *,
        default_reducer: ReducerName,
    ) -> EncodingManifest:
        if isinstance(value, CountValue):
            return EncodingManifest("count")
        if isinstance(value, (int, float)):
            return EncodingManifest("constant", value=float(value))
        field_value = FieldValue(value) if isinstance(value, str) else value
        register_field(field_value.source, "numeric")
        aggregate = register_aggregate(field_value, default_reducer)
        return EncodingManifest("field", source=field_value.source, aggregate=aggregate)

    if isinstance(color, CountValue):
        color_encoding = EncodingManifest("count")
    elif isinstance(color, str):
        # ``c`` follows Matplotlib's data-mapping role. Constant colors use
        # the separate ``color=`` argument in the public API.
        register_field(color, "numeric")
        aggregate = register_aggregate(FieldValue(color), "mean")
        color_encoding = EncodingManifest("field", source=color, aggregate=aggregate)
    else:
        register_field(color.source, "numeric")
        aggregate = register_aggregate(color, "mean")
        color_encoding = EncodingManifest(
            "field", source=color.source, aggregate=aggregate
        )

    known_markers = {
        "o",
        "circle",
        "s",
        "square",
        "^",
        "triangle",
        "D",
        "diamond",
        "x",
        "+",
    }
    if isinstance(marker, FieldValue):
        register_field(marker.source, "categorical")
        marker_encoding = EncodingManifest("field", source=marker.source)
    elif marker in known_markers:
        marker_encoding = EncodingManifest("constant", value=marker)
    else:
        register_field(marker, "categorical")
        marker_encoding = EncodingManifest("field", source=marker)

    if isinstance(size, (int, float)):
        size_encoding = EncodingManifest("constant", value=float(size))
    else:
        size_field = FieldValue(size) if isinstance(size, str) else size
        register_field(size_field.source, "numeric")
        # Marker size is an exact-point channel. Aggregate LOD geometry remains
        # the spatial square bin, so size is deliberately discarded there.
        size_encoding = EncodingManifest("field", source=size_field.source)

    alpha_encoding = numeric_encoding(alpha, default_reducer="mean")

    exact_fields = {
        source: f"field_{index:03d}" for index, source in enumerate(exact_sources)
    }
    aggregates = tuple(
        AggregateRequest(
            key=_aggregate_key(index),
            source=source,
            storage=exact_fields[source],
            reducer=reducer,
        )
        for index, (source, reducer) in enumerate(aggregate_requests)
    )
    return CompiledPlot(
        x=x,
        y=y,
        exact_fields=exact_fields,
        field_kinds=field_kinds,
        aggregate_plan=AggregatePlan(aggregates),
        scatter=ScatterManifest(
            color=color_encoding,
            marker=marker_encoding,
            size=size_encoding,
            alpha=alpha_encoding,
            cmap=cmap,
            label=label,
        ),
        axes=axes,
    )
