"""Build and view precision-preserving, out-of-core scatter plots."""

from .builder import BuildConfig, build_dataset
from .dataset import MassiveScatterDataset
from .manifest import Manifest
from .plot import Axes, Figure, subplots
from .spec import AggregatePlan, CountValue, FieldValue, count, field, max, mean, min, sum

__all__ = [
    "AggregatePlan",
    "Axes",
    "BuildConfig",
    "CountValue",
    "FieldValue",
    "Figure",
    "Manifest",
    "MassiveScatterDataset",
    "build_dataset",
    "count",
    "field",
    "max",
    "mean",
    "min",
    "subplots",
    "sum",
]
__version__ = "0.2.0"
