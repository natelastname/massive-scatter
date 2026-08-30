"""Build and view precision-preserving, out-of-core scatter plots."""

from .builder import BuildConfig, build_dataset
from .dataset import MassiveScatterDataset
from .manifest import Manifest

__all__ = ["BuildConfig", "Manifest", "MassiveScatterDataset", "build_dataset"]
__version__ = "0.1.0"
