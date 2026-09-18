"""TCSHW tensor geometry and an optional native-slice viewer."""

from .volume import Volume
from ._view import Display, view

__version__ = "0.0.0.4"
__all__ = ["Volume", "Display", "view", "__version__"]
