"""Data layer: explicit multi-format loaders and image-folder discovery.

This package sits beneath the connectors and the modality tracks. It knows how
to turn a path (or folder) into a pandas DataFrame (tabular) or an image
manifest DataFrame (vision), with honest, gated handling of optional formats.
"""

from start.data.loaders import (
    SUPPORTED_TABULAR_FORMATS,
    discover_image_folder,
    load_any_tabular,
    sniff_format,
)

__all__ = [
    "SUPPORTED_TABULAR_FORMATS",
    "discover_image_folder",
    "load_any_tabular",
    "sniff_format",
]
