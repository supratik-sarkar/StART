"""Data layer: explicit multi-format loaders, public data adapters, and deterministic schema contracts."""

from start.data.adapters import (
    ChunkClassification,
    DataFrameAdapter,
    DataSchema,
    DataSourceAdapter,
    LocalFileAdapter,
    SQLDataSourceAdapter,
    WarehouseAdapter,
    execute_chunked,
)
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
    "ChunkClassification",
    "DataSchema",
    "DataSourceAdapter",
    "DataFrameAdapter",
    "LocalFileAdapter",
    "SQLDataSourceAdapter",
    "WarehouseAdapter",
    "execute_chunked",
]
