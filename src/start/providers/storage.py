"""Storage providers."""

from __future__ import annotations

from pathlib import Path

from start.providers.base import StorageProvider


class LocalStorageProvider(StorageProvider):
    name = "local"

    def __init__(self, root: str | Path = "start_output") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_text(self, relpath: str, content: str) -> str:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return str(path)

    def read_text(self, relpath: str) -> str:
        return (self.root / relpath).read_text()
