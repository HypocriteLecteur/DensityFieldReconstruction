"""Typed description of a dataset source and its resolution context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Resolved dataset source used by loading workflows.

    Paths are stored as absolute paths so downstream code does not depend on
    the process working directory.
    """

    name: str
    data_path: Path
    config_path: Optional[Path] = None
    project_root: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("DatasetSpec.name must not be empty.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "data_path", self.data_path.expanduser().resolve())
        if self.config_path is not None:
            object.__setattr__(
                self, "config_path", self.config_path.expanduser().resolve()
            )
        if self.project_root is not None:
            object.__setattr__(
                self, "project_root", self.project_root.expanduser().resolve()
            )
