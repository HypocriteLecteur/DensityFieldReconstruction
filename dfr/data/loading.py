"""Public dataset loading facade over the legacy format factory."""

from __future__ import annotations

from pathlib import Path

from dfr.data.registry import DatasetSource, resolve_dataset
from dfr.dataset_io import DatasetFactory, DatasetInterface


def load_dataset(
    source: DatasetSource,
    *,
    project_root: str | Path | None = None,
    verbose: bool = False,
) -> DatasetInterface:
    """Load a scenario name, YAML config, explicit data path, or DatasetSpec.

    The returned object satisfies :class:`dfr.data.Dataset`. Existing callers
    may continue using :class:`dfr.dataset_io.DatasetFactory` directly while
    they migrate.
    """
    spec = resolve_dataset(source, project_root=project_root)
    dataset = DatasetFactory(verbose=verbose).get_dataset(spec.data_path)
    dataset._metadata.update(
        {
            "dataset_name": spec.name,
            "scenario_config": (
                str(spec.config_path) if spec.config_path is not None else None
            ),
            "project_root": (
                str(spec.project_root) if spec.project_root is not None else None
            ),
        }
    )
    return dataset
