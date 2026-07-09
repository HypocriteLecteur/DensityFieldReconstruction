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
    """Load one dataset source through the stable package facade.

    Parameters
    ----------
    source:
        Registered scenario name, YAML scenario config, explicit dataset file,
        or already resolved :class:`dfr.data.DatasetSpec`.
    project_root:
        Root used to resolve scenario names and relative paths. When omitted,
        the installed source checkout root is used rather than the process
        working directory.
    verbose:
        Forwarded to the legacy :class:`dfr.dataset_io.DatasetFactory`.

    Returns
    -------
    dfr.dataset_io.DatasetInterface
        A dataset satisfying :class:`dfr.data.Dataset`. Trajectories are
        world-coordinate arrays with shape ``(frames, agents, 3)`` and the
        dataset metadata records the resolved name, scenario config, and
        project root.

    Notes
    -----
    Loading is read-only. This function does not create output directories or
    write managed artifacts.
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
