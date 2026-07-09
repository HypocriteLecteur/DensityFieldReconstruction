"""Shared command-line plumbing for managed analysis entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dfr.artifacts import OutputConfig, RunArtifacts
from dfr.data.registry import default_project_root


def add_managed_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Add canonical managed-analysis output arguments to a parser.

    The added options are ``--project-root``, ``--output-root``, ``--run-id``,
    and the mutually exclusive ``--resume``/``--overwrite-run`` policy flags.
    Relative output roots are later resolved through :class:`OutputConfig`,
    keeping CLI behavior consistent with package workflows.
    """
    parser.add_argument(
        "--project-root", type=Path, default=default_project_root()
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id", default=None)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite-run", action="store_true")


def create_analysis_artifacts(
    args: argparse.Namespace,
    *,
    name: str,
    resolved_config: Any,
    entrypoint: str,
    device: str = "cuda",
) -> RunArtifacts:
    """Create a managed analysis run from the shared parser arguments.

    ``name`` becomes the human-readable run name, ``entrypoint`` is recorded in
    the manifest metadata, and ``resolved_config`` is serialized into the run's
    ``config.yaml``. The managed workflow is always ``analysis``.
    """
    return RunArtifacts.create(
        OutputConfig(
            workflow="analysis",
            name=name,
            root=args.output_root,
            run_id=args.run_id,
            project_root=args.project_root,
            resume=args.resume,
            overwrite=args.overwrite_run,
        ),
        resolved_config=resolved_config,
        device=device,
        metadata={"entrypoint": entrypoint},
    )
