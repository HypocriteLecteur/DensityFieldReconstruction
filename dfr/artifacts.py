"""Unified run directories, manifests, and artifact persistence for DFR."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import yaml

from dfr.data.registry import default_project_root


MANIFEST_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
ARTIFACT_CATEGORIES = (
    "data",
    "checkpoints",
    "metrics",
    "figures",
    "logs",
    "cache",
)
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_segment(value: str, label: str) -> str:
    normalized = value.strip()
    if not _SAFE_SEGMENT.fullmatch(normalized):
        raise ValueError(
            f"{label} must be one path-safe segment containing only letters, "
            f"numbers, '.', '_', or '-'; got {value!r}."
        )
    return normalized


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "run"


def to_serializable(value: Any) -> Any:
    """Convert common config/result values to JSON/YAML-safe Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _isoformat_utc(value)
    if isinstance(value, Enum):
        return to_serializable(value.value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            value = value.detach()
        return value.cpu().tolist()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_serializable(to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    raise TypeError(
        f"Value of type {type(value).__name__} cannot be serialized into a run config."
    )


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Configuration for one managed run directory.

    Relative roots are resolved from ``project_root`` (or the source checkout
    root), never from the process working directory.
    """

    workflow: str
    name: str
    root: Path = Path("outputs")
    run_id: Optional[str] = None
    project_root: Optional[Path] = None
    resume: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow", _safe_segment(self.workflow, "workflow"))
        if not self.name.strip():
            raise ValueError("OutputConfig.name must not be empty.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "root", Path(self.root).expanduser())
        if self.project_root is not None:
            object.__setattr__(
                self,
                "project_root",
                Path(self.project_root).expanduser().resolve(),
            )
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite are mutually exclusive.")
        if self.resume and self.run_id is None:
            raise ValueError("resume=True requires an explicit run_id.")
        if self.run_id is not None:
            object.__setattr__(self, "run_id", _safe_segment(self.run_id, "run_id"))

    @property
    def resolved_root(self) -> Path:
        """Absolute artifact root independent of the process cwd."""
        if self.root.is_absolute():
            return self.root.resolve()
        base = self.project_root or default_project_root()
        return (base / self.root).resolve()

    def resolved_run_id(self, now: Optional[datetime] = None) -> str:
        """Return the explicit ID or generate a timestamped, readable ID."""
        if self.run_id is not None:
            return self.run_id
        instant = now or _utc_now()
        stamp = instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{stamp}-{_slugify(self.name)}"

    def to_dict(self, resolved_run_id: Optional[str] = None) -> dict[str, Any]:
        """Return the fully resolved output configuration."""
        return {
            "workflow": self.workflow,
            "name": self.name,
            "root": str(self.resolved_root),
            "run_id": resolved_run_id or self.run_id,
            "resume": self.resume,
            "overwrite": self.overwrite,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "OutputConfig":
        """Restore an output configuration from YAML/JSON-safe values."""
        return cls(
            workflow=str(values["workflow"]),
            name=str(values["name"]),
            root=Path(values.get("root", "outputs")),
            run_id=values.get("run_id"),
            project_root=(
                Path(values["project_root"])
                if values.get("project_root") is not None
                else None
            ),
            resume=bool(values.get("resume", False)),
            overwrite=bool(values.get("overwrite", False)),
        )


@dataclass(slots=True)
class RunArtifacts:
    """Paths and persistence operations for one DFR run."""

    output: OutputConfig
    run_id: str
    run_dir: Path
    manifest: dict[str, Any] = field(repr=False)

    @classmethod
    def create(
        cls,
        output: OutputConfig,
        *,
        resolved_config: Any = None,
        device: str | torch.device | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
        git_commit: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> "RunArtifacts":
        """Create or resume a managed run and write its provenance files."""
        instant = now or _utc_now()
        run_id = output.resolved_run_id(instant)
        root = output.resolved_root
        run_dir = (root / output.workflow / run_id).resolve()
        _ensure_within(root, run_dir)

        existed = run_dir.exists()
        if existed and output.overwrite:
            shutil.rmtree(run_dir)
            existed = False
        elif existed and not output.resume:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. Set resume=True to "
                "reuse it or overwrite=True to replace it."
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        for category in ARTIFACT_CATEGORIES:
            (run_dir / category).mkdir(exist_ok=True)

        config_document = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "output": output.to_dict(resolved_run_id=run_id),
            "experiment": to_serializable(resolved_config),
        }
        config_path = run_dir / "config.yaml"
        if existed:
            _validate_resumed_config(config_path, config_document)
        else:
            _atomic_yaml(config_path, config_document)

        manifest_path = run_dir / "manifest.json"
        if existed:
            manifest = _load_and_validate_manifest(
                manifest_path, workflow=output.workflow, run_id=run_id
            )
            manifest["last_resumed_at_utc"] = _isoformat_utc(instant)
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        else:
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "workflow": output.workflow,
                "name": output.name,
                "run_id": run_id,
                "created_at_utc": _isoformat_utc(instant),
                "git_commit": git_commit or _discover_git_commit(),
                "package_version": _package_version(),
                "device": str(device) if device is not None else None,
                "metadata": to_serializable(metadata or {}),
                "resume_count": 0,
            }
        _atomic_json(manifest_path, manifest)
        return cls(output=output, run_id=run_id, run_dir=run_dir, manifest=manifest)

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def directory(self, category: str) -> Path:
        """Return one canonical artifact category directory."""
        if category not in ARTIFACT_CATEGORIES:
            raise ValueError(
                f"Unknown artifact category '{category}'. Choose from "
                f"{', '.join(ARTIFACT_CATEGORIES)}."
            )
        return self.run_dir / category

    @property
    def data_dir(self) -> Path:
        return self.directory("data")

    @property
    def checkpoints_dir(self) -> Path:
        return self.directory("checkpoints")

    @property
    def metrics_dir(self) -> Path:
        return self.directory("metrics")

    @property
    def figures_dir(self) -> Path:
        return self.directory("figures")

    @property
    def logs_dir(self) -> Path:
        return self.directory("logs")

    @property
    def cache_dir(self) -> Path:
        return self.directory("cache")

    def path(self, category: str, relative_path: str | Path) -> Path:
        """Resolve an artifact path and reject absolute/path-traversal inputs."""
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError(f"Artifact path must be relative, got: {relative}")
        base = self.directory(category).resolve()
        target = (base / relative).resolve()
        _ensure_within(base, target)
        return target

    def save_json(
        self,
        relative_path: str | Path,
        value: Any,
        *,
        category: str = "data",
        overwrite: bool = False,
    ) -> Path:
        target = self._prepare_target(category, relative_path, overwrite)
        _atomic_json(target, to_serializable(value))
        return target

    def save_npz(
        self,
        relative_path: str | Path,
        *,
        category: str = "data",
        overwrite: bool = False,
        **arrays: Any,
    ) -> Path:
        target = self._prepare_target(category, relative_path, overwrite)
        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(target)
        return target

    def save_checkpoint(
        self,
        relative_path: str | Path,
        value: Any,
        *,
        overwrite: bool = False,
    ) -> Path:
        target = self._prepare_target("checkpoints", relative_path, overwrite)
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(value, temporary)
        temporary.replace(target)
        return target

    def save_figure(
        self,
        relative_path: str | Path,
        figure: Any,
        *,
        overwrite: bool = False,
        **savefig_kwargs: Any,
    ) -> Path:
        target = self._prepare_target("figures", relative_path, overwrite)
        temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
        figure.savefig(temporary, **savefig_kwargs)
        temporary.replace(target)
        return target

    def _prepare_target(
        self, category: str, relative_path: str | Path, overwrite: bool
    ) -> Path:
        target = self.path(category, relative_path)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Artifact already exists: {target}. Pass overwrite=True to replace it."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


def warn_legacy_output(path: str | Path) -> None:
    """Warn when active code is about to write to a legacy output location."""
    candidate = Path(path)
    legacy_parts = {"figs", "results"}
    if legacy_parts.intersection(candidate.parts) or (
        "scenarios" in candidate.parts and "logs" in candidate.parts
    ):
        warnings.warn(
            f"Legacy output path '{candidate}' should migrate to a RunArtifacts "
            "directory under outputs/.",
            FutureWarning,
            stacklevel=2,
        )


def _ensure_within(base: Path, target: Path) -> None:
    try:
        target.relative_to(base)
    except ValueError as error:
        raise ValueError(f"Artifact path escapes managed root '{base}': {target}") from error


def _validate_resumed_config(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resume run without config.yaml: {path.parent}")
    with path.open("r", encoding="utf-8") as stream:
        existing = yaml.safe_load(stream)
    # Resume/overwrite are invocation policies, not scientific configuration.
    for document in (existing, expected):
        if isinstance(document, dict) and isinstance(document.get("output"), dict):
            document["output"]["resume"] = False
            document["output"]["overwrite"] = False
    if existing != expected:
        raise ValueError(
            f"Resolved configuration differs from existing run: {path}. "
            "Use a new run_id or overwrite=True."
        )


def _load_and_validate_manifest(
    path: Path, *, workflow: str, run_id: str
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resume run without manifest.json: {path.parent}")
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema in {path}")
    if manifest.get("workflow") != workflow or manifest.get("run_id") != run_id:
        raise ValueError(f"Manifest identity does not match requested run: {path}")
    return manifest


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def _atomic_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
    temporary.replace(path)


def _discover_git_commit() -> Optional[str]:
    root = default_project_root()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _package_version() -> str:
    try:
        from dfr import __version__
    except (ImportError, AttributeError):
        return "unknown"
    return __version__
