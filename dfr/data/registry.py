"""Resolve scenario names and explicit config/data paths without cwd coupling."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from dfr.data.spec import DatasetSpec


DatasetSource = Union[str, Path, DatasetSpec]


def default_project_root() -> Path:
    """Return the source checkout root containing the installed ``dfr`` package."""
    return Path(__file__).resolve().parents[2]


class ScenarioRegistry:
    """Resolve scenario names and dataset paths relative to an explicit root."""

    def __init__(self, project_root: str | Path | None = None):
        self.project_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else default_project_root()
        )
        self.scenarios_root = self.project_root / "scenarios"

    def available_scenarios(self) -> tuple[str, ...]:
        """Return scenario directory names containing ``config.yaml``."""
        if not self.scenarios_root.is_dir():
            return ()
        return tuple(
            sorted(
                path.parent.name
                for path in self.scenarios_root.glob("*/config.yaml")
                if path.is_file()
            )
        )

    def resolve_scenario(self, name: str) -> DatasetSpec:
        """Resolve one registered scenario name to an absolute DatasetSpec."""
        if not name or Path(name).name != name:
            raise ValueError(f"Scenario name must be a single directory name: {name!r}")
        config_path = self.scenarios_root / name / "config.yaml"
        if not config_path.is_file():
            available = ", ".join(self.available_scenarios()) or "<none>"
            raise FileNotFoundError(
                f"Unknown scenario '{name}' under '{self.scenarios_root}'. "
                f"Available scenarios: {available}."
            )
        return self.resolve_config(config_path, scenario_name=name)

    def resolve_config(
        self, config_path: str | Path, scenario_name: str | None = None
    ) -> DatasetSpec:
        """Resolve a YAML scenario config to its dataset source."""
        path = self._resolve_input_path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Scenario config does not exist: {path}")
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Scenario config must be YAML, got: {path}")

        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError(f"Scenario config must contain a YAML mapping: {path}")
        data_value = config.get("data_file")
        if not isinstance(data_value, str) or not data_value.strip():
            raise KeyError(f"Scenario config has no non-empty 'data_file': {path}")

        data_reference = Path(data_value).expanduser()
        if data_reference.is_absolute():
            data_path = data_reference.resolve()
        elif self._is_registered_config(path):
            # Existing scenario configs define data paths from the project root.
            data_path = (self.project_root / data_reference).resolve()
        else:
            # Standalone configs are portable: relative data sits beside config.
            data_path = (path.parent / data_reference).resolve()

        spec = DatasetSpec(
            name=scenario_name or path.parent.name,
            data_path=data_path,
            config_path=path,
            project_root=self.project_root,
        )
        self._validate_data_path(spec)
        return spec

    def resolve(self, source: DatasetSource) -> DatasetSpec:
        """Resolve a DatasetSpec, scenario name, YAML config, or data path."""
        if isinstance(source, DatasetSpec):
            self._validate_data_path(source)
            return source

        source_text = str(source)
        source_path = Path(source_text).expanduser()
        suffix = source_path.suffix.lower()
        looks_like_path = (
            suffix != ""
            or source_path.is_absolute()
            or len(source_path.parts) > 1
        )

        if not looks_like_path:
            return self.resolve_scenario(source_text)

        path = self._resolve_input_path(source_path)
        if suffix in {".yaml", ".yml"}:
            return self.resolve_config(path)

        spec = DatasetSpec(
            name=path.stem,
            data_path=path,
            project_root=self.project_root,
        )
        self._validate_data_path(spec)
        return spec

    def _resolve_input_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def _is_registered_config(self, path: Path) -> bool:
        try:
            path.relative_to(self.scenarios_root)
        except ValueError:
            return False
        return True

    @staticmethod
    def _validate_data_path(spec: DatasetSpec) -> None:
        if not spec.data_path.exists():
            context = (
                f" referenced by scenario config '{spec.config_path}'"
                if spec.config_path is not None
                else ""
            )
            raise FileNotFoundError(
                f"Dataset file for '{spec.name}' does not exist: "
                f"{spec.data_path}{context}"
            )
        if not spec.data_path.is_file():
            raise ValueError(
                f"Dataset source for '{spec.name}' is not a file: {spec.data_path}"
            )


def resolve_dataset(
    source: DatasetSource, *, project_root: str | Path | None = None
) -> DatasetSpec:
    """Resolve a dataset source with a one-shot registry."""
    return ScenarioRegistry(project_root=project_root).resolve(source)
