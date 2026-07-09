"""Typed, serializable configuration contracts for DFR workflows.

Configuration objects in this module are small dataclasses with validation and
``to_dict``/``from_dict`` helpers.  They are designed to be saved in managed
run configs, compared in tests, and passed directly into high-level workflow
functions.

Units and conventions
---------------------
- Frame IDs are integer dataset frame indices.
- Analysis and reconstruction scales are positive floats in world-coordinate
  units unless a lower-level analysis function explicitly documents normalized
  nearest-neighbour-distance units.
- Explicit camera poses are ``(x, y, z, qx, qy, qz, qw)`` values in world
  coordinates plus quaternion orientation.
- Device strings are passed through to the consuming workflow; CUDA-dependent
  workflows validate availability when they run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from dfr.artifacts import OutputConfig, to_serializable
from dfr.data.spec import DatasetSpec


RUN_CONFIG_SCHEMA_VERSION = 1


@dataclass
class TrainingParams:
    """Learning-rate and regularization controls for GMM optimization.

    These values are passed through to the reconstruction backend unchanged.
    Learning-rate fields ending in ``_c`` are the initial coefficients used by
    the legacy schedule, ``*_final_c`` fields are final schedule coefficients,
    and ``lr_max_steps`` is the per-frame optimization iteration count.
    """

    xyz_lr_c: float
    xyz_lr_final_c: float
    radius_lr_c: float
    radius_lr_final_c: float
    weights_lr_c: float
    weights_lr_final_c: float
    xyz_reg: float
    radius_reg: float
    radius_cutoff_inv: float
    lr_max_steps: int

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainingParams":
        return cls(**{key: values[key] for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass
class ReconstructionParams:
    """Adaptive-scale and visual-hull controls for reconstruction.

    ``targetd_num_mode`` is the historical field name used by existing config
    files; :attr:`target_mode_count` provides the corrected spelling. Voxel
    fields are expressed in world-coordinate units and cap the intermediate
    visual-hull grid used to initialize Gaussian components.
    """

    # Keep the historic misspelling for compatibility with existing callers.
    targetd_num_mode: int
    voxel_scale: float
    voxel_peak_threshold: float
    voxel_grid_max_size: int
    voxel_peaks_number: int

    @property
    def target_mode_count(self) -> int:
        """Correctly named compatibility view of ``targetd_num_mode``."""
        return self.targetd_num_mode

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ReconstructionParams":
        normalized = dict(values)
        if "target_mode_count" in normalized and "targetd_num_mode" not in normalized:
            normalized["targetd_num_mode"] = normalized.pop("target_mode_count")
        return cls(**{key: normalized[key] for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Camera layout consumed by reconstruction workflows.

    ``layout="encircling"`` derives camera poses around the selected frame from
    the scenario camera settings and the point-cloud extent. ``padding`` is a
    positive world-coordinate margin around that extent. ``layout="explicit"``
    requires one pose per camera, each shaped ``(x, y, z, qx, qy, qz, qw)`` in
    world coordinates with quaternion orientation. The current reconstruction
    backend expects ``device="cuda"`` and validates CUDA availability when the
    workflow runs.
    """

    count: int = 2
    layout: str = "encircling"
    padding: float = 1.0
    is_3d: bool = False
    device: str = "cuda"
    poses: Optional[tuple[tuple[float, ...], ...]] = None

    def __post_init__(self) -> None:
        if self.count < 2:
            raise ValueError("CameraConfig.count must be at least 2.")
        if self.layout not in {"encircling", "explicit"}:
            raise ValueError("CameraConfig.layout must be 'encircling' or 'explicit'.")
        if self.padding <= 0:
            raise ValueError("CameraConfig.padding must be positive.")
        if not self.device.strip():
            raise ValueError("CameraConfig.device must not be empty.")

        if self.poses is not None:
            normalized = tuple(tuple(float(value) for value in pose) for pose in self.poses)
            object.__setattr__(self, "poses", normalized)
        if self.layout == "explicit":
            if self.poses is None or len(self.poses) != self.count:
                raise ValueError(
                    "Explicit camera layout requires one pose per camera."
                )
            if any(len(pose) != 7 for pose in self.poses):
                raise ValueError(
                    "Each explicit pose must be [x, y, z, qx, qy, qz, qw]."
                )
        elif self.poses is not None:
            raise ValueError("Encircling camera layout does not accept explicit poses.")

    @classmethod
    def encircling(
        cls,
        count: int = 2,
        *,
        padding: float = 1.0,
        is_3d: bool = False,
        device: str = "cuda",
    ) -> "CameraConfig":
        """Create an automatically aimed orbit/encircling camera layout.

        The returned config contains no concrete poses; they are derived later
        from the dataset frame and scenario camera calibration.
        """
        return cls(
            count=count,
            layout="encircling",
            padding=padding,
            is_3d=is_3d,
            device=device,
        )

    @classmethod
    def explicit(
        cls,
        poses: tuple[tuple[float, ...], ...] | list[list[float]],
        *,
        device: str = "cuda",
    ) -> "CameraConfig":
        """Create a fixed-pose camera layout.

        ``poses`` must contain one seven-value pose per camera in
        ``(x, y, z, qx, qy, qz, qw)`` order. Values are normalized to immutable
        tuples so the config can be serialized and safely reused.
        """
        normalized = tuple(tuple(pose) for pose in poses)
        return cls(
            count=len(normalized),
            layout="explicit",
            device=device,
            poses=normalized,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "layout": self.layout,
            "padding": self.padding,
            "is_3d": self.is_3d,
            "device": self.device,
            "poses": [list(pose) for pose in self.poses] if self.poses else None,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CameraConfig":
        poses = values.get("poses")
        return cls(
            count=int(values.get("count", len(poses) if poses else 2)),
            layout=str(values.get("layout", "encircling")),
            padding=float(values.get("padding", 1.0)),
            is_3d=bool(values.get("is_3d", False)),
            device=str(values.get("device", "cuda")),
            poses=(tuple(tuple(pose) for pose in poses) if poses is not None else None),
        )


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Common frame, scale, seed, and device controls for analyses.

    ``frames`` are integer dataset frame indices. ``scales`` are strictly
    increasing positive floats; high-level analyses document whether they use
    world-coordinate scale units or normalized nearest-neighbour-distance
    units. ``seed`` controls stochastic analysis helpers where applicable.
    """

    frames: Optional[tuple[int, ...]] = None
    scales: Optional[tuple[float, ...]] = None
    seed: int = 12345
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.frames is not None:
            frames = tuple(int(frame) for frame in self.frames)
            if not frames:
                raise ValueError("AnalysisConfig.frames must not be empty.")
            object.__setattr__(self, "frames", frames)
        if self.scales is not None:
            scales = tuple(float(scale) for scale in self.scales)
            if not scales or any(scale <= 0 for scale in scales):
                raise ValueError("AnalysisConfig.scales must contain positive values.")
            if any(right <= left for left, right in zip(scales, scales[1:])):
                raise ValueError("AnalysisConfig.scales must be strictly increasing.")
            object.__setattr__(self, "scales", scales)
        if self.seed < 0:
            raise ValueError("AnalysisConfig.seed must be non-negative.")
        if not self.device.strip():
            raise ValueError("AnalysisConfig.device must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": list(self.frames) if self.frames is not None else None,
            "scales": list(self.scales) if self.scales is not None else None,
            "seed": self.seed,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AnalysisConfig":
        return cls(
            frames=(
                tuple(values["frames"]) if values.get("frames") is not None else None
            ),
            scales=(
                tuple(values["scales"]) if values.get("scales") is not None else None
            ),
            seed=int(values.get("seed", 12345)),
            device=str(values.get("device", "cuda")),
        )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Voxelized density-overlap evaluation controls.

    ``voxel_resolution`` is a positive world-coordinate grid spacing.
    ``bounds`` may explicitly provide three ``(min, max)`` world-coordinate
    pairs; when omitted, evaluation bounds are derived from the ground-truth
    positions and reconstruction scale. ``batch_size`` limits the number of
    sampled voxels evaluated per backend call.
    """

    voxel_resolution: float = 0.5
    batch_size: int = 500_000
    bounds: Optional[tuple[tuple[float, float], ...]] = None
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.voxel_resolution <= 0:
            raise ValueError("EvaluationConfig.voxel_resolution must be positive.")
        if self.batch_size < 1:
            raise ValueError("EvaluationConfig.batch_size must be positive.")
        if self.bounds is not None:
            bounds = tuple(tuple(float(value) for value in axis) for axis in self.bounds)
            if len(bounds) != 3 or any(len(axis) != 2 for axis in bounds):
                raise ValueError("EvaluationConfig.bounds must contain three (min, max) pairs.")
            if any(axis[1] <= axis[0] for axis in bounds):
                raise ValueError("Every evaluation bound must satisfy min < max.")
            object.__setattr__(self, "bounds", bounds)
        if not self.device.strip():
            raise ValueError("EvaluationConfig.device must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "voxel_resolution": self.voxel_resolution,
            "batch_size": self.batch_size,
            "bounds": [list(axis) for axis in self.bounds] if self.bounds else None,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "EvaluationConfig":
        bounds = values.get("bounds")
        return cls(
            voxel_resolution=float(values.get("voxel_resolution", 0.5)),
            batch_size=int(values.get("batch_size", 500_000)),
            bounds=(tuple(tuple(axis) for axis in bounds) if bounds is not None else None),
            device=str(values.get("device", "cuda")),
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Serializable composition of dataset, output, and workflow settings.

    A ``RunConfig`` is the object normally persisted into managed run
    ``config.yaml`` files. It may hold a registered dataset name or a resolved
    :class:`dfr.data.DatasetSpec`; optional camera, analysis, training,
    reconstruction, and evaluation sections are included only for workflows
    that need them.
    """

    dataset: DatasetSpec | str
    output: OutputConfig
    camera: Optional[CameraConfig] = None
    analysis: Optional[AnalysisConfig] = None
    training: Optional[TrainingParams] = None
    reconstruction: Optional[ReconstructionParams] = None
    evaluation: Optional[EvaluationConfig] = None
    seed: int = 12345

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, (DatasetSpec, str)):
            raise TypeError("RunConfig.dataset must be a DatasetSpec or scenario name.")
        if isinstance(self.dataset, str) and not self.dataset.strip():
            raise ValueError("RunConfig dataset name must not be empty.")
        if self.seed < 0:
            raise ValueError("RunConfig.seed must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_CONFIG_SCHEMA_VERSION,
            "dataset": (
                self.dataset.to_dict()
                if isinstance(self.dataset, DatasetSpec)
                else self.dataset
            ),
            "output": self.output.to_dict(),
            "camera": self.camera.to_dict() if self.camera else None,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "training": self.training.to_dict() if self.training else None,
            "reconstruction": (
                self.reconstruction.to_dict() if self.reconstruction else None
            ),
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RunConfig":
        schema = int(values.get("schema_version", RUN_CONFIG_SCHEMA_VERSION))
        if schema != RUN_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported RunConfig schema version: {schema}")
        dataset = values["dataset"]
        if isinstance(dataset, Mapping):
            dataset = DatasetSpec.from_dict(dict(dataset))
        return cls(
            dataset=dataset,
            output=OutputConfig.from_dict(values["output"]),
            camera=(
                CameraConfig.from_dict(values["camera"])
                if values.get("camera") is not None
                else None
            ),
            analysis=(
                AnalysisConfig.from_dict(values["analysis"])
                if values.get("analysis") is not None
                else None
            ),
            training=(
                TrainingParams.from_dict(values["training"])
                if values.get("training") is not None
                else None
            ),
            reconstruction=(
                ReconstructionParams.from_dict(values["reconstruction"])
                if values.get("reconstruction") is not None
                else None
            ),
            evaluation=(
                EvaluationConfig.from_dict(values["evaluation"])
                if values.get("evaluation") is not None
                else None
            ),
            seed=int(values.get("seed", 12345)),
        )

    def serializable(self) -> dict[str, Any]:
        """Alias used by callers that want a JSON/YAML-safe mapping."""
        return to_serializable(self.to_dict())
