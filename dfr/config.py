"""Typed configuration objects for reconstruction and training parameters."""

from dataclasses import dataclass


@dataclass
class TrainingParams:
    """Hyperparameters for GMM optimization."""
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
    def from_dict(cls, d: dict) -> "TrainingParams":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class ReconstructionParams:
    """Parameters for scale selection and visual hull reconstruction."""
    targetd_num_mode: int
    voxel_scale: float
    voxel_peak_threshold: float
    voxel_grid_max_size: int
    voxel_peaks_number: int

    @classmethod
    def from_dict(cls, d: dict) -> "ReconstructionParams":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
