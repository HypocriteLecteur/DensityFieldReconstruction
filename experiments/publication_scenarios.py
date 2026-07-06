"""Shared managed runner for the camera/noise publication table studies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from dfr import (
    CameraConfig,
    EvaluationConfig,
    OutputConfig,
    ScenarioRunSpec,
    evaluate,
    run_scenario,
)
from dfr.config import ReconstructionParams, TrainingParams


@dataclass(frozen=True, slots=True)
class DatasetPreset:
    name: str
    start: int
    stop: int | None
    step: int


@dataclass(frozen=True, slots=True)
class PublicationProfile:
    table: int
    description: str
    iterations: int
    datasets: tuple[DatasetPreset, ...]
    camera_counts: tuple[int, ...] = (2, 3, 5)
    noise_levels: tuple[float, ...] = (0.0,)


BIOLOGICAL_DATASETS = (
    DatasetPreset("swift", 0, None, 200),
    DatasetPreset("starling", 0, None, 1),
    DatasetPreset("jackdaw", 350, 550, 10),
    DatasetPreset("jackdaw2", 2700, 3460, 20),
)

PROFILES = {
    2: PublicationProfile(
        2,
        "camera-count ablation with 100 optimization iterations",
        100,
        BIOLOGICAL_DATASETS,
    ),
    3: PublicationProfile(
        3,
        "camera-count ablation with 500 optimization iterations",
        500,
        BIOLOGICAL_DATASETS,
    ),
    4: PublicationProfile(
        4,
        "projection-noise study (historical active preset: starling)",
        100,
        (DatasetPreset("starling", 0, None, 1),),
    ),
}


def training_params(iterations: int) -> TrainingParams:
    """Return the hyperparameters used by the historical table scripts."""
    return TrainingParams(
        xyz_lr_c=0.013836480453275012,
        xyz_lr_final_c=0.9885055055101057,
        radius_lr_c=0.04323227755615107,
        radius_lr_final_c=0.9868476122181389,
        weights_lr_c=0.0810712748566998,
        weights_lr_final_c=0.7979132269720964,
        xyz_reg=0.21978381872642633,
        radius_reg=0.6083537781516261,
        radius_cutoff_inv=0.6013595613763145,
        lr_max_steps=iterations,
    )


def reconstruction_params() -> ReconstructionParams:
    return ReconstructionParams(10, 0.5, 0.3, 32, 20)


def build_specs(
    profile: PublicationProfile,
    *,
    project_root: Path,
    output_root: Path,
    datasets: tuple[str, ...] | None = None,
    camera_counts: tuple[int, ...] | None = None,
    noise_levels: tuple[float, ...] | None = None,
    run_id_prefix: str | None = None,
    seed: int = 123456789,
    resume: bool = False,
    overwrite: bool = False,
) -> tuple[ScenarioRunSpec, ...]:
    """Expand one publication profile into explicit managed run specs."""
    selected_names = set(datasets) if datasets else None
    selected_datasets = tuple(
        item for item in profile.datasets
        if selected_names is None or item.name in selected_names
    )
    if selected_names is not None:
        missing = selected_names - {item.name for item in selected_datasets}
        if missing:
            raise ValueError(
                f"Datasets are not part of Table {profile.table}: {sorted(missing)}"
            )
    cameras = camera_counts or profile.camera_counts
    noise = noise_levels or profile.noise_levels
    prefix = run_id_prefix or f"table-{profile.table}"
    specs = []
    for noise_std in noise:
        for camera_count in cameras:
            for dataset in selected_datasets:
                run_id = (
                    f"{prefix}-{dataset.name}-cam-{camera_count}"
                    f"-noise-{noise_std:g}-iter-{profile.iterations}"
                )
                specs.append(
                    ScenarioRunSpec(
                        dataset=dataset.name,
                        start=dataset.start,
                        stop=dataset.stop,
                        step=dataset.step,
                        cameras=CameraConfig.encircling(
                            count=camera_count, device="cuda"
                        ),
                        training=training_params(profile.iterations),
                        reconstruction=reconstruction_params(),
                        use_ground_truth_scales=True,
                        projection_noise_std=noise_std,
                        seed=seed,
                        output=OutputConfig(
                            workflow="reconstruction",
                            name=f"Table {profile.table}: {dataset.name}",
                            root=output_root,
                            run_id=run_id,
                            project_root=project_root,
                            resume=resume,
                            overwrite=overwrite,
                        ),
                    )
                )
    if not specs:
        raise ValueError("The selected publication profile contains no runs.")
    return tuple(specs)


def create_parser(table: int) -> argparse.ArgumentParser:
    profile = PROFILES[table]
    parser = argparse.ArgumentParser(
        description=f"Reproduce Table {table}: {profile.description}."
    )
    parser.add_argument("action", choices=("reconstruct", "run"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--camera-counts", nargs="+", type=int)
    parser.add_argument("--noise-levels", nargs="+", type=float)
    parser.add_argument("--run-id-prefix")
    parser.add_argument("--seed", type=int, default=123456789)
    collision = parser.add_mutually_exclusive_group()
    collision.add_argument("--resume", action="store_true")
    collision.add_argument("--overwrite-run", action="store_true")
    parser.add_argument(
        "--evaluation-device",
        default="cuda",
        help="Device used by the optional evaluation in the 'run' action.",
    )
    return parser


def main(table: int, argv: list[str] | None = None) -> int:
    args = create_parser(table).parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    specs = build_specs(
        PROFILES[table],
        project_root=project_root,
        output_root=args.output_root,
        datasets=tuple(args.datasets) if args.datasets else None,
        camera_counts=tuple(args.camera_counts) if args.camera_counts else None,
        noise_levels=tuple(args.noise_levels) if args.noise_levels else None,
        run_id_prefix=args.run_id_prefix,
        seed=args.seed,
        resume=args.resume,
        overwrite=args.overwrite_run,
    )
    for spec in specs:
        reconstruction = run_scenario(spec, project_root=project_root)
        print(f"reconstruction: {reconstruction.run_dir}")
        if args.action == "run":
            output = spec.output
            evaluation = evaluate(
                reconstruction,
                config=EvaluationConfig(device=args.evaluation_device),
                output=OutputConfig(
                    workflow="evaluation",
                    name=f"Table {table}: {spec.dataset}",
                    root=args.output_root,
                    run_id=output.run_id,
                    project_root=project_root,
                    resume=args.resume,
                    overwrite=args.overwrite_run,
                ),
            )
            summary = evaluation.summary
            print(
                f"evaluation: {evaluation.run_dir} "
                f"recall={summary.recall:.3f} "
                f"hallucination={summary.hallucination:.3f} "
                f"dMOTA={summary.dmota:.3f}"
            )
    return 0
