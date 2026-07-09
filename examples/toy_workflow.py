"""CPU-safe toy workflow for the refactored DFR public API.

Run from the repository root:

    python examples/toy_workflow.py

The script creates a tiny local dataset under ``outputs/examples/toy_workflow``,
loads it with ``dfr.load_dataset``, computes a mode-count curve on CPU, and
saves one figure explicitly. It does not require the large ignored datasets,
CUDA, or the custom rasterizer extensions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import dfr
from dfr.analysis import analyze_dataset_modes
from dfr.plotting import plot_mode_count_curve, save_figure


def main(output_root: str | Path = "outputs/examples/toy_workflow") -> Path:
    """Run the toy load/analyze/plot workflow and return the saved figure path."""
    root = Path(output_root)
    data_dir = root / "dataset"
    scenario_dir = root / "scenarios" / "toy"
    data_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    trajectories = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.2, 0.0], [2.1, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.save(data_dir / "toy.npy", trajectories)
    (scenario_dir / "config.yaml").write_text(
        "data_file: dataset/toy.npy\n",
        encoding="utf-8",
    )

    dataset = dfr.load_dataset("toy", project_root=root)
    curve = analyze_dataset_modes(
        dataset,
        frame=0,
        scales=(0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
        device="cpu",
    )

    figure, _ = plot_mode_count_curve(curve)
    target = save_figure(figure, root / "mode_curve.png")
    print(f"Saved toy mode-count figure to {target}")
    return target


if __name__ == "__main__":
    main()
