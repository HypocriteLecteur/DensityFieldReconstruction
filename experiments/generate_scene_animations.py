"""
Generate professional 3D animations of flock trajectories and GT density fields.

Two animation types are produced per dataset:

1. **Trajectory animation** (``<name>.mp4``) — raw 3D positions with trailing
   motion traces (ghost paths) for each agent.  Camera frustums for spatial context.

2. **Density-field animation** (``<name>_density.mp4``) — GT density field
   rendered as nested semi-transparent shells (viridis colormap with PowerNorm),
   overlaid with agent positions and camera frustums.  Same style (axes, view
   angle, figsize) as the trajectory animations.

Output:  experiments/animations/<dataset_name>.mp4
         experiments/animations/<dataset_name>_density.mp4

Usage:
    python experiments/generate_scene_animations.py           # all datasets
    python experiments/generate_scene_animations.py jackdaw   # single dataset
"""

import sys
import os

sys.path.append(os.getcwd())

from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter

from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory, DatasetInterface
from dfr.camera_system import MultiCameraSystem
from dfr.camera_state import CameraState
from dfr.utils import generate_encircling_cameras
from experiments.run_scenarios_angle_sweep import _build_grid, _precompute_gt_density

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

DATASET_RUNS = [
    # {"name": "starling",  "start_step": 0,    "end_step": None, "step_length": 1},
    {"name": "swift",     "start_step": 0,    "end_step": None, "step_length": 25},
    {"name": "jackdaw",   "start_step": 350,  "end_step": 550,  "step_length": 3},
    # {"name": "jackdaw2",  "start_step": 2700, "end_step": 3460, "step_length": 5},
]

# Density-field runs match the trajectory step parameters so duration and FPS
# are identical.  Scales are interpolated from the sparser reconstruction_scale.npz.
DENSITY_FIELD_RUNS = [
    {"name": "swift",     "start_step": 0,    "end_step": None, "step_length": 25},
    {"name": "jackdaw",   "start_step": 350,  "end_step": 550,  "step_length": 3},
    {"name": "jackdaw2",  "start_step": 2700, "end_step": 3460, "step_length": 5},
]

# Original step_length used when generating reconstruction_scale.npz
# (from experiments/run_scenarios.py DATASET_RUNS).  Needed for interpolation.
_ORIG_STEP_LENGTH = {"swift": 200, "jackdaw": 10, "jackdaw2": 20}

OUTPUT_DIR = os.path.join(os.getcwd(), "experiments", "animations")
FPS = 15                    # frames per second in the output video
DPI = 150                   # output resolution
MAX_POINTS = 5_000          # downsample scatter per frame when N exceeds this
MAX_TRAIL_AGENTS = 400      # cap on number of agents that get trail traces
TRAIL_WINDOW = 30            # how many *animation* frames each trail spans
CAM_NUM = 2                 # number of camera frustums to draw
ELEV = 28                   # initial 3D view elevation (degrees)
AZIM = -60                  # initial 3D view azimuth (degrees)
MIN_ANIMATION_SECS = 2.5    # minimum animation duration — hold frames if needed

# Density-field rendering parameters (mirror dfr_plot.plot_jackdaw2_density_field)
DENSITY_VOXEL_RES_FACTOR = 2.5e-2    # voxel size relative to spatial extent
DENSITY_BATCH_SIZE = 50_000          # voxels per GPU batch during density eval
DENSITY_LAYER_FRACTIONS = [0.10, 0.02, 0.002]   # thresholds as fraction of max density
DENSITY_LAYER_SIZES = [8, 6, 4]                 # scatter marker sizes per layer
DENSITY_LAYER_ALPHAS = [(0.45, 0.95), (0.25, 0.80), (0.08, 0.50)]  # (min, max) alpha
DENSITY_GAMMA = 0.35                 # PowerNorm gamma for colormap

# ──────────────────────────────────────────────────────────────────────
#  Style setup
# ──────────────────────────────────────────────────────────────────────


def _setup_mpl_style() -> None:
    """Globally configure matplotlib for a clean professional look."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            # NOTE: do NOT set savefig.bbox='tight' — it varies frame sizes
            # and breaks FFMpegWriter.grab_frame().
        }
    )


# ──────────────────────────────────────────────────────────────────────
#  Data helpers
# ──────────────────────────────────────────────────────────────────────


def _step_range(dataset: DatasetInterface, run_params: dict) -> range:
    """Return the sequence of valid time-step indices for *run_params*."""
    n_frames = dataset.trajectories.shape[0]
    start = run_params["start_step"]
    end = run_params["end_step"] if run_params["end_step"] is not None else n_frames
    end = min(end, n_frames)
    return range(start, end, run_params["step_length"])


def _global_bounds(dataset: DatasetInterface, step_indices: range) -> np.ndarray:
    """Return (3, 2) bounds across all requested time steps."""
    all_mins = []
    all_maxs = []
    for t in tqdm(step_indices, desc="  Computing global bounds", leave=False):
        pos = dataset.positions_at_time_step(t)
        if pos.shape[0] > 0:
            all_mins.append(pos.min(axis=0))
            all_maxs.append(pos.max(axis=0))
    if not all_mins:
        return np.array([[-1, 1], [-1, 1], [-1, 1]])
    gmin = np.min(np.stack(all_mins), axis=0)
    gmax = np.max(np.stack(all_maxs), axis=0)
    pad = np.max(gmax - gmin) * 0.10
    return np.stack([gmin - pad, gmax + pad], axis=1)  # (3, 2)


# ──────────────────────────────────────────────────────────────────────
#  Trail precomputation
# ──────────────────────────────────────────────────────────────────────

def _precompute_trails(
    dataset: DatasetInterface,
    step_list: List[int],
    trail_window: int,
    max_agents: int,
) -> List[np.ndarray]:
    """Precompute a NaN-separated vertex array of agent trails for every frame.

    Each trail follows an agent backwards for up to *trail_window* animation
    frames (using the same ``step_length`` stride).  Individual trails are
    separated by a ``[NaN, NaN, NaN]`` sentinel row so they render as one
    efficient ``Line3D`` artist per frame.

    Returns
    -------
    list of ndarray
        One ``(V, 3)`` array per animation frame.  *V* is the total number of
        trail vertices (including NaN sentinels) for that frame.
    """
    trajectories = dataset.trajectories  # (n_raw_frames, n_agents, 3)
    n_raw = trajectories.shape[0]
    step_len = step_list[1] - step_list[0] if len(step_list) > 1 else 1

    all_trails: List[np.ndarray] = []

    for anim_idx, time_step in enumerate(
        tqdm(step_list, desc="  Precomputing trails", leave=False)
    ):
        # Valid agents at the *current* animation frame
        _, mask = dataset.positions_at_time_step_mask(time_step)
        valid_ids = np.where(mask)[0]  # original agent column indices

        # Cap trail agents for performance
        if len(valid_ids) > max_agents:
            rng = np.random.RandomState(time_step)  # deterministic per frame
            valid_ids = rng.choice(valid_ids, max_agents, replace=False)

        segments: List[np.ndarray] = []

        for agent_id in valid_ids:
            trail_pts: List[np.ndarray] = []

            # Walk backwards *trail_window* animation steps
            for w in range(trail_window, -1, -1):
                t = time_step - w * step_len
                if 0 <= t < n_raw:
                    pt = trajectories[t, agent_id]  # (3,)
                    if not np.isnan(pt[0]):
                        trail_pts.append(pt)

            if len(trail_pts) >= 2:
                segments.append(np.array(trail_pts, dtype=np.float32))
                # NaN sentinel to break the line between agents
                segments.append(
                    np.array([[np.nan, np.nan, np.nan]], dtype=np.float32)
                )

        if segments:
            all_trails.append(np.concatenate(segments, axis=0))
        else:
            all_trails.append(np.zeros((0, 3), dtype=np.float32))

    return all_trails


# ──────────────────────────────────────────────────────────────────────
#  Camera system
# ──────────────────────────────────────────────────────────────────────


def _build_camera_system(
    config: SimulationConfig, dataset, step_indices
) -> MultiCameraSystem:
    """Create a ``MultiCameraSystem`` with encircling cameras, mirroring the
    logic in ``run_scenarios.py``."""
    cam_positions, _ = generate_encircling_cameras(
        dataset,
        step_indices,
        config.intrinsics_params,
        config.H,
        config.W,
        cam_num=max(4, CAM_NUM),
        padding=1,
    )
    cam_poses = np.hstack(
        (
            cam_positions[:CAM_NUM],
            np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (CAM_NUM, 1)),
        )
    ).astype(np.float32)

    return MultiCameraSystem.create_homogeneous_system(
        state_class=CameraState,
        intrinsics=config.intrinsics_params,
        H=config.H,
        W=config.W,
        poses_or_RTs=cam_poses,
        near_clip=config.near_clip,
        far_clip=config.far_clip,
        size=config.size,
        device="cpu",
    )


def _draw_camera_frustum(
    ax, camera, color: str = "#4fc3f7", alpha: float = 0.45, linewidth: float = 1.0
):
    """Draw the frustum of a single camera as wireframe edges + position marker."""
    try:
        vertices = camera.state.get_world_frustum()
    except Exception:
        return []

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # near plane
        (4, 5), (5, 6), (6, 7), (7, 4),  # far plane
        (0, 4), (1, 5), (2, 6), (3, 7),  # side struts
    ]
    objs = []
    for i0, i1 in edges:
        (line,) = ax.plot(
            vertices[[i0, i1], 0],
            vertices[[i0, i1], 1],
            vertices[[i0, i1], 2],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )
        objs.append(line)

    center = camera.state.camera_center
    (pt,) = ax.plot(
        [center[0]], [center[1]], [center[2]],
        marker="s",
        color=color,
        markersize=6,
        alpha=0.9,
    )
    objs.append(pt)
    return objs


# ──────────────────────────────────────────────────────────────────────
#  Density-field helpers
# ──────────────────────────────────────────────────────────────────────


def _load_gt_scale_for_step(
    step_list: List[int], scales_gt: np.ndarray, name: str,
) -> List[float]:
    """Return a list of scales, one per entry in *step_list*.

    ``scales_gt`` was precomputed on a sparser original step grid (e.g. every
    200ᵗʰ frame for swift).  This function linearly interpolates onto the
    finer animation *step_list* so the density-field animation has the same
    duration and FPS as the trajectory animation.
    """
    orig_step = _ORIG_STEP_LENGTH.get(name)
    if orig_step is None or len(scales_gt) <= 1:
        return [float(scales_gt[-1]) for _ in step_list]

    # Original steps that have known scales
    start_step = step_list[0]
    orig_steps = np.arange(
        start_step,
        start_step + len(scales_gt) * orig_step,
        orig_step,
        dtype=np.float64,
    )
    # Truncate to actual scales_gt length (in case of off-by-one)
    if len(orig_steps) > len(scales_gt):
        orig_steps = orig_steps[:len(scales_gt)]

    # Linear interpolation onto the animation step grid
    interp = np.interp(
        np.array(step_list, dtype=np.float64),
        orig_steps,
        scales_gt.astype(np.float64),
    )
    return [float(s) for s in interp]


def _density_global_bounds(density_frames: List[dict]) -> np.ndarray:
    """Return (3, 2) bounds covering the voxel grids of all precomputed frames."""
    all_mins = []
    all_maxs = []
    for fd in density_frames:
        grid = fd["grid"]
        x = grid["x_ticks"].cpu().numpy()
        y = grid["y_ticks"].cpu().numpy()
        z = grid["z_ticks"].cpu().numpy()
        all_mins.append(np.array([x[0], y[0], z[0]]))
        all_maxs.append(np.array([x[-1], y[-1], z[-1]]))
    if not all_mins:
        return np.array([[-1, 1], [-1, 1], [-1, 1]])
    gmin = np.min(np.stack(all_mins), axis=0)
    gmax = np.max(np.stack(all_maxs), axis=0)
    return np.stack([gmin, gmax], axis=1)  # (3, 2)


def _precompute_density_frames(
    dataset: DatasetInterface,
    step_list: List[int],
    scales: List[float],
    voxel_res_factor: float = DENSITY_VOXEL_RES_FACTOR,
    batch_size: int = DENSITY_BATCH_SIZE,
    device: str = "cuda",
) -> List[dict]:
    """Precompute GT density 3D grids for every animation frame.

    Returns
    -------
    list of dict
        Each dict has keys ``density_3d`` (np, float32), ``grid``,
        ``positions`` (np, float32), and ``max_density`` (float).
    """
    frames: List[dict] = []

    for time_step, scale in zip(
        tqdm(step_list, desc="  Precomputing density", leave=False), scales
    ):
        positions = dataset.positions_at_time_step(time_step)

        # Skip frames with no agents
        if positions.shape[0] < 1:
            frames.append(
                {
                    "density_3d": np.zeros((1, 1, 1), dtype=np.float32),
                    "grid": {
                        "x_ticks": torch.tensor([-1.0, 1.0]),
                        "y_ticks": torch.tensor([-1.0, 1.0]),
                        "z_ticks": torch.tensor([-1.0, 1.0]),
                        "nx": 2,
                        "ny": 2,
                        "nz": 2,
                    },
                    "positions": positions,
                    "max_density": 0.0,
                }
            )
            continue

        grid = _build_grid(positions, scale, voxel_res_factor=voxel_res_factor, device=device)
        density_flat = _precompute_gt_density(positions, scale, grid, batch_size=batch_size, device=device)
        density_3d = density_flat.numpy().reshape(grid["nx"], grid["ny"], grid["nz"])

        frames.append(
            {
                "density_3d": density_3d,
                "grid": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in grid.items()},
                "positions": positions,
                "max_density": float(density_3d.max()),
            }
        )

    return frames


# ──────────────────────────────────────────────────────────────────────
#  Density-field animation generator
# ──────────────────────────────────────────────────────────────────────


def generate_density_field_animation(run_params: dict) -> Optional[str]:
    """Generate and save an MP4 of the GT density field for one dataset.

    The density field is rendered as nested semi-transparent shells (viridis
    colormap with PowerNorm) overlaid with agent positions and camera frustums.
    Style (axes, view angle, figsize, grids, frustums) matches the trajectory
    animations from ``generate_animation()``.

    Returns the output path on success, or ``None`` if skipped.
    """
    name = run_params["name"]
    print(f"\n{'=' * 60}\n  {name} (density field)\n{'=' * 60}")

    # ── 1. Load config & dataset ──────────────────────────────────────
    scenario_path = os.path.join(os.getcwd(), "scenarios", name)
    config_path = os.path.join(scenario_path, "config.yaml")
    if not os.path.exists(config_path):
        print(f"  [SKIP] config not found: {config_path}")
        return None

    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)
    steps = _step_range(dataset, run_params)
    step_list = list(steps)
    n_frames = len(step_list)
    if n_frames < 1:
        print("  [SKIP] empty step range")
        return None

    # Per-frame hold multiplier for very short sequences
    hold_mult = max(1, int(np.ceil(MIN_ANIMATION_SECS * FPS / n_frames)))

    print(
        f"  frames: {n_frames}"
        + (f"  |  hold x{hold_mult}" if hold_mult > 1 else "")
        + f"  |  step: {step_list[0]} -> {step_list[-1]}"
    )

    # ── 2. Load GT scales ─────────────────────────────────────────────
    scale_path = os.path.join(scenario_path, "reconstruction_scale.npz")
    if not os.path.exists(scale_path):
        print(f"  [SKIP] reconstruction_scale.npz not found: {scale_path}")
        return None
    gt_data = np.load(scale_path)
    scales_gt = gt_data["scales_gt"]
    scales = _load_gt_scale_for_step(step_list, scales_gt, name)
    print(f"  scales loaded: {len(scales)} values, range [{min(scales):.3f}, {max(scales):.3f}]")

    # ── 3. Cameras (static frustums) ──────────────────────────────────
    cam_system = _build_camera_system(config, dataset, steps)

    # ── 4. Precompute density grids for every frame ───────────────────
    density_frames = _precompute_density_frames(dataset, step_list, scales)
    bounds_3d = _density_global_bounds(density_frames)

    # ── 5. Set up figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Transparent axis panes
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Subtle grid
    ax.grid(
        True, which="major", color="#cccccc", linestyle="-",
        alpha=0.35, linewidth=0.5,
    )

    # Axis limits & equal aspect
    ax.set_xlim(bounds_3d[0])
    ax.set_ylim(bounds_3d[1])
    ax.set_zlim(bounds_3d[2])
    ax.set_box_aspect([
        float(np.ptp(bounds_3d[0])),
        float(np.ptp(bounds_3d[1])),
        float(np.ptp(bounds_3d[2])),
    ])

    ax.set_xlabel("X", color="#444444", fontsize=10)
    ax.set_ylabel("Y", color="#444444", fontsize=10)
    ax.set_zlabel("Z", color="#444444", fontsize=10)
    ax.tick_params(colors="#666666", labelsize=8)

    # Initial view angle (same as trajectory animation)
    ax.view_init(elev=ELEV, azim=AZIM)

    # Minimise white-space padding
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    # Camera frustums (static)
    cam_colors = ["#4fc3f7", "#81c784"]
    for idx, cam in enumerate(cam_system.cameras):
        _draw_camera_frustum(ax, cam, color=cam_colors[idx % len(cam_colors)])

    # ── 6. Animation loop ─────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_density.mp4")
    writer = FFMpegWriter(fps=FPS, bitrate=4000)
    writer.setup(fig, out_path, dpi=DPI)

    # Per-dataset colours for agent markers (same palette as trajectory anims)
    PALETTE = {
        "starling": "#e67e22",
        "swift":    "#2e86c1",
        "jackdaw":  "#27ae60",
        "jackdaw2": "#8e44ad",
    }
    AGENT_COLOR = PALETTE.get(name, "#1a5276")
    AGENT_SIZE = 25

    # Reusable artist lists — cleared and redrawn per frame
    layer_artists: List = []
    agent_artist: List = []

    # Global max density for consistent PowerNorm across frames
    global_dm = max(fd["max_density"] for fd in density_frames)
    if global_dm <= 0:
        global_dm = 1.0

    pbar = tqdm(total=n_frames * hold_mult, desc=f"  Rendering {name} density", unit="frame")
    for frame_idx, (time_step, fd) in enumerate(zip(step_list, density_frames)):
        density_3d = fd["density_3d"]
        grid = fd["grid"]
        positions = fd["positions"]

        x_np = grid["x_ticks"].numpy()
        y_np = grid["y_ticks"].numpy()
        z_np = grid["z_ticks"].numpy()

        # ── Remove previous frame's density & agent artists ─────────────
        for art in layer_artists:
            art.remove()
        layer_artists.clear()
        for art in agent_artist:
            art.remove()
        agent_artist.clear()

        # ── Density shells ──────────────────────────────────────────────
        norm = mcolors.PowerNorm(gamma=DENSITY_GAMMA, vmin=0, vmax=global_dm)

        for layer_idx, frac in enumerate(DENSITY_LAYER_FRACTIONS):
            thresh = global_dm * frac
            mask = density_3d >= thresh
            if not mask.any():
                continue
            ix, iy, iz = np.where(mask)
            pts = np.stack([x_np[ix], y_np[iy], z_np[iz]], axis=-1)
            vals = density_3d[mask]

            colors = plt.cm.viridis(norm(vals))
            alpha_min, alpha_max = DENSITY_LAYER_ALPHAS[layer_idx]
            alphas = norm(vals) * (alpha_max - alpha_min) + alpha_min
            colors[:, 3] = np.clip(alphas, alpha_min, alpha_max)

            sct = ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=colors,
                s=DENSITY_LAYER_SIZES[layer_idx],
                edgecolors="none",
                depthshade=False,
                rasterized=True,
            )
            layer_artists.append(sct)

        # ── Agent positions (forced on top of density shells) ───────────
        if positions.shape[0] > 0:
            agent_sct = ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=AGENT_COLOR, s=AGENT_SIZE, alpha=1.0,
                edgecolors="none", depthshade=True,
            )
            # Monkey-patch z-sort to force agents in front of density shells
            _orig_agent = agent_sct.do_3d_projection
            def _force_agent_front(orig=_orig_agent, obj=agent_sct):
                orig()
                obj._sort_zpos = -1e9
                return obj._sort_zpos
            agent_sct.do_3d_projection = _force_agent_front
            agent_artist.append(agent_sct)

        for _ in range(hold_mult):
            writer.grab_frame()

        pbar.update(hold_mult)
        pbar.set_postfix({"N": positions.shape[0]})

    # Cleanup
    writer.finish()
    plt.close(fig)
    pbar.close()
    print(f"  [OK] saved -> {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────
#  Core: single-animation generator (trajectory)
# ──────────────────────────────────────────────────────────────────────


def generate_animation(run_params: dict) -> Optional[str]:
    """Generate and save an MP4 for one dataset entry.

    Returns the output path on success, or ``None`` if skipped.
    """
    name = run_params["name"]
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")

    # ── 1. Load config & dataset ──────────────────────────────────────
    scenario_path = os.path.join(os.getcwd(), "scenarios", name)
    config_path = os.path.join(scenario_path, "config.yaml")
    if not os.path.exists(config_path):
        print(f"  [SKIP] config not found: {config_path}")
        return None

    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)
    steps = _step_range(dataset, run_params)
    step_list = list(steps)
    n_frames = len(step_list)
    if n_frames < 1:
        print("  [SKIP] empty step range")
        return None

    # Per-frame hold multiplier for very short sequences
    hold_mult = max(1, int(np.ceil(MIN_ANIMATION_SECS * FPS / n_frames)))

    print(
        f"  frames: {n_frames}"
        + (f"  |  hold x{hold_mult}" if hold_mult > 1 else "")
        + f"  |  step: {step_list[0]} -> {step_list[-1]}"
    )

    # ── 2. Cameras & global bounds ────────────────────────────────────
    cam_system = _build_camera_system(config, dataset, steps)
    bounds_3d = _global_bounds(dataset, steps)  # (3, 2)

    # ── 3. Precompute trails ──────────────────────────────────────────
    trail_vertices = _precompute_trails(dataset, step_list, TRAIL_WINDOW, MAX_TRAIL_AGENTS)

    # ── 4. Set up figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Transparent axis panes
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Subtle grid
    ax.grid(
        True, which="major", color="#cccccc", linestyle="-",
        alpha=0.35, linewidth=0.5,
    )

    # Axis limits & equal aspect
    ax.set_xlim(bounds_3d[0])
    ax.set_ylim(bounds_3d[1])
    ax.set_zlim(bounds_3d[2])
    ax.set_box_aspect([
        float(np.ptp(bounds_3d[0])),
        float(np.ptp(bounds_3d[1])),
        float(np.ptp(bounds_3d[2])),
    ])

    ax.set_xlabel("X", color="#444444", fontsize=10)
    ax.set_ylabel("Y", color="#444444", fontsize=10)
    ax.set_zlabel("Z", color="#444444", fontsize=10)
    ax.tick_params(colors="#666666", labelsize=8)

    # Initial view angle
    ax.view_init(elev=ELEV, azim=AZIM)

    # Minimise white-space padding around the 3D axes
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    # ── Title (suppressed — no title bar) ─────────────────────────────

    # Camera frustums (static — drawn once)
    cam_colors = ["#4fc3f7", "#81c784"]
    for idx, cam in enumerate(cam_system.cameras):
        _draw_camera_frustum(ax, cam, color=cam_colors[idx % len(cam_colors)])

    # ── 5. Animation loop ─────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{name}.mp4")
    writer = FFMpegWriter(fps=FPS, bitrate=4000)
    writer.setup(fig, out_path, dpi=DPI)

    scatter_handle: Optional[object] = None
    trail_handle: Optional[object] = None

    # Per-dataset colours (point, trail)
    PALETTE = {
        "starling": ("#e67e22", "#fdebd0"),  # orange
        "swift":    ("#2e86c1", "#d6eaf8"),  # blue
        "jackdaw":  ("#27ae60", "#d5f5e3"),  # green
        "jackdaw2": ("#8e44ad", "#e8daef"),  # purple
    }
    POINT_COLOR, TRAIL_COLOR = PALETTE.get(name, ("#1a5276", "#b0bec5"))
    MARKER_SIZE = 8.0

    pbar = tqdm(total=n_frames * hold_mult, desc=f"  Rendering {name}", unit="frame")
    for frame_idx, time_step in enumerate(step_list):
        positions = dataset.positions_at_time_step(time_step)

        # Downsample scatter if needed
        if positions.shape[0] > MAX_POINTS:
            idx = np.linspace(0, positions.shape[0] - 1, MAX_POINTS, dtype=int)
            positions = positions[idx]

        # ── Trails (a single Line3D per frame; NaN-separated segments) ──
        tv = trail_vertices[frame_idx]

        if trail_handle is None:
            (trail_handle,) = ax.plot(
                tv[:, 0], tv[:, 1], tv[:, 2],
                color=TRAIL_COLOR, alpha=0.35, linewidth=0.5,
                rasterized=True,
            )
        else:
            trail_handle.set_data_3d(tv[:, 0], tv[:, 1], tv[:, 2])

        # ── Scatter (current positions) ───────────────────────────────
        if scatter_handle is None:
            scatter_handle = ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=POINT_COLOR,
                s=MARKER_SIZE,
                alpha=0.85,
                edgecolors="none",
                depthshade=True,
                rasterized=True,
            )
        else:
            scatter_handle._offsets3d = (
                positions[:, 0], positions[:, 1], positions[:, 2],
            )

        for _ in range(hold_mult):
            writer.grab_frame()

        pbar.update(hold_mult)
        pbar.set_postfix({"N": positions.shape[0]})

    # Cleanup
    writer.finish()
    plt.close(fig)
    pbar.close()
    print(f"  [OK] saved -> {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    _setup_mpl_style()

    target = sys.argv[1] if len(sys.argv) > 1 else None

    # ── Trajectory animations ──────────────────────────────────────────
    traj_runs = (
        [r for r in DATASET_RUNS if r["name"] == target]
        if target
        else DATASET_RUNS
    )
    if target and not traj_runs:
        print(
            f"Unknown dataset: '{target}'."
            f"  Choices: {[r['name'] for r in DATASET_RUNS]}"
        )
        sys.exit(1)

    for run in traj_runs:
        try:
            generate_animation(run)
        except Exception as exc:
            print(f"  [FAIL] {run['name']}: {exc}")
            import traceback
            traceback.print_exc()

    # ── Density-field animations ───────────────────────────────────────
    dens_runs = (
        [r for r in DENSITY_FIELD_RUNS if r["name"] == target]
        if target
        else DENSITY_FIELD_RUNS
    )
    if target and not dens_runs:
        print(
            f"  [WARN] no density-field run for '{target}'."
            f"  Choices: {[r['name'] for r in DENSITY_FIELD_RUNS]}"
        )
    else:
        for run in dens_runs:
            try:
                generate_density_field_animation(run)
            except Exception as exc:
                print(f"  [FAIL] {run['name']} density: {exc}")
                import traceback
                traceback.print_exc()

    print(f"\nDone.  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
