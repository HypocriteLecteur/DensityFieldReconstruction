"""
Compute metrics from pre-trained models only — no training.

Loads MV-DFR checkpoint history and compares the initial model (iter 0, before
training) against the final model (iter 99, after training) on TP/FP/FN/dMOTA
metrics computed against the ground-truth point cloud.

Usage:
    python experiments/compute_metrics_from_pretrained.py
"""

import logging
import sys
import os
import numpy as np
from tqdm import tqdm


from dfr.simulation_config import SimulationConfig
from dfr.dataset_io import DatasetFactory
from dfr.density_field_model import GaussianModel
from dfr.evaluation import EvaluationSummary, compute_density_overlap_masses

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# Dataset definitions  (mirrors run_scenarios_table_2.py)
# ---------------------------------------------------------------------------
DATASET_RUNS = [
    {
        'name': 'swift',
        'log_name': None,           # filled per camera sweep below
        'start_step': 0,
        'end_step': None,
        'step_length': 200,
    },
    {
        'name': 'starling',
        'log_name': None,
        'start_step': 0,
        'end_step': None,
        'step_length': 1,
    },
    {
        'name': 'jackdaw',
        'log_name': None,
        'start_step': 350,
        'end_step': 550,
        'step_length': 10,
    },
    {
        'name': 'jackdaw2',
        'log_name': None,
        'start_step': 2700,
        'end_step': 3460,
        'step_length': 20,
    },
]

# Which iterations to compare
ITER_INIT = 0     # before training
ITER_FINAL = 99   # after training


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def aggregate_metrics(metric_data: dict) -> dict:
    """Aggregate per-frame counts into global recall/hallucination/dMOTA."""
    sum_tp = np.sum(metric_data['tp'])
    sum_fp = np.sum(metric_data['fp'])
    sum_fn = np.sum(metric_data['fn'])
    total_N = np.sum(metric_data['N'])
    total_weights = np.sum(metric_data['w'])
    if total_N <= 0:
        return {
            'recall': 0.0,
            'hallucination': 0.0,
            'dmota': 0.0,
            'total_N': total_N,
            'sum_tp': sum_tp,
            'sum_fp': sum_fp,
            'sum_fn': sum_fn,
        }

    summary = EvaluationSummary(
        true_positive_mass=float(sum_tp),
        false_positive_mass=float(sum_fp),
        false_negative_mass=float(sum_fn),
        ground_truth_mass=float(total_N),
        predicted_mass=float(total_weights),
    )
    return {
        'recall':        summary.recall,
        'hallucination': summary.hallucination,
        'dmota':         summary.dmota,
        'total_N':       total_N,
        'sum_tp':        sum_tp,
        'sum_fp':        sum_fp,
        'sum_fn':        sum_fn,
    }


def empty_metric_buffers():
    return {
        'tp': [], 'fp': [], 'fn': [], 'N': [], 'w': [],
        'coverage_recall': [], 'miss': [], 'hallucination': [], 'dMOTA': [],
    }


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def compute_metrics_single_scenario(run_params: dict):
    """For one (dataset, cam_count) combination:
       - load iter 0 and iter 99 from each frame's checkpoint,
       - compute TP/FP/FN/dMOTA for both,
       - cache results,
       - also pull training-time statistics from statistics.npz.

    Returns a dict with keys:
        'label'            — human-readable scenario label
        'init'             — aggregated post-hoc metrics at iter 0
        'final'            — aggregated post-hoc metrics at iter 99
        'gmm_init_mean'    — mean #components at iter 0
        'gmm_final_mean'   — mean #components at iter 99
        'train_loss_mean'  — mean training rendering loss (from statistics.npz)
        'train_ise_mean'   — mean ISE dissimilarity (from statistics.npz)
        'train_time_mean'  — mean training time per frame (ms)
        'scale_mean'       — mean reconstruction scale
    """
    force_update = False

    name = run_params['name']
    log_name = run_params['log_name']
    start_step = run_params['start_step']
    end_step = run_params['end_step']
    step_length = run_params['step_length']

    scenario_path = os.path.join(os.getcwd(), "scenarios", name)
    log_file_path = os.path.join(scenario_path, "logs", log_name)
    config_path = os.path.join(scenario_path, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = SimulationConfig(config_path)
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    max_steps = dataset.trajectories.shape[0]
    effective_end_step = end_step if end_step is not None and end_step <= max_steps else max_steps
    step_range = range(start_step, effective_end_step, step_length)

    if len(step_range) == 0:
        logger.warning(f"Empty step range for {name} ({log_name}). Skipping.")
        return None

    # Ground-truth scale
    gt_data_path = os.path.join(scenario_path, 'reconstruction_scale.npz')
    if not os.path.exists(gt_data_path):
        raise FileNotFoundError(f"GT scale file not found: {gt_data_path}")
    gt_data = np.load(gt_data_path)
    scale_history = gt_data['scales_gt']

    # ---- Training-time statistics ----
    train_loss_mean = None
    train_ise_mean = None
    train_time_mean = None
    scale_mean = None

    stats_path = os.path.join(log_file_path, "statistics.npz")
    if os.path.exists(stats_path):
        try:
            stats = np.load(stats_path)
            if 'final_training_loss' in stats and len(stats['final_training_loss']) > 0:
                train_loss_mean = np.mean(stats['final_training_loss']).item()
            if 'final_density_field_loss' in stats and len(stats['final_density_field_loss']) > 0:
                train_ise_mean = np.mean(stats['final_density_field_loss']).item()
            if 'train_gaussian_scale_space' in stats and len(stats['train_gaussian_scale_space']) > 0:
                train_time_mean = np.mean(stats['train_gaussian_scale_space']).item()
            if 'scale' in stats and len(stats['scale']) > 0:
                scale_mean = np.mean(stats['scale']).item()
        except Exception as e:
            logger.warning(f"Could not load statistics.npz for {name}/{log_name}: {e}")

    # ---- Metric buffers for init (iter 0) and final (iter 99) ----
    buf_init = empty_metric_buffers()
    buf_final = empty_metric_buffers()

    gmm_counts_init = []
    gmm_counts_final = []

    # Cache paths
    cache_init = os.path.join(scenario_path, f"metrics_init_{log_name}.npz")
    cache_final = os.path.join(scenario_path, f"metrics_final_{log_name}.npz")

    if force_update or not os.path.exists(cache_final):
        for idx, time_step in enumerate(tqdm(step_range, desc=f"{name}/{log_name}")):
            step_dir = os.path.join(log_file_path, f"t_{time_step:03d}")
            checkpoint_path = os.path.join(step_dir, "checkpoint_level_0.pth")

            # Load checkpoint history once, extract both iters
            try:
                history = GaussianModel.load_training_history(checkpoint_path)
                model_init = GaussianModel.load_iter(history, iter=ITER_INIT)
                model_final = GaussianModel.load_iter(history, iter=ITER_FINAL)
            except Exception as e:
                logger.error(f"Failed to load models for {name} t={time_step}: {e}")
                continue

            positions = dataset.positions_at_time_step(time_step)
            N = positions.shape[0]
            if N == 0:
                continue

            scale = scale_history[idx]

            min_coords = np.min(positions, axis=0)
            max_coords = np.max(positions, axis=0)
            bounds = np.vstack((
                min_coords - 3 * scale,
                max_coords + 3 * scale,
            )).T
            voxel_res = np.max(max_coords - min_coords) * 5e-3

            # --- iter 0 metrics ---
            tp_i, fp_i, fn_i = compute_density_overlap_masses(
                ground_truth_means=positions, ground_truth_sigma=scale,
                predicted_means=model_init._xyz,
                predicted_weights=model_init._weights,
                predicted_sigmas=model_init._radius,
                bounds=bounds, voxel_resolution=voxel_res,
                batch_size=50000, device='cuda',
            )
            w_i = model_init._weights.sum().item()
            buf_init['tp'].append(tp_i)
            buf_init['fp'].append(fp_i)
            buf_init['fn'].append(fn_i)
            buf_init['N'].append(N)
            buf_init['w'].append(w_i)
            buf_init['coverage_recall'].append(tp_i / N)
            buf_init['miss'].append(fn_i / N)
            buf_init['hallucination'].append(fp_i / w_i if w_i > 0 else 0.0)
            buf_init['dMOTA'].append(1.0 - (fn_i + fp_i) / N)
            gmm_counts_init.append(model_init._xyz.shape[0])

            # --- iter 99 metrics ---
            tp_f, fp_f, fn_f = compute_density_overlap_masses(
                ground_truth_means=positions, ground_truth_sigma=scale,
                predicted_means=model_final._xyz,
                predicted_weights=model_final._weights,
                predicted_sigmas=model_final._radius,
                bounds=bounds, voxel_resolution=voxel_res,
                batch_size=50000, device='cuda',
            )
            w_f = model_final._weights.sum().item()
            buf_final['tp'].append(tp_f)
            buf_final['fp'].append(fp_f)
            buf_final['fn'].append(fn_f)
            buf_final['N'].append(N)
            buf_final['w'].append(w_f)
            buf_final['coverage_recall'].append(tp_f / N)
            buf_final['miss'].append(fn_f / N)
            buf_final['hallucination'].append(fp_f / w_f if w_f > 0 else 0.0)
            buf_final['dMOTA'].append(1.0 - (fn_f + fp_f) / N)
            gmm_counts_final.append(model_final._xyz.shape[0])

        # Save caches
        if len(buf_final['N']) == 0:
            raise ValueError(f"No valid metric data generated for {name}/{log_name}.")

        buf_init_arrays = {k: np.array(v) for k, v in buf_init.items()}
        buf_final_arrays = {k: np.array(v) for k, v in buf_final.items()}
        np.savez(cache_init, **buf_init_arrays)
        np.savez(cache_final, **buf_final_arrays)
        buf_init = buf_init_arrays
        buf_final = buf_final_arrays
    else:
        buf_init_loaded = np.load(cache_init)
        buf_init = {k: np.array(v) for k, v in buf_init_loaded.items()}
        buf_final_loaded = np.load(cache_final)
        buf_final = {k: np.array(v) for k, v in buf_final_loaded.items()}

    init_agg = aggregate_metrics(buf_init)
    final_agg = aggregate_metrics(buf_final)

    label = f"{name}/{log_name}"
    return {
        'label': label,
        'init': init_agg,
        'final': final_agg,
        'gmm_init_mean': np.mean(gmm_counts_init).item() if gmm_counts_init else None,
        'gmm_final_mean': np.mean(gmm_counts_final).item() if gmm_counts_final else None,
        'train_loss_mean': train_loss_mean,
        'train_ise_mean': train_ise_mean,
        'train_time_mean': train_time_mean,
        'scale_mean': scale_mean,
    }


def compute_metrics_multi_scenarios():
    """Run over all (dataset, cam_count) combinations and print comparisons."""
    all_results = []

    for cam_num in [2, 3, 5]:
        log_name = f'base_reg_cam_{cam_num}'

        for run in DATASET_RUNS:
            run['log_name'] = log_name

        logger.info(f"{'='*60}")
        logger.info(f"Camera count: {cam_num}  |  log: {log_name}")
        logger.info(f"{'='*60}")

        for run_params in DATASET_RUNS:
            try:
                result = compute_metrics_single_scenario(run_params)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed on {run_params['name']}/{run_params['log_name']}: {e}"
                )

    # ===================================================================
    # Table 1 — iter 0 vs iter 99: post-hoc density metrics
    # ===================================================================
    header = (f"{'Scenario':<35s} "
              f"{'Recall(0)':>9s} {'Recall(99)':>10s} "
              f"{'Hall(0)':>8s} {'Hall(99)':>9s} "
              f"{'dMOTA(0)':>9s} {'dMOTA(99)':>10s} "
              f"{'#GMM(0)':>8s} {'#GMM(99)':>9s}")

    print("\n" + "=" * len(header.expandtabs()))
    print(f"ITER {ITER_INIT} vs ITER {ITER_FINAL}  —  post-hoc density metrics")
    print("=" * len(header.expandtabs()))
    # shorter column labels for alignment
    print(f"{'Scenario':<35s} "
          f"{'Recall0':>9s} {'Recall99':>10s} "
          f"{'Hall0':>8s} {'Hall99':>9s} "
          f"{'dMOTA0':>9s} {'dMOTA99':>10s} "
          f"{'#GMM0':>8s} {'#GMM99':>9s}")
    print("-" * len(header.expandtabs()))

    for r in all_results:
        i = r['init']
        f = r['final']
        gm0 = f"{r['gmm_init_mean']:.0f}" if r['gmm_init_mean'] is not None else "N/A"
        gm99 = f"{r['gmm_final_mean']:.0f}" if r['gmm_final_mean'] is not None else "N/A"
        print(f"{r['label']:<35s} "
              f"{i['recall']:9.4f} {f['recall']:10.4f} "
              f"{i['hallucination']:8.4f} {f['hallucination']:9.4f} "
              f"{i['dmota']:9.4f} {f['dmota']:10.4f} "
              f"{gm0:>8s} {gm99:>9s}")

    # ===================================================================
    # Table 2 — Delta (improvement from training)
    # ===================================================================
    print("\n" + "=" * 100)
    print(f"IMPROVEMENT  (iter 99 − iter 0)")
    print("=" * 100)
    print(f"{'Scenario':<35s} {'ΔRecall':>9s} {'ΔHall':>9s} {'ΔdMOTA':>9s} {'Δ#GMM':>9s}")
    print("-" * 100)

    for r in all_results:
        i = r['init']
        f = r['final']
        d_recall = f['recall'] - i['recall']
        d_hall = f['hallucination'] - i['hallucination']
        d_dmota = f['dmota'] - i['dmota']
        d_gmm = ""
        if r['gmm_init_mean'] is not None and r['gmm_final_mean'] is not None:
            d_gmm = f"{r['gmm_final_mean'] - r['gmm_init_mean']:+.0f}"
        print(f"{r['label']:<35s} {d_recall:+.4f}   {d_hall:+.4f}   {d_dmota:+.4f}   {d_gmm:>9s}")

    # ===================================================================
    # Table 3 — Training-time statistics (for reference)
    # ===================================================================
    print("\n" + "=" * 100)
    print("TRAINING-TIME STATISTICS  (from statistics.npz)")
    print("=" * 100)
    print(f"{'Scenario':<35s} {'TrLoss':>10s} {'ISE':>10s} {'TrTime':>10s} {'Scale':>10s}")
    print("-" * 100)

    for r in all_results:
        tloss = f"{r['train_loss_mean']:.4f}" if r['train_loss_mean'] is not None else "N/A"
        tise = f"{r['train_ise_mean']:.4f}" if r['train_ise_mean'] is not None else "N/A"
        ttime = f"{r['train_time_mean']:.0f} ms" if r['train_time_mean'] is not None else "N/A"
        tscale = f"{r['scale_mean']:.3f}" if r['scale_mean'] is not None else "N/A"
        print(f"{r['label']:<35s} {tloss:>10s} {tise:>10s} {ttime:>10s} {tscale:>10s}")

    # ===================================================================
    # Table 4 — Side-by-side: dMOTA(0), dMOTA(99), ISE(train)
    # ===================================================================
    print("\n" + "=" * 100)
    print("SUMMARY: dMOTA before/after training vs training ISE loss")
    print("=" * 100)
    print(f"{'Scenario':<35s} {'dMOTA(0)':>10s} {'dMOTA(99)':>10s} {'ISE(train)':>12s} {'ΔdMOTA':>9s}")
    print("-" * 100)

    for r in all_results:
        dm0 = f"{r['init']['dmota']:.4f}"
        dm99 = f"{r['final']['dmota']:.4f}"
        ise = f"{r['train_ise_mean']:.4f}" if r['train_ise_mean'] is not None else "N/A"
        dd = f"{r['final']['dmota'] - r['init']['dmota']:+.4f}"
        print(f"{r['label']:<35s} {dm0:>10s} {dm99:>10s} {ise:>12s} {dd:>9s}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    compute_metrics_multi_scenarios()
