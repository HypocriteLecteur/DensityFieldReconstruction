"""
Checkpoint serialization for GaussianModel.

Extracted from density_field_model.py to separate serialization concerns
from model parameters and training logic.
"""

import os
import torch


def build_checkpoint(model) -> dict:
    """Capture the current model state as a serializable dict.

    Args:
        model: GaussianModel instance with _xyz, _radius, _weights,
               optimizer, training config, and rasterizer attributes.

    Returns:
        dict with all data needed to restore the model later.
    """
    checkpoint = {
        '_xyz': model._xyz.detach().clone(),
        '_radius': model._radius.detach().clone(),
        '_weights': model._weights.detach().clone(),

        '_xyz_grad': model._xyz.grad.detach().clone() if model._xyz.grad is not None else None,
        '_radius_grad': model._radius.grad.detach().clone() if model._radius.grad is not None else None,
        '_weights_grad': model._weights.grad.detach().clone() if model._weights.grad is not None else None,

        'xyz_reg': model.xyz_reg,
        'radius_reg': model.radius_reg,
        'radius_cutoff_inv': model.radius_cutoff_inv,
        'xyz_lr_c': model.xyz_lr_c,
        'xyz_lr_final_c': model.xyz_lr_final_c,
        'radius_lr_c': model.radius_lr_c,
        'radius_lr_final_c': model.radius_lr_final_c,
        'weights_lr_c': model.weights_lr_c,
        'weights_lr_final_c': model.weights_lr_final_c,

        'optimizer_state_dict': model.optimizer.state_dict(),

        'optimizer_type': model.optimizer_type,
        'rasterizer_h': model.rasterizer_h,
        'rasterizer_w': model.rasterizer_w,
        'rasterizer_p_max': model.rasterizer_p_max,

        'cam_num': len(model.rasterizer_list),
    }
    return checkpoint


def write_checkpoints(training_history: list, path: str):
    """Save the full training history to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(training_history, path)


def save_history(metrics_history: list, path: str):
    """Save metrics history to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(metrics_history, path)


def load_training_history(path: str, device='cuda') -> list:
    """Load training history from a checkpoint file."""
    return torch.load(path, map_location=device, weights_only=False)


def restore_model_from_checkpoint(model_cls, training_history: list, iter: int,
                                  device='cuda'):
    """Restore a GaussianModel from a specific iteration in the training history.

    Args:
        model_cls: The GaussianModel class (or compatible subclass).
        training_history: List of checkpoint dicts from build_checkpoint().
        iter: Iteration index to restore.
        device: Target device.

    Returns:
        A new GaussianModel instance with restored parameters and optimizer state.
    """
    from gaussian_rasterizer_simple_large import GaussianRasterizerSimpleLarge

    checkpoint = training_history[iter + 1]

    model = model_cls(optimizer_type=checkpoint['optimizer_type'])
    model.rasterizer_h = checkpoint.get('rasterizer_h', 1000)
    model.rasterizer_w = checkpoint.get('rasterizer_w', 1000)
    model.rasterizer_p_max = checkpoint.get('rasterizer_p_max', 512)

    model._xyz = torch.nn.Parameter(checkpoint['_xyz'].to(device).requires_grad_(True))
    model._radius = torch.nn.Parameter(checkpoint['_radius'].to(device).requires_grad_(True))
    model._weights = torch.nn.Parameter(checkpoint['_weights'].to(device).requires_grad_(True))

    model._xyz.grad = checkpoint['_xyz_grad'].to(device) if checkpoint['_xyz_grad'] is not None else None
    model._radius.grad = checkpoint['_radius_grad'].to(device) if checkpoint['_radius_grad'] is not None else None
    model._weights.grad = checkpoint['_weights_grad'].to(device) if checkpoint['_weights_grad'] is not None else None

    model.rasterizer_list = [
        GaussianRasterizerSimpleLarge(
            H=model.rasterizer_h, W=model.rasterizer_w, P_max=model.rasterizer_p_max
        )
        for _ in range(checkpoint['cam_num'])
    ]

    model.training_setup(
        xyz_reg=checkpoint['xyz_reg'],
        radius_reg=checkpoint['radius_reg'],
        radius_cutoff_inv=checkpoint['radius_cutoff_inv'],
        xyz_lr_c=checkpoint['xyz_lr_c'],
        xyz_lr_final_c=checkpoint['xyz_lr_final_c'],
        radius_lr_c=checkpoint['radius_lr_c'],
        radius_lr_final_c=checkpoint['radius_lr_final_c'],
        weights_lr_c=checkpoint['weights_lr_c'],
        weights_lr_final_c=checkpoint['weights_lr_final_c'],
    )

    if 'optimizer_state_dict' in checkpoint and model.optimizer is not None:
        try:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print(f"Warning: Optimizer state could not be loaded: {e}. Starting with a fresh state.")

    return model
