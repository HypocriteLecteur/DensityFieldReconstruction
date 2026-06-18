"""
Center estimation: triangulate the 3D swarm center from 2D camera observations.

Extracted from DensityReconstructor to make the pipeline step independently testable.
"""

import warnings
import numpy as np
import torch
import cv2


def estimate_center_from_point_sets(cameras, point_sets):
    """Triangulate the 3D center from the mean of 2D projections.

    Args:
        cameras: List of Camera objects (only first two are used).
        point_sets: List of 2D point arrays, one per camera.

    Returns:
        center: (3,) numpy array in world coordinates.
    """
    if len(cameras) > 2:
        warnings.warn(
            "Only the first two cameras will be used for swarm center estimation.",
            UserWarning
        )

    cam1 = cameras[0].state
    cam2 = cameras[1].state

    P1_proj = cam1.intrinsics_params @ cam1.P_np
    P2_proj = cam2.intrinsics_params @ cam2.P_np

    pnts4D = cv2.triangulatePoints(
        P1_proj, P2_proj,
        np.mean(point_sets[0], axis=0),
        np.mean(point_sets[1], axis=0)
    )

    center = (pnts4D[:3, :] / pnts4D[3].T).reshape((3,))
    return center


def estimate_center_from_images(cameras, images):
    """Triangulate the 3D center from image-intensity centroids.

    Args:
        cameras: List of Camera objects (only first two are used).
        images: List of torch.Tensor density images (H, W).

    Returns:
        center: (3,) numpy array in world coordinates.
    """
    if len(images) < 2:
        raise ValueError("Need at least two images for triangulation.")

    centroids_np = []
    for img in images:
        H, W = img.shape[-2:]

        total_intensity = img.sum()
        if total_intensity.item() == 0:
            raise ValueError(
                f"Image from camera {len(centroids_np)} is empty (sum=0)."
            )

        x_coords = torch.arange(W, dtype=torch.float32, device=img.device)
        y_coords = torch.arange(H, dtype=torch.float32, device=img.device)

        x_weighted = (x_coords * img.sum(dim=-2)).sum() / total_intensity
        y_weighted = (y_coords * img.sum(dim=-1)).sum() / total_intensity

        centroids_np.append(np.array([
            x_weighted.cpu().item() + 0.5,
            y_weighted.cpu().item() + 0.5
        ]))

    cam1 = cameras[0].state
    cam2 = cameras[1].state

    P1_proj = cam1.intrinsics_params @ cam1.P_np
    P2_proj = cam2.intrinsics_params @ cam2.P_np

    pnts4D = cv2.triangulatePoints(
        P1_proj, P2_proj,
        centroids_np[0], centroids_np[1]
    )

    center = (pnts4D[:3, :] / pnts4D[3]).T.reshape((3,))
    return center
