"""2D camera projection and image-plane density plotting primitives."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Ellipse


def transparent_colormap(
    color=(0.255, 0.412, 0.882, 1.0),
    *,
    name: str = "dfr_transparent",
    samples: int = 256,
) -> LinearSegmentedColormap:
    """Return a transparent-to-color colormap for image-plane densities."""
    if samples < 2:
        raise ValueError("samples must be at least 2.")
    top = np.asarray(color, dtype=float)
    if top.shape == (3,):
        top = np.concatenate([top, [1.0]])
    if top.shape != (4,):
        raise ValueError("color must be RGB or RGBA.")
    bottom = np.array([1.0, 1.0, 1.0, 0.0])
    alpha = np.linspace(0, top[3], samples)
    rgb = bottom[:3] + (top[:3] - bottom[:3]) * np.linspace(0, 1, samples)[:, None]
    rgba = np.column_stack([rgb, alpha])
    return LinearSegmentedColormap.from_list(name, rgba)


def plot_projection_points(
    points,
    *,
    image_shape: tuple[int, int],
    ax=None,
    color: str = "royalblue",
    point_size: float = 10,
    alpha: float = 0.65,
):
    """Plot 2D projected points on image-plane axes."""
    points = _points(points)
    height, width = _image_shape(image_shape)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    if len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=color,
            s=point_size,
            alpha=alpha,
            edgecolors="none",
        )
    _format_image_axes(ax, width, height)
    fig.tight_layout(pad=0)
    return fig, ax


def plot_density_image(
    density,
    *,
    image_shape: Optional[tuple[int, int]] = None,
    density_cutoff: float = 1e-2,
    num_levels: int = 8,
    ax=None,
    cmap=None,
    line_color: str = "#4169e1",
    line_alpha: float = 0.5,
    line_width: float = 0.3,
):
    """Plot an image-plane density as transparent filled contours."""
    image = _density(density)
    height, width = _image_shape(image_shape or image.shape)
    if image.shape != (height, width):
        raise ValueError("density shape must match image_shape.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    levels, vmax = _contour_levels(image, density_cutoff, num_levels)
    if levels is not None:
        y_px = np.arange(height)
        x_px = np.arange(width)
        norm = PowerNorm(gamma=0.40, vmin=0, vmax=vmax)
        cmap = cmap or transparent_colormap()
        ax.contourf(
            x_px,
            y_px,
            image,
            levels=levels,
            cmap=cmap,
            norm=norm,
            antialiased=True,
        )
        ax.contour(
            x_px,
            y_px,
            image,
            levels=levels,
            colors=line_color,
            linewidths=line_width,
            alpha=line_alpha,
        )
    _format_image_axes(ax, width, height, max_is_inclusive=True)
    fig.tight_layout(pad=0)
    return fig, ax


def plot_projected_gmm_density(
    density,
    means_2d,
    covariances_2d,
    weights,
    *,
    image_shape: Optional[tuple[int, int]] = None,
    density_cutoff: float = 1e-2,
    num_levels: int = 8,
    ax=None,
    cmap=None,
):
    """Plot projected 2D GMM density with one-sigma covariance ellipses."""
    means = _points(means_2d)
    covariances = _covariances(covariances_2d, len(means))
    weights = _weights(weights, len(means))
    fig, ax = plot_density_image(
        density,
        image_shape=image_shape,
        density_cutoff=density_cutoff,
        num_levels=num_levels,
        ax=ax,
        cmap=cmap,
        line_alpha=0.0,
    )
    if len(means):
        relative = weights / weights.max() if weights.max() > 0 else np.ones_like(weights)
        for mean, covariance, rel_weight in zip(means, covariances, relative):
            ellipse = _covariance_ellipse(mean, covariance, rel_weight)
            if ellipse is not None:
                ax.add_patch(ellipse)
    return fig, ax


def _covariance_ellipse(mean, covariance, relative_weight: float) -> Optional[Ellipse]:
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        return None
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    alpha = 0.15 + 0.70 * float(relative_weight)
    return Ellipse(
        (mean[0], mean[1]),
        2 * np.sqrt(values[0]),
        2 * np.sqrt(values[1]),
        angle=angle,
        facecolor="none",
        edgecolor="black",
        linestyle="--",
        linewidth=1.0,
        alpha=alpha,
        zorder=3,
    )


def _format_image_axes(ax, width: int, height: int, *, max_is_inclusive=False) -> None:
    limit_offset = 1 if max_is_inclusive else 0
    ax.set_xlim(0, width - limit_offset)
    ax.set_ylim(height - limit_offset, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _contour_levels(image: np.ndarray, cutoff: float, count: int):
    if cutoff <= 0:
        raise ValueError("density_cutoff must be positive.")
    if count < 2:
        raise ValueError("num_levels must be at least 2.")
    vmax = float(np.nanmax(image))
    if not np.isfinite(vmax) or vmax <= 0:
        return None, vmax
    return np.geomspace(vmax * cutoff, vmax, count), vmax


def _points(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array.reshape(0, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("2D points must have shape (points, 2).")
    return array


def _density(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("density must have shape (height, width).")
    return array


def _covariances(values, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (length, 2, 2):
        raise ValueError("covariances_2d must have shape (components, 2, 2).")
    return array


def _weights(values, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(array) != length:
        raise ValueError("weights must contain one value per 2D mean.")
    return array


def _image_shape(value) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("image_shape must be (height, width).")
    height, width = (int(value[0]), int(value[1]))
    if height < 1 or width < 1:
        raise ValueError("image_shape dimensions must be positive.")
    return height, width
