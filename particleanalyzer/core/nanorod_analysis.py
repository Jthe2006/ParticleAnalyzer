"""Nanorod dimensions derived from exact convex-hull calipers."""

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class NanorodMetrics:
    """Projected 2D dimensions and orientations for one segmented particle."""

    length: float
    width: float
    aspect_ratio: float
    orientation: float
    feret_max: float
    feret_min: float
    feret_mean: float
    feret_max_angle: float
    feret_min_angle: float
    iso_feret_ratio: float


_ZERO_METRICS = NanorodMetrics(
    length=0.0,
    width=0.0,
    aspect_ratio=0.0,
    orientation=0.0,
    feret_max=0.0,
    feret_min=0.0,
    feret_mean=0.0,
    feret_max_angle=0.0,
    feret_min_angle=0.0,
    iso_feret_ratio=0.0,
)


def calculate_nanorod_metrics(contour: np.ndarray) -> NanorodMetrics:
    """Measure a straight nanorod with an orthogonal Feret-caliper pair.

    The width is the minimum Feret diameter of the convex hull. The length is
    the hull span perpendicular to that minimum-width direction. This avoids
    treating the diagonal of a flat-ended rod as its physical length.

    The unrestricted maximum/minimum/mean Feret descriptors are returned for
    compatibility with the existing ParticleAnalyzer output. The ISO-style
    Feret ratio is ``minimum / maximum``; the nanorod aspect ratio is the more
    customary ``length / width`` and is therefore at least one.
    """

    points = np.asarray(contour, dtype=np.float64).reshape((-1, 2))
    if len(points) < 3 or not np.all(np.isfinite(points)):
        return _ZERO_METRICS

    hull = cv2.convexHull(points.astype(np.float32)).reshape((-1, 2)).astype(float)
    if len(hull) < 3 or cv2.contourArea(hull.astype(np.float32)) <= 0.0:
        return _ZERO_METRICS

    edges = np.roll(hull, -1, axis=0) - hull
    edge_lengths = np.linalg.norm(edges, axis=1)
    valid_edges = edge_lengths > np.finfo(float).eps
    if not np.any(valid_edges):
        return _ZERO_METRICS

    tangents = edges[valid_edges] / edge_lengths[valid_edges, np.newaxis]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    normal_projections = hull @ normals.T
    widths = np.ptp(normal_projections, axis=0)
    min_index = int(np.argmin(widths))

    width = float(widths[min_index])
    major_direction = tangents[min_index]
    length = float(np.ptp(hull @ major_direction))
    if width <= 0.0 or length <= 0.0:
        return _ZERO_METRICS

    max_distance_squared = -1.0
    max_vector = np.zeros(2, dtype=float)
    for index, point in enumerate(hull[:-1]):
        vectors = hull[index + 1 :] - point
        distances_squared = np.einsum("ij,ij->i", vectors, vectors)
        pair_index = int(np.argmax(distances_squared))
        if distances_squared[pair_index] > max_distance_squared:
            max_distance_squared = float(distances_squared[pair_index])
            max_vector = vectors[pair_index]

    feret_max = math.sqrt(max_distance_squared)
    feret_min = width
    hull_perimeter = cv2.arcLength(
        hull.astype(np.float32).reshape((-1, 1, 2)), closed=True
    )
    feret_mean = float(hull_perimeter / math.pi)

    orientation = _normalize_angle(major_direction)
    feret_min_angle = _normalize_angle(normals[min_index])
    feret_max_angle = _normalize_angle(max_vector)
    aspect_ratio = length / width
    iso_feret_ratio = feret_min / feret_max if feret_max > 0.0 else 0.0

    values = (
        length,
        width,
        aspect_ratio,
        feret_max,
        feret_min,
        feret_mean,
        iso_feret_ratio,
    )
    if not all(math.isfinite(value) for value in values):
        return _ZERO_METRICS

    return NanorodMetrics(
        length=length,
        width=width,
        aspect_ratio=max(1.0, aspect_ratio),
        orientation=orientation,
        feret_max=feret_max,
        feret_min=feret_min,
        feret_mean=feret_mean,
        feret_max_angle=feret_max_angle,
        feret_min_angle=feret_min_angle,
        iso_feret_ratio=max(0.0, min(1.0, iso_feret_ratio)),
    )


def contour_touches_image_border(
    contour: np.ndarray,
    image_shape: Sequence[int],
    margin: float = 0.0,
) -> bool:
    """Return whether a contour is truncated by an image boundary."""

    points = np.asarray(contour, dtype=float).reshape((-1, 2))
    if len(points) == 0 or len(image_shape) < 2:
        return False

    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        return False

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return bool(
        np.any(x_coords <= margin)
        or np.any(y_coords <= margin)
        or np.any(x_coords >= width - 1 - margin)
        or np.any(y_coords >= height - 1 - margin)
    )


def passes_nanorod_filter(
    metrics: NanorodMetrics,
    min_aspect_ratio: float,
    touches_border: bool,
    exclude_border_rods: bool,
) -> bool:
    """Return whether a valid projected rod satisfies the selected QC gate."""

    threshold = float(min_aspect_ratio)
    if not math.isfinite(threshold):
        return False
    if metrics.length <= 0.0 or metrics.width <= 0.0:
        return False
    if exclude_border_rods and touches_border:
        return False
    return metrics.aspect_ratio >= threshold


def _normalize_angle(direction: np.ndarray) -> float:
    """Normalize an undirected 2D axis angle to the half-open [0, 180) range."""

    angle = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    return angle % 180.0
