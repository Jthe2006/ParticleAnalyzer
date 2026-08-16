"""Shape metrics used to keep near-circular particle projections."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ShapeMetrics:
    """Dimensionless 2D shape metrics for one segmented particle."""

    circularity: float
    axis_ratio: float


def calculate_shape_metrics(
    area: float,
    perimeter: float,
    contour: np.ndarray,
) -> ShapeMetrics:
    """Calculate circularity and a moment-based minor-to-major axis ratio.

    Invalid or degenerate contours receive a score of zero so they are rejected
    whenever shape filtering is enabled.
    """

    values = (area, perimeter)
    if not all(math.isfinite(float(value)) for value in values):
        return ShapeMetrics(circularity=0.0, axis_ratio=0.0)

    circularity = (
        4.0 * math.pi * float(area) / float(perimeter) ** 2
        if area > 0 and perimeter > 0
        else 0.0
    )
    axis_ratio = _calculate_axis_ratio(contour)

    return ShapeMetrics(
        circularity=max(0.0, min(1.0, circularity)),
        axis_ratio=axis_ratio,
    )


def _calculate_axis_ratio(contour: np.ndarray) -> float:
    """Calculate a filled-contour axis ratio from second central moments."""

    points = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
    if len(points) < 3:
        return 0.0

    moments = cv2.moments(points)
    if moments["m00"] <= 0:
        return 0.0

    variance_x = moments["mu20"] / moments["m00"]
    variance_y = moments["mu02"] / moments["m00"]
    covariance = moments["mu11"] / moments["m00"]
    trace = variance_x + variance_y
    discriminant = math.sqrt(
        max(0.0, (variance_x - variance_y) ** 2 + 4.0 * covariance**2)
    )
    major_variance = (trace + discriminant) / 2.0
    minor_variance = (trace - discriminant) / 2.0

    if major_variance <= 0 or minor_variance < 0:
        return 0.0

    return max(0.0, min(1.0, math.sqrt(minor_variance / major_variance)))


def passes_shape_filter(
    metrics: ShapeMetrics,
    enabled: bool,
    min_circularity: float,
    min_axis_ratio: float,
) -> bool:
    """Return whether a particle should be retained by the 2D shape filter."""

    if not enabled:
        return True

    if metrics.circularity <= 0.0 or metrics.axis_ratio <= 0.0:
        return False

    return (
        metrics.circularity >= float(min_circularity)
        and metrics.axis_ratio >= float(min_axis_ratio)
    )
