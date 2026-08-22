"""Nanorod dimensions derived from exact convex-hull calipers."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class NanorodOverlapAssessment:
    """Quality-control decision for a possibly merged or duplicate mask."""

    exclude: bool
    reason: str
    oversized: bool
    duplicate: bool
    pair_limit_reached: bool = False


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


def calculate_contour_solidity(contour: np.ndarray) -> float:
    """Return contour area divided by convex-hull area."""

    points = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
    if len(points) < 3:
        return 0.0
    area = float(cv2.contourArea(points))
    hull_area = float(cv2.contourArea(cv2.convexHull(points)))
    if area <= 0.0 or hull_area <= 0.0:
        return 0.0
    return max(0.0, min(1.0, area / hull_area))


def assess_nanrod_overlaps(
    contours: Sequence[np.ndarray],
    image_shape: Sequence[int],
    confidences: Sequence[float] | None = None,
    min_aspect_ratio: float = 2.0,
    min_reference_count: int = 10,
    area_factor: float = 4.0,
    width_factor: float = 2.2,
    duplicate_iou: float = 0.50,
    containment_iom: float = 0.80,
    max_pair_checks: int = 250_000,
    gray_image: np.ndarray | None = None,
) -> list[NanorodOverlapAssessment]:
    """Identify duplicate masks and oversized multi-rod clusters.

    Typical rod width and area are estimated robustly from isolated,
    elongated, convex, non-border candidates in the same image. A candidate
    is considered an oversized cluster when its width is far above the
    reference width, or when excess area is accompanied by abnormal width or
    concavity. Polygon overlap then removes duplicate masks and large parent
    masks that contain smaller rod-like children. This deliberately does not
    try to reconstruct rods whose boundaries are hidden by overlap.
    """

    count = len(contours)
    if count == 0:
        return []

    if confidences is None:
        confidence_values = [0.0] * count
    else:
        confidence_values = list(confidences[:count])
        confidence_values.extend([0.0] * (count - len(confidence_values)))

    candidates = []
    normalized_gray = _normalize_grayscale_image(gray_image)
    for index, contour in enumerate(contours):
        points = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
        metrics = calculate_nanorod_metrics(points)
        area = float(cv2.contourArea(points)) if len(points) >= 3 else 0.0
        solidity = calculate_contour_solidity(points)
        touches_border = contour_touches_image_border(points, image_shape)
        bbox, local_mask, pixel_area = _rasterize_local_contour(points)
        confidence = float(confidence_values[index])
        if not math.isfinite(confidence):
            confidence = 0.0
        candidates.append(
            {
                "metrics": metrics,
                "area": area,
                "area_per_length": (
                    area / metrics.length if metrics.length > 0.0 else 0.0
                ),
                "solidity": solidity,
                "touches_border": touches_border,
                "bbox": bbox,
                "mask": local_mask,
                "pixel_area": pixel_area,
                "confidence": confidence,
            }
        )

    reference_candidates = [
        candidate
        for candidate in candidates
        if candidate["metrics"].aspect_ratio >= 2.2
        and candidate["metrics"].length >= 5.0
        and candidate["metrics"].width >= 2.0
        and candidate["area"] > 0.0
        and candidate["solidity"] >= 0.90
        and not candidate["touches_border"]
    ]
    median_area = 0.0
    median_width = 0.0
    median_area_per_length = 0.0
    if len(reference_candidates) >= max(1, int(min_reference_count)):
        median_area = float(np.median([item["area"] for item in reference_candidates]))
        median_width = float(
            np.median([item["metrics"].width for item in reference_candidates])
        )
        median_area_per_length = float(
            np.median([item["area_per_length"] for item in reference_candidates])
        )

    oversized = []
    reasons: list[list[str]] = [[] for _ in range(count)]
    duplicate = [False] * count
    for index, candidate in enumerate(candidates):
        is_too_wide = bool(
            median_width > 0.0
            and candidate["metrics"].width > width_factor * median_width
        )
        is_large_and_irregular = bool(
            median_area > 0.0
            and median_width > 0.0
            and (
                (
                    candidate["area"] > area_factor * median_area
                    and (
                        candidate["metrics"].width > 1.6 * median_width
                        or candidate["solidity"] < 0.85
                    )
                )
                or (
                    candidate["area"] > 2.5 * median_area
                    and candidate["metrics"].width > 1.6 * median_width
                    and candidate["solidity"] < 0.90
                )
            )
        )
        is_parallel_cluster = bool(
            not is_too_wide
            and not is_large_and_irregular
            and normalized_gray is not None
            and median_width > 0.0
            and median_area_per_length > 0.0
            and candidate["metrics"].aspect_ratio >= 1.4
            and candidate["metrics"].width > 1.55 * median_width
            and candidate["area_per_length"] > 1.45 * median_area_per_length
            and _has_parallel_intensity_bands(
                normalized_gray,
                np.asarray(contours[index], dtype=np.float32).reshape((-1, 1, 2)),
                reference_width=median_width,
                reference_area_per_length=median_area_per_length,
            )
        )
        is_oversized = bool(
            is_too_wide or is_large_and_irregular or is_parallel_cluster
        )
        oversized.append(is_oversized)
        if is_parallel_cluster:
            reasons[index].append("parallel_cluster")
        elif is_oversized:
            reasons[index].append("oversized_cluster")

    # Only compare candidates whose x-ranges overlap. This sweep avoids the
    # quadratic all-pairs cost for the usual sparse field, while the explicit
    # budget keeps a pathological fully dense field responsive.
    pair_events = []
    pair_checks = 0
    pair_limit_reached = False
    order = sorted(range(count), key=lambda index: candidates[index]["bbox"][0])
    active: list[int] = []
    for second in order:
        second_x, second_y, _second_width, second_height = candidates[second][
            "bbox"
        ]
        active = [
            first
            for first in active
            if candidates[first]["bbox"][0] + candidates[first]["bbox"][2]
            > second_x
        ]
        for first in active:
            first_y = candidates[first]["bbox"][1]
            first_height = candidates[first]["bbox"][3]
            if (
                first_y + first_height <= second_y
                or second_y + second_height <= first_y
            ):
                continue
            pair_checks += 1
            if pair_checks > max(0, int(max_pair_checks)):
                pair_limit_reached = True
                break
            overlap = _local_mask_overlap(candidates[first], candidates[second])
            if overlap is None:
                continue
            intersection, union, smaller_area = overlap
            if intersection <= 0 or union <= 0 or smaller_area <= 0:
                continue
            iou = intersection / union
            iom = intersection / smaller_area
            first_area = candidates[first]["pixel_area"]
            second_area = candidates[second]["pixel_area"]
            larger = first if first_area >= second_area else second
            smaller = second if larger == first else first
            area_ratio = max(first_area, second_area) / max(
                1, min(first_area, second_area)
            )
            # A rod-like child inside a substantially larger mask is more
            # informative than generic IoU: reject the merged parent first,
            # even when the pair also crosses the duplicate-IoU threshold.
            if (
                iom >= containment_iom
                and area_ratio >= 1.5
                and candidates[smaller]["metrics"].aspect_ratio
                >= max(1.0, float(min_aspect_ratio))
            ):
                pair_events.append(("contained_parent", iom, larger, smaller))
            elif iou >= duplicate_iou:
                pair_events.append(("duplicate", iou, first, second))
        if pair_limit_reached:
            break
        active.append(second)

    pair_events.sort(
        key=lambda event: (event[0] == "contained_parent", event[1]), reverse=True
    )
    for event_type, _score, first, second in pair_events:
        if event_type == "contained_parent":
            if not reasons[second] and "contained_parent" not in reasons[first]:
                reasons[first].append("contained_parent")
            continue
        if reasons[first] and reasons[second]:
            continue
        if reasons[first]:
            continue
        if reasons[second]:
            continue
        first_priority = _candidate_priority(
            candidates[first], oversized[first], min_aspect_ratio
        )
        second_priority = _candidate_priority(
            candidates[second], oversized[second], min_aspect_ratio
        )
        rejected = second if first_priority >= second_priority else first
        reasons[rejected].append("duplicate_mask")
        duplicate[rejected] = True

    return [
        NanorodOverlapAssessment(
            exclude=bool(reasons[index]),
            reason=";".join(reasons[index]),
            oversized=oversized[index],
            duplicate=duplicate[index],
            pair_limit_reached=pair_limit_reached,
        )
        for index in range(count)
    ]


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


def _normalize_grayscale_image(image: np.ndarray | None) -> np.ndarray | None:
    """Return an 8-bit grayscale image without per-particle normalization."""

    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.ndim != 2 or array.size == 0:
        return None
    if array.dtype == np.uint8:
        return array

    values = array.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [1.0, 99.0])
    if not math.isfinite(float(low)) or high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = np.clip((values - low) * (255.0 / (high - low)), 0.0, 255.0)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.uint8)


def _has_parallel_intensity_bands(
    gray_image: np.ndarray,
    contour: np.ndarray,
    reference_width: float,
    reference_area_per_length: float,
) -> bool:
    """Detect two sustained dark transverse bands inside one wide TEM mask."""

    if reference_width <= 0.0 or reference_area_per_length <= 0.0:
        return False
    points = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
    if len(points) < 3:
        return False

    rect = cv2.minAreaRect(points)
    center = np.asarray(rect[0], dtype=np.float32)
    box = cv2.boxPoints(rect).astype(np.float32)
    edges = np.roll(box, -1, axis=0) - box
    edge_lengths = np.linalg.norm(edges, axis=1)
    longest_index = int(np.argmax(edge_lengths))
    length = float(edge_lengths[longest_index])
    width = float(np.min(edge_lengths))
    if width < 2.0 or length < 5.0:
        return False

    direction = edges[longest_index] / max(length, 1e-6)
    rotation = np.asarray(
        [
            [direction[0], direction[1]],
            [-direction[1], direction[0]],
        ],
        dtype=np.float32,
    )
    half_length = max(2, int(np.ceil(0.32 * length)))
    half_width = max(3, int(np.ceil(0.75 * width)))
    output_width = 2 * half_length + 1
    output_height = 2 * half_width + 1
    destination_center = np.asarray(
        [(output_width - 1) / 2.0, (output_height - 1) / 2.0],
        dtype=np.float32,
    )
    transform = np.empty((2, 3), dtype=np.float32)
    transform[:, :2] = rotation
    transform[:, 2] = destination_center - rotation @ center

    inverse_transform = cv2.invertAffineTransform(transform)
    output_corners = np.asarray(
        [
            [0.0, 0.0],
            [output_width - 1.0, 0.0],
            [output_width - 1.0, output_height - 1.0],
            [0.0, output_height - 1.0],
        ],
        dtype=np.float32,
    ).reshape((-1, 1, 2))
    source_corners = cv2.transform(output_corners, inverse_transform).reshape((-1, 2))
    if (
        np.any(source_corners[:, 0] < 0.0)
        or np.any(source_corners[:, 1] < 0.0)
        or np.any(source_corners[:, 0] > gray_image.shape[1] - 1.0)
        or np.any(source_corners[:, 1] > gray_image.shape[0] - 1.0)
    ):
        return False

    patch = cv2.warpAffine(
        gray_image.astype(np.float32, copy=False),
        transform,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_points = cv2.transform(points, transform).astype(np.int32)
    rotated_mask = np.zeros((output_height, output_width), dtype=np.uint8)
    cv2.fillPoly(rotated_mask, [rotated_points], 1)
    if not np.any(rotated_mask):
        return False
    profile = patch.mean(axis=1)
    kernel = cv2.getGaussianKernel(7, 0.75).ravel()
    smooth = np.convolve(np.pad(profile, 3, mode="reflect"), kernel, mode="valid")
    troughs = (
        np.flatnonzero(
            (smooth[1:-1] <= smooth[:-2])
            & (smooth[1:-1] < smooth[2:])
        )
        + 1
    )
    contrast = float(np.percentile(smooth, 90) - np.percentile(smooth, 10))
    required_score = max(8.0, 0.20 * contrast)

    for position, first in enumerate(troughs):
        for second in troughs[position + 1 :]:
            separation_ratio = float(second - first) / width
            if not 0.25 <= separation_ratio <= 0.80:
                continue
            saddle = float(np.max(smooth[first : second + 1]))
            left_outer = float(np.max(smooth[: first + 1]))
            right_outer = float(np.max(smooth[second:]))
            first_prominence = min(left_outer, saddle) - float(smooth[first])
            second_prominence = min(saddle, right_outer) - float(smooth[second])
            saddle_rise = saddle - max(
                float(smooth[first]), float(smooth[second])
            )
            if min(first_prominence, second_prominence, saddle_rise) >= required_score:
                return True
    return False


def _rasterize_local_contour(contour: np.ndarray):
    if len(contour) < 3:
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=bool), 0
    x_coord, y_coord, width, height = cv2.boundingRect(contour)
    if width <= 0 or height <= 0:
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=bool), 0
    shifted = contour.astype(np.int32).copy()
    shifted[:, 0, 0] -= x_coord
    shifted[:, 0, 1] -= y_coord
    local_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(local_mask, [shifted], 1)
    boolean_mask = local_mask.astype(bool)
    return (
        (x_coord, y_coord, width, height),
        boolean_mask,
        int(np.count_nonzero(boolean_mask)),
    )


def _local_mask_overlap(first, second):
    first_x, first_y, first_width, first_height = first["bbox"]
    second_x, second_y, second_width, second_height = second["bbox"]
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_width, second_x + second_width)
    bottom = min(first_y + first_height, second_y + second_height)
    if left >= right or top >= bottom:
        return None

    first_region = first["mask"][
        top - first_y : bottom - first_y,
        left - first_x : right - first_x,
    ]
    second_region = second["mask"][
        top - second_y : bottom - second_y,
        left - second_x : right - second_x,
    ]
    intersection = int(np.count_nonzero(first_region & second_region))
    if intersection == 0:
        return None
    union = first["pixel_area"] + second["pixel_area"] - intersection
    smaller_area = min(first["pixel_area"], second["pixel_area"])
    return intersection, union, smaller_area


def _candidate_priority(candidate, is_oversized: bool, min_aspect_ratio: float):
    return (
        candidate["metrics"].aspect_ratio
        >= max(1.0, float(min_aspect_ratio)),
        not is_oversized,
        candidate["confidence"],
        candidate["metrics"].aspect_ratio,
        candidate["solidity"],
        -candidate["area"],
    )


def _normalize_angle(direction: np.ndarray) -> float:
    """Normalize an undirected 2D axis angle to the half-open [0, 180) range."""

    angle = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    return angle % 180.0
