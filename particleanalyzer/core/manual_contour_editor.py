"""Pure helpers for validating and editing particle contour results."""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import pandas as pd


class PolygonValidationError(ValueError):
    """Raised when a proposed particle polygon is not a valid simple polygon."""


def normalize_polygon(points: Any, image_shape: Sequence[int]) -> np.ndarray:
    """Validate *points* and return an ``int32`` OpenCV contour."""

    height, width = _image_dimensions(image_shape)
    coordinates = _coerce_coordinate_array(points)
    if not np.all(np.isfinite(coordinates)):
        raise PolygonValidationError("Polygon coordinates must be finite.")
    if (
        np.any(coordinates[:, 0] < 0)
        or np.any(coordinates[:, 1] < 0)
        or np.any(coordinates[:, 0] > width - 1)
        or np.any(coordinates[:, 1] > height - 1)
    ):
        raise PolygonValidationError("Polygon coordinates are outside the image.")

    rounded = np.rint(coordinates).astype(np.int32)
    cleaned: list[tuple[int, int]] = []
    for x_coord, y_coord in rounded:
        point = (int(x_coord), int(y_coord))
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    if len(set(cleaned)) < 3:
        raise PolygonValidationError("A polygon needs at least three unique points.")
    if len(set(cleaned)) != len(cleaned):
        raise PolygonValidationError("A polygon cannot revisit a vertex.")
    if _polygon_self_intersects(cleaned):
        raise PolygonValidationError("Polygon edges cannot self-intersect.")

    contour = np.asarray(cleaned, dtype=np.int32).reshape((-1, 1, 2))
    if float(abs(cv2.contourArea(contour))) <= 0.0:
        raise PolygonValidationError("Polygon area must be positive.")
    return contour


def render_editor_preview(
    base_image: np.ndarray,
    points_df: pd.DataFrame | None,
    draft_points: Any,
    selected_ids: Any,
    output_table: pd.DataFrame | None,
) -> np.ndarray:
    """Render existing, selected, and draft polygons on a same-size image copy."""

    image = np.asarray(base_image)
    if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("base_image must be a non-empty image array")
    preview = image.copy()

    rows = _points_rows(points_df)
    particle_ids = _particle_ids(output_table, len(rows))
    if output_table is not None and len(output_table) != len(rows):
        raise ValueError("output_table and points_df must have aligned rows")
    selected = {
        _canonical_particle_id(value) for value in _selection_values(selected_ids)
    }

    normal_contours: list[np.ndarray] = []
    selected_contours: list[np.ndarray] = []
    for particle_id, stored_points in zip(particle_ids, rows):
        contour = _drawing_contour(stored_points, preview.shape)
        if contour is None:
            continue
        if _canonical_particle_id(particle_id) in selected:
            selected_contours.append(contour)
        else:
            normal_contours.append(contour)

    green = _drawing_color(preview, (0, 255, 0))
    orange = _drawing_color(preview, (0, 165, 255))
    magenta = _drawing_color(preview, (255, 0, 255))
    if normal_contours:
        cv2.polylines(preview, normal_contours, True, green, 2, cv2.LINE_8)
    if selected_contours:
        cv2.polylines(preview, selected_contours, True, orange, 2, cv2.LINE_8)

    draft = _drawing_contour(draft_points, preview.shape)
    if draft is not None:
        if len(draft) >= 2:
            cv2.polylines(
                preview,
                [draft],
                len(draft) >= 3,
                magenta,
                2,
                cv2.LINE_8,
            )
        for x_coord, y_coord in draft.reshape((-1, 2)):
            cv2.circle(
                preview,
                (int(x_coord), int(y_coord)),
                3,
                magenta,
                -1,
                cv2.LINE_8,
            )
    return preview


def mutate_particle_tables(
    action: str,
    record_with_points: Any,
    selected_ids: Any,
    output_table: pd.DataFrame,
    points_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy-on-write add, replace, or delete with particle-number selection."""

    if not isinstance(output_table, pd.DataFrame) or not isinstance(
        points_df, pd.DataFrame
    ):
        raise TypeError("output_table and points_df must be pandas DataFrames")
    if len(output_table.columns) == 0:
        raise ValueError("output_table needs a particle-number column")
    if len(output_table) != len(points_df):
        raise ValueError("output_table and points_df must have aligned rows")
    if "points" not in points_df.columns:
        if len(points_df) == 0:
            points_df = points_df.copy()
            points_df["points"] = pd.Series(dtype=object)
        else:
            raise ValueError("points_df needs a 'points' column")

    normalized_action = str(action).strip().lower()
    if normalized_action not in {"add", "replace", "delete"}:
        raise ValueError("action must be 'add', 'replace', or 'delete'")

    table_copy = output_table.copy(deep=True).reset_index(drop=True)
    points_copy = points_df.copy(deep=True).reset_index(drop=True)
    points_copy["points"] = [
        copy.deepcopy(value) for value in points_copy["points"].tolist()
    ]
    id_column = table_copy.columns[0]
    positions = _selected_positions(table_copy, selected_ids)

    record: dict[str, Any] | None = None
    if normalized_action in {"add", "replace"}:
        record = _record_mapping(record_with_points)
        if "points" not in record:
            raise ValueError("record_with_points needs a 'points' value")
        stored_points = _stored_points(record["points"])

    if normalized_action == "add":
        assert record is not None
        new_row = {
            column: copy.deepcopy(record.get(column, np.nan))
            for column in table_copy.columns
        }
        table_copy = pd.concat(
            [table_copy, pd.DataFrame([new_row], columns=table_copy.columns)],
            ignore_index=True,
        )
        point_row = {column: np.nan for column in points_copy.columns}
        point_row["points"] = stored_points
        points_copy = pd.concat(
            [points_copy, pd.DataFrame([point_row], columns=points_copy.columns)],
            ignore_index=True,
        )
    elif normalized_action == "replace":
        assert record is not None
        if len(positions) != 1:
            raise ValueError("replace requires exactly one selected particle")
        position = positions[0]
        for column in table_copy.columns:
            if column != id_column and column in record:
                table_copy.at[position, column] = copy.deepcopy(record[column])
        points_copy.at[position, "points"] = stored_points
    else:
        if not positions:
            raise ValueError("delete requires at least one selected particle")
        removed = set(positions)
        keep = [index for index in range(len(table_copy)) if index not in removed]
        table_copy = table_copy.iloc[keep].reset_index(drop=True)
        points_copy = points_copy.iloc[keep].reset_index(drop=True)

    table_copy[id_column] = np.arange(1, len(table_copy) + 1, dtype=int)
    table_copy.reset_index(drop=True, inplace=True)
    points_copy.reset_index(drop=True, inplace=True)
    return table_copy, points_copy


def write_contour_exports(
    points_df: pd.DataFrame,
    image_shape: Sequence[int],
    image_name: str,
    output_dir: str = "output",
) -> tuple[str, str]:
    """Write a union binary mask and matching single-image COCO annotations."""

    if not isinstance(points_df, pd.DataFrame):
        raise TypeError("points_df must be a pandas DataFrame")
    if len(points_df) and "points" not in points_df.columns:
        raise ValueError("points_df needs a 'points' column")
    height, width = _image_dimensions(image_shape)
    contours = [
        normalize_polygon(value, (height, width))
        for value in (
            points_df["points"].tolist() if "points" in points_df.columns else []
        )
    ]

    mask = np.zeros((height, width), dtype=np.uint8)
    annotations: list[dict[str, Any]] = []
    for annotation_id, contour in enumerate(contours, 1):
        cv2.fillPoly(mask, [contour], 255)
        flattened = contour.reshape((-1, 2))
        x_min = float(np.min(flattened[:, 0]))
        y_min = float(np.min(flattened[:, 1]))
        x_max = float(np.max(flattened[:, 0]))
        y_max = float(np.max(flattened[:, 1]))
        annotations.append(
            {
                "id": annotation_id,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [flattened.astype(float).ravel().tolist()],
                "area": float(abs(cv2.contourArea(contour))),
                "bbox": [
                    x_min,
                    y_min,
                    x_max - x_min,
                    y_max - y_min,
                ],
                "iscrowd": 0,
            }
        )

    safe_image_name = str(image_name)
    if not safe_image_name or os.path.basename(safe_image_name) != safe_image_name:
        raise ValueError("image_name must be a non-empty file name")
    os.makedirs(output_dir, exist_ok=True)
    mask_path = os.path.join(output_dir, f"binary_mask_{safe_image_name}.png")
    coco_path = os.path.join(output_dir, f"coco_file_{safe_image_name}.json")
    if not cv2.imwrite(mask_path, mask):
        raise OSError(f"Unable to write binary mask: {mask_path}")

    coco = {
        "images": [
            {
                "id": 1,
                "file_name": f"{safe_image_name}.png",
                "width": width,
                "height": height,
            }
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "Particle", "supercategory": "none"}],
    }
    with open(coco_path, "w", encoding="utf-8") as output_file:
        json.dump(coco, output_file, ensure_ascii=False, indent=2)
    return mask_path, coco_path


def _image_dimensions(image_shape: Sequence[int]) -> tuple[int, int]:
    try:
        height, width = int(image_shape[0]), int(image_shape[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise PolygonValidationError(
            "image_shape must contain height and width"
        ) from exc
    if height <= 0 or width <= 0:
        raise PolygonValidationError("image dimensions must be positive")
    return height, width


def _coerce_coordinate_array(points: Any) -> np.ndarray:
    if points is None:
        raise PolygonValidationError("Polygon points are required.")
    try:
        coordinates = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PolygonValidationError(
            "Polygon points must be numeric coordinate pairs."
        ) from exc
    if coordinates.size == 0 or coordinates.size % 2:
        raise PolygonValidationError("Polygon points must be coordinate pairs.")
    try:
        return coordinates.reshape((-1, 2))
    except ValueError as exc:
        raise PolygonValidationError(
            "Polygon points must be coordinate pairs."
        ) from exc


def _polygon_self_intersects(points: list[tuple[int, int]]) -> bool:
    count = len(points)
    for index in range(count):
        first = points[index]
        second = points[(index + 1) % count]
        third = points[(index + 2) % count]
        if _orientation(first, second, third) == 0:
            first_vector = (second[0] - first[0], second[1] - first[1])
            second_vector = (third[0] - second[0], third[1] - second[1])
            if (
                first_vector[0] * second_vector[0] + first_vector[1] * second_vector[1]
                < 0
            ):
                return True

    for first_index in range(count):
        first_start = points[first_index]
        first_end = points[(first_index + 1) % count]
        for second_index in range(first_index + 1, count):
            if second_index == first_index:
                continue
            if (first_index + 1) % count == second_index:
                continue
            if (second_index + 1) % count == first_index:
                continue
            second_start = points[second_index]
            second_end = points[(second_index + 1) % count]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return True
    return False


def _segments_intersect(a, b, c, d) -> bool:
    first = _orientation(a, b, c)
    second = _orientation(a, b, d)
    third = _orientation(c, d, a)
    fourth = _orientation(c, d, b)
    if first == 0 and _on_segment(a, c, b):
        return True
    if second == 0 and _on_segment(a, d, b):
        return True
    if third == 0 and _on_segment(c, a, d):
        return True
    if fourth == 0 and _on_segment(c, b, d):
        return True
    return (first > 0) != (second > 0) and (third > 0) != (fourth > 0)


def _orientation(a, b, c) -> int:
    cross_product = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (cross_product > 0) - (cross_product < 0)


def _on_segment(a, point, b) -> bool:
    return min(a[0], b[0]) <= point[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= point[
        1
    ] <= max(a[1], b[1])


def _drawing_contour(points: Any, image_shape: Sequence[int]) -> np.ndarray | None:
    if points is None:
        return None
    try:
        coordinates = _coerce_coordinate_array(points)
    except PolygonValidationError:
        return None
    finite = coordinates[np.all(np.isfinite(coordinates), axis=1)]
    if len(finite) == 0:
        return None
    height, width = int(image_shape[0]), int(image_shape[1])
    finite[:, 0] = np.clip(np.rint(finite[:, 0]), 0, width - 1)
    finite[:, 1] = np.clip(np.rint(finite[:, 1]), 0, height - 1)
    cleaned: list[tuple[int, int]] = []
    for x_coord, y_coord in finite.astype(np.int32):
        point = (int(x_coord), int(y_coord))
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if not cleaned:
        return None
    return np.asarray(cleaned, dtype=np.int32).reshape((-1, 1, 2))


def _drawing_color(image: np.ndarray, bgr: tuple[int, int, int]):
    if image.ndim == 2:
        return round(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])
    channels = image.shape[2]
    if channels == 1:
        return round(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])
    if channels == 4:
        return (*bgr, 255)
    return bgr


def _points_rows(points_df: pd.DataFrame | None) -> list[Any]:
    if points_df is None:
        return []
    if not isinstance(points_df, pd.DataFrame):
        raise TypeError("points_df must be a pandas DataFrame")
    if "points" not in points_df.columns:
        if len(points_df) == 0:
            return []
        raise ValueError("points_df needs a 'points' column")
    return points_df["points"].tolist()


def _particle_ids(output_table: pd.DataFrame | None, row_count: int) -> list[Any]:
    if output_table is None:
        return list(range(1, row_count + 1))
    if not isinstance(output_table, pd.DataFrame):
        raise TypeError("output_table must be a pandas DataFrame")
    if len(output_table.columns) == 0:
        if row_count == 0:
            return []
        raise ValueError("output_table needs a particle-number column")
    return output_table.iloc[:, 0].tolist()


def _selection_values(selected_ids: Any) -> list[Any]:
    if selected_ids is None:
        return []
    if isinstance(selected_ids, (str, bytes)) or np.isscalar(selected_ids):
        return [selected_ids]
    try:
        return list(selected_ids)
    except TypeError:
        return [selected_ids]


def _canonical_particle_id(value: Any):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        stripped = value.strip()
        try:
            numeric = float(stripped)
        except ValueError:
            return stripped
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        return stripped
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    return str(value)


def _selected_positions(table: pd.DataFrame, selected_ids: Any) -> list[int]:
    requested = [
        _canonical_particle_id(value) for value in _selection_values(selected_ids)
    ]
    if not requested:
        return []
    existing = [_canonical_particle_id(value) for value in table.iloc[:, 0].tolist()]
    if len(existing) != len(set(existing)):
        raise ValueError("Particle numbers must be unique")
    mapping = {particle_id: index for index, particle_id in enumerate(existing)}
    missing = [particle_id for particle_id in requested if particle_id not in mapping]
    if missing:
        raise KeyError(f"Unknown selected particle number(s): {missing}")
    return sorted({mapping[particle_id] for particle_id in requested})


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, pd.DataFrame):
        if len(record) != 1:
            raise ValueError("record_with_points DataFrame must contain one row")
        return record.iloc[0].to_dict()
    if isinstance(record, pd.Series):
        return record.to_dict()
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError(
        "record_with_points must be a mapping, Series, or one-row DataFrame"
    )


def _stored_points(points: Any) -> list[list[list[int]]]:
    coordinates = _coerce_coordinate_array(points)
    if not np.all(np.isfinite(coordinates)):
        raise PolygonValidationError("Polygon coordinates must be finite.")
    rounded = np.rint(coordinates).astype(np.int32)
    cleaned: list[tuple[int, int]] = []
    for x_coord, y_coord in rounded:
        point = (int(x_coord), int(y_coord))
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(set(cleaned)) < 3:
        raise PolygonValidationError("A polygon needs at least three unique points.")
    if len(set(cleaned)) != len(cleaned) or _polygon_self_intersects(cleaned):
        raise PolygonValidationError("Polygon edges cannot self-intersect.")
    contour = np.asarray(cleaned, dtype=np.int32).reshape((-1, 1, 2))
    if float(abs(cv2.contourArea(contour))) <= 0.0:
        raise PolygonValidationError("Polygon area must be positive.")
    return contour.tolist()
