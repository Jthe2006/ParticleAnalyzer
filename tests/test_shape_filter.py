import math

import cv2
import numpy as np

from particleanalyzer.core.ParticleAnalyzer import ParticleAnalyzer
from particleanalyzer.core.shape_filter import (
    ShapeMetrics,
    calculate_shape_metrics,
    passes_shape_filter,
)


def _metrics(points):
    contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    return calculate_shape_metrics(
        area=cv2.contourArea(contour),
        perimeter=cv2.arcLength(contour, closed=True),
        contour=contour,
    )


def test_circle_passes_default_thresholds():
    circle = cv2.ellipse2Poly((50, 50), (20, 20), 0, 0, 360, 5)
    metrics = _metrics(circle)

    assert metrics.circularity > 0.85
    assert metrics.axis_ratio > 0.98
    assert passes_shape_filter(metrics, True, 0.75, 0.80)


def test_elongated_particle_fails_axis_ratio_threshold():
    ellipse = cv2.ellipse2Poly((50, 50), (30, 12), 0, 0, 360, 5)
    metrics = _metrics(ellipse)

    assert metrics.axis_ratio < 0.50
    assert not passes_shape_filter(metrics, True, 0.60, 0.80)


def test_four_point_rectangle_uses_moments_instead_of_ellipse_fit():
    rectangle = [(0, 0), (40, 0), (40, 10), (0, 10)]
    metrics = _metrics(rectangle)

    assert math.isclose(metrics.axis_ratio, 0.25, abs_tol=0.01)
    assert not passes_shape_filter(metrics, True, 0.50, 0.80)


def test_degenerate_contour_scores_zero():
    metrics = _metrics([(0, 0), (10, 0), (20, 0)])

    assert metrics == ShapeMetrics(circularity=0.0, axis_ratio=0.0)
    assert not passes_shape_filter(metrics, True, 0.0, 0.0)


def test_disabled_filter_preserves_particle():
    metrics = ShapeMetrics(circularity=0.0, axis_ratio=0.0)

    assert passes_shape_filter(metrics, False, 1.0, 1.0)


def test_thresholds_are_inclusive():
    metrics = ShapeMetrics(circularity=0.75, axis_ratio=0.80)

    assert passes_shape_filter(metrics, True, 0.75, 0.80)


def _analysis_config(points, spherical_filter_enabled=True):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    return {
        "points": points,
        "output_image": image,
        "gray_image": np.zeros((100, 100), dtype=np.uint8),
        "thickness": 1,
        "particle_data": [],
        "annotations": [],
        "raw_mask": np.ones((100, 100), dtype=np.uint8),
        "particle_counter": 1,
        "show_fillPoly": False,
        "show_polylines": False,
        "show_Feret_diametr": False,
        "fill_type_color": "Random",
        "fill_color": "rgb(0, 255, 0, 1)",
        "fill_alpha": 0.3,
        "outline_color": "rgb(0, 255, 0, 1)",
        "scale_input": 1.0,
        "scale": 1.0,
        "scale_factor_glob": 1.0,
        "scale_selector": {
            "scale": False,
            "correction_factor": 1.0,
            "unit": "px",
        },
        "round_value": 4,
        "spherical_filter_enabled": spherical_filter_enabled,
        "min_circularity": 0.75,
        "min_axis_ratio": 0.80,
    }


def test_rejected_particle_has_no_output_side_effects():
    analyzer = ParticleAnalyzer.__new__(ParticleAnalyzer)
    analyzer.lang = "en"
    analyzer.default_lang = "en"
    config = _analysis_config([(5, 5), (85, 5), (85, 20), (5, 20)])
    original_image = config["output_image"].copy()

    next_counter = analyzer._analyze_particle(**config)

    assert next_counter == 1
    assert config["particle_data"] == []
    assert config["annotations"] == []
    np.testing.assert_array_equal(config["output_image"], original_image)


def test_accepted_particle_is_numbered_and_exported_once():
    analyzer = ParticleAnalyzer.__new__(ParticleAnalyzer)
    analyzer.lang = "en"
    analyzer.default_lang = "en"
    circle = cv2.ellipse2Poly((50, 50), (20, 20), 0, 0, 360, 5)
    config = _analysis_config(circle)

    next_counter = analyzer._analyze_particle(**config)

    assert next_counter == 2
    assert len(config["particle_data"]) == 1
    assert len(config["annotations"]) == 1
    assert config["particle_data"][0]["Circularity"] >= 0.75
    assert config["particle_data"][0]["Axis ratio"] >= 0.80
