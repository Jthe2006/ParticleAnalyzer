import math

import cv2
import numpy as np
import pandas as pd
import pytest

from particleanalyzer.core.ParticleAnalyzer import ParticleAnalyzer
from particleanalyzer.core.StatisticsBuilder import StatisticsBuilder
from particleanalyzer.core.nanorod_analysis import (
    NanorodMetrics,
    calculate_nanorod_metrics,
    contour_touches_image_border,
    passes_nanorod_filter,
)


def _rotated_rectangle(length=60.0, width=20.0, angle=0.0):
    return cv2.boxPoints(((75.0, 75.0), (length, width), angle)).reshape((-1, 1, 2))


def _angle_error(first, second):
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def test_orthogonal_calipers_do_not_use_rectangle_diagonal_as_length():
    metrics = calculate_nanorod_metrics(_rotated_rectangle())

    assert math.isclose(metrics.length, 60.0, abs_tol=1e-4)
    assert math.isclose(metrics.width, 20.0, abs_tol=1e-4)
    assert math.isclose(metrics.aspect_ratio, 3.0, abs_tol=1e-4)
    assert math.isclose(metrics.feret_max, math.hypot(60.0, 20.0), abs_tol=1e-4)
    assert metrics.feret_max > metrics.length


def test_dimensions_and_orientation_are_rotation_invariant():
    for angle in (0.0, 17.0, 45.0, 83.0, 137.0):
        metrics = calculate_nanorod_metrics(
            _rotated_rectangle(length=80.0, width=16.0, angle=angle)
        )

        assert math.isclose(metrics.length, 80.0, abs_tol=2e-4)
        assert math.isclose(metrics.width, 16.0, abs_tol=2e-4)
        assert math.isclose(metrics.aspect_ratio, 5.0, abs_tol=2e-4)
        assert _angle_error(metrics.orientation, angle % 180.0) < 1e-4


def test_round_particle_has_unit_aspect_ratio():
    circle = cv2.ellipse2Poly((50, 50), (20, 20), 0, 0, 360, 2)
    metrics = calculate_nanorod_metrics(circle)

    assert math.isclose(metrics.aspect_ratio, 1.0, abs_tol=0.01)
    assert math.isclose(metrics.iso_feret_ratio, 1.0, abs_tol=0.04)


def test_degenerate_contour_returns_zero_metrics():
    metrics = calculate_nanorod_metrics([(0, 0), (10, 0), (20, 0)])

    assert metrics == NanorodMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_nanorod_filter_is_inclusive_and_applies_border_qc():
    metrics = calculate_nanorod_metrics(_rotated_rectangle())

    assert passes_nanorod_filter(metrics, 3.0, False, True)
    assert not passes_nanorod_filter(metrics, 3.01, False, True)
    assert not passes_nanorod_filter(metrics, 2.0, True, True)
    assert passes_nanorod_filter(metrics, 2.0, True, False)


def test_border_detection_uses_the_full_image_boundary():
    at_border = np.array([(0, 10), (20, 10), (20, 20), (0, 20)])
    interior = at_border + np.array((1, 0))

    assert contour_touches_image_border(at_border, (100, 100, 3))
    assert not contour_touches_image_border(interior, (100, 100, 3))


def _analysis_config(points, **overrides):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    config = {
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
        "spherical_filter_enabled": True,
        "min_circularity": 0.75,
        "min_axis_ratio": 0.80,
        "nanorod_mode_enabled": True,
        "min_nanorod_aspect_ratio": 2.0,
        "exclude_border_rods": True,
    }
    config.update(overrides)
    return config


def _analyzer(language="en"):
    analyzer = ParticleAnalyzer.__new__(ParticleAnalyzer)
    analyzer.lang = language
    analyzer.default_lang = "en"
    return analyzer


def test_nanorod_mode_overrides_near_circular_filter_and_exports_metrics():
    config = _analysis_config([(20, 40), (80, 40), (80, 60), (20, 60)])

    next_counter = _analyzer()._analyze_particle(**config)

    assert next_counter == 2
    assert len(config["annotations"]) == 1
    row = config["particle_data"][0]
    assert row["Nanorod aspect ratio (L/W)"] == 3.0
    assert row["Rod length [px]"] == 60.0
    assert row["Rod width [px]"] == 20.0
    assert row["Border touching"] is False


def test_round_particle_is_rejected_in_nanorod_mode_without_side_effects():
    circle = cv2.ellipse2Poly((50, 50), (20, 20), 0, 0, 360, 5)
    config = _analysis_config(circle)
    original_image = config["output_image"].copy()

    next_counter = _analyzer()._analyze_particle(**config)

    assert next_counter == 1
    assert config["particle_data"] == []
    assert config["annotations"] == []
    np.testing.assert_array_equal(config["output_image"], original_image)


def test_border_touching_rod_can_be_excluded_or_retained():
    border_rod = [(0, 40), (60, 40), (60, 60), (0, 60)]
    rejected = _analysis_config(border_rod)
    retained = _analysis_config(border_rod, exclude_border_rods=False)

    assert _analyzer()._analyze_particle(**rejected) == 1
    assert rejected["particle_data"] == []
    assert _analyzer()._analyze_particle(**retained) == 2
    assert retained["particle_data"][0]["Border touching"] is True


def test_scaling_changes_rod_dimensions_but_not_aspect_ratio():
    points = [(20, 40), (80, 40), (80, 60), (20, 60)]
    unscaled = _analysis_config(points)
    scaled = _analysis_config(points, scale_factor_glob=2.5)

    _analyzer()._analyze_particle(**unscaled)
    _analyzer()._analyze_particle(**scaled)

    row = unscaled["particle_data"][0]
    scaled_row = scaled["particle_data"][0]
    assert scaled_row["Rod length [px]"] == row["Rod length [px]"] * 2.5
    assert scaled_row["Rod width [px]"] == row["Rod width [px]"] * 2.5
    assert scaled_row["Nanorod aspect ratio (L/W)"] == row[
        "Nanorod aspect ratio (L/W)"
    ]


@pytest.mark.parametrize("language", ["en", "ru", "zh-cn", "zh-tw"])
def test_statistics_and_distribution_include_nanorod_ratios(language):
    rows = []
    for length in (40, 60, 80):
        config = _analysis_config(
            [(10, 40), (10 + length, 40), (10 + length, 60), (10, 60)],
            exclude_border_rods=False,
        )
        _analyzer(language)._analyze_particle(**config)
        rows.extend(config["particle_data"])

    dataframe = pd.DataFrame(rows).drop(
        columns=["points", "centroid_x", "centroid_y"]
    )
    builder = StatisticsBuilder(
        dataframe,
        _analysis_config([])["scale_selector"],
        round_value=4,
        number_of_bins=5,
        lang=language,
    )

    stats = builder.build_stats_table().data
    figure, _ = builder.build_distribution_fig(np.zeros((100, 100, 3)))

    parameter_column = builder._get_translation("Параметр")
    assert "Nanorod aspect ratio (L/W)" in stats[parameter_column].values
    assert "ISO Feret ratio (min/max)" in stats[parameter_column].values
    assert figure.layout.height == 1380
    plot_titles = [annotation.text for annotation in figure.layout.annotations]
    assert builder._get_translation(
        "Nanorod aspect ratio (L/W) distribution"
    ) in plot_titles
    if language != "en":
        assert builder._get_translation("Nanorod aspect ratio (L/W)") not in (
            dataframe.columns
        )
