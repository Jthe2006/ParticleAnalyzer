import math
from unittest.mock import Mock

import cv2
import numpy as np
import pandas as pd
import pytest

from particleanalyzer.core.nanorod_analysis import (
    NanorodMetrics,
    assess_nanrod_overlaps,
    calculate_nanorod_metrics,
    contour_touches_image_border,
    passes_nanorod_filter,
)
from particleanalyzer.core.ParticleAnalyzer import ParticleAnalyzer
from particleanalyzer.core.StatisticsBuilder import StatisticsBuilder


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
        "exclude_overlapping_rods": False,
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


def test_border_exclusion_is_independent_of_shape_analysis_mode():
    border_particle = [(0, 30), (30, 30), (30, 60), (0, 60)]
    rejected = _analysis_config(
        border_particle,
        nanorod_mode_enabled=False,
        spherical_filter_enabled=False,
        exclude_border_rods=True,
    )
    retained = _analysis_config(
        border_particle,
        nanorod_mode_enabled=False,
        spherical_filter_enabled=False,
        exclude_border_rods=False,
    )

    assert _analyzer()._analyze_particle(**rejected) == 1
    assert rejected["particle_data"] == []
    assert _analyzer()._analyze_particle(**retained) == 2
    assert retained["particle_data"][0]["Border touching"] is True


def test_border_exclusion_precedes_spherical_filter():
    border_circle = cv2.ellipse2Poly((10, 50), (10, 10), 0, 0, 360, 5)
    config = _analysis_config(
        border_circle,
        nanorod_mode_enabled=False,
        spherical_filter_enabled=True,
        exclude_border_rods=True,
    )

    assert _analyzer()._analyze_particle(**config) == 1
    assert config["particle_data"] == []


def _axis_aligned_rod(left, top, length=30, width=10):
    return np.array(
        [
            (left, top),
            (left + length, top),
            (left + length, top + width),
            (left, top + width),
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))


def _overlap_qc_candidates():
    reference_rods = [
        _axis_aligned_rod(10 + 55 * (index % 5), 10 + 45 * (index // 5))
        for index in range(10)
    ]
    duplicate = reference_rods[0].copy()
    oversized_parent = _axis_aligned_rod(300, 70, length=80, width=30)
    contained_rod = _axis_aligned_rod(325, 80)
    return reference_rods + [duplicate, oversized_parent, contained_rod]


def _parallel_pair_fixture(doublet: bool):
    gray_image = np.full((240, 400), 190, dtype=np.uint8)
    references = [
        _axis_aligned_rod(10 + 65 * (index % 5), 10 + 45 * (index // 5), 42, 7)
        for index in range(10)
    ]
    for reference in references:
        cv2.fillPoly(gray_image, [reference], 100)

    candidate = _axis_aligned_rod(300, 150, 42, 14)
    if doublet:
        cv2.rectangle(gray_image, (300, 150), (342, 156), 110, thickness=-1)
        cv2.rectangle(gray_image, (300, 158), (342, 164), 115, thickness=-1)
    else:
        cv2.fillPoly(gray_image, [candidate], 110)
    return references + [candidate], gray_image


def _rotated_parallel_pair_fixture(angle):
    height, width, scale = 260, 450, 2
    high_resolution = np.full(
        (height * scale, width * scale), 190, dtype=np.uint8
    )
    references = [
        _axis_aligned_rod(10 + 65 * (index % 5), 10 + 45 * (index // 5), 42, 7)
        for index in range(10)
    ]
    for reference in references:
        scaled_reference = np.rint(
            reference.reshape((-1, 2)) * scale
        ).astype(np.int32)
        cv2.fillPoly(high_resolution, [scaled_reference], 100)

    center = np.asarray((370.0, 170.0))
    radians = math.radians(angle)
    normal = np.asarray((-math.sin(radians), math.cos(radians)))
    candidate = cv2.boxPoints((tuple(center), (42.0, 14.0), angle)).reshape(
        (-1, 1, 2)
    )
    for offset, intensity in ((-4.0, 110), (4.0, 115)):
        rod_center = tuple((center + offset * normal) * scale)
        rod = cv2.boxPoints(
            (rod_center, (42.0 * scale, 6.0 * scale), angle)
        )
        cv2.fillPoly(high_resolution, [np.rint(rod).astype(np.int32)], intensity)
    high_resolution = cv2.GaussianBlur(high_resolution, (0, 0), 1.2 * scale)
    gray_image = cv2.resize(
        high_resolution, (width, height), interpolation=cv2.INTER_AREA
    )
    return references + [candidate], gray_image


def test_overlap_qc_removes_duplicate_and_oversized_parent_but_keeps_rods():
    contours = _overlap_qc_candidates()
    assessments = assess_nanrod_overlaps(
        contours,
        image_shape=(220, 420, 3),
        confidences=[0.9] * 10 + [0.1, 0.95, 0.8],
    )

    assert all(not assessment.exclude for assessment in assessments[:10])
    assert assessments[10].exclude
    assert assessments[10].duplicate
    assert assessments[10].reason == "duplicate_mask"
    assert assessments[11].exclude
    assert assessments[11].oversized
    assert "oversized_cluster" in assessments[11].reason
    assert not assessments[12].exclude


def test_containment_qc_works_when_too_few_rods_exist_for_size_reference():
    parent = _axis_aligned_rod(20, 20, length=80, width=30)
    child = _axis_aligned_rod(45, 30)

    assessments = assess_nanrod_overlaps(
        [parent, child],
        image_shape=(100, 140, 3),
        min_reference_count=100,
    )

    assert assessments[0].exclude
    assert assessments[0].reason == "contained_parent"
    assert not assessments[1].exclude


def test_containment_keeps_valid_child_over_higher_confidence_invalid_parent():
    parent = _axis_aligned_rod(20, 20, length=31, width=17)
    child = _axis_aligned_rod(20, 23, length=30, width=10)

    assessments = assess_nanrod_overlaps(
        [parent, child],
        image_shape=(80, 100, 3),
        confidences=[0.95, 0.80],
        min_aspect_ratio=2.0,
        min_reference_count=100,
    )

    assert assessments[0].exclude
    assert assessments[0].reason == "contained_parent"
    assert not assessments[1].exclude


def test_isolated_long_rod_is_not_mislabeled_as_an_overlapping_cluster():
    references = [
        _axis_aligned_rod(10 + 55 * (index % 5), 10 + 45 * (index // 5))
        for index in range(10)
    ]
    long_rod = _axis_aligned_rod(10, 140, length=150, width=10)

    assessments = assess_nanrod_overlaps(
        references + [long_rod],
        image_shape=(200, 300, 3),
    )

    assert all(not assessment.exclude for assessment in assessments)


def test_moderately_large_wide_concave_cluster_is_excluded():
    references = [
        _axis_aligned_rod(10 + 55 * (index % 5), 10 + 45 * (index // 5))
        for index in range(10)
    ]
    cluster = np.asarray(
        [
            (300, 60),
            (360, 60),
            (360, 78),
            (338, 78),
            (338, 68),
            (323, 68),
            (323, 78),
            (300, 78),
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    assessments = assess_nanrod_overlaps(
        references + [cluster],
        image_shape=(180, 400, 3),
    )

    assert assessments[-1].exclude
    assert assessments[-1].reason == "oversized_cluster"


def test_parallel_grayscale_bands_confirm_a_side_by_side_merged_mask():
    contours, gray_image = _parallel_pair_fixture(doublet=True)

    assessments = assess_nanrod_overlaps(
        contours,
        image_shape=gray_image.shape,
        gray_image=gray_image,
    )

    assert assessments[-1].exclude
    assert assessments[-1].reason == "parallel_cluster"


def test_one_thick_grayscale_band_is_not_mislabeled_as_parallel_rods():
    contours, gray_image = _parallel_pair_fixture(doublet=False)

    assessments = assess_nanrod_overlaps(
        contours,
        image_shape=gray_image.shape,
        gray_image=gray_image,
    )

    assert not assessments[-1].exclude


@pytest.mark.parametrize("angle", [-80, -45, -20, 0, 20, 45, 80])
def test_parallel_grayscale_detection_is_rotation_invariant(angle):
    contours, gray_image = _rotated_parallel_pair_fixture(angle)

    assessments = assess_nanrod_overlaps(
        contours,
        image_shape=gray_image.shape,
        gray_image=gray_image,
    )

    assert assessments[-1].exclude
    assert assessments[-1].reason == "parallel_cluster"


def test_spatial_sweep_ignores_vertical_nonoverlaps_before_pair_budget():
    rods = [_axis_aligned_rod(10, 20 * index) for index in range(40)]
    duplicate = rods[-1].copy()

    assessments = assess_nanrod_overlaps(
        rods + [duplicate],
        image_shape=(850, 60, 3),
        confidences=[0.9] * len(rods) + [0.1],
        min_reference_count=100,
        max_pair_checks=1,
    )

    assert assessments[-1].exclude
    assert assessments[-1].duplicate
    assert not assessments[-1].pair_limit_reached


def test_overlap_assessment_reports_when_pair_budget_is_exhausted():
    rod = _axis_aligned_rod(10, 10)

    assessments = assess_nanrod_overlaps(
        [rod, rod.copy(), rod.copy()],
        image_shape=(50, 60, 3),
        min_reference_count=100,
        max_pair_checks=1,
    )

    assert all(assessment.pair_limit_reached for assessment in assessments)


def test_sahi_multipolygons_are_filled_without_artificial_bridge():
    polygons = [
        [2, 2, 6, 2, 6, 6, 2, 6],
        [14, 2, 18, 2, 18, 6, 14, 6],
    ]

    mask = _analyzer()._sahi_polygon_to_binary_mask(polygons, height=10, width=22)

    assert mask[4, 4] == 1
    assert mask[4, 16] == 1
    assert mask[4, 10] == 0


def test_batch_overlap_filter_has_no_export_or_drawing_side_effects(monkeypatch):
    contours = _overlap_qc_candidates()
    image = np.zeros((220, 420, 3), dtype=np.uint8)
    config = _analysis_config(
        [],
        output_image=image,
        gray_image=np.zeros(image.shape[:2], dtype=np.uint8),
        exclude_border_rods=False,
        exclude_overlapping_rods=True,
    )
    candidates = []
    for index, contour in enumerate(contours):
        raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(raw_mask, [contour], 1)
        candidates.append(
            {
                "points": contour,
                "raw_mask": raw_mask,
                "confidence": 0.1 if index == 10 else 0.9,
            }
        )
    info = Mock()
    monkeypatch.setattr("gradio.Info", info)

    particle_data, annotations = _analyzer()._process_particle_candidates(
        candidates,
        image,
        thickness=1,
        config=config,
    )

    assert len(particle_data) == 11
    assert len(annotations) == 11
    assert [row["№"] for row in particle_data] == list(range(1, 12))
    info.assert_called_once()


def test_batch_overlap_filter_can_be_disabled():
    contours = _overlap_qc_candidates()
    image = np.zeros((220, 420, 3), dtype=np.uint8)
    config = _analysis_config(
        [],
        output_image=image,
        gray_image=np.zeros(image.shape[:2], dtype=np.uint8),
        exclude_border_rods=False,
        exclude_overlapping_rods=False,
    )
    candidates = []
    for contour in contours:
        raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(raw_mask, [contour], 1)
        candidates.append(
            {"points": contour, "raw_mask": raw_mask, "confidence": 0.9}
        )

    particle_data, annotations = _analyzer()._process_particle_candidates(
        candidates,
        image,
        thickness=1,
        config=config,
    )

    assert len(particle_data) == len(contours)
    assert len(annotations) == len(contours)


def test_overlap_filter_operates_without_nanorod_or_spherical_mode(monkeypatch):
    contour = _axis_aligned_rod(20, 20)
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    config = _analysis_config(
        [],
        output_image=image,
        gray_image=np.zeros(image.shape[:2], dtype=np.uint8),
        nanorod_mode_enabled=False,
        spherical_filter_enabled=False,
        exclude_overlapping_rods=True,
    )
    candidates = []
    for confidence in (0.9, 0.1):
        raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(raw_mask, [contour], 1)
        candidates.append(
            {
                "points": contour,
                "raw_mask": raw_mask,
                "confidence": confidence,
            }
        )
    monkeypatch.setattr("gradio.Info", Mock())

    particle_data, annotations = _analyzer()._process_particle_candidates(
        candidates,
        image,
        thickness=1,
        config=config,
    )

    assert len(particle_data) == 1
    assert len(annotations) == 1


def test_overlap_qc_only_compares_candidates_that_pass_spherical_filter():
    circle = cv2.ellipse2Poly((50, 50), (10, 10), 0, 0, 360, 5).reshape(
        (-1, 1, 2)
    )
    elongated_parent = _axis_aligned_rod(34, 42, length=32, width=16)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    config = _analysis_config(
        [],
        output_image=image,
        gray_image=np.zeros(image.shape[:2], dtype=np.uint8),
        nanorod_mode_enabled=False,
        spherical_filter_enabled=True,
        exclude_overlapping_rods=True,
    )
    candidates = []
    for contour, confidence in ((elongated_parent, 0.99), (circle, 0.50)):
        raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(raw_mask, [contour], 1)
        candidates.append(
            {
                "points": contour,
                "raw_mask": raw_mask,
                "confidence": confidence,
            }
        )

    particle_data, annotations = _analyzer()._process_particle_candidates(
        candidates,
        image,
        thickness=1,
        config=config,
    )

    assert len(particle_data) == 1
    assert len(annotations) == 1
    assert particle_data[0]["Axis ratio"] >= 0.8


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
