import json

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import pytest

from particleanalyzer.core.manual_contour_editor import (
    PolygonValidationError,
    mutate_particle_tables,
    normalize_polygon,
    render_editor_preview,
    write_contour_exports,
)
from particleanalyzer.core.ParticleAnalyzer import ParticleAnalyzer
from particleanalyzer.core.utils import (
    capture_analysis_edit_context,
    particle_removal,
)

IMAGE_SHAPE = (32, 48, 3)
ID_COLUMN = "\N{NUMERO SIGN}"


def _normalized(points):
    return normalize_polygon(points, IMAGE_SHAPE)


def _point_array(points):
    return np.asarray(points, dtype=np.int32).reshape((-1, 2))


def _initial_tables():
    first = _normalized([(3, 3), (12, 3), (12, 8), (3, 8)])
    second = _normalized([(20, 10), (31, 10), (31, 16), (20, 16)])
    output_table = pd.DataFrame(
        {
            ID_COLUMN: [1, 2],
            "Measurement": [10.0, 20.0],
        }
    )
    points_df = pd.DataFrame({"points": [first.tolist(), second.tolist()]})
    return output_table, points_df


def _assert_tables_aligned(output_table, points_df):
    assert output_table.iloc[:, 0].tolist() == list(range(1, len(output_table) + 1))
    assert len(points_df) == len(output_table)
    assert (
        list(output_table.index)
        == list(points_df.index)
        == list(range(len(output_table)))
    )


def test_analysis_edit_context_uses_the_completed_analysis_snapshot():
    points_df = pd.DataFrame({"points": []})
    points_df.attrs["analysis_edit_context"] = {
        "scale": 62.0,
        "scale_input": 4.0,
        "scale_selector": "Micrometers (µm)",
        "solution": "1280x1280",
        "sahi_mode": True,
        "round_value": 5,
    }

    assert capture_analysis_edit_context(points_df) == (
        62.0,
        4.0,
        "Micrometers (µm)",
        "1280x1280",
        True,
        5,
    )


def test_normalize_polygon_returns_an_opencv_contour():
    points = [(2, 4), (15, 4), (15, 11), (2, 11)]

    contour = normalize_polygon(points, IMAGE_SHAPE)

    assert contour.dtype == np.int32
    assert contour.shape == (4, 1, 2)
    np.testing.assert_array_equal(contour.reshape((-1, 2)), points)


@pytest.mark.parametrize(
    "points",
    [
        [(2, 2), (10, 2)],
        [(2, 2), (8, 2), (14, 2)],
        [(2, 2), (12, 12), (2, 12), (12, 2)],
        [(2, 2), (48, 2), (2, 12)],
    ],
    ids=["fewer-than-three", "collinear", "self-intersecting", "out-of-bounds"],
)
def test_normalize_polygon_rejects_invalid_geometry(points):
    with pytest.raises(PolygonValidationError):
        normalize_polygon(points, IMAGE_SHAPE)


@pytest.mark.parametrize(
    "points",
    [
        [(2, 2), (10, 2)],
        [(2, 2), (8, 2), (14, 2)],
        [(2, 2), (12, 12), (2, 12), (12, 2)],
        [(-1, 2), (8, 2), (2, 12)],
    ],
)
def test_invalid_polygon_validation_is_atomic_for_particle_tables(points):
    output_table, points_df = _initial_tables()
    original_output = output_table.copy(deep=True)
    original_points = points_df.copy(deep=True)

    with pytest.raises(PolygonValidationError):
        normalized = normalize_polygon(points, IMAGE_SHAPE)
        mutate_particle_tables(
            "add",
            {
                ID_COLUMN: 99,
                "Measurement": 99.0,
                "points": normalized.tolist(),
            },
            [],
            output_table,
            points_df,
        )

    pd.testing.assert_frame_equal(output_table, original_output)
    pd.testing.assert_frame_equal(points_df, original_points)


def test_add_replace_and_delete_keep_tables_aligned_and_ids_contiguous():
    output_table, points_df = _initial_tables()
    original_output = output_table.copy(deep=True)
    original_points = points_df.copy(deep=True)
    added = _normalized([(34, 4), (43, 4), (43, 9), (34, 9)])

    output_table, points_df = mutate_particle_tables(
        "add",
        {ID_COLUMN: 47, "Measurement": 30.0, "points": added.tolist()},
        [],
        output_table,
        points_df,
    )

    expected_output, expected_points = _initial_tables()
    pd.testing.assert_frame_equal(original_output, expected_output)
    pd.testing.assert_frame_equal(original_points, expected_points)
    _assert_tables_aligned(output_table, points_df)
    assert output_table["Measurement"].tolist() == [10.0, 20.0, 30.0]
    np.testing.assert_array_equal(
        _point_array(points_df.iloc[2]["points"]), added[:, 0]
    )

    replacement = _normalized([(18, 19), (34, 19), (34, 25), (18, 25)])
    output_table, points_df = mutate_particle_tables(
        "replace",
        {
            ID_COLUMN: 91,
            "Measurement": 200.0,
            "points": replacement.tolist(),
        },
        [2],
        output_table,
        points_df,
    )

    _assert_tables_aligned(output_table, points_df)
    assert output_table["Measurement"].tolist() == [10.0, 200.0, 30.0]
    np.testing.assert_array_equal(
        _point_array(points_df.iloc[1]["points"]), replacement[:, 0]
    )

    output_table, points_df = mutate_particle_tables(
        "delete",
        None,
        [1, 3],
        output_table,
        points_df,
    )

    _assert_tables_aligned(output_table, points_df)
    assert output_table["Measurement"].tolist() == [200.0]
    np.testing.assert_array_equal(
        _point_array(points_df.iloc[0]["points"]), replacement[:, 0]
    )


def test_render_editor_preview_does_not_mutate_the_base_image():
    base_image = np.full(IMAGE_SHAPE, 37, dtype=np.uint8)
    original = base_image.copy()
    output_table, points_df = _initial_tables()
    draft = [(6, 20), (13, 27), (3, 27)]

    preview = render_editor_preview(
        base_image,
        points_df,
        draft,
        [2],
        output_table,
    )

    np.testing.assert_array_equal(base_image, original)
    assert preview is not base_image
    assert preview.shape == base_image.shape
    assert preview.dtype == base_image.dtype
    assert np.any(preview != base_image)


def test_render_editor_preview_with_no_contours_or_draft_is_a_clean_copy():
    base_image = np.full(IMAGE_SHAPE, 73, dtype=np.uint8)
    empty_points = pd.DataFrame(columns=["points"])
    empty_output = pd.DataFrame(columns=[ID_COLUMN, "Measurement"])

    preview = render_editor_preview(
        base_image,
        empty_points,
        [],
        [],
        empty_output,
    )

    assert preview is not base_image
    np.testing.assert_array_equal(preview, base_image)


def test_write_contour_exports_creates_empty_mask_and_coco(tmp_path):
    points_df = pd.DataFrame(columns=["points"])

    mask_path, coco_path = write_contour_exports(
        points_df,
        IMAGE_SHAPE,
        "empty_manual",
        tmp_path,
    )

    assert mask_path == str(tmp_path / "binary_mask_empty_manual.png")
    assert coco_path == str(tmp_path / "coco_file_empty_manual.json")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    mask = np.squeeze(mask)
    assert mask.shape == IMAGE_SHAPE[:2]
    assert np.count_nonzero(mask) == 0
    with open(coco_path, encoding="utf-8") as coco_file:
        coco = json.load(coco_file)
    assert coco["images"] == [
        {
            "id": 1,
            "file_name": "empty_manual.png",
            "width": IMAGE_SHAPE[1],
            "height": IMAGE_SHAPE[0],
        }
    ]
    assert coco["annotations"] == []


def test_write_contour_exports_uses_polygon_area_not_bbox_area(tmp_path):
    triangle = _normalized([(2, 2), (10, 2), (2, 8)])
    points_df = pd.DataFrame({"points": [triangle.tolist()]})

    mask_path, coco_path = write_contour_exports(
        points_df,
        IMAGE_SHAPE,
        "triangle_manual",
        tmp_path,
    )

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.squeeze(mask)
    expected_mask = np.zeros(IMAGE_SHAPE[:2], dtype=np.uint8)
    cv2.fillPoly(expected_mask, [triangle], 255)
    np.testing.assert_array_equal(mask, expected_mask)

    with open(coco_path, encoding="utf-8") as coco_file:
        coco = json.load(coco_file)
    assert [annotation["id"] for annotation in coco["annotations"]] == [1]
    annotation = coco["annotations"][0]
    expected_area = float(cv2.contourArea(triangle))
    assert annotation["area"] == pytest.approx(expected_area)
    assert annotation["area"] != pytest.approx(
        annotation["bbox"][2] * annotation["bbox"][3]
    )
    assert annotation["segmentation"] == [[2.0, 2.0, 10.0, 2.0, 2.0, 8.0]]


def _bare_analyzer(tmp_path):
    analyzer = ParticleAnalyzer.__new__(ParticleAnalyzer)
    analyzer.default_lang = "en"
    analyzer.output_dir = str(tmp_path)
    return analyzer


def _measure_pixels(analyzer, points, particle_number=1, exclude_border=False):
    return analyzer.measure_manual_contour(
        points=points,
        particle_number=particle_number,
        in_image=np.full((100, 100, 3), 128, dtype=np.uint8),
        scale=1.0,
        scale_input=1.0,
        scale_selector="Pixels",
        solution="Original",
        sahi_mode=False,
        round_value=4,
        exclude_border_particles=exclude_border,
        selected_language="en",
    )


def test_manual_measurement_uses_normal_metrics_and_independent_border_qc(tmp_path):
    analyzer = _bare_analyzer(tmp_path)
    interior = _measure_pixels(
        analyzer,
        [(10, 40), (70, 40), (70, 60), (10, 60)],
    )
    border = _measure_pixels(
        analyzer,
        [(0, 40), (60, 40), (60, 60), (0, 60)],
        exclude_border=True,
    )

    assert interior["Nanorod aspect ratio (L/W)"] == pytest.approx(3.0)
    assert interior["Border touching"] is False
    assert border is None


def test_apply_manual_add_remeasures_and_rewrites_geometry_exports(tmp_path):
    analyzer = _bare_analyzer(tmp_path)
    source_image = np.full((100, 100, 3), 128, dtype=np.uint8)
    seed = _measure_pixels(
        analyzer,
        [(10, 40), (40, 40), (40, 50), (10, 50)],
    )
    output_table = pd.DataFrame(
        [{key: value for key, value in seed.items() if key != "points"}]
    )
    points_df = pd.DataFrame({"points": [seed["points"]]})

    result = analyzer.apply_manual_contour_edit(
        "add",
        [(50, 40), (90, 40), (90, 50), (50, 50)],
        [],
        points_df,
        output_table,
        source_image,
        1.0,
        1.0,
        "Pixels",
        "Original",
        False,
        4,
        False,
        "manual_edit",
        "en",
    )

    updated_table, updated_points = result[:2]
    assert updated_table.iloc[:, 0].tolist() == [1, 2]
    assert updated_table["Nanorod aspect ratio (L/W)"].tolist() == [3.0, 4.0]
    assert len(updated_points) == 2
    assert result[2] == []
    assert result[6] == []
    assert result[7] == "idle"
    assert (tmp_path / "binary_mask_manual_edit.png").exists()
    with open(tmp_path / "coco_file_manual_edit.json", encoding="utf-8") as file:
        coco = json.load(file)
    assert [annotation["id"] for annotation in coco["annotations"]] == [1, 2]


def test_invalid_manual_apply_is_atomic_and_keeps_the_draft(tmp_path):
    analyzer = _bare_analyzer(tmp_path)
    source_image = np.full((100, 100, 3), 128, dtype=np.uint8)
    seed = _measure_pixels(
        analyzer,
        [(10, 40), (40, 40), (40, 50), (10, 50)],
    )
    output_table = pd.DataFrame(
        [{key: value for key, value in seed.items() if key != "points"}]
    )
    points_df = pd.DataFrame({"points": [seed["points"]]})
    original_table = output_table.copy(deep=True)
    original_points = points_df.copy(deep=True)

    with pytest.raises(gr.Error):
        analyzer.apply_manual_contour_edit(
            "add",
            [(10, 10), (20, 20)],
            [],
            points_df,
            output_table,
            source_image,
            1.0,
            1.0,
            "Pixels",
            "Original",
            False,
            4,
            False,
            "manual_invalid",
            "en",
        )

    pd.testing.assert_frame_equal(output_table, original_table)
    pd.testing.assert_frame_equal(points_df, original_points)
    assert not (tmp_path / "binary_mask_manual_invalid.png").exists()


def test_delete_selected_particle_renumbers_and_rewrites_exports(tmp_path, monkeypatch):
    analyzer = _bare_analyzer(tmp_path)
    source_image = np.full((100, 100, 3), 128, dtype=np.uint8)
    records = [
        _measure_pixels(
            analyzer,
            [(10, 20), (40, 20), (40, 30), (10, 30)],
            particle_number=1,
        ),
        _measure_pixels(
            analyzer,
            [(50, 60), (90, 60), (90, 70), (50, 70)],
            particle_number=2,
        ),
    ]
    output_table = pd.DataFrame(
        [
            {key: value for key, value in record.items() if key != "points"}
            for record in records
        ]
    )
    points_df = pd.DataFrame({"points": [record["points"] for record in records]})
    selected = output_table.iloc[[0]]
    monkeypatch.chdir(tmp_path)

    result = particle_removal(
        selected,
        points_df,
        output_table,
        4,
        "Pixels",
        selected_language="en",
        in_image=source_image,
        solution="Original",
        sahi_mode=False,
        image_name="manual_delete",
    )

    cleared_selection = result[0]
    updated_points = result[4]
    updated_table = result[5]
    assert cleared_selection.empty
    assert updated_table.iloc[:, 0].tolist() == [1]
    assert len(updated_points) == 1
    with open(
        tmp_path / "output" / "coco_file_manual_delete.json", encoding="utf-8"
    ) as file:
        coco = json.load(file)
    assert [annotation["id"] for annotation in coco["annotations"]] == [1]

    second_result = particle_removal(
        cleared_selection,
        updated_points,
        updated_table,
        4,
        "Pixels",
        selected_language="en",
        in_image=source_image,
        solution="Original",
        sahi_mode=False,
        image_name="manual_delete",
    )
    pd.testing.assert_frame_equal(second_result[4], updated_points)
    pd.testing.assert_frame_equal(second_result[5], updated_table)
