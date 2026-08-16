import math
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from particleanalyzer.core.ParticleAnalyzer import ParticleAnalyzer
from particleanalyzer.core.nanorod_chart import build_nanorod_aspect_ratio_chart
from particleanalyzer.core.utils import reset_interface, reset_interface2, statistic_an


def _measurements():
    return pd.DataFrame(
        {
            "№": [1, 2, 3],
            "Rod length [nm]": [40.0, 60.0, 80.0],
            "Rod width [nm]": [20.0, 20.0, 20.0],
            "Nanorod aspect ratio (L/W)": [2.0, 3.0, 4.0],
            "Rod orientation [°]": [0.0, 45.0, 90.0],
            "Border touching": [False, True, False],
        }
    )


def test_chart_uses_canonical_aspect_ratio_and_dimensions():
    figure = build_nanorod_aspect_ratio_chart(_measurements(), 10, 2.5, "en")

    histogram = next(trace for trace in figure.data if trace.type == "histogram")
    interior = next(trace for trace in figure.data if trace.name == "Interior rods")
    border = next(trace for trace in figure.data if trace.name == "Border-touching rods")

    np.testing.assert_allclose(histogram.x, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(interior.x, [20.0, 20.0])
    np.testing.assert_allclose(interior.y, [40.0, 80.0])
    np.testing.assert_allclose(interior.marker.color, [2.0, 4.0])
    np.testing.assert_allclose(border.x, [20.0])
    np.testing.assert_allclose(border.y, [60.0])
    assert border.marker.symbol == "triangle-up-open"


def test_cutoff_ray_has_the_selected_l_over_w_slope():
    figure = build_nanorod_aspect_ratio_chart(_measurements(), 10, 2.5, "en")
    cutoff = next(trace for trace in figure.data if trace.name == "Minimum L/W = 2.50")

    assert math.isclose(cutoff.y[-1] / cutoff.x[-1], 2.5)


@pytest.mark.parametrize("language", ["en", "ru", "zh-cn", "zh-tw"])
def test_chart_labels_are_localized_without_changing_dataframe_keys(language):
    dataframe = _measurements()
    figure = build_nanorod_aspect_ratio_chart(dataframe, 10, 2.0, language)

    assert "Nanorod aspect ratio (L/W)" in dataframe.columns
    assert figure.layout.height == 570
    assert figure.layout.title.text
    assert len(figure.layout.annotations) >= 3


@pytest.mark.parametrize(
    "dataframe",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame({"Nanorod aspect ratio (L/W)": [2.0]}),
        pd.DataFrame(
            {
                "Rod length [px]": [np.nan, 10.0],
                "Rod width [px]": [2.0, 0.0],
                "Nanorod aspect ratio (L/W)": [np.nan, np.inf],
            }
        ),
    ],
)
def test_empty_or_invalid_data_returns_an_informative_chart(dataframe):
    figure = build_nanorod_aspect_ratio_chart(dataframe, 10, 2.0, "en")

    assert len(figure.data) == 0
    assert any(
        "No nanorod measurements available" in annotation.text
        for annotation in figure.layout.annotations
    )


def test_single_constant_measurement_renders_without_spread_errors():
    dataframe = _measurements().iloc[[0]].copy()

    figure = build_nanorod_aspect_ratio_chart(dataframe, 10, 2.0, "en")

    assert any(trace.type == "histogram" for trace in figure.data)
    assert "± 0.000" in figure.layout.title.text


def test_analysis_error_tuple_includes_the_nanorod_chart_output():
    analyzer = ParticleAnalyzer.__new__(ParticleAnalyzer)

    assert len(analyzer._create_error_return()) == 20


def test_reset_and_filter_callbacks_include_the_nanorod_chart_output():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert len(reset_interface()) == 16
        assert len(reset_interface2()) == 10

    empty_arguments = {
        parameter: None for parameter in inspect.signature(statistic_an).parameters
    }
    assert statistic_an(**empty_arguments) == (None, None, None, None, None)


def test_filter_callback_skips_refresh_until_contours_are_available():
    arguments = {
        parameter: None for parameter in inspect.signature(statistic_an).parameters
    }
    arguments.update(
        {
            "df": pd.DataFrame({"№": [1]}),
            "points_df": None,
            "in_image": np.zeros((10, 10, 3), dtype=np.uint8),
        }
    )

    result = statistic_an(**arguments)

    assert result == tuple({"__type__": "update"} for _ in range(5))
