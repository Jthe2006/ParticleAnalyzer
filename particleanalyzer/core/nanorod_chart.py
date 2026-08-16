"""Plotly dashboard for projected nanorod aspect-ratio results."""

import math
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from particleanalyzer.core.languages import translations


def build_nanorod_aspect_ratio_chart(
    dataframe: Optional[pd.DataFrame],
    number_of_bins: int,
    min_aspect_ratio: float,
    lang: str = "en",
) -> go.Figure:
    """Build an L/W histogram and rod-length-versus-width scatter dashboard."""

    translate = lambda text: translations.get(lang, {}).get(text, text)
    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.42, 0.58],
        subplot_titles=(
            translate("Aspect-ratio distribution"),
            translate("Rod length versus width"),
        ),
        horizontal_spacing=0.12,
    )

    required_ratio_column = "Nanorod aspect ratio (L/W)"
    length_column = _find_column(dataframe, "Rod length [")
    width_column = _find_column(dataframe, "Rod width [")
    if (
        dataframe is None
        or dataframe.empty
        or required_ratio_column not in dataframe.columns
        or length_column is None
        or width_column is None
    ):
        return _add_empty_state(figure, translate)

    measurements = pd.DataFrame(index=dataframe.index)
    measurements["length"] = pd.to_numeric(
        dataframe[length_column], errors="coerce"
    )
    measurements["width"] = pd.to_numeric(dataframe[width_column], errors="coerce")
    measurements["aspect_ratio"] = pd.to_numeric(
        dataframe[required_ratio_column], errors="coerce"
    )
    measurements["particle"] = dataframe.iloc[:, 0].astype(str)
    measurements["orientation"] = pd.to_numeric(
        dataframe.get("Rod orientation [°]", np.nan), errors="coerce"
    )
    measurements["border"] = _border_status(dataframe)

    finite_values = np.isfinite(
        measurements[["length", "width", "aspect_ratio"]]
    ).all(axis=1)
    valid_values = (
        finite_values
        & (measurements["length"] > 0.0)
        & (measurements["width"] > 0.0)
        & (measurements["aspect_ratio"] >= 1.0)
    )
    invalid_count = int((~valid_values).sum())
    measurements = measurements.loc[valid_values].copy()
    if measurements.empty:
        return _add_empty_state(figure, translate, invalid_count=invalid_count)

    aspect_ratio = measurements["aspect_ratio"].to_numpy(dtype=float)
    mean_ratio = float(np.mean(aspect_ratio))
    median_ratio = float(np.median(aspect_ratio))
    q1, q3 = np.quantile(aspect_ratio, [0.25, 0.75])
    std_ratio = float(np.std(aspect_ratio, ddof=1)) if len(aspect_ratio) > 1 else 0.0
    threshold = _finite_threshold(min_aspect_ratio)
    bin_count = max(1, min(100, int(number_of_bins)))

    figure.add_trace(
        go.Histogram(
            x=aspect_ratio,
            nbinsx=min(bin_count, max(1, len(aspect_ratio))),
            name=translate("Retained rods"),
            marker=dict(color="#4C78A8", line=dict(color="white", width=1)),
            opacity=0.85,
            hovertemplate="L/W=%{x:.3f}<br>" + translate("Count") + "=%{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=aspect_ratio,
            y=np.zeros(len(aspect_ratio)),
            mode="markers",
            name=translate("Individual rods"),
            marker=dict(color="#1F3552", size=7, symbol="line-ns"),
            hovertemplate="L/W=%{x:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    _add_ratio_marker(
        figure,
        mean_ratio,
        translate("Mean"),
        "#F58518",
        "dash",
        "top left",
    )
    _add_ratio_marker(
        figure,
        median_ratio,
        translate("Median"),
        "#54A24B",
        "dot",
        "top right",
    )
    _add_ratio_marker(
        figure,
        threshold,
        translate("Minimum L/W"),
        "#E45756",
        "solid",
        "bottom right",
    )

    border_styles = (
        (False, "circle", translate("Interior rods"), "#1F3552"),
        (True, "triangle-up-open", translate("Border-touching rods"), "#D62728"),
        (None, "diamond-open", translate("Unknown border status"), "#777777"),
    )
    for border_value, symbol, name, outline_color in border_styles:
        if border_value is None:
            selected = measurements["border"].isna()
        else:
            selected = measurements["border"] == border_value
        subset = measurements.loc[selected]
        if subset.empty:
            continue

        hover_text = [
            "<br>".join(
                [
                    f"{translate('Particle')}: {row.particle}",
                    f"L/W: {row.aspect_ratio:.3f}",
                    f"{length_column}: {row.length:.3f}",
                    f"{width_column}: {row.width:.3f}",
                    f"{translate('Rod orientation')}: "
                    + (
                        f"{row.orientation % 180.0:.1f}°"
                        if math.isfinite(row.orientation)
                        else translate("Not available")
                    ),
                    f"{translate('Border touching')}: {name}",
                ]
            )
            for row in subset.itertuples()
        ]
        figure.add_trace(
            go.Scattergl(
                x=subset["width"],
                y=subset["length"],
                mode="markers",
                name=name,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=11,
                    symbol=symbol,
                    color=subset["aspect_ratio"],
                    coloraxis="coloraxis",
                    line=dict(color=outline_color, width=1.5),
                ),
            ),
            row=1,
            col=2,
        )

    max_width = float(measurements["width"].max()) * 1.08
    max_length = float(measurements["length"].max()) * 1.08
    _add_ratio_ray(figure, 1.0, max_width, max_length, "L/W = 1", "#9D9D9D")
    _add_ratio_ray(
        figure,
        threshold,
        max_width,
        max_length,
        f"{translate('Minimum L/W')} = {threshold:.2f}",
        "#E45756",
    )
    if not math.isclose(median_ratio, threshold, abs_tol=1e-9):
        _add_ratio_ray(
            figure,
            median_ratio,
            max_width,
            max_length,
            f"{translate('Median')} = {median_ratio:.2f}",
            "#54A24B",
            dash="dot",
        )

    invalid_text = (
        f" · {translate('Invalid rows')}: {invalid_count}" if invalid_count else ""
    )
    title = (
        f"<b>{translate('Nanorod aspect-ratio chart')}</b><br>"
        f"<sup>n={len(measurements)} · {translate('Median')}={median_ratio:.3f} "
        f"[{q1:.3f}–{q3:.3f}] · {translate('Mean')}={mean_ratio:.3f} "
        f"± {std_ratio:.3f}{invalid_text}</sup>"
    )
    figure.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=570,
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.08,
        legend=dict(orientation="h", yanchor="bottom", y=-0.27, x=0.5, xanchor="center"),
        margin=dict(l=60, r=40, t=105, b=120),
        coloraxis=dict(
            colorscale="Viridis",
            colorbar=dict(title="L/W", x=1.02, len=0.72),
        ),
    )
    figure.update_xaxes(title_text="L/W", row=1, col=1, showgrid=True)
    figure.update_yaxes(title_text=translate("Count"), row=1, col=1, showgrid=True)
    figure.update_xaxes(
        title_text=width_column,
        range=[0.0, max_width],
        row=1,
        col=2,
        showgrid=True,
    )
    figure.update_yaxes(
        title_text=length_column,
        range=[0.0, max_length],
        row=1,
        col=2,
        showgrid=True,
    )
    figure.add_annotation(
        text=translate(
            "Projected aspect ratio = orthogonal-caliper length / minimum-Feret width; it is dimensionless and describes a 2D projection."
        ),
        x=0.5,
        y=-0.18,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12, color="#555555"),
    )
    return figure


def _find_column(dataframe: Optional[pd.DataFrame], prefix: str) -> Optional[str]:
    if dataframe is None:
        return None
    return next((column for column in dataframe.columns if column.startswith(prefix)), None)


def _border_status(dataframe: pd.DataFrame) -> pd.Series:
    if "Border touching" not in dataframe.columns:
        return pd.Series(np.nan, index=dataframe.index, dtype=object)

    def normalize(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return np.nan

    return dataframe["Border touching"].map(normalize)


def _finite_threshold(value: float) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 1.0
    return threshold if math.isfinite(threshold) and threshold > 0.0 else 1.0


def _add_ratio_marker(figure, value, label, color, dash, position):
    figure.add_vline(
        x=value,
        row=1,
        col=1,
        line=dict(color=color, width=2, dash=dash),
        annotation_text=f"{label}: {value:.2f}",
        annotation_position=position,
    )


def _add_ratio_ray(figure, ratio, max_width, max_length, name, color, dash="dash"):
    if ratio <= 0.0:
        return
    end_x = min(max_width, max_length / ratio)
    figure.add_trace(
        go.Scatter(
            x=[0.0, end_x],
            y=[0.0, ratio * end_x],
            mode="lines",
            name=name,
            line=dict(color=color, width=1.8, dash=dash),
            hovertemplate=f"{name}<extra></extra>",
        ),
        row=1,
        col=2,
    )


def _add_empty_state(figure, translate, invalid_count=0):
    message = translate("No nanorod measurements available.")
    if invalid_count:
        message += f" {translate('Invalid rows')}: {invalid_count}."
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=17, color="#666666"),
    )
    figure.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white")
    return figure
