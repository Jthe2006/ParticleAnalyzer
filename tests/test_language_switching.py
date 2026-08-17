import ast
from contextvars import Context
from inspect import signature
from pathlib import Path

import pandas as pd
import pytest

from particleanalyzer.core.ImagePreprocessor import ImagePreprocessor
from particleanalyzer.core.LLMAnalysis import LLMAnalysis
from particleanalyzer.core.ParticleAnalyzer import ParticleAnalyzer
from particleanalyzer.core.PointManager import PointManager
from particleanalyzer.core.ui_styles import custom_head
from particleanalyzer.core.languages import translations
from particleanalyzer.core.language_context import (
    LanguageContext,
    normalize_language_code,
    resolve_language,
)
from particleanalyzer.core.utils import (
    activate_interface_language,
    particle_removal,
    loadLanguagePreference,
    prepareLanguageSelection,
    reloadWithLanguage,
    scale_input_unit_measurement,
    scale_input_visibility,
    statistic_an,
    translate_chatbot,
)


@pytest.mark.parametrize(
    ("language_code", "expected"),
    [
        ("en", "en"),
        ("en-US", "en"),
        ("ru_RU.UTF-8", "ru"),
        ("zh-CN", "zh-cn"),
        ("zh-SG", "zh-cn"),
        ("zh-Hans-CN", "zh-cn"),
        ("zh-TW", "zh-tw"),
        ("zh-HK", "zh-tw"),
        ("zh-Hant", "zh-tw"),
        ("繁體中文", "zh-tw"),
        ("browser", "auto"),
        ("AUTO", "auto"),
        ("fr-FR", None),
        (None, None),
    ],
)
def test_normalize_language_code(language_code, expected):
    assert normalize_language_code(language_code) == expected


def test_explicit_language_preference_overrides_browser_header():
    assert resolve_language("ru", "zh-CN,zh;q=0.9,en;q=0.8") == "ru"
    assert resolve_language("zh-TW", "en-US,en;q=0.9") == "zh-tw"


@pytest.mark.parametrize("preference", [None, "", "auto", "browser"])
def test_automatic_language_preference_uses_weighted_browser_header(preference):
    assert resolve_language(preference, "fr-FR;q=1,ru-RU;q=0.8,en-US;q=0.4") == "ru"


def test_browser_language_quality_and_zero_weight_are_respected():
    assert resolve_language("auto", "ru;q=0,zh-TW;q=0.5,en;q=0.9") == "en"


@pytest.mark.parametrize(
    ("preference", "accept_language", "fallback", "expected"),
    [
        ("auto", "fr-FR,de-DE;q=0.8", "ru-RU", "ru"),
        ("auto", None, "zh-Hant", "zh-tw"),
        ("klingon", "ru", "en", "en"),
        ("auto", "*", "unsupported", "en"),
    ],
)
def test_resolve_language_uses_safe_fallback(
    preference, accept_language, fallback, expected
):
    assert resolve_language(preference, accept_language, fallback) == expected


def test_language_context_normalizes_values_and_can_be_reset():
    token = LanguageContext.set_language("zh-TW")
    try:
        assert LanguageContext.get_language() == "zh-tw"
    finally:
        LanguageContext.reset_language(token)


def test_language_context_is_isolated_between_execution_contexts():
    original = LanguageContext.set_language("en")
    isolated_context = Context()
    try:
        isolated_context.run(LanguageContext.set_language, "ru")
        assert isolated_context.run(LanguageContext.get_language) == "ru"
        assert LanguageContext.get_language() == "en"
    finally:
        LanguageContext.reset_language(original)


def test_interface_preference_updates_backend_context_and_returns_saved_value():
    token = LanguageContext.set_language("en")
    try:
        preference, concrete, selector_update = activate_interface_language(
            "zh-TW", "zh-TW"
        )
        assert (preference, concrete) == ("zh-TW", "zh-tw")
        assert selector_update["value"] == "zh-TW"
        assert selector_update["choices"][0] == ("自動 / 瀏覽器", "auto")
        assert LanguageContext.get_language() == "zh-tw"
        preference, concrete, selector_update = activate_interface_language(
            "unsupported", "ru"
        )
        assert (preference, concrete) == ("auto", "ru")
        assert selector_update["choices"][0] == (
            "Автоматически / Браузер",
            "auto",
        )
        assert LanguageContext.get_language() == "ru"
    finally:
        LanguageContext.reset_language(token)


def test_frontend_adapter_supports_every_interface_language():
    for language in ("en", "ru", "zh-CN", "zh-TW"):
        assert language in custom_head
    assert "Object.defineProperty(navigator" in custom_head
    assert "particleanalyzer-interface-preference-v1" in custom_head
    assert "localStorage.getItem" in loadLanguagePreference
    assert "resolveLocale" in prepareLanguageSelection
    assert "localStorage.setItem" in reloadWithLanguage
    assert "window.location.reload" in reloadWithLanguage


def test_preprocessor_translation_does_not_mutate_shared_language():
    preprocessor = ImagePreprocessor.__new__(ImagePreprocessor)
    preprocessor.lang = "en"

    translated = preprocessor._get_translation("Загрузка изображения...", "ru")

    assert translated == translations["ru"]["Загрузка изображения..."]
    assert preprocessor.lang == "en"


@pytest.mark.parametrize(
    "callback",
    [
        ParticleAnalyzer.analyze_image,
        ParticleAnalyzer.handle_file_upload,
        PointManager.handle_select,
        LLMAnalysis.analyze,
        translate_chatbot,
        scale_input_visibility,
        scale_input_unit_measurement,
        particle_removal,
        statistic_an,
    ],
)
def test_language_argument_is_optional_for_existing_python_callers(callback):
    assert signature(callback).parameters["selected_language"].default == "auto"


def test_analyze_image_preserves_progress_argument_position():
    parameters = list(signature(ParticleAnalyzer.analyze_image).parameters)
    assert parameters.index("pr") < parameters.index("selected_language")


class _RussianRequest:
    headers = {"Accept-Language": "ru-RU,ru;q=0.9"}


def test_empty_llm_result_uses_automatic_browser_language():
    analyzer = LLMAnalysis.__new__(LLMAnalysis)

    result = analyzer.analyze(pd.DataFrame(), "unused", _RussianRequest())

    assert result[0]["content"] == translations["ru"]["No particles detected"]


def test_every_static_ui_translation_key_exists_in_all_languages():
    ui_path = Path(__file__).parents[1] / "particleanalyzer" / "core" / "ui.py"
    tree = ast.parse(ui_path.read_text(encoding="utf-8"))
    ui_keys = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "i18n"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }

    for language in ("en", "ru", "zh-cn", "zh-tw"):
        assert not (ui_keys - translations[language].keys()), language
