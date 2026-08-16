"""Language selection helpers shared by the UI and analysis callbacks.

The application uses lower-case language keys internally because those are the
keys in :mod:`particleanalyzer.core.languages`. Browser locale variants are
normalized to one of those keys before they reach the analysis code.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Final

SUPPORTED_LANGUAGE_CODES: Final[tuple[str, ...]] = (
    "en",
    "ru",
    "zh-cn",
    "zh-tw",
)

_AUTO_LANGUAGE_CODES: Final[frozenset[str]] = frozenset(
    {"auto", "browser", "browser-default", "system", "system-default"}
)
_LANGUAGE_NAME_ALIASES: Final[dict[str, str]] = {
    "english": "en",
    "russian": "ru",
    "русский": "ru",
    "simplified-chinese": "zh-cn",
    "traditional-chinese": "zh-tw",
    "简体中文": "zh-cn",
    "簡體中文": "zh-cn",
    "繁体中文": "zh-tw",
    "繁體中文": "zh-tw",
}


def _clean_language_code(language_code: object) -> str:
    """Return a comparison-friendly locale string without changing meaning."""

    if language_code is None:
        return ""

    code = str(language_code).strip().replace("_", "-").lower()
    # Operating-system locale strings can include an encoding or a modifier,
    # for example ``ru_RU.UTF-8`` or ``zh_CN@pinyin``.
    return code.split(".", 1)[0].split("@", 1)[0]


def normalize_language_code(language_code: object) -> str | None:
    """Normalize a locale or UI preference to an application language key.

    Supported concrete results are ``en``, ``ru``, ``zh-cn`` and ``zh-tw``.
    The UI aliases ``auto`` and ``browser`` normalize to the sentinel ``auto``.
    Unsupported or empty values return ``None`` so callers can apply their own
    fallback policy.
    """

    code = _clean_language_code(language_code)
    if not code:
        return None
    if code in _AUTO_LANGUAGE_CODES:
        return "auto"
    if code in _LANGUAGE_NAME_ALIASES:
        return _LANGUAGE_NAME_ALIASES[code]

    subtags = code.split("-")
    primary_language = subtags[0]
    if primary_language == "en":
        return "en"
    if primary_language == "ru":
        return "ru"
    if primary_language != "zh":
        return None

    # Script subtags are authoritative. If no script is present, regions that
    # conventionally use Traditional Chinese are handled explicitly; bare
    # ``zh`` and other Chinese variants default to Simplified Chinese.
    if "hant" in subtags:
        return "zh-tw"
    if "hans" in subtags:
        return "zh-cn"
    if any(region in subtags for region in ("tw", "hk", "mo")):
        return "zh-tw"
    return "zh-cn"


def _fallback_language(fallback: object) -> str:
    normalized = normalize_language_code(fallback)
    if normalized in SUPPORTED_LANGUAGE_CODES:
        return normalized
    return "en"


def _language_from_accept_header(accept_language: object) -> str | None:
    """Select the best supported language from an Accept-Language header."""

    if accept_language is None:
        return None

    candidates: list[tuple[float, int, str]] = []
    for position, raw_entry in enumerate(str(accept_language).split(",")):
        parts = [part.strip() for part in raw_entry.split(";")]
        if not parts or not parts[0] or parts[0] == "*":
            continue

        quality = 1.0
        malformed_quality = False
        for parameter in parts[1:]:
            name, separator, value = parameter.partition("=")
            if separator and name.strip().lower() == "q":
                try:
                    quality = float(value.strip())
                except ValueError:
                    malformed_quality = True
                break
        if malformed_quality or quality <= 0:
            continue

        language = normalize_language_code(parts[0])
        if language in SUPPORTED_LANGUAGE_CODES:
            # Earlier entries win when quality values are equal, matching the
            # preference order expressed in the header.
            candidates.append((min(quality, 1.0), -position, language))

    if not candidates:
        return None
    return max(candidates)[2]


def resolve_language(
    preference: object = None,
    accept_language: object = None,
    fallback: object = "en",
) -> str:
    """Resolve an explicit UI preference and browser header to one language.

    A supported explicit preference always wins. ``None``, an empty value,
    ``auto`` or ``browser`` delegates to the weighted ``Accept-Language``
    header. Invalid explicit preferences and unsupported browser headers use
    the normalized fallback, with English as the final safe default.
    """

    fallback_language = _fallback_language(fallback)
    cleaned_preference = _clean_language_code(preference)
    normalized_preference = normalize_language_code(preference)
    preference_is_automatic = not cleaned_preference or normalized_preference == "auto"

    if not preference_is_automatic:
        if normalized_preference in SUPPORTED_LANGUAGE_CODES:
            return normalized_preference
        return fallback_language

    return _language_from_accept_header(accept_language) or fallback_language


class LanguageContext:
    """Context-local language used by nested analysis helpers.

    Gradio can process multiple sessions concurrently. ``ContextVar`` keeps a
    callback's language from leaking into another request while retaining the
    existing ``set_language``/``get_language`` interface.
    """

    _current_lang: ContextVar[str] = ContextVar(
        "particleanalyzer_language", default="en"
    )

    @classmethod
    def set_language(cls, lang: object) -> Token[str]:
        language = normalize_language_code(lang)
        if language not in SUPPORTED_LANGUAGE_CODES:
            language = "en"
        return cls._current_lang.set(language)

    @classmethod
    def get_language(cls) -> str:
        return cls._current_lang.get()

    @classmethod
    def reset_language(cls, token: Token[str]) -> None:
        """Restore the value that preceded a matching ``set_language`` call."""

        cls._current_lang.reset(token)
