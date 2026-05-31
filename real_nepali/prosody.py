"""Prosody-aware phone tokenization for real_nepali v0.3.

The v0.2 frontend intentionally reused the base G2P word path and dropped
sentence punctuation. For TTS, that made a period, question mark, exclamation,
and comma indistinguishable. This module keeps the same word G2P policy but
adds explicit punctuation symbols that can survive into Piper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from nepali_frontend.g2p import phonemizer as base_phonemizer
from nepali_frontend.normalize import text as text_normalizer

from . import g2p as real_nepali_g2p

PUNCTUATION_TOKENS: dict[str, str] = {
    "।": "<period>",
    "॥": "<period>",
    ".": "<period>",
    "?": "<question>",
    "!": "<exclaim>",
    ",": "<comma>",
    "،": "<comma>",
    ";": "<comma>",
    ":": "<comma>",
}

STYLE_TOKENS: dict[str, str] = {
    "neutral": "<neutral>",
    "statement": "<neutral>",
    "question": "<questioning>",
    "questioning": "<questioning>",
    "emphatic": "<emphatic>",
    "exclamation": "<excited>",
    "excited": "<excited>",
    "happy": "<happy>",
    "sad": "<sad>",
    "whispered": "<whispered>",
}

PROSODY_TOKENS: tuple[str, ...] = (
    "<period>", "<question>", "<exclaim>", "<comma>", "<pause>",
    "<neutral>", "<questioning>", "<emphatic>", "<happy>",
    "<sad>", "<whispered>", "<excited>",
)

TOKEN_PATTERN = re.compile(r"[।॥?!.,،;:]|[ऀ-ॣॲ-ॿ]+")
REAL_NEPALI_PROFILES = {
    "real_nepali",
    "real_nepali_v0.2",
    "real_nepali_v0.3",
    "real_nepali_v0.3_prosody",
    "standard_clear_nepali",
}


@dataclass
class ProsodyResult:
    """Phone stream plus audit fields for a prosody-aware utterance."""

    phones: list[str] = field(default_factory=list)
    word_sources: list[str] = field(default_factory=list)
    prosody_sources: list[str] = field(default_factory=list)
    style_token: str | None = None
    punctuation_tokens: list[str] = field(default_factory=list)
    normalized_text: str = ""

    @property
    def sources(self) -> list[str]:
        return self.prosody_sources + self.word_sources


def style_token(style_tag: str | None) -> str:
    """Map a metadata style/intent tag to the v0.3 model token."""
    key = (style_tag or "neutral").strip().lower().replace("-", "_")
    return STYLE_TOKENS.get(key, "<neutral>")


def _phonemize_word(text: str, profile: str):
    if profile in REAL_NEPALI_PROFILES:
        return real_nepali_g2p.phonemize_word(text, profile=profile)
    if profile == "base":
        return base_phonemizer.phonemize_word(text)
    raise ValueError(f"unknown G2P profile: {profile}")


def _append_word_boundary(phones: list[str]) -> None:
    if phones and phones[-1] != "|":
        phones.append("|")


def _append_punctuation(
    phones: list[str],
    token: str,
    punctuation_tokens: list[str],
) -> None:
    if phones and phones[-1] == "|":
        phones.pop()
    phones.append(token)
    punctuation_tokens.append(token)
    if token == "<comma>":
        phones.append("|")


def phonemize_text_with_prosody(
    text: str,
    *,
    profile: str = "real_nepali_v0.3",
    style_tag: str | None = None,
    include_style_token: bool = False,
    include_punctuation_tokens: bool = True,
    normalize: bool = True,
) -> ProsodyResult:
    """Phonemize text while preserving punctuation as phone tokens.

    Style tokens remain available as an explicit opt-in for experiments, but
    are off by default. The v0.3 Algenib repair tests showed that utterance-wide
    style tokens caused too much acoustic drift.
    """
    normalized = text
    if normalize:
        normalized, _ = text_normalizer.normalize(text)

    result = ProsodyResult(normalized_text=normalized)
    if include_style_token:
        result.style_token = style_token(style_tag)
        result.phones.append(result.style_token)
        result.prosody_sources.append(f"style:{result.style_token}")

    for token in TOKEN_PATTERN.findall(normalized):
        punct_token = PUNCTUATION_TOKENS.get(token)
        if punct_token is not None:
            if include_punctuation_tokens:
                _append_punctuation(
                    result.phones,
                    punct_token,
                    result.punctuation_tokens,
                )
                result.prosody_sources.append(f"punct:{punct_token}")
            continue

        word = _phonemize_word(token, profile)
        if not word.phones:
            continue
        result.phones.extend(word.phones)
        _append_word_boundary(result.phones)
        result.word_sources.append(word.source)

    while result.phones and result.phones[-1] == "|":
        result.phones.pop()
    return result
