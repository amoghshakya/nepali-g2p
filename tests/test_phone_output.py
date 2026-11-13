from __future__ import annotations

import pytest

from nepali_frontend.g2p import phonemizer
from nepali_frontend.phone_output import to_ipa
from real_nepali import g2p as real_g2p


def test_base_phonemizer_can_return_ipa_without_changing_default():
    assert phonemizer.phonemize_word("चार").phones == ["ts", "aa", "r"]
    assert phonemizer.phonemize_word(
        "चार", representation="ipa"
    ).phones == ["t͡s", "a", "r"]


def test_ipa_output_preserves_boundaries_and_gemination():
    assert to_ipa(["r:", ".", "ax"]) == ["r:", ".", "ʌ"]


def test_real_nepali_converts_after_profile_rewrite():
    result = real_g2p.phonemize_word("चार", representation="ipa")
    assert result.phones == ["t͡ʃ", "a", "r"]
    assert result.base_phones == ["ts", "aa", "r"]


def test_invalid_representation_is_explicit():
    with pytest.raises(ValueError, match="representation"):
        phonemizer.phonemize_word("चार", representation="raw")
