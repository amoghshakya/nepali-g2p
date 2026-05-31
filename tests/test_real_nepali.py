"""Tests for the clear-standard real_nepali G2P profile."""

from __future__ import annotations

from nepali_frontend.g2p import phonemizer as old_phonemizer
from real_nepali import g2p
from real_nepali.manifest import rephonemize_row


def phones(text: str) -> list[str]:
    return g2p.phonemize_word(text).phones


def test_real_nepali_rewrites_affricates():
    assert phones("चार") == ["ch", "aa", "r"]
    assert phones("छ") == ["chh", "ax"]
    assert phones("आज") == ["aa", ".", "j", "ax"]


def test_real_nepali_rewrites_affricates_inside_words():
    assert phones("मान्छे") == ["m", "aa", "n", ".", "chh", "e"]
    assert phones("सञ्चार") == ["s", "ax", "n", ".", "ch", "aa", "r"]


def test_old_frontend_remains_kathmandu_policy():
    assert old_phonemizer.phonemize_word("चार").phones == ["ts", "aa", "r"]
    assert old_phonemizer.phonemize_word("आज").phones == ["aa", ".", "dz", "ax"]


def test_result_keeps_base_phones_for_audit():
    result = g2p.phonemize_word("चीन")
    assert result.phones == ["ch", "i", "n"]
    assert result.base_phones == ["ts", "i", "n"]
    assert any(d.get("rule") == "clear_standard_affricates" for d in result.decisions)


def test_phonemize_text_uses_profile_for_each_word():
    out = g2p.phonemize_text("चार छ आज")
    assert [" ".join(r.phones) for r in out] == ["ch aa r", "chh ax", "aa . j ax"]


def test_v02_profile_alias():
    result = g2p.phonemize_word("चार", profile="real_nepali_v0.2")
    assert result.phones == ["ch", "aa", "r"]


def test_v03_profile_alias_keeps_clear_standard_policy():
    result = g2p.phonemize_word("चार", profile="real_nepali_v0.3")
    assert result.phones == ["ch", "aa", "r"]


def test_manifest_records_selected_profile_in_source_path():
    row = {
        "text": "चार",
        "phones": "",
        "g2p_path": "",
    }
    out = rephonemize_row(row, profile="real_nepali_v0.2")
    assert out["phones"] == "ch aa r"
    assert out["g2p_path"].startswith("real_nepali_v0.2:")
