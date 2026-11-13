"""Conversions between project phone labels and output representations."""

from __future__ import annotations

from collections.abc import Iterable

from . import data


def to_ipa_phone(phone: str, *, overrides: dict[str, str] | None = None) -> str:
    """Convert one project phone label to its IPA candidate.

    Boundary markers and special tokens without an IPA value are preserved.
    Unknown labels fail explicitly so output cannot silently become corrupt.
    """
    if phone in {".", "|"}:
        return phone
    base = phone.removesuffix(":")
    mapping = data.ipa_map()
    if overrides and base in overrides:
        ipa = overrides[base]
        return f"{ipa}:" if phone.endswith(":") else ipa
    if base not in mapping:
        raise ValueError(f"no IPA mapping for project phone {phone!r}")
    ipa = mapping[base]
    if not ipa:
        return phone
    return f"{ipa}:" if phone.endswith(":") else ipa


def to_ipa(
    phones: Iterable[str], *, overrides: dict[str, str] | None = None
) -> list[str]:
    """Convert a sequence of project phone labels to IPA candidates."""
    return [to_ipa_phone(phone, overrides=overrides) for phone in phones]
