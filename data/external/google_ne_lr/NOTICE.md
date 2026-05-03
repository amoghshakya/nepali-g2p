# google/language-resources — Nepali (`ne/`) mirror

This directory is a verbatim mirror of the Nepali resources from
[google/language-resources](https://github.com/google/language-resources).
Files are unmodified.

## Pinned source

| Field | Value |
| --- | --- |
| Repository | `google/language-resources` |
| Commit | `a90601024f07ab7e96c911243324e0adeb16da96` |
| Fetched | 2026-04-30 |
| Repo status | Archived 2022-11-05 (read-only upstream) |

## Files

| Local file | Upstream path | SHA-256 |
| --- | --- | --- |
| `lexicon.tsv` | `ne/data/lexicon.tsv` | `a9f616db4ecc35ecde4e519e1438c5501d8574b3623d43370fb0089a31ea8a83` |
| `phonology.json` | `ne/festvox/phonology.json` | `62df6f433e33924946b00a443abd71913c37628d28b720a9f1ce698753313b7f` |
| `ne_README.md` | `ne/festvox/README.md` | `5ac216ec7be1c86759b00aa487d137f4af55a015a9229498eead382bd7a9b18f` |
| `LICENSE.upstream` | `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |

## Licensing

Two licenses apply because upstream uses different licenses for code vs data.

- **`lexicon.tsv`**: declared in the file header as **CC BY 4.0** —
  Copyright 2018 Google Inc. — <https://creativecommons.org/licenses/by/4.0/>.
  Required attribution must accompany any redistribution or derivative work
  that surfaces these pronunciations.
- **`phonology.json`, `ne_README.md`**: covered by the repository's
  `LICENSE.upstream` — **Apache License, Version 2.0**.

## Companion paper

Alexander Gutkin et al., "A Step-by-Step Process for Building TTS Voices
Using Open Source Data and Framework for Bangla, Javanese, Khmer, Nepali,
Sinhala, and Sundanese," Proceedings of SLTU 2018.
<https://research.google/pubs/a-step-by-step-process-for-building-tts-voices-using-open-source-data-and-framework-for-bangla-javanese-khmer-nepali-sinhala-and-sundanese/>

## Use in this project

This mirror is the **input source** for the project lexicon. It is not the
project lexicon itself. Conversion into project artifacts (phone alias
mapping, candidate lexicon, gold lexicon) lives elsewhere under
`data/frontend/` and must record `source=google_ne_lr_2018` plus this
NOTICE for traceability.

Do not edit files in this directory. Re-mirror by overwriting from the
pinned commit if upstream is ever revised.
