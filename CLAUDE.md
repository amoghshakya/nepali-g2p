# Working notes for Claude Code on this repo

This file tells Claude Code (or any future contributor working with an AI
assistant) how to make changes here without breaking the project's invariants.

## What this repo is

A Devanagari-to-phone frontend for **Kathmandu Newari-influenced Nepali** TTS,
plus the curated lexicon and reference data. The phonology policy is anchored
on Khatiwada (2009) and is non-trivial — see `docs/paper.md` §3 and
`README.md` for the dialect rationale.

This is **not** a generic Indian-language G2P. Pan-Indic conventions
(`च` → `/tʃ/`, vowel-length contrast, three-way sibilant) are deliberately
not what we do.

## Hard rules — do not change without policy review

These are the load-bearing decisions. If you change one, you change the
character of the entire downstream model.

1. **`च` → `/ts/`, `ज` → `/dz/`** (alveolar, not palato-alveolar). Source:
   Khatiwada (2009: 375). Encoded in `data/frontend/phones.tsv` and the
   lexicon. Switching to `/tʃ/` is a v2 effort, not a code change.
2. **Single sibilant `s` in default mode.** `sh` (श) and `sx` (ष) exist in
   the inventory but are reserved for the opt-in `careful_sanskritized`
   style. Do not start emitting them by default.
3. **No phonemic vowel length.** `ी` and `ि` both → `i`. There is no `ii`
   phone in the default inventory.
4. **Lexicon promotion is human-only.** `data/frontend/candidates_lexicon.tsv`
   carries `status=candidate, source=google_ne_lr_2018`. Never write a
   script that auto-promotes candidates to `gold_lexicon.tsv`. Promotion
   needs a reviewer name and date per row.
5. **The audit trace is the contract.** Every phone the frontend emits must
   carry a `source` ∈ `{lexicon, suffix_strip, g2p_rule, english_route,
   manual_override, unknown}`. New rule paths must add a source value, not
   silently inherit one.

## Canonical files

| concern | file |
|---|---|
| Phone inventory (single source of truth) | `data/frontend/phones.tsv` |
| Project-phone ↔ IPA correspondence | `data/frontend/ipa_map.tsv` |
| Curated word → phones (gold) | `data/frontend/gold_lexicon.tsv` |
| Imported word → phones (candidates) | `data/frontend/candidates_lexicon.tsv` |
| Akshara-to-phone rule cases | `data/frontend/akshara_cases.tsv` |
| Minimal-pair eval set | `data/frontend/minimal_pairs.tsv` |
| Upstream Google lexicon (CC-BY-4.0) | `data/external/google_ne_lr/` |
| NepTTS-Bench sentences | `data/external/neptts_bench/sentences.json` |

## Code layout

```
nepali_frontend/
  __init__.py
  data.py              ← lexicon loading, DATA_DIR resolves data/frontend/
  trace.py             ← audit-trace shape (per docs/frontend-trace-contract)
  g2p/
    phonemizer.py      ← public entry: phonemize_text, phonemize_word
    akshara.py         ← Devanagari → orthographic syllable units
    base_map.py        ← akshara → base phone tokens
    post_rules.py      ← schwa deletion, anusvara resolution, gemination
  normalize/           ← Unicode + text-normalization (NFC, ZWJ, danda, digits)
  tokenize/            ← word/sentence tokenization, script tagging
  prosody/             ← clause chunker
```

The G2P resolution order is **lexicon → suffix-strip → akshara+rules**.
Each token's source is set at the stage it resolves at.

## Conventions

- The `'.'` token in a phone list marks **syllable boundary inside a word**.
- The `'|'` token marks **word boundary inside a phrase**.
- Geminates are `<phone>:` (e.g. `r:`, `t:`). The akshara parser detects
  virama-joined identical consonants.
- The schwa label is `ax` (project label), IPA `/ʌ/` (Khatiwada 2009: 377).
  Do not relabel to `ə` — it would invalidate the IPA correspondence.

## Tests

```bash
pip install -e ".[test]"
pytest
```

Tests live in `tests/` and cover akshara parsing, normalization, the
phonemizer, post-rules, and the trace shape. **Do not commit changes to
`data/frontend/` without re-running tests.** A change to `phones.tsv`
that adds or renames a phone will break the lexicon-load tests; run
`pytest tests/test_phonemizer.py` first.

## What "ground a claim" means in this project

When asked to make a phonological change, do not invent a justification.
Either:
- **Cite a primary source** — Khatiwada (2009 / 2019), Regmi (2025),
  Pokharel (1989), Bandhu et al. (1971), or Acharya (1991). Direct quotes
  for these are recorded in `docs/paper.md` §3.2 and §3.3 (in this repo)
  and in the source-of-record `policy-research-findings.md` (in the
  project's working repo, not committed here).
- **Or run an empirical measurement** on the lexicon or the bench and
  attach the numbers (denominators included).

If you can't do either, do not make the change.

## License

MIT for code in `nepali_frontend/` and curated TSVs in `data/frontend/`.
Upstream third-party data in `data/external/` carries upstream licenses
(CC-BY-4.0 for the Google `ne/` lexicon and NepTTS-Bench). See each
subdirectory's `NOTICE.md` for attribution requirements. Any redistribution
of those files must follow upstream attribution rules.

## Citation discipline

- Khatiwada, R. (2009). "Nepali." *Journal of the IPA*, 39(3): 373–380.
- Khatiwada, R. (2019). "Retroflexion in Nepali." *Gipan*, 4: 19–29.
- Regmi, B. N. (2025). "Variation in Nepali Verb Roots." *Gipan*, 7(1):
  121–135.
- Sodimana et al. (2018). "A Step-by-Step Process for Building TTS Voices…"
  *SLTU 2018*. (For `data/external/google_ne_lr/`.)
- Masica, C. P. (1991). *The Indo-Aryan Languages*. CUP. (For the schwa
  deletion tradition.)

Always credit the upstream Google lexicon when redistributing or
discussing it; the `data/external/google_ne_lr/NOTICE.md` records the
upstream commit hash and SHA-256s of each file.
