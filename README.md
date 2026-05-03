# nepali-text-frontend

Text frontend for Nepali TTS — turns Devanagari into a phone-token stream that an
acoustic model can speak. eSpeak-free.

The phonology follows **Khatiwada (2009) — Kathmandu Nepali**, which has substantial
Newari substrate. If you're a native speaker from outside the Kathmandu Valley, the
output will sound mildly Newari-accented (e.g. `च` → /ts/ rather than /tʃ/). That is
faithful to the source phonology, not a bug. See [Phonology policy](#phonology-policy)
below.

## What this gives you

- **G2P** — Devanagari word → list of project phones (65 audible + special tokens)
- **Tokenization** — sentence chunking, word splitting, danda handling
- **Normalization** — numerals, abbreviations, punctuation
- **Prosody chunker** — clause-level phrasing for natural pause placement
- **Audit trace** — every phone tagged with its source (lexicon, suffix-strip, g2p_rule)
  so you can debug per-word coverage

## Install

```bash
pip install -e .
```

(or once we publish to PyPI: `pip install nepali-text-frontend`)

## Usage

```python
from nepali_frontend.g2p import phonemizer as ph

words = ph.phonemize_text("नेपाल हाम्रो देश हो।")
for w in words:
    print(w.text, "->", w.phones, "(source:", w.source + ")")

# नेपाल -> ['n', 'e', '.', 'p', 'aa', 'l']           (source: lexicon)
# हाम्रो -> ['h', 'aa', 'm', '.', 'm', 'r', 'o']     (source: lexicon)
# देश   -> ['d', 'e', 's']                            (source: lexicon)
# हो    -> ['h', 'o']                                 (source: lexicon)
```

Single-word lookup with full trace:

```python
r = ph.phonemize_word("पश्चिम")
print(r.phones)   # ['p', 'ax', 's', '.', 'ts', 'i', 'm']
print(r.source)   # 'lexicon'
```

## Phone inventory (65 audible + 4 special)

- 6 oral vowels: `i e ax aa o u`
- 5 nasal vowels: `in en axn aan un`
- 13 diphthongs (oral + nasal): `axj axw aaj aaw oj ew ow axjn axwn aajn aawn ojn own`
- 16 stops (4 places × {plain, aspirated, voiced, voiced-aspirated}): `p ph b bh / t th d dh / tx txh dx dxh / k kh g gh`
- 4 affricates: `ts tsh dz dzh`
- 5 nasals + 6 sonorants/fricatives + 4 careful + 2 loan
- Geminate forms (`<phone>:` for any consonant)
- Specials: `_` (PAD), `^` (BOS), `$` (EOS), `#` (BLANK)

Full table in [`data/frontend/phones.tsv`](data/frontend/phones.tsv).

## Phonology policy

We anchor on Khatiwada, R. (2009). *Nepali*. *Journal of the International Phonetic
Association*, 39(3), 373–380. This documents Kathmandu Nepali, which has features
that diverge from non-Kathmandu/Hill Nepali:

| Letter | Khatiwada (this G2P) | Pan-Nepali / Indo-Aryan-style |
|---|---|---|
| `च`    | `/ts/`               | `/tʃ/` (English "ch")           |
| `ज`    | `/dz/`               | `/dʒ/` ("j" in jeep)            |
| `व`    | `/w/`                | `/v/` in some dialects          |
| `ि`/`ी` | merged to `i` (no length) | distinguished in some traditions |

If you build a downstream model on this G2P, the model will reproduce these choices
faithfully — i.e. it will say `chīn` (China) as `tsin` rather than `tʃin`. We treat
this as a feature for Kathmandu/Newari-influenced Nepali and a known limitation for
other dialects. A future v2 of this frontend may expose a flag to switch policies.

## Sources of phones (the audit trace)

For every word, the G2P annotates which path was used:

| source | meaning |
|---|---|
| `lexicon`        | direct hit in the curated lexicon (highest confidence) |
| `suffix_strip`   | base form found after stripping a known inflectional suffix |
| `g2p_rule`       | rule-based fallback (Akshara → phone mapping) |

This shows up on every `Word` object as `.source`. Useful when you find a
mispronunciation and want to know whether to fix the lexicon or the rules.

## What's in `data/`

| path | what |
|---|---|
| `data/frontend/phones.tsv` | the canonical 65-phone inventory |
| `data/frontend/gold_lexicon.tsv` | manually curated word→phones (highest confidence) |
| `data/frontend/candidates_lexicon.tsv` | lexicon candidates seeded from Google language-resources |
| `data/frontend/ipa_map.tsv` | project phones ↔ IPA correspondence |
| `data/frontend/akshara_cases.tsv` | Akshara → phone rule cases for the rule-based fallback |
| `data/frontend/minimal_pairs.tsv` | curated minimal pairs for evaluation |
| `data/external/google_ne_lr/` | upstream Nepali lexicon (CC-BY-4.0) — see `NOTICE.md` |
| `data/external/neptts_bench/` | NepTTS-Bench sentence list for evaluation (CC-BY-4.0) |

## Trained TTS using this frontend

A multi-speaker Piper-VITS Nepali TTS trained with this G2P:
**[`ampixa/nepali-voices-v0`](https://huggingface.co/ampixa/nepali-voices-v0)** —
419 speakers, ~22h training audio, ONNX-exportable for ARM/CPU inference.

A 5-voice clean cold-start follow-up will land at `ampixa/nepali-tts-v1`.

## License

MIT for the code in `nepali_frontend/`. Upstream data files in `data/external/` carry
their own licenses (CC-BY-4.0 for the Google lexicon; see each directory's `NOTICE.md`).
The curated TSVs in `data/frontend/` are MIT-licensed alongside the code.

## Citation

```
@misc{nepali_text_frontend_2026,
  title  = {nepali-text-frontend: Devanagari G2P for Nepali TTS},
  author = {Ampixa},
  year   = {2026},
  url    = {https://github.com/ampixa/nepali-text-frontend}
}
```

Phonology source:

```
@article{khatiwada2009nepali,
  title   = {Nepali},
  author  = {Khatiwada, R.},
  journal = {Journal of the International Phonetic Association},
  volume  = {39}, number = {3}, pages = {373--380},
  year    = {2009},
}
```
