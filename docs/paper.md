# A Dialect-Faithful G2P and Multi-Speaker VITS for Kathmandu Nepali Text-to-Speech

**Ampixa**
*ashish.thapa477@gmail.com*

---

## Abstract

We describe a lexicon-first, audit-traced grapheme-to-phoneme (G2P) frontend for
Nepali text-to-speech and a multi-speaker VITS acoustic model trained with it.
The frontend's phoneme inventory is anchored on Khatiwada (2009)'s IPA-Handbook
illustration of standard Kathmandu Nepali — a dialect with non-trivial Newari
substrate influence — and is justified, item by item, against six recent and
classical primary sources. We make six concrete policy decisions that diverge
from pan-Indic conventions: alveolar (rather than palato-alveolar) affricates,
single sibilant /s/ (rather than the orthographic three-way contrast),
restricted loan phones /f/ /z/, dropped palatal nasal /ɲ/, no phonemic vowel
length contrast, and Google's 13-label phonetic diphthong scheme. On the
NepTTS-Bench `phonological_minimal_pairs` set (149 pair-sentences), the G2P
emits distinct phone sequences for the two members in 99.5% of pairwise
comparisons (script: `eval/run_neptts_bench.py`); the result measures the
G2P alone, not the acoustic model. We train a 419-speaker VITS model
(`ampixa/nepali-voices-v0`, ~22 h training audio, warmstart from the
multilingual `piper-plus-base`) and have a 5-speaker clean cold-start
(`ampixa/nepali-tts-v1`) in progress at the time of writing. We document the
engineering hazards we hit in the VITS training pipeline (cuDNN-host-side
overhead, Cython monotonic alignment search, mid-precision STFT, frozen
duration predictor in cross-lingual fine-tuning, single-speaker collapse
pathology) and report what fixed each one, with hardware/config noted for
each measurement. We are explicit about the model's two remaining failure
modes: a phonotactic gap on rare phoneme contexts (e.g. /ts i n/ in चीन),
and a dialect mismatch that, in informal native-speaker feedback, listeners
from outside the Kathmandu Valley have described as a Newari accent — the
latter is the policy doing what we asked it to.

---

## 1. Introduction

Nepali (ISO 639-3 `nep`, glottocode `nepa1254`) is the official language of
Nepal and one of the 22 scheduled languages of India, with roughly 16 million
L1 speakers. It is a member of the Indo-Aryan branch and is written in
Devanagari. The published academic TTS literature for Nepali is small —
Khadka et al. (SIGUL 2023) and Dongol & Bal (ICON 2023) are the most-cited
recent baselines (§2.3) — and existing open-weights Nepali TTS releases on
the Hugging Face Hub (e.g. `tuskbyte/nepali_male_v1`, voices in
`ai4bharat/indic-parler-tts`) are limited in either training-data documentation,
phoneme-level interface, or both, when surveyed in May 2026. There is room
for an open, audit-traced Nepali TTS that documents its phonological policy
end-to-end.

We focus on the text-frontend bottleneck rather than the acoustic-modelling
bottleneck because the frontend is where the project-specific phonological
choices have to live: existing free Nepali G2P systems either (a) lump Nepali
into Hindi rules (eSpeak NG's `ne_rules` retains the orthographic three-way
sibilant and full phonemic vowel length, both of which are not contrastive
in spoken Nepali per Khatiwada 2009 — see §3.2), or (b) provide no
phonological audit trace, making it impossible to attribute a TTS
mispronunciation to a lexicon error vs a rule fallback.

This paper documents three contributions:

1. A grounded, peer-reviewed-source-anchored phonological policy for the
   Kathmandu Nepali default mode, with six divergences from pan-Indic G2P
   conventions justified by direct quotes from primary sources (§3).
2. A lexicon-first G2P implementation with per-phone audit trace and
   99.5% minimal-pair contrast preservation on NepTTS-Bench (§4).
3. A multi-speaker VITS training pipeline built on `piper-plus-base`, with
   the engineering hazards we hit and the fixes (§5), and two released
   checkpoints on Hugging Face (§6).

The frontend is released as `ampixa/nepali-text-frontend` (MIT) and the
acoustic models as `ampixa/nepali-voices-v0` and `ampixa/nepali-tts-v1`
(CC-BY-SA-4.0, forced by the share-alike licenses on training data).

We are explicit about an honest limitation: the model speaks **Kathmandu Newari-
influenced Nepali**, the dialect Khatiwada documents. Listeners from outside
the Kathmandu Valley perceive the output as having a Newari accent. This is a
policy decision encoded in the G2P, not a model defect (§7.2).

## 2. Related Work

### 2.1 Nepali phonology — primary sources

The contrastive phoneme inventory of standard Kathmandu Nepali is described
across a small but coherent literature:

- **Bandhu, Dahal, Holzhausen, & Hale (1971).** *Nepali Segmental Phonology*.
  SIL/Tribhuvan University. The foundational consonant-and-nasalisation
  analysis. WALS catalogues 11 datapoints from this work.
- **Pokharel, M. P. (1989).** *Experimental Analysis of Nepali Sound System*.
  PhD thesis, Pune University. Source for affricate palatography (showing
  alveolar place) and the canonical 10-diphthong inventory cited by every
  later author.
- **Khatiwada, R. (2009).** "Nepali." *Journal of the International Phonetic
  Association*, 39(3), 373–380. The IPA-Handbook illustration; the canonical
  reference for modern Kathmandu Nepali. We use this as our policy anchor.
- **Khatiwada, R. (2019).** "Retroflexion in Nepali." *Gipan*, 4: 19–29.
  Direct palatographic + linguographic study of 9 speakers. Settles the
  retroflex-as-phonological-category question.
- **Regmi, B. N. (2025).** "Variation in Nepali Verb Roots: A Phonological
  Perspective." *Gipan*, 7(1): 121–135. Recent TU paper providing explicit
  Devanagari → phoneme mappings (e.g. ञ → /n/, श → /s/) and a 46-segment
  inventory.

Supporting acoustic and phonetic studies — Clements & Khatiwada (2007) on
affricate place, Khatiwada (2008) on aspiration cues, Schwarz, Sonderegger
& Goad (2019) on laryngeal contrasts, Agrawal, Bansal & Sinha (2019) on
comparative retroflex production — all support the same picture. We read
Khatiwada (2009), Khatiwada (2019), and Regmi (2025) end-to-end; the rest
are accepted via citations in those three.

### 2.2 Existing Nepali G2P systems

| System | Approach | Why we did not use it |
|---|---|---|
| **eSpeak NG `ne_rules`** | Rule-based, runtime, formant-style assumptions | Hindi-style schwa deletion rules, preserves phonemic vowel length and three-way sibilant, palatal-implying labels for affricates — all wrong for spoken Nepali (Khatiwada 2009: 376–378). Rule output is hard to audit per-phone. |
| **Google `language-resources` `ne/`** | 54K-entry hand-curated lexicon (Sodimana, Pipatsrisawat, Ha, Jansche, Kjartansson, Pasternack, Silfverberg, Quéré, & Henderson 2018, "A Step-by-Step Process for Building TTS Voices Using Open Source Data and Frameworks for Bangla, Javanese, Khmer, Nepali, Sinhala, and Sundanese", *SLTU 2018*), CC-BY-4.0, archived 2022-11 | Adopted as our *candidate* lexicon. Disagreements on /f/, /z/, /ny/, /ii/ retention required a project-specific re-derivation. Used as seed, not as gold. |
| **Indic Parler-TTS** (AI4Bharat) | Multilingual end-to-end with internal G2P | Black-box; no phone-level audit trace; pan-Indic phoneme set is wrong for Nepali contrast preservation. Useful as a multilingual baseline, not as a frontend. |
| **`bhashini-ai/g2p`, `dmort27/epitran`, `lingjzhu/CharsiuG2P`, `CUNY-CL/wikipron`** | Various rule-based / neural G2P collections | None covers Nepali. Rejected by direct file-inspection. |

We also note the TDIL Pronunciation Lexicon Dictionary (request-based, opaque
licensing) and the AI4Bharat IndicMFA aligner (replaces G2P with G2G, not a
pronunciation lexicon) — neither was usable for our purposes.

### 2.3 Existing Nepali TTS

The published Nepali TTS landscape we surveyed is small and Tacotron /
FastPitch-era:

- Khadka, G. C., Paudel, Shah, & Joshi (SIGUL 2023). Tacotron2 + HiFi-GAN /
  WaveGlow on OpenSLR-43. The paper reports MOS 4.03 naturalness and 4.01
  accuracy after fine-tuning and post-processing on the authors' own listener
  panel; MOS scale, listener count, and protocol details are in the original
  publication and we do not re-evaluate them here.
- Dongol & Bal (ICON 2023). FastPitch + HiFi-GAN; reports MOS 3.70 (Munal
  voice) and 3.40 (Danphe voice) on a similar listener-panel methodology.
- AI4Bharat Indic Parler-TTS lists Nepali among supported languages with
  28.65 h in the training mix per the model card. We did not benchmark its
  Nepali performance ourselves; users on the bench report mixed results,
  but we do not have a controlled evaluation to cite.

Within the systems we inspected, none exposes a phoneme-level audit trace
and none documents a Khatiwada-anchored phoneme inventory. We do not claim
this is true of every Nepali TTS system; the survey is bounded by what we
read.

## 3. Phonological Policy

### 3.1 The dialect anchor

Khatiwada (2009) explicitly targets the eastern variety used in Kathmandu mass
media:

> "The author has adopted the variety used in mass media." (p. 373)

This dialect carries Newari (Tibeto-Burman) substrate influence on consonants
and vowels. We adopt it as the default policy target because (a) it is the
best-documented Nepali variety in the cited literature, (b) the available
metadata for our training corpora (OpenSLR-43, OpenSLR-143, IndicVoices-R
Nepali subset) suggests that recordings were made in Nepal with speakers
plausibly drawn from the Kathmandu standard, though per-utterance regional
metadata is not always available, and (c) it is internally consistent — if
a TTS model trained on those speakers is to sound coherent, the policy must
not force Indo-Aryan-broad pronunciation onto those audio targets.

The cost is that, in informal native-speaker feedback we have collected
during model evaluation, listeners from outside the Kathmandu Valley have
described the output as Newari-accented. We have not run a controlled
listening-panel evaluation; this remark is qualitative. We document the
trade-off in the public README of the frontend repository so consumers
understand the policy choice up front.

### 3.2 Six policy decisions, per primary source

The following table summarises the decisions and their primary-source
justifications. Direct quotes are reproduced from the sources we read
end-to-end.

| # | Item | Decision | Primary source |
|---|---|---|---|
| 1 | Affricate place | Alveolar `ts tsh dz dzh` (not palato-alveolar) | Khatiwada (2009: 375): "Palatograms of the Nepali affricates /ts tsʰ dz dzʱ/ … have shown that these sounds are produced with laminal contact in the alveolar region. Auditorily, however, they often sound rather like palato-alveolar sounds to English and French speakers." Confirmed independently by Pokharel (1989) palatograms and Clements & Khatiwada (2007). |
| 2 | Sibilants श / ष / स | Default mode emits `s` only; `sh` (श) and `sx` (ष) reserved for an opt-in `careful_sanskritized` style. | Khatiwada (2009: 376): "Nepali has only two contrastive fricatives, /s/ and /ɦ/, but in spontaneous speech the voiced and voiceless aspirated labials and velars can also be realized as the corresponding homorganic fricatives." Regmi (2025: 123): "ञ ण श ष are pronounced as n, n, s and s respectively in Nepali." |
| 3 | Loan phones /f/ /z/ | Kept as project phones, used only when (a) lexicon entry is loan-tagged, (b) the source is Latin-script English routed via the code-switch path, or (c) the Devanagari spelling uses an explicit nukta (फ़, ज़). Native words default to `ph` and `j`. | Khatiwada (2009: 374): "The consonant chart above does not include consonants found uniquely in loanwords, including those from Sanskrit." |
| 4 | Palatal nasal `ny` | Drop from default. ञ → `n` in default; `ny` reserved for `careful_sanskritized`. | Khatiwada (2009: 377): "Though the Nepali orthographic system represents five nasals ⟨m n ŋ ɳ ɲ⟩, only the first three are phonologically pertinent." Regmi (2025: 123) corroborates. Agrawal, Bansal & Sinha (2019) document /ɳ/ → [n] productionally for Nepali speakers. |
| 5 | Vowel length `ii`/`uu` | No phonemic length contrast. ी → `i`, ू → `u`. | Khatiwada (2009: 378): "Although Sanskrit — the ancestral language of Nepali — had a phonological distinction in vowel length (for example, /i/ versus /iː/ and /u/ versus /uː/), there is no such distinction in spoken Nepali." The only length effect, [paːɽi] from /pʌɦaɖi/, is allophonic from /ɦ/-deletion (p. 378). |
| 6 | Diphthong representation | Adopt Google's 13 phonetic labels: `axj axw aaj aaw oj ew ow` + 6 nasalised counterparts. Pokharel's three "missing" phonological diphthongs (ei, iu, ui) encoded as 2-syllable vowel sequences. | Khatiwada (2009: 378, fn 4) cites Pokharel (1989) ten-diphthong list verbatim. Khatiwada also writes (p. 378): "It is not very easy to determine in many cases whether a given sequence is monosyllabic or disyllabic." Empirical pass on the 54K-entry Google lexicon: 0 within-syllable `e i / i u / u i`; 15, 7, 10 cross-syllable instances. The 7-vs-10 split is phonetically motivated (high-nucleus diphthongs behave as 2-syllable acoustically). |

We also adopted the secondary findings from the same research log:

- **Schwa is /ʌ/, not /ə/.** Khatiwada (2009: 377): "[ʌ] is proposed as the
  basic norm and all others are considered phonetic variants of this phoneme."
  Project label `ax`; IPA correspondence updated.
- **Three nasals only.** No phonemic /ɲ/ or /ɳ/ in default mode (item 4).
  `nx` retained for ण in opt-in careful mode.
- **No /õ/.** Khatiwada (2009: 378): "All Nepali vowels except /o/ have
  distinctive nasal counterparts. The absence of a phonological /õ/ … is
  quite striking." Project draft `on` dropped.
- **/j/ and /w/ as full phonemes, not glides.** We follow Regmi (2025) over
  Khatiwada's parenthetical-glide treatment because the downstream G2P needs
  them as cluster onsets and offglides regardless of phonological analysis.

### 3.3 Allophonic processes (handled in post-rules, not in inventory)

Nepali has well-documented allophonic processes that do not change the phoneme
inventory but affect pronunciation:

- **Schwa deletion.** We treat the four-condition schwa-retention scheme
  (conjunct-final retention; verb-form retention unless halanta-marked;
  adverb / postposition retention; compound-postposition halanta attachment)
  as an **operational heuristic** drawn from the Indo-Aryan schwa-deletion
  tradition (Masica, *The Indo-Aryan Languages*, 1991, ch. 5; the Wikipedia
  encyclopedia article that summarises this tradition for Nepali; Regmi
  2025: 123 corroborates Nepali-specific behaviour for inherent अ). We do
  not treat the four-rule decomposition as a peer-reviewed Nepali-specific
  result and we plan native-speaker validation in a follow-up. Hindi-style
  "VC_CV deletion" is **not** the right rule for Nepali — concrete pair:
  सुलोचना is `sulocnā` in Hindi but `sulocanā` in Nepali.
- **Voiced aspirate weakening.** Khatiwada (2009: 376): "Often in spontaneous
  speech, the voiced aspirates lose their aspiration intervocalically and
  word-finally." Handled per-entry in the lexicon, not as a default rule.
- **Retroflex variability.** Khatiwada (2019): out of 9 speakers, 75% of
  retroflex productions are cacuminal (rim-of-tongue post-alveolar), 25% are
  truly sub-apical retroflex; vocalic conditioning is significant. We keep
  `tx txh dx dxh` as separate phones (retroflex *is* a phonological category,
  per Khatiwada 2019: 28) but the acoustic target for voice talent is
  cacuminal.
- **Gemination.** Khatiwada (2009: 376): "all consonants have geminate
  counterparts. Geminates occur only medially." Gemination is lexically
  contrastive (e.g. चपल /tsʌpʌl/ 'unstable' vs चप्पल /tsʌppʌl/ 'slipper').
  Detected by virama-joined identical consonants in the akshara parser.
- **/ɦ/-deletion length.** Khatiwada (2009: 378): /pʌɦaɖi/ → [paːɽi]. The
  only place phonetic length emerges, and it is allophonic. Encoded as a
  duration hint on lexicon entries flagged with the `h_deletion` allophone
  tag, not as a separate phone.

## 4. The G2P Frontend

### 4.1 Architecture

```
raw text
  → Unicode/script normalization (NFC, ZWJ/ZWNJ, danda canonicalisation,
    digit normalisation)
  → tokenization + script tagging (Devanagari / Latin / digit /
    punctuation / mixed)
  → text normalization (numerals, dates, currency, abbreviations)
  → lexical lookup (gold → candidates → suffix-strip)
  → akshara parsing (Devanagari → orthographic syllable units)
  → base G2P map (independent vowel, consonant + matra, conjunct, halant,
    nukta, anusvara, chandrabindu, visarga, danda)
  → phonological post-rules (schwa deletion, anusvara resolution,
    gemination)
  → audit trace (per-phone source tag)
  → phone token sequence + JSON trace
```

The trace contract is documented in `docs/frontend-trace-contract.md` as a
design intent: every phone in the output should carry a `source` field with
values `{lexicon, suffix_strip, g2p_rule, english_route, manual_override,
unknown}`. The intent is to convert pronunciation bugs from "the model
sounds wrong somewhere" into specific lexicon rows or rules to edit. The
test suite covers the basic source attribution but does not yet enforce the
no-`unknown_phone`-leaks invariant in CI; that is tracked as future work.

### 4.2 Phone inventory

The default-mode inventory has 65 audible phones plus 4 special tokens:

- **Vowels:** 6 oral (`i e ax aa o u`) + 5 nasal (`in en axn aan un`).
  No /õ/, no phonemic length.
- **Diphthongs:** 7 oral (`axj axw aaj aaw oj ew ow`) + 6 nasalised
  (`axjn axwn aajn aawn ojn own`).
- **Stops:** 16 (4 places × {plain, aspirated, voiced, voiced-aspirated}):
  `p ph b bh / t th d dh / tx txh dx dxh / k kh g gh`.
- **Affricates:** 4 (`ts tsh dz dzh`), alveolar.
- **Nasals:** 3 (`m n ng`); `nx` and `ny` are opt-in.
- **Sonorants/fricatives:** 6 (`r s h l y w`).
- **Loan / careful:** `f z sh sx` (gated by source tag or style flag).
- **Geminates:** `<phone>:` for any consonant.
- **Specials:** `_` (PAD), `^` (BOS), `$` (EOS), `#` (BLANK).

The inventory and IPA correspondences are in `data/frontend/phones.tsv` and
`data/frontend/ipa_map.tsv`.

### 4.3 Lexicon stack

We seed from `google/language-resources/ne` (Gutkin et al., SLTU 2018), pinned
to commit `a90601024f07ab7e96c911243324e0adeb16da96`, file SHA-256s recorded
in `data/external/google_ne_lr/NOTICE.md`. The seed:

- 54,178 entries, structurally clean (0 malformed lines).
- 89.7% Devanagari (48,585 entries) — directly usable.
- Phone inventory aligns on 36/54 phones with our project draft after a small
  rename. Disagreements are explicit and audit-able.

The candidate file (`data/frontend/candidates_lexicon.tsv`) carries
`status=candidate, source=google_ne_lr_2018` per row. Promotion to gold
(`data/frontend/gold_lexicon.tsv, status=accepted`) is human-driven, never
automatic. The conversion applies a phone-alias table
(`data/frontend/phone_alias_google.tsv`) and re-derives Devanagari evidence
where Google's policy collapses what ours preserves (e.g. श → `s` in default
mode, `sh` in careful mode).

### 4.4 Resolution order

For each Devanagari word token:

1. **Gold lexicon hit** — direct use (`source=lexicon`, highest confidence).
2. **Candidates lexicon hit** — direct use (`source=lexicon`,
   `confidence=candidate`).
3. **Suffix-strip rescue** — if the surface form is a known inflected form,
   try base + suffix lookup (`source=suffix_strip`). Productive for ~11% of
   tokens in natural Nepali text.
4. **Akshara + base G2P** — parse to orthographic syllables, apply base map,
   apply post-rules (`source=g2p_rule`). Catches everything else.
5. **`unknown_phone` warning** — never reached in practice on Nepali text,
   only on truly unknown Devanagari sequences.

For each Devanagari word token, exactly one of these stages assigns its
`source`. On natural Nepali production text (Wikipedia + iriis-ne news
copy), the per-word source distribution measured by the audit trace on
sampled corpora is roughly: lexicon ≈ 74–80%, suffix-strip ≈ 11%, g2p_rule
≈ 9–15% — the three stages are mutually exclusive and sum to ~100% per
sample. The per-source ratios depend on text source; the percentages above
are estimates from sampled subsets, not corpus-exhaustive measurements.
A separate measurement on IndicVoices-R + SLR54 transcripts yields a
26.7% rule-fallback rate, reflecting more out-of-lexicon vocabulary in
crowdsourced read speech (§7.3).

### 4.5 Coverage and minimal-pair contrast

The NepTTS-Bench (Ampixa) sentence set contains 365 sentences across 9
categories: 160 natural-speech, 149 phonological-minimal-pairs, 10
question-intonation, 10 robustness (tongue twisters), 10 phrases, 8 emotion,
8 contrastive-stress, 6 homographs, 4 newspaper-ambiguities. The minimal-pair
category covers Nepali-specific contrasts: aspiration (velar, dental, labial,
palatal, retroflex), retroflex-vs-dental, oral-vs-nasal vowel, gemination,
schwa deletion, and nasal consonants. Each sentence in this category is
tagged with a `pair_id`; pair members share a `pair_id` prefix, and each
member has a `contrast_word` that is the lexically-contrastive token.

We measure contrast preservation as: for every pair group with ≥ 2 members,
emit each member's `contrast_word` through the G2P, then count pairwise
comparisons whose phone sequences differ. Across the 149 minimal-pair
sentences (script: `eval/run_neptts_bench.py`) the G2P emits distinct
phone sequences for the two members in 99.5% of pairwise comparisons.
The remaining 0.5% are cases where two valid spellings collapse to the
same underlying form by design (e.g. orthographic sibilant variants both
→ `s` in default mode). This number measures the **G2P alone**: it does
not say the *acoustic* model produces audibly distinct realisations of the
two phone sequences, only that the frontend hands the model a different
phone sequence to render.

## 5. TTS Training Pipeline

### 5.1 Architecture and base model

We build on `ayousanz/piper-plus` (a maintained Python-3.12 / PyTorch-2.7
fork of the rhasspy/piper trainer) and fine-tune from
`ayousanz/piper-plus-base` — a multilingual VITS pretrained on 6 languages
(EN, JA, ZH, ES, FR, PT), 173-symbol vocabulary, prosody-dim 16.

Why VITS specifically, given the 2024–2026 trend toward flow-matching
(F5-TTS, E2-TTS) and codec-LM (CosyVoice2) models? Three reasons:

1. **Deployment fit.** The downstream goal is hosting on Hetzner ARM
   (Ampere Altra). The Piper project's published deployment story (ONNX
   export, ~80–150 MB artefact, CPU-only inference) matches our target.
   We have not benchmarked our specific checkpoint at the CPU yet; the
   RTF range of 0.05–0.15 reported in §7-style future work is an estimate
   based on Piper's published Raspberry Pi 4 benchmarks scaled to Ampere
   Altra core counts, not a direct measurement on our model.
2. **Audit-friendly for our G2P.** Phone-token input is preserved through
   the model; we can A/B specific phoneme sequences against listening tests
   and the audit trace.
3. **Existing tooling.** Among multilingual Devanagari-friendly bases we
   surveyed, the `piper-plus` fork was the one that did not require eSpeak
   as a runtime phonemizer. Other bases may exist that we did not survey.

### 5.2 Training data

| source | hours | utts | speakers | license |
|---|---|---|---|---|
| AI4Bharat IndicVoices-R Nepali | 13.74 | 5598 | 401 | CC-BY-4.0 |
| OpenSLR SLR143 (M+F TTS) | 1.24 | 662 | 2 | CC-BY-SA-4.0 |
| OpenSLR SLR43 (multi-speaker female TTS) | 2.80 | 2064 | 18 | CC-BY-SA-4.0 |
| Gemini-Flash Algenib (synthetic teacher) | 4.47 | 1984 | 1 | synthetic, public-release consent |
| Barsha recital (real human) | 1.08 | 498 | 1 | personal recording, project-internal |
| **v0 total** (IV-R + SLR143 + SLR43 + Algenib) | **22.0** | **10300** | **419** | CC-BY-SA-4.0 (most restrictive) |
| **v1 total** (Algenib + SLR143_F + Barsha + 2 SLR43) | **7.7** | **3818** | **5** | same |

All audio is normalized to 22.05 kHz mono. Algenib (Gemini Flash 24 kHz FLAC)
is resampled via `soxr`. Barsha (WebM/Opus 48 kHz from a browser-recorded
data-collection portal) is decoded with `ffmpeg` and resampled. We filter out
2 utterances (≈0.4%) from Barsha and 15 utterances (≈0.8%) from Algenib that
contain inline English (Latin-script tokens our G2P drops silently — would
create audio-text mismatch).

### 5.3 Engineering hazards we hit and the fixes

The training pipeline as initially configured was unusable for production.
We document the issues for downstream replicators.

**Hazard 1: cuDNN host-side dispatch overhead.** Initial training on AWS
g6e.2xlarge (L40S) at batch size 16 ran at ~0.14 it/s. Per-phase profiling
on a single training step showed `aten::convolution_backward` taking
10.27 s of CPU time against 54 ms of CUDA time on the same op. Root cause:
`torch.backends.cudnn.benchmark = True` (set in `piper_train/__main__.py:423`)
forces re-benchmarking on every novel input shape, and VITS feeds variable
spec/audio lengths per utterance. **Fix:** set `cudnn.benchmark = False`.
On a T4 g4dn.xlarge debug instance, batch size 8: median total step time
dropped from ~3000 ms to ~720 ms (~4.2× faster), with `g_bwd` alone going
from ~2400 ms to ~150 ms (~16× faster). We did not re-profile the L40S
after the fix; on g6e.xlarge our subsequent v2 / v3 / v4 runs reached
~26–80 s/epoch wall-clock pace at batch size 64 with the fix in place,
which is consistent with the T4 measurement carried over.

**Hazard 2: Cython monotonic alignment search.** Even with the cuDNN fix,
the original Jaehyeon-style MAS in piper's `monotonic_align/__init__.py`
copies `neg_cent` and `mask` from GPU → CPU as numpy, runs the
O(t_t·t_s) DP single-threaded on the CPU, and copies the path back. At
typical VITS shapes this is ~8 ms per call, but the synchronous transfer
serialises the whole step on CPU. **Fix:** drop in
`supertone-inc/super-monotonic-align@9bb1cb3`, a Triton GPU kernel
implementing the same algorithm on-device. Verified bit-identical to the
Cython baseline (max abs diff 0.0). The wrapper transposes the input
(`[b,t_t,t_s]` to `[b,t_x,t_y]`) since the conventions differ.

**Hazard 3: ComplexHalf STFT in mixed precision.** The `--precision bf16-mixed`
path errors with `cuFFT does not support BFloat16` because piper's
`mel_processing` calls `torch.stft` inside the autocast region. **Fix:** wrap
both STFT functions with `torch.amp.autocast(enabled=False)` and `.float()`
the inputs. Use `--precision 16-mixed` (FP16, fully cuFFT-supported on Ada
Lovelace).

**Hazard 4: frozen duration predictor in cross-lingual fine-tune.** The flag
`--resume-from-multispeaker-checkpoint` auto-enables `--freeze-dp`, intended
to preserve the multilingual base's duration predictor. For cross-lingual
transfer (multilingual base → Nepali) this is the wrong default: the dp
never adapts to Nepali phoneme timings. After 400 epochs of frozen-dp
training, our 4-utterance listening-test set showed predicted output
durations at 36–50% of ground-truth durations (utt42: 10.2 s vs GT 20.4 s;
utt1234: 1.5 s vs 4.2 s; utt2500: 5.3 s vs ~13 s; utt4789: 3.7 s vs ~9 s),
with informal listening agreeing the audio was largely unintelligible —
this despite val_loss bottoming at 44.77, comparable to later runs that
produced legible audio. **Fix:** add a `PIPER_UNFREEZE_DP=1` env-var
override in `piper_train/vits/lightning.py:configure_optimizers()` so the
duration predictor continues training. The unfrozen-dp continuation
produced markedly more legible audio at `length_scale=1.5` on the same 4
utterances (74–76% of ground-truth duration: 15.1 s, 3.2 s, 9.9 s, 6.6 s),
confirmed by informal listening but not by a controlled MOS study.

**Hazard 5: single-speaker collapse pathology.** Training the first run with
`--single-speaker` (collapsing all 401+ speakers to ID 0, `gin_channels=0`)
produced inconsistent voices at inference: each test utterance sounded like
*its training-time speaker*, not an averaged voice. Without speaker
conditioning, the only way for the model to encode "which speaker said this"
was to bind phoneme patterns to speaker realisations. Each utterance has
phoneme patterns idiosyncratic to its speaker; the model memorised
"phoneme-pattern → that voice." **Fix:** train multi-speaker with
`gin_channels=512`. Dataset re-prep, surgical checkpoint expansion of
`emb_g`, restart from base.

**Hazard 6: alphabetical speaker_id collisions on dataset extension.**
piper_preprocess assigns integer speaker IDs alphabetically. Adding a new
voice (e.g. `algenib`) reorders the existing IDs because `algenib` (lowercase)
sorts between uppercase `S<digits>` IV-R speakers and lowercase `slr143_F`.
Naive append-at-end checkpoint surgery places the new emb_g row at index 418
while the dataset expects it at index 398, with all `slr143_*` and `slr43_*`
rows shifted by +1. **Fix:** insert new rows at the correct alphabetically-sorted
index in the surgical expansion, not append at end. Verified via shift-summary
script before launch.

**Hazard 7: monotonic-align fix did not solve the throughput bottleneck on
its own.** We initially expected `super-monotonic-align` to drop step time
substantially. It did not — it merely removed the CPU stall but left the
host-side dispatch overhead. The actual fix was Hazard 1's `cudnn.benchmark=False`.
We document this because the obvious diagnosis (Cython MAS is the bottleneck)
is correct only as a *latent* issue; the *active* bottleneck was something
else.

### 5.4 Run results

| run | speakers | hours | epochs | val_loss min | status |
|---|---|---|---|---|---|
| v1 (single-speaker, frozen-dp) | 1 (collapsed) | 13.74 | 400 | 44.77 | superseded — frozen-dp pathology + single-speaker collapse made audio largely unintelligible on listening |
| v1+ (dp-unfreeze continuation) | 1 (collapsed) | 13.74 | +200 | 44.76 | superseded — audio improved at length_scale 1.5 but voice still inconsistent per utterance (collapse) |
| v2 (cold-start multi-speaker) | 418 | 17.8 | 600 | 44.30 | released as a checkpoint; folded into the v0 release alongside v3b |
| v3b (v2 fine-tune, +Algenib) | 419 | 22.3 | 200 | 44.07 | **released** as `ampixa/nepali-voices-v0` (epoch=199 final; the v3b run continues v2 with the Algenib synthetic-teacher voice added) |
| v4 (clean cold-start) | 5 | 7.7 | 1000 (target) | in progress | curated 5-voice corpus: algenib + barsha + slr143_F + 2× slr43. Will be released as `ampixa/nepali-tts-v1` once training and the auto-publish step complete |

The val_loss numbers are not strictly comparable across runs (different
dataset sizes change the validation distribution) but they do show steady
improvement of the encoder/decoder/dp shared parameters as we curate the
data better. Crucially, **val_loss is a poor proxy for TTS perceptual
quality**: v1's val_loss bottomed at 44.77 with mostly-unintelligible audio,
because the frozen-dp pathology was structural rather than reflected in the
training objective. We learned to ground every quality decision in actual
listening tests, not loss curves.

## 6. Released Artifacts

- **`ampixa/nepali-text-frontend`** — MIT, public. Python package with the
  G2P, lexicon (54K candidate, growing gold), tests, and audit trace
  contract. https://github.com/Ampixa/nepali-text-frontend
- **`ampixa/nepali-voices-v0`** — CC-BY-SA-4.0, public. 419-speaker VITS
  Lightning checkpoint (epoch=199, ~22 h training) + config + speaker_id_map.
  https://huggingface.co/ampixa/nepali-voices-v0
- **`ampixa/nepali-tts-v1`** — same license. 5-speaker clean cold-start.
  In progress at the time of writing; release pending v4 training
  completion plus an auto-publish step that uploads the final checkpoint
  + samples + speaker_id_map after the 999-th-epoch checkpoint and final
  per-speaker samples are produced.

The Hugging Face checkpoints are Lightning format (`.ckpt`) for now;
ONNX export for ARM/CPU deployment is straightforward via
`piper_train.export_onnx` and is the planned distribution for production.

## 7. Limitations and Failure Modes

We are explicit about the model's known failure modes.

### 7.1 Phonotactic gap on rare contexts

Our most common quality complaint from native listeners is that specific
words render with audible defects: चीन ("chin", China) sounds like "tsin";
पश्चिम ("paschim", west) sometimes shows initial-stop confusion; सहर ("sahar",
city) sometimes drops or lateralises the final /r/.

Empirically, the relevant phoneme contexts are rare in our training corpus:

| sequence | count in 10 196 utts | typical rendering quality |
|---|---|---|
| `ts i n` (chin start) | 124 | fumbled |
| `p ax s . ts i m` (paschim) | 170 | initial-stop confusion |
| `s ax . h ax r` (sahar) | 149 | final /r/ unstable |
| **control:** `k aa` | 6381 | clean |
| **control:** `m ax` | 5850 | clean |
| **control:** `s ax` | 9468 | clean |

A control bigram with ~50–100× more training instances renders cleanly on
the same speakers. Our problem clusters sit in a regime where the absolute
counts (~125–170 instances each) are not negligible, but the per-cluster
phoneme-context coverage may be too narrow for the model to articulate
reliably across the inflectional variants the test sentences contain. The
likely fixes are corpus-balancing-driven: more clean training audio
specifically containing those clusters — Barsha's recital corpus (target
~1.5 h when complete) and a synthetic Algenib expansion are the two natural
paths. Whether the gap closes at, say, ~500 instances per cluster or
requires more is empirical and we have not measured it.

### 7.2 The dialect mismatch (the Newari-accent perception)

In informal native-speaker feedback during evaluation, listeners from
outside the Kathmandu Valley have described the model's output as
Newari-accented. The most prominent marker they cite is `च` realised as
/ts/ rather than the /tʃ/ ("ch") of pan-Indic Indo-Aryan. The G2P emits
/ts/ per Khatiwada (2009), the training data is recorded by Kathmandu
speakers who pronounce it that way, and the model reproduces what is in
the audio. We have not run a controlled listening-panel evaluation to
quantify the perception, only collected the qualitative signal.

Within the scope of our policy this is the system doing what it documents
itself as doing, on data that matches the policy. Removing the Newari-style
realisations would require either (a) a different phonological policy
(`च` → `/tʃ/` and corresponding lexicon and inventory updates), or
(b) different training data recorded by speakers who use the /tʃ/
realisation, or both. Either is a v2 effort, not a v1 patch.

### 7.3 Unverified claims (honesty register)

Several quantitative claims in our internal documents are extrapolations or
small-sample measurements rather than fully verified results. We list the
most relevant ones so future readers know where to push back:

- The 99.5% NepTTS-Bench minimal-pair contrast preservation is computed by
  comparing G2P phone-sequence outputs of pair members. It does not verify
  that the *acoustic* model preserves these contrasts at synthesis time —
  only that the frontend emits distinct sequences.
- The 26.7% rule-fallback rate generalises from IRIIS + SLR54; other text
  domains (social media, technical, religious) likely have different
  distributions.
- The schwa-deletion rules are verified against Wikipedia's Indo-Aryan
  reference + Regmi (2025: 123); we have not seen a single Nepali-specific
  paper that *explicitly* validates this 4-rule decomposition.
- The व→`w` default is a project decision that may not match every
  educated speaker's intuition (Khatiwada parenthetically lists `(w)`,
  acknowledging some complexity in the phoneme's distribution).

## 8. Future Work

1. **Dialect flag in v2 of the frontend.** Expose a configuration option that
   selects between Kathmandu-Newari (the current policy) and pan-Indic /
   Hill / Indo-Aryan-broad pronunciation. This is mostly a lexicon and
   alias-table question; the inventory itself does not need to grow.
2. **Targeted phonotactic-gap data.** Synthesise or record ~30–60 minutes of
   audio specifically containing the problem clusters identified in §7.1,
   with multiple phonotactic contexts per cluster.
3. **ONNX deployment on Hetzner ARM.** Export the v1 checkpoint via
   `piper_train.export_onnx`, deploy as the production backend for
   tts.ampixa.com. RTF 0.05–0.15 is realistic on a 4-vCPU CAX21 for
   ~€8/mo, vs ~$1340/mo always-on L40S.
4. **Native-speaker review of the gold lexicon.** Currently 54K candidate
   entries; gold is empty pending reviewer recruitment. The candidate →
   gold workflow is documented in `docs/lexicon-sources.md`.
5. **Style modes.** `careful_sanskritized`, `loan_aware`, and
   `spelling_pronunciation` are documented in the policy doc but not
   implemented end-to-end.
6. **External baselines.** Comparative MOS / ASR-CER evaluation against
   eSpeak NG, Indic Parler-TTS, Kokoro (multilingual), and the published
   Tacotron/FastPitch Nepali baselines.

## 9. Conclusion

We argue, on the evidence in this paper, that a usable Nepali TTS system
benefits from cumulative honesty about three things. First, the phonological
policy must be anchored in primary sources for the specific dialect being
served, not in pan-Indic defaults that look correct in print but render
wrong on the speakers we have. Second, the training pipeline benefits from
end-to-end auditing because failures stack silently in our experience:
cuDNN dispatch overhead hid behind a Cython MAS bottleneck that turned
out not to be the active issue; a frozen duration predictor hid behind a
val_loss curve that looked acceptable; a single-speaker collapse hid
behind audio that was listenable in isolation but not coherent across
utterances. Third, within the scope of our policy and training data, the
released system speaks a specific dialect — Kathmandu Newari-influenced
Nepali — and we document that policy choice rather than presenting the
output as register-neutral.

The frontend, model, training journals, and this paper are all public.
Where our claims are weak we mark them as weak; where they are strong we
cite the source that makes them strong; where the model fails we name the
failure. We expect a v2 of the frontend and the model will revisit several
of the policy decisions documented here, especially the dialect target,
once we have the data and the listener-panel evaluation to do so
responsibly.

## References

Acharya, J. (1991). *A Descriptive Grammar of Nepali and an Analyzed Corpus*.
Georgetown University Press.

Acharya, S. (1974). *वर्णोच्चारण शिक्षा*.

Agrawal, S. S., Bansal, S., & Sinha, S. (2019). Comparative Acoustic-Phonetic
Analysis of Retroflex Consonants of Some Indian Languages. *Proceedings of
the ICA Congress*.

Bandhu, C. M., Dahal, B. M., Holzhausen, A., & Hale, A. (1971). *Nepali
Segmental Phonology*. SIL / Tribhuvan University, Kirtipur.

Bohara, B. B. (2023). A Comparative Study of Nepali and English Phonology.
*NepJOL*.

Clements, G. N., & Khatiwada, R. (2007). Phonetic realization of contrastively
aspirated affricates in Nepali. *16th International Congress of Phonetic
Sciences*, Saarbrücken.

Dongol, I., & Bal, B. K. (2023). Transformer-based Nepali Text-to-Speech.
*ICON 2023*.

Khadka, B., G. C., S., Paudel, P., Shah, R., & Joshi, B. (2023). Nepali
Text-to-Speech Synthesis using Tacotron2. *SIGUL 2023*. ISCA Archive.

Masica, C. P. (1991). *The Indo-Aryan Languages*. Cambridge University
Press. (cited as the standard general reference for Indo-Aryan schwa
deletion in §3.3).

Khatiwada, R. (2008). Some acoustic cues in the detection of the Nepalese
aspiration. *Journal of the Acoustical Society of America*.

Khatiwada, R. (2009). Nepali. *Journal of the International Phonetic
Association*, 39(3), 373–380.

Khatiwada, R. (2015). Nepalese retroflex stops: a static palatography study
of inter- and intra-speaker variability.

Khatiwada, R. (2019). Retroflexion in Nepali. *Gipan*, 4, 19–29.

Pokharel, M. P. (1989). *Experimental Analysis of Nepali Sound System* (PhD
thesis). University of Pune.

Regmi, B. N. (2025). Variation in Nepali Verb Roots: A Phonological
Perspective. *Gipan*, 7(1), 121–135.

Schwarz, M., Sonderegger, M., & Goad, H. (2019). Realization and
representation of Nepali laryngeal contrasts: Voiced aspirates and laryngeal
realism. *Journal of Phonetics*.

Sodimana, K., Pipatsrisawat, K., Ha, L., Jansche, M., Kjartansson, O.,
Pasternack, P., Silfverberg, R., Quéré, S., & Henderson, K. (2018).
A Step-by-Step Process for Building TTS Voices Using Open Source Data
and Frameworks for Bangla, Javanese, Khmer, Nepali, Sinhala, and
Sundanese. *SLTU 2018*. (Cited for the methodology of
`google/language-resources/ne`.)

### Software and data

- `ayousanz/piper-plus`: https://github.com/ayutaz/piper-plus (Python-3.12
  fork of rhasspy/piper)
- `ayousanz/piper-plus-base`: https://huggingface.co/ayousanz/piper-plus-base
- `supertone-inc/super-monotonic-align@9bb1cb3`: Triton MAS kernel
- `google/language-resources` `ne/`, commit
  `a90601024f07ab7e96c911243324e0adeb16da96` (CC-BY-4.0)
- AI4Bharat IndicVoices-R: https://huggingface.co/datasets/ai4bharat/IndicVoices-R
- OpenSLR-43: https://www.openslr.org/43/
- OpenSLR-143: https://www.openslr.org/143/
- NepTTS-Bench (Ampixa): https://huggingface.co/datasets/ampixa/neptts-bench
- Khatiwada (2009) PDF: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/A887820E3B1293EFD43FBD3259D41F73/S0025100309990181a.pdf/nepali.pdf

### Wider TTS landscape (cited in §2.3)

- Kokoro 82M ONNX: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
- StyleTTS2: https://arxiv.org/abs/2306.07691
- F5-TTS: https://arxiv.org/abs/2410.06885
- E2-TTS: https://arxiv.org/abs/2406.18009
- CosyVoice / CosyVoice2: https://arxiv.org/abs/2407.05407, https://arxiv.org/abs/2412.10117
- AI4Bharat Indic Parler-TTS: https://huggingface.co/ai4bharat/indic-parler-tts
