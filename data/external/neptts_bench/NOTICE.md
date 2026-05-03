# NepTTS-Bench (sentences mirror)

This directory mirrors the sentence list from
[Ampixa/neptts-bench](https://github.com/Ampixa/neptts-bench), the
existing Nepali TTS benchmark. The full repository also contains a
rating-app, MOS predictor, evaluation package, and reference audio —
those are not mirrored here; we only use the text test cases.

## Pinned source

| Field | Value |
| --- | --- |
| Repository | `Ampixa/neptts-bench` |
| Branch | `main` |
| File mirrored | `benchmark/sentences.json` |
| Fetched | 2026-04-30 |
| Total sentences | 365 |
| Format | JSON list of objects: `sent_id`, `text_devanagari`, `text_romanized`, `word_count`, `category`, `phonetic_targets`, `contrast_word`, `pair_id` |

## Benchmark categories

| Category | Sentences | Use for |
| --- | ---: | --- |
| `phonological_minimal_pairs` | 149 | Contrast preservation (~75 pairs) |
| `natural_speech` | 160 | Coverage / robustness on real text |
| `question_intonation` | 10 | Prosody (deferred for v0) |
| `robustness` | 10 | Edge cases, tongue twisters, rare words |
| `phrases` | 10 | Multi-word units |
| `emotion` | 8 | Prosody (deferred) |
| `contrastive_stress` | 8 | Prosody (deferred) |
| `homographs` | 6 | Disambiguation (lexicon work) |
| `newspaper_ambiguities` | 4 | Ambiguity handling |

## License

**MIT License**. See `LICENSE`.

## Citation

```bibtex
@article{neptts-bench-2026,
  title={NepTTS-Bench: A Comprehensive Benchmark for Nepali Text-to-Speech Evaluation},
  author={Ampixa},
  year={2026}
}
```

## Use in this project

- `eval/run_neptts_bench.py` runs our frontend on every sentence,
  scores phonological-minimal-pair contrast preservation, and reports
  coverage on natural-speech sentences.
- Results land in `reports/neptts_bench_scorecard.md`.
- Prosody categories are run for trace-logging but not scored at v0
  (we have no prosody chunker yet).
