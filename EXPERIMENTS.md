# Reproducing the paper

Every number in `paper/ContextCanvas.tex` comes from files in `paper/generated/`, written by the
scripts below. Never edit those files or type results into the paper by hand.

## 0. Setup (once)

```bash
pip install pydantic kuzu networkx openai   # or your existing environment
python download_propbank.py            # pins PropBank 3.4 to an exact commit -> data/propbank_lock.json
ollama pull qwen3.5:9b                 # small extractor and reader (extractor_small, small)
ollama pull qwen3.8:27b                # reader (main)
ollama pull gemma4:26b                 # reader (second model family)
```

* Commit `data/propbank_lock.json`. Later downloads fetch exactly that commit and fail if its checksum changes.
* If `download_propbank.py` reports that `v3.4.0` is not a tag, it prints the tags that exist.
  Re-run with `--ref <tag>`; if no release tag exists, pin a commit SHA with `--ref <sha>`.
* Changing the PropBank release changes sense selection. Re-run everything after the first pin,
  including the pilot.

## 1. Model configurations

| config | extractor | reader | role in the paper |
|---|---|---|---|
| `configs/main.json` | qwen3.8:27b | qwen3.8:27b | main results, ablations |
| `configs/reader_gemma.json` | qwen3.8:27b | gemma4:26b | second model family (reader only) |
| `configs/extractor_small.json` | qwen3.5:9b | qwen3.8:27b | effect of extractor size |
| `configs/small.json` | qwen3.5:9b | qwen3.5:9b | everything small |

All configurations share one extraction cache (`results/extraction_cache`), keyed by extractor
model, prompt and text. Each extractor extracts the paragraphs **once**: `main` and
`reader_gemma` share the 27B extraction, `extractor_small` and `small` share the 9B one. Every
other system re-runs only answering.

Why the 27B extractor: on the 20-question pilot it raised facts mode from 9 to 13 correct and
retrieval mode from 12 to 14, for about 15% more indexing time (80 vs. 70 s per question).

## 2. Pilot (development questions only)

Run the whole pipeline on 20 **development** questions before spending days on the real runs:

```bash
python -m evaluation.experiments --data external_data/musique_ans_dev.jsonl --config configs/main.json \
    --out results/pilot_main --pool dev --n 20 --systems main,ablations
python -m evaluation.report --results results/pilot_main --paper-dir results/pilot_main/generated
```

`--pool dev` samples only from the 50 development questions, so the pilot can't leak anything
about the held-out questions. Check:

1. `results/pilot_main/generated/results.md`: EM roughly where the development runs were (about 60–70%).
2. `predictions.jsonl` of `contextcanvas`: few `"error"` entries, and the extraction did not
   fail. A failed extraction shows up as many questions with `failure_stage` = `extraction_zero_events` or
   `retrieval_empty`.
3. The `usage` fields: seconds per question × 500 gives you the runtime estimate.
4. Repeat for `configs/reader_gemma.json` and `configs/small.json` (with `--systems main`), so each model is confirmed to work.

If anything needs fixing, fix it, delete `results/pilot_*`, and pilot again. After the
held-out runs start, no more code changes.

## 3. Freeze

Run `pytest tests/`; everything must pass. Commit the code, configs and lock file, and tag the
commit (for example `paper-v1`). The experiment driver refuses to continue an output directory
whose config file changed.

## 4. Held-out runs

```bash
D=external_data/musique_ans_dev.jsonl
python -m evaluation.experiments --data $D --config configs/main.json --out results/main --n 500 --seed 13 --systems main,ablations
python -m evaluation.experiments --data $D --config configs/reader_gemma.json --out results/reader_gemma --n 500 --seed 13 --systems main
python -m evaluation.experiments --data $D --config configs/extractor_small.json --out results/extractor_small --n 500 --seed 13 \
    --systems contextcanvas,contextcanvas_retrieval,contextcanvas_retrieval_nofill
python -m evaluation.experiments --data $D --config configs/small.json --out results/small --n 500 --seed 13 --systems main
```

Using the same `--n`, `--seed` and `--exclude-first` (default 50) gives every configuration the
same questions; the report checks this. Runs resume after interruption.

## 5. Storage benchmark

```bash
python run_harness.py        # writes benchmark_summary.json
```

## 6. Extraction annotation (hybrid: two humans + LLM judge)

Follow `ANNOTATION_GUIDELINES.md`. In short, after step 4:

```bash
python -m evaluation.annotation export --cache results/extraction_cache --extractor qwen3.8:27b --out annotation/sample.csv --n 200 --seed 7
python -m evaluation.annotation subset annotation/sample.csv --n 35 --seed 11 --out annotation/human_subset.csv
# humans: calibrate, then fill annotator_a.csv and annotator_b.csv (copies of human_subset.csv) independently
export OPENAI_API_KEY=...        # or the variable named in configs/judge.json
python -m evaluation.llm_judge annotation/sample.csv --config configs/judge.json --out annotation/annotator_llm.csv
python -m evaluation.annotation score annotation/annotator_a.csv annotation/annotator_b.csv \
    annotation/annotator_llm.csv --labels "Annotator A,Annotator B,LLM judge" \
    --judge-meta annotation/annotator_llm.meta.json --out paper/generated/tab_annotation.tex
```

* Set `model` in `configs/judge.json` first. Choose a strong model from a different family than
  the extractor and readers. `base_url` can point to any OpenAI-compatible endpoint.
* Run the judge only after the humans' calibration, once the rules are final. The judge records
  a hash of the rules; editing them later means running it again.
* Keep `annotation/annotator_llm.log.jsonl`. It holds every raw judge response and is the record
  of the judgment, since hosted models change or are retired.
* `--extractor` restricts the sample to the main configuration's extraction model; the cache is
  shared by all configurations.

## 7. Tables and numbers

```bash
python -m evaluation.report --results results/main --storage benchmark_summary.json \
    --reference contextcanvas_retrieval \
    --paper-dir paper/generated \
    --models "Main=results/main,Gemma reader=results/reader_gemma,Small extractor=results/extractor_small,Small (9B only)=results/small"
cd paper && latexmk -pdf ContextCanvas.tex
```

Then resolve every red **Author note** box in the PDF.

## Runtime

Each extractor processes about 10,000 paragraphs (500 questions × 20). On the pilot, the 27B
extractor needed about 80 s per question (≈11 hours for 500) and the 9B extractor about 70 s
(≈10 hours). Every later system and configuration re-runs only answering: one or two reader calls
per question, roughly 5–15 s each.
