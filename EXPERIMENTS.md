# Reproducing the paper

Every number in `paper/ContextCanvas.tex` comes from files in `paper/generated/`, written by the
scripts below. Never edit those files or type results into the paper by hand.

## 0. Setup (once)

```bash
pip install pydantic kuzu networkx openai   # or your existing environment
python download_propbank.py            # pins PropBank 3.4 to an exact commit -> data/propbank_lock.json
ollama pull qwen3.5:9b                 # extractor (all configurations except extractor_gemma)
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
| `configs/main.json` | qwen3.5:9b | qwen3.8:27b | main results, ablations |
| `configs/reader_gemma.json` | qwen3.5:9b | gemma4:26b | second model family (reader only) |
| `configs/small.json` | qwen3.5:9b | qwen3.5:9b | does a small reader suffice? |
| `configs/extractor_gemma.json` | gemma4:12b | qwen3.8:27b | optional: does the extractor matter? |

All configurations share one extraction cache (`results/extraction_cache`), keyed by extractor
model, prompt and text. The first three therefore extract the paragraphs only **once**; the
others re-run only answering. `extractor_gemma` re-extracts everything, so it is the most
expensive and is optional.

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
python -m evaluation.experiments --data $D --config configs/small.json --out results/small --n 500 --seed 13 --systems main
# optional
python -m evaluation.experiments --data $D --config configs/extractor_gemma.json --out results/extractor_gemma --n 500 --seed 13 --systems contextcanvas
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
python -m evaluation.annotation export --cache results/extraction_cache --extractor qwen3.5:9b --out annotation/sample.csv --n 200 --seed 7
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
    --paper-dir paper/generated \
    --models "Main=results/main,Gemma reader=results/reader_gemma,Small (9B only)=results/small"
cd paper && latexmk -pdf ContextCanvas.tex
```

Then resolve every red **Author note** box in the PDF.

## Runtime

The first held-out run extracts about 10,000 paragraphs (500 questions × 20). With a 9B
extractor this takes several times less than with the 27B model you used for development, but
measure it in the pilot before planning. Every later system and configuration re-runs only
answering: one or two reader calls per question.
