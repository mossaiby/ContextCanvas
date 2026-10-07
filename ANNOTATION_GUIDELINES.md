# Extraction annotation guidelines

## Purpose

End-to-end accuracy mixes up three error sources: extraction, retrieval and reasoning. This
study measures **extraction alone**: when the system turns a paragraph into events, how often
are those events correct, and how much does it miss?

The study is a **hybrid**:

* An **LLM judge** (a strong model, reached through an OpenAI-compatible API, from a different
  family than the extractor and the readers) annotates the
  whole sample, following the rules in this document, which serve verbatim as its prompt.
* **Two human annotators** independently annotate a smaller subset of the same paragraphs.
* On that subset we measure agreement between the humans (Cohen's κ, human A vs human B) and
  between each human and the judge. If the judge agrees with the humans about as well as they
  agree with each other, its numbers on the full sample are credible, and that is measured
  rather than assumed.

The humans must work **independently**: each judges every row without seeing or discussing the
other's answers, and without seeing the judge's answers.

## Who

* Two people who read English well and understand what "who did what to whom" means.
  Linguistics or NLP experience helps but is not required.
* Ideally at least one of them was **not** involved in building ContextCanvas. Someone who
  knows how the system works tends to read its output generously.
* A third person (or the two annotators together, after both finish) resolves disagreements
  for the adjudicated version.

## Effort

The human subset is about 35 paragraphs with ~8 events each, so ~280 rows. At 20–40 seconds
per row, plan **2–3 hours per annotator**. The judge annotates the full sample (200 paragraphs
by default) without human effort.

## Setup

1. Export the full sample, then draw the human subset from it:
   ```bash
   python -m evaluation.annotation export --cache results/extraction_cache --extractor qwen3.5:9b --out annotation/sample.csv --n 200 --seed 7
   python -m evaluation.annotation subset annotation/sample.csv --n 35 --seed 11 --out annotation/human_subset.csv
   ```
2. Copy `human_subset.csv` to `annotator_a.csv` and `annotator_b.csv`. Give one copy to each person.
3. **Calibration (about 1 hour, together):** both annotate the same 10 paragraphs taken from a
   *development* run, not from the sample. Discuss every disagreement and refine these rules.
   This is the only time they compare answers. **Freeze the rules afterwards:** the judge must
   use the final version, so run it only after calibration.
4. **Main pass (separately):** each fills their own file. No discussion until both are done.
5. **Judge:** `python -m evaluation.llm_judge annotation/sample.csv --config configs/judge.json --out annotation/annotator_llm.csv`
6. Score all three:
   ```bash
   python -m evaluation.annotation score annotation/annotator_a.csv annotation/annotator_b.csv \
       annotation/annotator_llm.csv --labels "Annotator A,Annotator B,LLM judge" \
       --judge-meta annotation/annotator_llm.meta.json --out paper/generated/tab_annotation.tex
   ```

Open the CSV in a spreadsheet program (LibreOffice, Excel). Don't change any column except the
four you fill in, and don't reorder, add or delete rows.

## What each row is

Each paragraph appears as a block of rows. The paragraph text is shown on its first row only.
Each row is one event the system extracted:

| column | meaning |
|---|---|
| `lemma`, `sense_id` | the relation, e.g. `employ`, `employ.01` |
| `roles` | the participants, e.g. `{":ARG0": "DLR", ":ARG1": "Ulrich Walter"}` |
| `time`, `place` | the time or place attached to the event, if any |

The `role_meanings` column gives the meaning of each role for that sense (for example,
`parent.01`: ARG0 = parent, ARG1 = child). If it is empty, search for the sense at
<http://propbank.github.io/v3.4.0/frames/>.

<!-- judge-rules:start -->
## What to fill in

### `event_correct` (1 or 0, every row)

**1** if the paragraph states, or directly and unambiguously implies, this relation between
these participants. **0** otherwise.

* Synonyms are fine: `employ(DLR, Ulrich Walter)` is correct for "Walter worked at the DLR".
* A participant may be a shortened or fuller form of the name in the text ("Walter" or "Ulrich Walter").
* **0** if any participant is wrong, invented, or a whole clause instead of a name.
* **0** if the relation is not stated, even if it is true in the real world. Judge the paragraph, not your knowledge.
* **0** for vague or empty events (`be(Paris)` with nothing else).
* Duplicates: if two rows say the same thing, mark both on their own merits.

### `roles_correct` (1 or 0, only when `event_correct` = 1)

**1** if every participant is in the right role, so the direction of the relation is right.
**0** if any participant is in the wrong role. Leave it blank when `event_correct` = 0.

* `parent.01 {ARG0: Peter, ARG1: Johan}` for "Johan is Peter's son" → **1** (ARG0 is the parent).
* The same with the names swapped → **0**.
* `own.01 {ARG0: Bombardier Aerospace, ARG1: Bombardier Inc.}` for "Bombardier Aerospace is a division of Bombardier Inc." → **0** (the owner must be ARG0).

### `context_correct` (1, 0 or blank, only when `event_correct` = 1)

Fill this only if the row shows a `time` or `place`.
**1** if that time or place is stated for **this** event. **0** if it belongs to another
event or is wrong. Example: "Walter, born in 1953, joined DLR in 1988": `employ … time 1953` → **0**.

### `missed_facts` (a number, on the first row of each paragraph only)

Count the relations between **two named entities** that the paragraph states but no row
captures, whether that row was marked correct or not.

* Count each missing relation once.
* Count only relations between named things (people, places, organisations, works,
  products). Don't count descriptions ("is a large city"), numbers, or dates on their own.
* A relation counts as captured only if some row states it with the right participants.
  Wrong roles are fine for this column.
* Use 0 if nothing is missing.

## Common cases

| Situation | Decision |
|---|---|
| "X (born 1950 in Y)" extracted as `bear.02 {ARG1: X, ARG2: Y}` | event 1, roles 1 |
| A pronoun or "the film" resolved to the right name | treat as the name: may be 1 |
| A pronoun resolved to the wrong name | event 0 |
| A place split into a chain: `locate(Canyon, Texas)`, `locate(Texas, United States)` from "Canyon, Texas, United States" | each 1 |
| `alias.01 {ARG0: German Aerospace Center, ARG1: DLR}` when the text says "German Aerospace Center (DLR)" | 1 |
| A relation stated only as a possibility ("may have been born in…") | 1 only if the event is not presented as certain; otherwise 0 |

When unsure, decide. Human annotators: note the row number and reason in a separate notes file;
unsure cases are what the calibration and adjudication steps are for.
<!-- judge-rules:end -->

## About the LLM judge

The text between the `judge-rules` markers in this file is the judge's instruction, word for
word. The judge script records a hash of it, so **any edit to the rules after the judge has run
means running the judge again**. The judge sees the same columns as the humans, including the
role meanings, and returns the same four judgments. Every raw response is logged.

Two known weaknesses, to check in the agreement table:

* `missed_facts` is the hardest column for a judge. Models are better at verifying stated facts
  than at noticing what is absent. If judge–human agreement is poor there, report recall from
  the humans only.
* A judge may be lenient towards plausible but unstated relations. Compare its event precision
  with the humans' on the subset; a judge that is consistently more generous inflates precision.

## Reporting

The paper reports, per annotator on the shared subset and for the judge on the full sample:
event precision, role-direction accuracy, time/place accuracy and missed facts per paragraph;
and Cohen's κ for event and role correctness between every pair of annotators. As a rough guide,
κ ≥ 0.6 is usually read as substantial agreement. If κ is low, the guidelines were unclear:
refine them on new calibration paragraphs and annotate again; never just pick the better
annotator. Report the annotators' background and the time spent in the paper.
