# Which interview questions are well posed for an LLM user proxy?

This repository contains the public analysis package for
[What an LLM can and cannot find for product teams in user interview transcripts](https://guanjie.li/Thinking-out-louder/2026/06/18/what-LLM-can-and-cannot-find.html).
It is designed to help other researchers **replicate the method** with their own
models, harnesses, samples, and research questions.

Replication does not mean recovering the exact published scores. Model versions
and inference systems change, and another transcript sample should produce
different numbers. The reusable contribution is the measurement logic: combine
repeated LLM answerability judgments with a separate human content check, and
inspect both axes before treating a research question as suitable for an LLM
user proxy.

## What is included

This repository has two layers. Layer 1 reruns the published analysis without
API credentials or transcript downloads. The optional Layer 2 replication kit
prepares a new public transcript sample and provides an executable reference
path through human annotation, repeated answerability judgments, user-proxy
generation, gold-based content grading, and final analysis.

The repository itself contains no transcript text, proxy responses, or
participant-level researcher annotations.

```text
analyze_question_suitability.py
requirements-replication.txt
data/
  questions.csv                       canonical 23-question instrument
  answerability_runs.csv              repeated A/B/C judgments
  human_pass_rate_per_question.csv    aggregate pass counts, denominators, and rates
outputs/
  question_ranking.csv                generated reference output
prompts/                              historical prompt artifacts and schemas
scripts/
  prepare_dataset.py                  pinned public dataset download
  sample_cohort.py                    deterministic stratified sampling
  generate_human_reference_templates.py
                                      one 23-row annotation file per transcript
  merge_human_references.py           strict validation and long-table export
  run_answerability.py                resumable repeated classifier runner
  run_user_proxy.py                   resumable full-context proxy runner
  run_content_grader.py               resumable gold-based content grader
  aggregate_human_reference.py        strict per-question aggregation
templates/
  human_reference_template.csv        canonical annotation columns
docs/
  human_reference_protocol.md         minimal annotation protocol
tests/
  ...                                 analysis and replication-kit tests
```

The `bundle_id` values refer to participants in Anthropic's public
[AnthropicInterviewer dataset](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer).
They provide provenance for the released labels without including participant
prose.

## The two reference points

The study asks two different questions.

1. **Is the answerability judgment stable?** Repeated LLM grader runs classify
   whether each transcript answers each research question.
2. **Is the resulting content right?** A separate content pass rate compares
   the proxy response with the researcher's reading made before the LLM run.

The first produces the well-posedness axis. The second produces the aggregate
`human_pass_rate`. Neither is a substitute for the other.

## Answerability labels

Each grader run assigns one of three labels:

| Label | Meaning |
| --- | --- |
| A | The answer is directly stated in the transcript. |
| B | The answer is not stated but is reasonably inferable. |
| C | The transcript does not answer the question. |

The analysis collapses A and B to **answerable** and C to **not answerable**.
The A/B/C distinction follows the answerability scheme introduced by Park et
al. (2024), [*Generative Agent Simulations of 1,000 People*](https://arxiv.org/abs/2411.10109).

These terms refer to different levels of the measurement:

- an **answerability judgment** is one A/B/C label from one grader run;
- `pct_answerable` is the prevalence of A/B judgments across the sample;
- `well_posedness` measures whether repeated judgments are stable for the same
  transcript-question pair, whether they consistently say answerable or not.

A question can therefore have low `pct_answerable` but high `well_posedness` if
the grader consistently labels most transcripts C.

## Bounded well-posedness estimator

For one participant and question, let `p` be the fraction of repeated grader
runs labeled A or B.

```text
within_variance = mean over participants of p * (1 - p)
well_posedness  = 1 - within_variance / 0.25
```

`p * (1 - p)` ranges from 0 to 0.25, so well-posedness is always in `[0, 1]`.
A score near 1 means the grader consistently sees the question as answerable or
not answerable for the same transcript. A lower score means it changes its mind
across repeated runs.

This is **within-participant** instability. It is intentionally separate from
`pct_answerable`, the proportion of transcripts that address the topic. A rare
but clear question should not be confused with an ambiguous question.

### Correction history

An earlier version of the article's interactive chart applied a `9/8`
finite-sample correction to the nine-run variance even though the appendix
specified the bounded empirical estimator above. The chart was corrected in
July 2026. The current article, appendix, reference output, and analysis code
now use the same bounded estimator. This history is disclosed because the
pre-correction chart showed slightly lower well-posedness values.

## Run the analysis

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analyze_question_suitability.py
python -m unittest discover -s tests
```

The analysis validates the input schemas before calculating the output. It
rejects unexpected question IDs, invalid or missing A/B/C labels, duplicate
participant-question rows, and out-of-range normalized metrics.

## Prepare a new transcript sample

The dataset preparation utilities are optional. They are not needed to rerun
the released Layer 1 analysis, but they begin the Layer 2 replication workflow.

```bash
pip install -r requirements-replication.txt
python scripts/prepare_dataset.py
python scripts/sample_cohort.py --seed 42
```

`prepare_dataset.py` downloads all 1,250 public AnthropicInterviewer
transcripts and writes a local Parquet file plus a provenance manifest. The
historical dataset revision is pinned in the script; callers may explicitly
select another revision. Transcript text and generated cohorts remain under
the ignored `local_data/` directory.

`sample_cohort.py` draws a deterministic sample without replacement. Its
default 30/10/10 split mirrors the historical study's workforce, creative, and
scientist composition, but the seed is intentionally user-selectable. The
historical participant IDs remain auditable in `answerability_runs.csv`; using
the identical cohort is not required for a methodological replication.

## End-to-end replication

The reference workflow runs from the pinned public Hugging Face dataset to a
new 23-row question ranking. The only substantive manual step is Step 4: a
researcher reads each transcript and creates an independent human reference
before any proxy answers are generated. Run all commands from the repository
root with Python 3.10 or newer.

### One-time setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-replication.txt
export OPENAI_API_KEY="YOUR_KEY"
```

The commands below use OpenAI as one concrete reference path. To use Anthropic,
set `ANTHROPIC_API_KEY` and replace the provider and model arguments. Always
record the exact model identifiers used; do not treat a moving alias as a
frozen experimental condition. As an alternative to exporting a key, add
`--env-file .env` to each model-runner command; key values and the file path are
not written to manifests or ledgers.

### 1. Download the pinned AnthropicInterviewer revision

```bash
python scripts/prepare_dataset.py
```

This downloads revision `c9e1ec1e6b093712b9c42235c7303ece647490e9`, validates
the expected 1,250 rows, and writes:

- `local_data/anthropic_interviewer.parquet`
- `local_data/anthropic_interviewer.manifest.json`

Both files stay under the git-ignored `local_data/` directory.

### 2. Draw the deterministic 30/10/10 cohort

```bash
python scripts/sample_cohort.py --seed 42
```

The default sample contains 30 workforce, 10 creative, and 10 scientist
transcripts, selected without replacement. It writes `local_data/cohort.csv`
with the split and sampling seed for each of the 50 participants. Choose and
report another seed if the replication is intended to test a different cohort.
The sample size is not fixed: change `--workforce`, `--creatives`, and
`--scientists`, and every downstream script will use the resulting cohort size.

### 3. Generate one human-reference file per sampled transcript

```bash
python scripts/generate_human_reference_templates.py
```

This creates one 23-row file per participant under
`local_data/human_reference/by_transcript/`. It refuses to overwrite an
existing annotation file.

### 4. Manually complete and freeze the human reference

**This is the manual step.** Read one transcript end to end, then complete its
23-row CSV according to `docs/human_reference_protocol.md`. Repeat for every
sampled transcript. Each annotation filename is the participant's
`transcript_id`; use that ID to open the matching row in
`local_data/anthropic_interviewer.parquet` with
your preferred Parquet viewer or analysis environment. For Type A or B, at
least one of `gold_answer`, `gold_evidence`, or `notes` must contain the
researcher's reference. For Type C, the C decision is sufficient;
`absence_check` is optional. A `skip` requires `skip_reason`.

After annotation, validate and merge the files:

```bash
python scripts/merge_human_references.py
```

The merger requires exactly one file per sampled transcript and all 23
canonical rows in each file. It writes `local_data/human_reference.csv`.
Freeze this file before running any model-facing stage: later edits would
contaminate the independent reference point.

### 5. Run repeated answerability classification

```bash
python scripts/run_answerability.py \
  --provider openai \
  --model YOUR_ANSWERABILITY_MODEL_ID \
  --runs 9 \
  --eligibility local_data/human_reference.csv
```

The answerability grader sees only one transcript and one question. The
eligibility file is used solely to omit researcher `skip` rows; human labels and
reference prose are never sent in this stage. The output is
`local_data/answerability_runs.csv`, with nine A/B/C judgments for each
evaluated participant-question pair.

### 6. Calculate well-posedness, confidence intervals, and answerable rate

```bash
python analyze_question_suitability.py \
  --runs local_data/answerability_runs.csv \
  --well-posedness-only \
  --output local_data/well_posedness_summary.csv
```

This produces 23 rows containing `well_posedness`, the 90% participant-bootstrap
interval (`ci_lo`, `ci_hi`), and `pct_answerable`. It deliberately does not use
the human reference or content-grader results.

### 7. Run the user proxy

```bash
python scripts/run_user_proxy.py \
  --provider openai \
  --model YOUR_USER_PROXY_MODEL_ID \
  --eligibility local_data/human_reference.csv
```

The proxy sees the full transcript and one research question, but no human
type, gold field, or notes. It writes `local_data/proxy_answers.csv`.

### 8. Grade proxy content against the human reference

```bash
python scripts/run_content_grader.py \
  --provider openai \
  --model YOUR_CONTENT_GRADER_MODEL_ID
```

The grader joins `local_data/proxy_answers.csv` with the frozen
`local_data/human_reference.csv`, applies the appropriate Type A/B/C rubric,
and writes `local_data/content_grader_results.csv`.

### 9. Calculate per-question content correctness

```bash
python scripts/aggregate_human_reference.py \
  --input local_data/content_grader_results.csv \
  --output local_data/human_pass_rate_per_question.csv
```

This writes one strict content pass count, evaluated count, and pass rate for
each of the 23 questions. Type A `partial` judgments remain failures under the
published strict metric.

### 10. Build the final 23-row question ranking

```bash
python analyze_question_suitability.py \
  --runs local_data/answerability_runs.csv \
  --human-summary local_data/human_pass_rate_per_question.csv \
  --output local_data/question_ranking.csv
```

The final file joins well-posedness, its bootstrap interval, answerability base
rate, and strict content pass rate. The analysis verifies that the answerability
and content denominators agree for every question. `skip` pairs are removed
from the answerability, proxy, and content-grader universes together.

### Running cost, smoke tests, and resume behavior

Every model runner uses an append-only JSONL ledger and immutable manifest.
Rerunning an interrupted command with the same condition skips successful
calls and attempts only missing ones. A changed model, prompt, input, repeat
count, or run setting requires a new ledger path; the runner refuses to mix
conditions.

Before a full run, use `--max-pairs` with separate smoke-test ledger and output
paths. Do not reuse those smoke-test paths for the full condition. For example:

```bash
python scripts/run_user_proxy.py \
  --provider openai \
  --model YOUR_USER_PROXY_MODEL_ID \
  --eligibility local_data/human_reference.csv \
  --max-pairs 2 \
  --ledger local_data/proxy_smoke_calls.jsonl \
  --output local_data/proxy_smoke_answers.csv
```

A complete 50-transcript run with no skips makes 10,350 answerability calls,
1,150 proxy calls, and 1,150 content-grader calls. Estimate provider cost and
rate limits before launching it. Both `anthropic` and `openai` are supported
reference adapters; their SDK behavior and inference infrastructure are not
standardized by this repository.

## Run repeated answerability judgments

Set the API key for your provider (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`),
choose an exact model identifier, and start with a small smoke test:

```bash
python scripts/run_answerability.py \
  --provider anthropic \
  --model YOUR_MODEL_ID \
  --max-pairs 2 \
  --runs 2 \
  --ledger local_data/smoke_calls.jsonl \
  --output local_data/smoke_runs.csv
```

The runner appends every call to `local_data/answerability_calls.jsonl`, so an
interrupted run can resume without repeating successful calls. It exports the
Layer 1 compatible `local_data/answerability_runs.csv` only after every
requested call succeeds.

The runner supports Anthropic tool use and OpenAI Structured Outputs while
keeping the classifier instructions, transcript, question, and output fields
the same. Each ledger has an immutable manifest containing hashes of the
transcript, cohort, question, and prompt inputs plus the provider, model, and
run settings. A changed condition must use a new ledger, preventing accidental
mixing across models or harness configurations.

API-call count is a design choice, not a fixed requirement:

```text
calls = evaluated transcript-question pairs x repeated runs
```

A complete 50-transcript, 23-question matrix repeated nine times would make
10,350 calls. The historical study made **10,287 calls** because seven Q03
pairs were explicitly skipped: `(22 x 50 + 43) x 9`. A smaller replication
might use 10 transcripts and five runs (up to 1,150 calls); a larger one might
use 100 transcripts and five runs (up to 11,500 calls). Report the evaluated
pair count, per-question exclusions, and repeat count so readers can interpret
the resulting uncertainty and compare conditions. Estimate cost and rate
limits before launching any substantial run.

The most robust way to omit researcher abstentions is to pass the frozen local
human-reference CSV directly with `--eligibility`. The runner requires exactly
one row for every cohort-question pair, includes every A/B/C row, and omits only
rows whose `type` is `skip`. It never sends the human type or any other
annotation field to the model. This keeps the answerability and human-content
denominators aligned without manually copying IDs or leaking an answer into the
prompt.

```bash
python scripts/run_answerability.py \
  --provider openai \
  --model YOUR_MODEL_ID \
  --eligibility local_data/human_reference.csv
```

For workflows that do not use the supplied human-reference template, a local
CSV containing only `bundle_id,qid` can instead be passed with `--exclusions`.
Both modes reject unknown or duplicate pairs, are mutually exclusive, and add
the input file hash to the run manifest. These private files can remain under
`local_data/`; the public report needs only the resulting evaluated count for
each question.

The runner requires an explicit provider and model rather than silently
substituting whichever model is current. For example, an OpenAI smoke test can
use `--provider openai --model gpt-5-nano --reasoning-effort minimal`;
substantive replications should choose a model appropriate to their validation
goal rather than optimizing only for minimum cost. You may pass a local key
file with `--env-file .env`; the key and file path are not written to the run
manifest or ledger.

The exact classifier and user-proxy prompts are stored under `prompts/`. The
answerability classifier sees only the transcript and question; it does not see
the proxy response or any researcher annotation. The user-proxy runner may use
the frozen human reference to omit `skip` pairs, but human types and gold fields
are discarded before any model request is constructed.

## Build the human reference axis

Use `scripts/generate_human_reference_templates.py` and
`docs/human_reference_protocol.md` to create a study-specific reference before
running the user proxy. The generator writes one file per sampled transcript,
using the column contract shown in `templates/human_reference_template.csv`,
and includes an explicit researcher `skip` option. After annotation,
`scripts/merge_human_references.py` validates and combines the files.

After grading proxy responses with the published content-grader prompts,
aggregate a private grader-output file into a shareable per-question summary:

```bash
python scripts/aggregate_human_reference.py \
  --input local_data/content_grader_results.csv \
  --output local_data/human_pass_rate_per_question.csv
```

The public summary contains only `qid`, pass count, evaluated count, and pass
rate. Type A `partial` judgments remain failures under the published strict
metric. Completed annotations, evidence, and proxy prose remain local.

## Reusing the method

To replicate the method:

1. define and freeze the research questions;
2. select a transcript sample and record its source and sampling procedure;
3. run the same answerability grader repeatedly for each transcript-question
   pair;
4. export one row per pair with `bundle_id`, `qid`, and `run01`, `run02`, ...
   columns containing A/B/C labels;
5. construct an independent human reference point appropriate to the product
   decision and record its aggregate pass count and evaluated count per question;
6. run this analysis and inspect individual questions, uncertainty intervals,
   and base rates rather than only an overall average.

Pin and report the model identifier, prompt, system/user message placement,
sampling or reasoning settings, structured-output approach, harness/library
version, number of runs, and run date. Changing those conditions is a useful
replication, but it may change the scores.

The original cohort is not required. Sampling a new cohort tests whether the
pattern generalizes. Likewise, the original 23 questions are examples of
measurement instruments, not a universal UXR question set. New questions need
their own construct boundaries and human calibration.

## Historical data notes

- Q07 and Q25 were excluded because their topics appeared in only about 2 of 50
  transcripts in the historical sample. A replicator should make exclusions
  based on coverage in their own sample.
- Like every other canonical question, Q03 began with 50
  participant-question pairs. The researcher explicitly marked seven Q03 pairs
  as `skip` because the human answerability decision remained genuinely
  borderline. Those seven pairs were excluded before proxy generation and
  grading; they were not coded as unanswered questions or model failures. The
  remaining 43 pairs entered both calculations: 35 passed the human content
  check, and all 43 contributed to the well-posedness estimate. No pairs were
  skipped for the other questions, so all 50 entered their calculations. The
  analysis validates that the two evaluated denominators match and reports
  `n_participants` explicitly.

## Human annotation boundary

The public file contains only per-question aggregate pass counts, evaluated
counts, and rates. Completed prior readings, expected participant answers,
evidence selections, absence checks, researcher notes, and exploratory
adjudication are intentionally not published.

Those private annotations encode researcher judgment and are not necessary for
applying the method to a new study. A replication should create its own human
reference point under the documented, use-case-specific protocol. The supplied
generator creates one blank 23-question annotation file per sampled transcript;
the strict merger converts completed files into the private long-table input
used by the reference runners.

## Limits

- Well-posedness measures grader consistency, not truth.
- Human pass rates depend on the researcher's question definition and decision
  context; they are not universal ground truth.
- Confidence intervals overlap for many middle-ranked questions at this sample
  size. Interpret broad regimes and recurring failure patterns, not tiny rank
  differences.
- Harness and model choices can introduce systematic effects. Internal
  consistency does not guarantee comparability across configurations.
- Directed question answering and open-ended qualitative discovery are
  different jobs. This method evaluates the former.
