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
prepares a new public transcript sample, runs repeated answerability judgments,
and documents how to construct the independent human reference axis.

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
  run_answerability.py                resumable repeated classifier runner
  aggregate_human_reference.py        strict per-question aggregation
templates/
  human_reference_template.csv        blank private-work template
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

1. **Is answerability stable?** Repeated LLM grader runs classify whether each
   transcript answers each research question.
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
the released Layer 1 analysis, but they provide the first part of the Layer 2
replication kit.

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

## Run repeated answerability judgments

Set `ANTHROPIC_API_KEY`, choose an exact model identifier, and start with a
small smoke test:

```bash
python scripts/run_answerability.py \
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

Each ledger has an immutable manifest containing hashes of the transcript,
cohort, question, and prompt inputs plus the model and run settings. A changed
condition must use a new ledger, preventing accidental mixing across models or
harness configurations.

Remove `--max-pairs` and use `--runs 9` with the default ledger for the
historical design. With the default 50-transcript cohort and 23 questions, that
design makes **10,350 API calls**. Estimate cost and rate limits before
launching it. The runner requires an explicit model rather than silently
substituting whichever model is current.

The exact classifier and user-proxy prompts are stored under `prompts/`. The
answerability classifier sees only the transcript and question; it does not see
the proxy response or any researcher annotation.

## Build the human reference axis

Use `templates/human_reference_template.csv` and
`docs/human_reference_protocol.md` to create a study-specific reference before
running the user proxy. The template contains only the fields required by the
final method and includes an explicit researcher `skip` option.

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
reference point under a documented, use-case-specific protocol. A simplified
blank template and protocol may be added later as part of a broader replication
kit.

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
