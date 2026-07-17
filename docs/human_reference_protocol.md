# Minimal human reference protocol

This protocol creates a study-specific human reference point without assuming
that the researcher is universal ground truth. Complete it before running the
user proxy for the same transcript-question pairs.

## Unit of annotation

Annotate one `bundle_id` and `qid` pair per row in
`templates/human_reference_template.csv`.

## Answerability type

- **A — explicitly stated:** The participant directly answers the question or
  an essentially equivalent interviewer question. Record the relevant
  `gold_evidence`; add a concise `gold_answer` when useful.
- **B — reasonably inferable:** No direct answer is present, but the transcript
  supports a sufficiently clear inference. Record both the evidence chain and
  the inferred `gold_answer`.
- **C — not answered:** The topic is not addressed and no reasonable inference
  is available. Record a short `absence_check` describing what was checked.
- **skip — researcher abstention:** The researcher cannot defend an A, B, or C
  decision. Record a `skip_reason` rather than forcing a label.

## Freeze and exclusion rules

1. Read the transcript and complete the reference row without consulting a
   user-proxy response for that pair.
2. Freeze the row before generating or grading the proxy response.
3. Exclude `skip` rows before proxy generation and grading.
4. Exclude the same `skip` rows from both the human pass-rate denominator and
   the repeated-answerability denominator. Never convert a researcher
   abstention into Type C or a model failure.
5. Report evaluated counts per question so exclusions remain visible.

The answerability runner can apply this decision directly with
`--eligibility PATH_TO_FROZEN_REFERENCE.csv`. It validates that the reference
contains exactly one row for every cohort-question pair, runs all A/B/C rows,
and omits only `skip` rows. Human types, answers, evidence, and notes are never
included in model input. This filtering step fixes the evaluation universe; it
does not provide the model with a correct answer.

This minimal protocol includes only the fields and decisions required by the
final study.
