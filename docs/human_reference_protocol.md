# Minimal human reference protocol

This protocol creates a study-specific human reference point without assuming
that the researcher is universal ground truth. Complete it before running the
user proxy for the same transcript-question pairs.

## Unit of annotation

The reference interface uses one CSV per sampled transcript, with one row for
each of the 23 canonical questions. Generate the files after sampling:

```bash
python scripts/generate_human_reference_templates.py
```

Read one transcript end to end, then complete its 23-row file under
`local_data/human_reference/by_transcript/`. This batching is an annotation
convenience, not a required research design: another interface is acceptable
if it exports the same fields and preserves one `bundle_id,qid` pair per row.

## Answerability type

- **A — explicitly stated:** The participant directly answers the question or
  an essentially equivalent interviewer question. Record at least one usable
  human reference in `gold_evidence`, `gold_answer`, or `notes`.
- **B — reasonably inferable:** No direct answer is present, but the transcript
  supports a sufficiently clear inference. Record the inference, its evidence
  chain, or an equivalent explanation in at least one of the three reference
  fields. More detail may improve grader confidence but is not mandatory.
- **C — not answered:** The topic is not addressed and no reasonable inference
  is available. The Type C decision itself is the minimum reference. An
  `absence_check` or `notes` entry is recommended when the boundary was not
  obvious, but is not required.
- **skip — researcher abstention:** The researcher cannot defend an A, B, or C
  decision. Record a `skip_reason` rather than forcing a label.

Optional `notes` may record boundary decisions that will help interpret the
content grade. Do not consult a proxy response while writing any field.

## Validate and combine completed files

After every cohort file is complete, run:

```bash
python scripts/merge_human_references.py
```

The merger checks the full cohort-by-question universe, exact question text,
valid A/B/C/skip types, and that Type A/B rows contain at least one usable human
reference field. It writes the frozen machine-facing reference to
`local_data/human_reference.csv`. Do not edit the merged file directly; correct
its per-transcript source and merge again.

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
