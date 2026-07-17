# Prompt artifacts

These files preserve the final prompt text relevant to the published study
without preserving the legacy experiment harness.

## Answerability classifier

`answerability_classifier_system.txt` is the system instruction used by the
final stage-1 classifier. It sees only one transcript and one research question;
it never sees a user-proxy response or researcher annotation. Its structured
output is defined in `answerability_output_schema.json`: A, B, or C plus a
rationale and confidence level.

The reference runner sends the transcript as a second system block and sends
the following user message:

```text
QUESTION:
{{QUESTION}}

Classify this (transcript, question) pair using the tool.
```

## User proxy

`user_proxy_system.txt` is the final full-context user-proxy prompt for the
Anthropic interviews. Replace `{{TRANSCRIPT}}` with one transcript and send the
research question as the user message. It is published for methodological
inspection; the answerability runner does not need to invoke the proxy.

## Content grader

`content_grader_system.txt`, `content_grader_user_template.txt`, and the three
`content_grader_type_*.txt` files preserve the final gold-based grading prompt.
A replication combines the system prompt with the rubric matching the
researcher's A/B/C annotation and provides the question, researcher gold
fields, and proxy output in the user message. Structured labels are defined in
`content_grader_output_schemas.json`.

The historical structured labels were:

- Type A: `correct`, `partial`, `wrong`, `abstained`
- Type B: `inferred_correctly`, `abstained`, `wrong`
- Type C: `abstained_correctly`, `non_abstained`

For the published strict human-reference metric, only `correct`,
`inferred_correctly`, and `abstained_correctly` counted as passes. In
particular, a Type A `partial` judgment was a failure in the published
aggregation.

The completed researcher gold records are intentionally absent. These prompt
files make the measurement rule inspectable without publishing participant-level
human annotations.

## Harness boundary

Model provider, model version, reasoning or sampling settings, concurrency,
retry policy, and structured-output implementation remain experimental
conditions. A replication should hold them constant within a run and report
them. Exact scores are not expected to transfer across harnesses.
