# Two-transcript end-to-end integration test

This is a small execution check for the public replication harness, not an
estimate of population performance. On 2026-07-17, the complete OpenAI path was
run from two public AnthropicInterviewer transcripts through repeated
answerability classification, user-proxy generation, content grading against a
previously frozen human reference, aggregation, and final question analysis.

## Frozen test condition

- Selection universe: the published 50-transcript primary cohort.
- Eligibility: all 23 canonical question rows present, no researcher `skip`
  rows, and no malformed text in the pre-existing human-reference fields.
- Sampling: two transcripts selected without replacement with seed `20260717`.
- Selected public IDs: `work_0466` and `work_0245`.
- Provider and model: OpenAI, `gpt-5.6-luna`.
- Reasoning effort: `none`.
- Answerability repeats: 9 independent calls per transcript-question pair.
- User-proxy generation: 1 call per pair.
- Content grading: 1 call per pair after proxy generation.

The answerability grader saw only a transcript and one question. It did not see
the proxy answer, human type decision, human reference prose, or content-grader
result. The content grader separately received the proxy response and the
applicable frozen human-reference fields.

## Execution result

All stages completed:

| Stage | Requested | Valid outputs |
| --- | ---: | ---: |
| Answerability grader | 46 pairs x 9 repeats = 414 | 414 |
| User proxy | 46 pairs x 1 | 46 |
| Content grader | 46 pairs x 1 | 46 |

The final strict content pass count was **35/46 (0.761)**. Across the 23
questions, 22 had well-posedness `1.000` in this tiny sample. Q01 had
well-posedness `0.654` with a 90% participant-bootstrap interval of
`0.309-1.000`: one transcript was labeled A in all nine runs, while the other
was labeled B seven times and C twice.

These numbers prove that the interfaces compose and that the outputs reach the
final analysis. They should not be interpreted as stable model-quality or
question-ranking estimates: there are only two participants, so the confidence
intervals and per-question content rates are intentionally uninformative.

## Resume behavior exercised

An initial answerability pass at concurrency 10 produced 232 successful calls
and 182 provider rate-limit failures. Rerunning the same command at concurrency
2 reused the immutable manifest and append-only ledger, skipped the 232
successful calls, and filled only the missing calls. The exported wide table
contained all 414 required judgments. This checks the documented resume path;
it is not a claim that the reference runner implements a universal provider
retry policy.

## Reproducibility record

The private run artifacts were hashed after completion:

| Artifact | SHA-256 |
| --- | --- |
| `answerability_runs.csv` | `0b431f7640ec524570cfb25f7d70a2400bced61f472b33fb370774daa1b3683c` |
| `proxy_answers.csv` | `4e87f66f825584fb6e47ef45831c2345e189139fe8200db86be3a696970213ee` |
| `content_grader_results.csv` | `fb9fedaeeed77232a141b5342c73023f10169616e8397eb29476a5f1fa7ad126` |
| `human_pass_rate_per_question.csv` | `2212836acc2d76ae8c6d74c88fdabee6fbf72918dc195f86ddbc9f0237d28128` |
| `question_ranking.csv` | `ed8cf43f1175f894c703fbb1504a0ed7993c7169b52420f6fed390ead7d7bf55` |

The public repository does not include transcript text, human-reference prose,
proxy responses, content-grader rationales, API ledgers, or secrets. Releasing
those private rows is unnecessary to demonstrate that the reference harness
runs end to end. Readers can reproduce the same interfaces with the public
dataset and their own human references, as documented in the main README.
