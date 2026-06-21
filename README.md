# Which interview questions are well-posed for an LLM user-proxy

This is the data analysis behind the post **["What an LLM can and cannot find for product teams in user interview transcripts"](https://guanjie.li/Thinking-out-louder/2026/06/18/what-LLM-can-and-cannot-find.html)**.
It reproduces the post's two-axis table from two small, **de-identified** input files.

It does **not** include the user-proxy pipeline, the interview transcripts, or the proxy's generated
answers. Everything here runs on per-question / per-participant **labels only**
(the grader's answerable/not calls and pass/fail flags); there is no participant prose in this repo.

## The idea

A user-proxy reads one participant's interview transcript and predicts how they would answer a
research question; a separate LLM grader scores that prediction. Before asking whether the proxy's
answer is *right*, we ask whether the question is **well-posed** for this setup at all: can the grader
even consistently decide whether the question is *answerable* from the transcript? A question that an
LLM can't consistently classify as answerable doesn't have a stable enough target for any proxy to be
reliably held to.

## How well-posedness is measured

The grader runs **multiple times** (here, `r` runs) on each of the transcripts. On each run it
makes a binary call: is this question **answerable** from the transcript, or **not**? For each
(question, participant) that is a coin flipped `r` times.

```
p              = fraction of the r runs that called the question answerable (one participant)
within_var     = mean over participants of  p · (1 − p)      # within-participant variance
well_posedness = 1 − within_var / 0.25                       # 0.25 = max of p(1−p), at p = 0.5
```

If the signal is clear the grader lands the same way every time (`p` = 0 or 1, variance 0,
well-posedness 1). If it's borderline it flip-flops (`p` near ½, variance near its 0.25 max,
well-posedness near 0).

We use **within-participant** variance (the r re-runs for one person), averaged over participants —
**not** total variance. Total variance also contains *between-participant* spread (some transcripts
address the question, some don't), which is **base rate**, not well-posedness. We report well-posedness
and `pct_answerable` (the base rate) separately, so a rare-but-clear question is not mistaken for an
ambiguous one.

This axis is **human-free** (LLM self-consistency only). The post crosses it with one human axis:

- **human_pass_rate** — the content pass rate when the proxy's answer is checked against the
  researcher's own prior reading of the transcript. *"Given an answer, is it actually right?"*

A question is only reliable when it clears **both** axes. The interesting failures are the off-diagonal
cells: a well-posed question the proxy still answers wrongly, and a low-well-posedness question whose
content nonetheless checks out (the grader just wavers on a rubric line). See the post for the verdict
on each question and the three recurring hard cases (emotional/affective, fine-grained behavioral, and
normative questions).

## How the human pass rate is measured

The y-axis is grounded in the researcher's own prior reading of the transcripts, recorded **before any
LLM was run**. For each (question, participant) the researcher read the transcript and noted the
expected answer and its answerability type (the same A/B/C). The proxy's answer is then scored against
that gold reading on a **type-aware pass** criterion:

| gold type | the proxy passes if it… |
|-----------|--------------------------|
| A (stated) | gives the correct answer |
| B (inferable) | correctly infers it |
| C (not answerable) | correctly abstains |

`human_pass_rate` for a question is the fraction of participants whose proxy answer passed, in a single
grading run. We ship only the 23 per-question aggregates (`human_pass_rate_per_question.csv`), not the
per-participant gold judgments because they are not the reusable part. The
gold answers, rubric, and grader prompts are **use-case-specific**. A team grading for a different
product purpose would write different gold answers and a different bar for "correct", so *our* encoding
wouldn't carry over.

**To reproduce on your own data:** build your own gold standard (read your transcripts; for each
question record the expected answer and its answerability type under your own rubric), grade your
proxy's answers against it with whatever pass criterion fits your use, and take the per-question pass
fraction. Drop those numbers into `human_pass_rate_per_question.csv` (`qid, human_pass_rate`) and the
script pairs them with well-posedness automatically. The well-posedness axis is computed for you; this
axis is yours to define.

## Where the answerable/not label comes from (needed to reproduce)

The post describes the grader's call as simply **answerable vs not**. Under the hood, each grader run
first assigns the question one of **three** answerability types for that transcript:

| label | meaning |
|-------|---------|
| **A** | the answer is **directly stated** in the transcript |
| **B** | the answer is **not stated but reasonably inferable** from it |
| **C** | the transcript does **not** address the question (not answerable) |

This three-way "is the answer stated / inferable / absent" typing follows the answerability scheme
introduced by Park et al. (2024), *Generative Agent Simulations of 1,000 People*
([arXiv:2411.10109](https://arxiv.org/abs/2411.10109)). To get the binary axis used in the post,
**collapse A and B into "answerable" and C into "not answerable"**. This is what
`analyze_question_suitability.py` does (`ANSWERABLE = {"A", "B"}`). The raw A/B/C labels are kept in
the data file so the collapse is reproducible and auditable.

## Files

```
analyze_question_suitability.py     the analysis → outputs/question_ranking.csv
requirements.txt                    pandas, numpy
data/
  answerability_runs.csv            per (question, participant): the grader's A/B/C answerability
                                    call for each of the r runs
  human_pass_rate_per_question.csv  per question: the human prior-reading content pass rate (23 numbers)
outputs/
  question_ranking.csv              generated table (see below)
```

`data/answerability_runs.csv` columns: `bundle_id`, `qid`, `evaluation_question`, then one column per
run (`run01`, `run02`, …) each holding that run's answerability label (A/B/C). `bundle_id` identifies
a participant in Anthropic's publicly released
[**AnthropicInterviewer**](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer) interview
transcripts (the source material for this study) so it references only already-public data, not
anything private; the file itself contains no transcript text. (The number of `run*` columns is the
`r` above; the script discovers them automatically, so any `r` works.)

## Run

```bash
pip install -r requirements.txt
python analyze_question_suitability.py
```

## Output table (`outputs/question_ranking.csv`)

One row per question, ranked so **rank 1 = least well-posed** (the hardest to transmit):

| column | meaning |
|--------|---------|
| `well_posedness` | `1 − within_var / 0.25`, in `[0, 1]` (1 = the grader is perfectly self-consistent) |
| `ci_lo`, `ci_hi` | 90% CI for `well_posedness`, cluster bootstrap over participants |
| `within_participant_var` | the underlying within-participant answerable/not variance (`well_posedness = 1 − this / 0.25`) |
| `human_pass_rate` | the human prior-reading content pass rate (the post's y-axis) |
| `pct_answerable` | base rate: fraction of participants for whom the topic is present |
| `between_participant_var` | between-participant spread of the answerable rate (reported separately, never summed in) |

## Reusing this on your own data

The two files in `data/` are the de-identified labels from *this* study — shipped so the post's table
is fully reproducible. The analysis itself is meant to be **reused**: run your own proxy + grader
pipeline, then produce an `answerability_runs.csv` with the same schema (one row per
question × participant; the grader's A/B/C answerability call in a `run01`, `run02`, … column for
each of your `r` re-runs) and a per-question human pass-rate file, and the script gives you the same
two-axis table for your questions. The script discovers however many `run*` columns you have, so any
`r` works; set `DROP` to your own too-rare questions.

## Honest limits

- **No single human-free number decides suitability.** Well-posedness surfaces candidates; the human
  content check confirms. A well-posed question the proxy still answers *wrongly* is invisible to the
  automated axis and caught only by the human read. A small human calibration sample stays
  necessary by design.
- **Well-posedness is the LLM's own consistency, not ground truth.** It says the answerable/not signal
  is (or isn't) cleanly classifiable by this pipeline; it is silent on whether an answer is correct.
- **The bootstrap CIs overlap across the middle of the range** at N = 50, so fine per-question rank is
  not separable there — read the extremes and the regimes, not exact positions. N = 50 is admittedly
  on the small side: it cleanly separates the clearly-well-posed from the clearly-not, but not the
  middle. I didn't have the time or resources to run a larger sample. If you reuse this and need
  finer-grained, statistically separable per-question rankings, feel free to obtain a larger transcript sample.
  the method scales straightforwardly with N.
- **Two questions (Q07, Q25) are dropped *in this dataset*** because their topic appears in only
  ~2/50 transcripts — too rare to estimate a stable variance. That exclusion list is dataset-specific:
  if you reuse this on your own data, check `pct_answerable` / per-question sample size and drop
  whatever is too rare for you (`DROP` at the top of the script).
