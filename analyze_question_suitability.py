"""Well-posedness analysis for the post "What an LLM can and cannot find".

Reproduces the post's two-axis table from two small, de-identified inputs in ./data/. It does NOT
include the user-proxy pipeline, the interview transcripts, or the proxy's generated answers; only
per-question / per-participant labels.

Method (matches the post):
  The grader runs multiple times on each transcript and, on each run, makes a binary call: is this
  question ANSWERABLE from the transcript, or not? For each (question, participant) that is a coin
  flipped r times. The clarity signal is the WITHIN-participant variance of that coin, averaged over
  participants, then rescaled to a bounded [0, 1] "well-posedness":

    p          = fraction of the r runs that called the question answerable, for one participant
    within_var = mean_over_participants[ p (1 - p) ]          # within-participant Bernoulli variance
    well_posedness = 1 - within_var / 0.25                    # 0.25 = max of p(1-p), at p = 0.5

  WITHIN-participant (the r re-runs for one person), NOT total variance: total also carries
  between-participant spread (some transcripts address the question, some don't), which is base rate,
  not well-posedness. We report well-posedness, and pct_answerable separately as the base rate.

The two axes of the post:
  x = well_posedness        (grader self-consistency on "answerable or not"; human-free)
  y = human_pass_rate       (the researcher's prior-reading content pass rate)

Run: python analyze_question_suitability.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROW = HERE / "data/answerability_runs.csv"
HUMAN = HERE / "data/human_pass_rate_per_question.csv"
OUT = HERE / "outputs/question_ranking.csv"

# Questions to exclude because their topic is too rare in the data to estimate a stable within-
# participant variance (here: present in only ~2/50 transcripts). This list is SPECIFIC TO THIS
# DATASET, not universal. If you run the analysis on your own data, inspect pct_answerable / the
# per-question sample size and drop whatever is too rare for *you* to measure.
DROP = {"Q25", "Q07"}
# Each run column (run01, run02, ...) holds the grader's answerability classification for that run.
# It has three labels (see README): A and B both mean "answerable" (directly stated / inferable),
# C means "not answerable". We collapse to binary. The number of run columns is the r above.
ANSWERABLE = {"A", "B"}
MAX_VAR = 0.25                                            # theoretical max of p(1-p), at p = 0.5
BOOT_REPS, BOOT_SEED, CI = 5000, 42, 90                  # 90% cluster bootstrap over participants


def load_cells() -> pd.DataFrame:
    """One row per (participant, question): the answerable fraction across the r runs."""
    df = pd.read_csv(ROW)
    df = df[df["qid"].str.startswith("Q") & ~df["qid"].isin(DROP)].copy()
    runs = [c for c in df.columns if c.startswith("run")]
    df["p"] = df[runs].isin(ANSWERABLE).mean(axis=1)        # answerable fraction over the r runs
    df["within"] = df["p"] * (1 - df["p"])                  # within-participant Bernoulli variance
    return df[["qid", "bundle_id", "p", "within"]]


def human_pass() -> pd.Series:
    return pd.read_csv(HUMAN).set_index("qid")["human_pass_rate"]


def boot_ci(within: np.ndarray) -> tuple[float, float]:
    """90% CI for well_posedness, resampling whole participants (clusters)."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(within)
    wp = [1 - within[rng.integers(0, n, n)].mean() / MAX_VAR for _ in range(BOOT_REPS)]
    lo, hi = (100 - CI) / 2, 100 - (100 - CI) / 2
    return float(np.percentile(wp, lo)), float(np.percentile(wp, hi))


def main() -> None:
    cells = load_cells()
    human = human_pass()
    rows = []
    for qid, sub in cells.groupby("qid"):
        within = sub["within"].mean()
        lo, hi = boot_ci(sub["within"].values)
        rows.append({
            "qid": qid,
            "well_posedness": round(1 - within / MAX_VAR, 3),
            "ci_lo": round(lo, 3),
            "ci_hi": round(hi, 3),
            "within_participant_var": round(within, 4),
            "human_pass_rate": round(float(human.get(qid, np.nan)), 2),
            "pct_answerable": round(sub["p"].mean(), 2),
            "between_participant_var": round(sub["p"].var(ddof=0), 3),
        })
    # rank 1 = least well-posed (the hardest to transmit)
    t = pd.DataFrame(rows).sort_values(
        ["well_posedness", "qid"], ascending=[True, True]).reset_index(drop=True)
    t.insert(0, "rank", t.index + 1)
    qtext = pd.read_csv(ROW).drop_duplicates("qid").set_index("qid")["evaluation_question"]
    t["question"] = t["qid"].map(qtext)

    OUT.parent.mkdir(exist_ok=True)
    t.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(HERE)}  ({len(t)} questions)")
    print(t.drop(columns="question").to_string(index=False))


if __name__ == "__main__":
    main()
