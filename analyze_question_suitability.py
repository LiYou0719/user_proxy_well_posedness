"""Analyze question well-posedness from repeated answerability judgments.

The canonical estimator is intentionally bounded:

    p = fraction of repeated runs labeled A or B
    within_variance = mean_over_participants[p * (1 - p)]
    well_posedness = 1 - within_variance / 0.25

No finite-sample correction is applied. The resulting well-posedness score is
therefore always in [0, 1].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS_PATH = HERE / "data/answerability_runs.csv"
QUESTIONS_PATH = HERE / "data/questions.csv"
HUMAN_PATH = HERE / "data/human_pass_rate_per_question.csv"
OUTPUT_PATH = HERE / "outputs/question_ranking.csv"

ANSWERABLE = {"A", "B"}
VALID_LABELS = ANSWERABLE | {"C"}
MAX_VARIANCE = 0.25
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 42
CI_PERCENT = 90


def load_questions(path: Path = QUESTIONS_PATH) -> pd.DataFrame:
    questions = pd.read_csv(path, dtype=str)
    required = {"qid", "question"}
    if set(questions.columns) != required:
        raise ValueError(f"questions file must contain exactly {sorted(required)}")
    if questions.isna().any().any() or (questions == "").any().any():
        raise ValueError("questions file contains empty values")
    if questions["qid"].duplicated().any():
        raise ValueError("questions file contains duplicate qids")
    if not questions["qid"].str.fullmatch(r"Q\d{2}").all():
        raise ValueError("canonical qids must use the Q00 format")
    if len(questions) != 23:
        raise ValueError(f"expected 23 canonical questions, found {len(questions)}")
    return questions


def load_cells(
    path: Path = RUNS_PATH,
    questions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    questions = load_questions() if questions is None else questions
    expected_qids = set(questions["qid"])
    runs = pd.read_csv(path, dtype=str)
    required = {"bundle_id", "qid"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"answerability file is missing columns: {sorted(missing)}")

    run_columns = sorted(c for c in runs.columns if c.startswith("run"))
    if not run_columns:
        raise ValueError("answerability file has no run columns")
    unexpected_columns = set(runs.columns) - required - set(run_columns)
    if unexpected_columns:
        raise ValueError(
            f"answerability file has unexpected columns: {sorted(unexpected_columns)}"
        )
    if runs[list(required) + run_columns].isna().any().any():
        raise ValueError("answerability file contains missing identifiers or labels")
    if runs.duplicated(["bundle_id", "qid"]).any():
        raise ValueError("answerability file contains duplicate participant-question rows")

    actual_qids = set(runs["qid"])
    if actual_qids != expected_qids:
        missing_qids = sorted(expected_qids - actual_qids)
        extra_qids = sorted(actual_qids - expected_qids)
        raise ValueError(
            f"answerability qids do not match questions; missing={missing_qids}, "
            f"extra={extra_qids}"
        )

    observed_labels = set(runs[run_columns].stack())
    invalid_labels = sorted(observed_labels - VALID_LABELS)
    if invalid_labels:
        raise ValueError(f"invalid answerability labels: {invalid_labels}")

    cells = runs[["bundle_id", "qid"]].copy()
    cells["p_answerable"] = runs[run_columns].isin(ANSWERABLE).mean(axis=1)
    cells["within_variance"] = cells["p_answerable"] * (
        1 - cells["p_answerable"]
    )
    return cells


def load_human_pass_rates(
    path: Path = HUMAN_PATH,
    questions: pd.DataFrame | None = None,
) -> pd.Series:
    questions = load_questions() if questions is None else questions
    expected_qids = set(questions["qid"])
    human = pd.read_csv(path)
    if set(human.columns) != {"qid", "human_pass_rate"}:
        raise ValueError("human pass-rate file must contain qid and human_pass_rate")
    if human["qid"].duplicated().any():
        raise ValueError("human pass-rate file contains duplicate qids")
    if set(human["qid"]) != expected_qids:
        raise ValueError("human pass-rate qids do not match canonical questions")
    if human["human_pass_rate"].isna().any() or not human[
        "human_pass_rate"
    ].between(0, 1).all():
        raise ValueError("human pass rates must be between 0 and 1")
    return human.set_index("qid")["human_pass_rate"]


def bootstrap_ci(
    within: np.ndarray,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    ci_percent: int = CI_PERCENT,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(within)
    sampled_means = np.empty(repetitions)
    for i in range(repetitions):
        sampled_means[i] = within[rng.integers(0, n, n)].mean()
    scores = 1 - sampled_means / MAX_VARIANCE
    tail = (100 - ci_percent) / 2
    return float(np.percentile(scores, tail)), float(
        np.percentile(scores, 100 - tail)
    )


def build_ranking(
    cells: pd.DataFrame,
    questions: pd.DataFrame,
    human_pass_rates: pd.Series,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> pd.DataFrame:
    rows = []
    for qid, group in cells.groupby("qid"):
        within_values = group["within_variance"].to_numpy()
        within = float(within_values.mean())
        well_posedness = 1 - within / MAX_VARIANCE
        ci_lo, ci_hi = bootstrap_ci(
            within_values, repetitions=bootstrap_repetitions
        )
        rows.append(
            {
                "qid": qid,
                "n_participants": len(group),
                "well_posedness": round(well_posedness, 3),
                "ci_lo": round(ci_lo, 3),
                "ci_hi": round(ci_hi, 3),
                "within_participant_var": round(within, 4),
                "human_pass_rate": round(float(human_pass_rates.loc[qid]), 2),
                "pct_answerable": round(float(group["p_answerable"].mean()), 2),
                "between_participant_var": round(
                    float(group["p_answerable"].var(ddof=0)), 3
                ),
            }
        )

    ranking = pd.DataFrame(rows).sort_values(
        ["well_posedness", "qid"], ascending=[True, True]
    )
    ranking = ranking.reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    question_lookup = questions.set_index("qid")["question"]
    ranking["question"] = ranking["qid"].map(question_lookup)

    bounded_columns = ["well_posedness", "ci_lo", "ci_hi", "human_pass_rate"]
    if not ranking[bounded_columns].apply(lambda column: column.between(0, 1)).all().all():
        raise ValueError("a normalized metric fell outside [0, 1]")
    return ranking


def main() -> None:
    questions = load_questions()
    cells = load_cells(questions=questions)
    human_pass_rates = load_human_pass_rates(questions=questions)
    ranking = build_ranking(cells, questions, human_pass_rates)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    ranking.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH.relative_to(HERE)} ({len(ranking)} questions)")
    print(ranking.drop(columns="question").to_string(index=False))


if __name__ == "__main__":
    main()
