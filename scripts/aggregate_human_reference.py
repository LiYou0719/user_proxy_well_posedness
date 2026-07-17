"""Aggregate strict content-grader judgments without publishing human records."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PASS_LABEL = {
    "A": "correct",
    "B": "inferred_correctly",
    "C": "abstained_correctly",
}
VALID_JUDGMENTS = {
    "A": {"correct", "partial", "wrong", "abstained"},
    "B": {"inferred_correctly", "abstained", "wrong"},
    "C": {"abstained_correctly", "non_abstained"},
}


def aggregate_human_reference(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"bundle_id", "qid", "type", "judgment"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"grader output is missing columns: {sorted(missing)}")

    data = rows[list(required)].copy()
    for column in required:
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["type"] = data["type"].str.upper()
    data = data[data["type"] != "SKIP"].copy()
    if data[["bundle_id", "qid", "type", "judgment"]].eq("").any().any():
        raise ValueError("evaluated rows contain empty identifiers, types, or judgments")
    if data.duplicated(["bundle_id", "qid"]).any():
        raise ValueError("grader output contains duplicate participant-question rows")

    invalid_types = sorted(set(data["type"]) - set(VALID_JUDGMENTS))
    if invalid_types:
        raise ValueError(f"invalid researcher types: {invalid_types}")
    invalid_rows = data[
        data.apply(
            lambda row: row["judgment"] not in VALID_JUDGMENTS[row["type"]], axis=1
        )
    ]
    if not invalid_rows.empty:
        sample = invalid_rows.iloc[0]
        raise ValueError(
            f"invalid judgment {sample['judgment']!r} for Type {sample['type']}"
        )

    data["passed"] = data.apply(
        lambda row: row["judgment"] == PASS_LABEL[row["type"]], axis=1
    )
    grouped = data.groupby("qid", sort=True)["passed"]
    summary = grouped.agg(n_passed="sum", n_evaluated="count").reset_index()
    summary["n_passed"] = summary["n_passed"].astype(int)
    summary["human_pass_rate"] = summary["n_passed"] / summary["n_evaluated"]
    return summary[["qid", "n_passed", "n_evaluated", "human_pass_rate"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = aggregate_human_reference(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(summary)} questions)")


if __name__ == "__main__":
    main()
