"""Validate per-transcript human references and merge them into one long CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VALID_TYPES = {"A", "B", "C", "SKIP"}
COLUMNS = [
    "bundle_id",
    "qid",
    "question",
    "type",
    "gold_answer",
    "gold_evidence",
    "absence_check",
    "skip_reason",
    "notes",
]


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) != set(COLUMNS):
        missing = sorted(set(COLUMNS) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(COLUMNS))
        raise ValueError(f"annotation columns differ; missing={missing}, extra={extra}")
    data = frame[COLUMNS].copy()
    for column in COLUMNS:
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["type"] = data["type"].str.upper()
    return data


def _validate_annotation_rows(
    data: pd.DataFrame,
    bundle_id: str,
    questions: pd.DataFrame,
    source: Path,
) -> None:
    if len(data) != len(questions):
        raise ValueError(
            f"{source}: expected {len(questions)} question rows, found {len(data)}"
        )
    if set(data["bundle_id"]) != {bundle_id}:
        raise ValueError(f"{source}: every bundle_id must equal {bundle_id!r}")
    if data["qid"].duplicated().any():
        raise ValueError(f"{source}: duplicate qids")

    expected = questions.set_index("qid")["question"].to_dict()
    if set(data["qid"]) != set(expected):
        missing = sorted(set(expected) - set(data["qid"]))
        extra = sorted(set(data["qid"]) - set(expected))
        raise ValueError(f"{source}: qids differ; missing={missing}, extra={extra}")
    mismatched = data[data.apply(lambda row: expected[row["qid"]] != row["question"], axis=1)]
    if not mismatched.empty:
        raise ValueError(f"{source}: question text differs for {mismatched.iloc[0]['qid']}")

    invalid = sorted(set(data["type"]) - VALID_TYPES)
    if invalid:
        if "" in invalid:
            raise ValueError(f"{source}: annotation has unfinished type cells")
        raise ValueError(f"{source}: invalid types: {invalid}")

    for row in data.itertuples(index=False):
        if row.type in {"A", "B"} and not (
            row.gold_answer or row.gold_evidence or row.notes
        ):
            raise ValueError(
                f"{source}: {row.qid} Type {row.type} requires at least one "
                "human reference in gold_answer, gold_evidence, or notes"
            )
        if row.type == "SKIP" and not row.skip_reason:
            raise ValueError(f"{source}: {row.qid} skip requires skip_reason")


def merge_references(
    cohort: pd.DataFrame,
    questions: pd.DataFrame,
    input_dir: Path,
) -> pd.DataFrame:
    """Return a validated, deterministic cohort-by-question long table."""
    if "transcript_id" not in cohort.columns:
        raise ValueError("cohort is missing transcript_id")
    if cohort["transcript_id"].isna().any() or cohort["transcript_id"].duplicated().any():
        raise ValueError("cohort transcript_id values must be present and unique")
    if set(questions.columns) != {"qid", "question"}:
        raise ValueError("questions must contain exactly qid and question")

    expected_ids = [str(value) for value in cohort["transcript_id"]]
    present = {path.stem for path in input_dir.glob("*.csv")}
    missing = sorted(set(expected_ids) - present)
    extra = sorted(present - set(expected_ids))
    if missing or extra:
        raise ValueError(
            "annotation files must match the cohort exactly; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    parts: list[pd.DataFrame] = []
    for bundle_id in expected_ids:
        source = input_dir / f"{bundle_id}.csv"
        data = _clean(pd.read_csv(source, dtype=str, keep_default_na=False))
        _validate_annotation_rows(data, bundle_id, questions, source)
        parts.append(data)
    combined = pd.concat(parts, ignore_index=True)
    if combined.duplicated(["bundle_id", "qid"]).any():
        raise ValueError("merged annotations contain duplicate bundle_id/qid pairs")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path, default=Path("local_data/cohort.csv")
    )
    parser.add_argument(
        "--questions", type=Path, default=Path("data/questions.csv")
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("local_data/human_reference/by_transcript"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("local_data/human_reference.csv")
    )
    args = parser.parse_args()

    combined = merge_references(
        pd.read_csv(args.cohort, dtype=str),
        pd.read_csv(args.questions, dtype=str),
        args.input_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
