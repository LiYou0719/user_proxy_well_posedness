"""Generate one 23-question human-reference CSV per sampled transcript.

The files are annotation interfaces: a researcher reads one transcript and
completes all canonical questions before moving to the next transcript.  The
completed files are later validated and combined by ``merge_human_references``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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


def generate_templates(
    cohort: pd.DataFrame,
    questions: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write one non-destructive annotation template per cohort member."""
    if "transcript_id" not in cohort.columns:
        raise ValueError("cohort is missing transcript_id")
    if cohort["transcript_id"].isna().any() or cohort["transcript_id"].duplicated().any():
        raise ValueError("cohort transcript_id values must be present and unique")
    if set(questions.columns) != {"qid", "question"}:
        raise ValueError("questions must contain exactly qid and question")
    if questions.isna().any().any() or questions["qid"].duplicated().any():
        raise ValueError("question identifiers and text must be present and unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / f"{bundle_id}.csv" for bundle_id in cohort["transcript_id"]]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite {len(existing)} existing annotation file(s); "
            f"first existing file: {existing[0]}"
        )

    written: list[Path] = []
    for bundle_id, path in zip(cohort["transcript_id"], targets):
        template = questions.copy()
        template.insert(0, "bundle_id", str(bundle_id))
        for column in COLUMNS[3:]:
            template[column] = ""
        template[COLUMNS].to_csv(path, index=False)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path, default=Path("local_data/cohort.csv")
    )
    parser.add_argument(
        "--questions", type=Path, default=Path("data/questions.csv")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local_data/human_reference/by_transcript"),
    )
    args = parser.parse_args()

    written = generate_templates(
        pd.read_csv(args.cohort, dtype=str),
        pd.read_csv(args.questions, dtype=str),
        args.output_dir,
    )
    print(f"wrote {len(written)} annotation templates to {args.output_dir}")


if __name__ == "__main__":
    main()
