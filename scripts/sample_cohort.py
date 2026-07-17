"""Draw a deterministic stratified cohort from prepared transcripts."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

DEFAULT_COUNTS = {"workforce": 30, "creatives": 10, "scientists": 10}


def sample_cohort(
    transcripts: pd.DataFrame,
    counts: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    """Sample without replacement, then deterministically shuffle row order."""
    required = {"transcript_id", "split"}
    missing = required - set(transcripts.columns)
    if missing:
        raise ValueError(f"transcript data is missing columns: {sorted(missing)}")
    if transcripts["transcript_id"].duplicated().any():
        raise ValueError("transcript data contains duplicate transcript IDs")

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for split, count in counts.items():
        if count < 0:
            raise ValueError("sample counts cannot be negative")
        candidates = sorted(
            transcripts.loc[transcripts["split"] == split, "transcript_id"].tolist()
        )
        if len(candidates) < count:
            raise ValueError(
                f"split {split!r} has {len(candidates)} transcripts; requested {count}"
            )
        selected.extend(
            {"transcript_id": transcript_id, "split": split}
            for transcript_id in rng.sample(candidates, count)
        )

    rng.shuffle(selected)
    cohort = pd.DataFrame(selected)
    cohort.insert(0, "rank", range(1, len(cohort) + 1))
    cohort["sampling_seed"] = seed
    return cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("local_data/anthropic_interviewer.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("local_data/cohort.csv"))
    parser.add_argument(
        "--seed",
        type=lambda value: int(value, 0),
        default=42,
        help="Integer seed; decimal and 0x-prefixed values are accepted.",
    )
    parser.add_argument("--workforce", type=int, default=DEFAULT_COUNTS["workforce"])
    parser.add_argument("--creatives", type=int, default=DEFAULT_COUNTS["creatives"])
    parser.add_argument("--scientists", type=int, default=DEFAULT_COUNTS["scientists"])
    args = parser.parse_args()

    transcripts = pd.read_parquet(args.input)
    counts = {
        "workforce": args.workforce,
        "creatives": args.creatives,
        "scientists": args.scientists,
    }
    cohort = sample_cohort(transcripts, counts, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(cohort)} transcripts, seed={args.seed})")


if __name__ == "__main__":
    main()
