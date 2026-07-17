"""Download and validate the public AnthropicInterviewer transcripts.

The historical study loaded the dataset revision pinned below. Transcript text
is written only to ``local_data/``, which is excluded from version control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

DATASET_ID = "Anthropic/AnthropicInterviewer"
DATASET_CONFIG = "AnthropicInterviewer"
DATASET_REVISION = "c9e1ec1e6b093712b9c42235c7303ece647490e9"
SPLITS = ("workforce", "creatives", "scientists")
EXPECTED_ROWS = {"workforce": 1000, "creatives": 125, "scientists": 125}


def load_transcripts(
    revision: str = DATASET_REVISION,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Load all three splits and return transcript_id, text, and split."""
    from datasets import load_dataset

    parts: list[pd.DataFrame] = []
    for split in SPLITS:
        dataset = load_dataset(
            DATASET_ID,
            name=DATASET_CONFIG,
            split=split,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        frame = dataset.to_pandas()
        missing = {"transcript_id", "text"} - set(frame.columns)
        if missing:
            raise ValueError(f"split {split!r} is missing columns: {sorted(missing)}")
        if revision == DATASET_REVISION and len(frame) != EXPECTED_ROWS[split]:
            raise ValueError(
                f"pinned split {split!r} has {len(frame)} rows; "
                f"expected {EXPECTED_ROWS[split]}"
            )
        part = frame[["transcript_id", "text"]].copy()
        part["split"] = split
        parts.append(part)

    transcripts = pd.concat(parts, ignore_index=True)
    if transcripts[["transcript_id", "text", "split"]].isna().any().any():
        raise ValueError("dataset contains missing transcript identifiers, text, or splits")
    if transcripts["transcript_id"].duplicated().any():
        raise ValueError("dataset contains duplicate transcript IDs")
    return transcripts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset(
    transcripts: pd.DataFrame,
    output_path: Path,
    revision: str,
) -> Path:
    """Write a local Parquet file and a JSON provenance manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transcripts.to_parquet(output_path, index=False)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "revision": revision,
        "rows": len(transcripts),
        "rows_by_split": {
            split: int(count)
            for split, count in transcripts["split"].value_counts().sort_index().items()
        },
        "columns": list(transcripts.columns),
        "parquet_sha256": sha256(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("local_data/hf_cache"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local_data/anthropic_interviewer.parquet"),
    )
    args = parser.parse_args()

    transcripts = load_transcripts(args.revision, args.cache_dir)
    manifest_path = write_dataset(transcripts, args.output, args.revision)
    print(f"wrote {args.output} ({len(transcripts)} transcripts)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
