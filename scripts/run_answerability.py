"""Run repeated transcript-question answerability classifications.

Each API call sees only one transcript and one question. Results are appended
to a JSONL ledger so interrupted runs can resume safely. A Layer 1 compatible
wide CSV is exported only when every requested call has a valid A/B/C label.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts/answerability_classifier_system.txt"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "prompts/answerability_output_schema.json"
PROMPT_VERSION = "historical-answerability-v1"
VALID_LABELS = {"A", "B", "C"}

CLASSIFICATION_TOOL = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_run_manifest(args: argparse.Namespace, prompt: str) -> Path:
    """Create or validate the immutable configuration for one ledger."""
    manifest_path = args.ledger.with_suffix(".manifest.json")
    configuration = {
        "model": args.model,
        "runs": args.runs,
        "max_tokens": args.max_tokens,
        "max_pairs": args.max_pairs,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": text_sha256(prompt),
        "output_schema_sha256": file_sha256(SCHEMA_PATH),
        "transcripts_sha256": file_sha256(args.transcripts),
        "cohort_sha256": file_sha256(args.cohort),
        "questions_sha256": file_sha256(args.questions),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise ValueError(
                f"run configuration differs from {manifest_path}; "
                "use a new --ledger and --output for a new condition"
            )
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )
    return manifest_path


def render_transcript(text: str) -> str:
    return (
        "The following is an interview transcript between an AI interviewer "
        '("Assistant") and a participant ("User"). The transcript records '
        "the participant's own words and views about their use of AI tools "
        "in their professional work.\n\n"
        "----- BEGIN TRANSCRIPT -----\n\n"
        f"{text.strip()}\n\n"
        "----- END TRANSCRIPT -----"
    )


def user_message(question: str) -> str:
    return (
        f"QUESTION:\n{question.strip()}\n\n"
        "Classify this (transcript, question) pair using the tool."
    )


def parse_tool_result(response: Any) -> dict[str, str]:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            payload = dict(block.input)
            label = str(payload.get("type_inferred", "")).strip().upper()
            confidence = str(payload.get("type_confidence", "")).strip().lower()
            if label not in VALID_LABELS:
                raise ValueError(f"invalid classifier label: {label!r}")
            if confidence not in {"high", "medium", "low"}:
                raise ValueError(f"invalid classifier confidence: {confidence!r}")
            return {
                "label": label,
                "rationale": str(payload.get("type_rationale", "")).strip(),
                "confidence": confidence,
            }
    raise ValueError("response did not contain the classification tool result")


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def completed_keys(records: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    done = set()
    for record in records:
        if record.get("label") in VALID_LABELS and not record.get("error"):
            done.add((record["bundle_id"], record["qid"], int(record["run"])))
    return done


def export_wide(
    records: list[dict[str, Any]],
    pairs: pd.DataFrame,
    runs: int,
    output_path: Path,
) -> None:
    successful = [
        record
        for record in records
        if record.get("label") in VALID_LABELS and not record.get("error")
    ]
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in successful:
        key = (record["bundle_id"], record["qid"], int(record["run"]))
        if key in by_key:
            raise ValueError(f"duplicate successful ledger entry: {key}")
        by_key[key] = record

    rows = []
    missing = []
    for pair in pairs.itertuples(index=False):
        row = {"bundle_id": pair.bundle_id, "qid": pair.qid}
        for run in range(1, runs + 1):
            key = (pair.bundle_id, pair.qid, run)
            if key not in by_key:
                missing.append(key)
            else:
                row[f"run{run:02d}"] = by_key[key]["label"]
        rows.append(row)
    if missing:
        raise ValueError(f"cannot export: {len(missing)} requested calls are incomplete")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def build_pairs(
    transcripts: pd.DataFrame,
    cohort: pd.DataFrame,
    questions: pd.DataFrame,
) -> pd.DataFrame:
    transcript_ids = set(transcripts["transcript_id"])
    missing_ids = sorted(set(cohort["transcript_id"]) - transcript_ids)
    if missing_ids:
        raise ValueError(f"cohort IDs missing from transcript data: {missing_ids[:5]}")
    if cohort["transcript_id"].duplicated().any():
        raise ValueError("cohort contains duplicate transcript IDs")
    if questions["qid"].duplicated().any():
        raise ValueError("questions contain duplicate qids")

    pairs = cohort[["transcript_id"]].merge(
        questions[["qid", "question"]], how="cross"
    )
    return pairs.rename(columns={"transcript_id": "bundle_id"})


async def run_calls(args: argparse.Namespace) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    from anthropic import AsyncAnthropic

    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    ensure_run_manifest(args, prompt)
    transcripts = pd.read_parquet(args.transcripts)
    cohort = pd.read_csv(args.cohort)
    questions = pd.read_csv(args.questions)
    pairs = build_pairs(transcripts, cohort, questions)
    if args.max_pairs is not None:
        pairs = pairs.head(args.max_pairs).copy()

    text_by_id = transcripts.set_index("transcript_id")["text"].to_dict()
    records = read_ledger(args.ledger)
    done = completed_keys(records)
    ledger_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    client = AsyncAnthropic()

    async def call_one(bundle_id: str, qid: str, question: str, run: int) -> None:
        key = (bundle_id, qid, run)
        if key in done:
            return
        record: dict[str, Any] = {
            "bundle_id": bundle_id,
            "qid": qid,
            "run": run,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": text_sha256(prompt),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "label": "",
            "rationale": "",
            "confidence": "",
            "error": "",
        }
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=args.model,
                    max_tokens=args.max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": prompt,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "The participant transcript follows.\n\n"
                            + render_transcript(text_by_id[bundle_id]),
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                    tools=[CLASSIFICATION_TOOL],
                    tool_choice={"type": "tool", "name": CLASSIFICATION_TOOL["name"]},
                    messages=[{"role": "user", "content": user_message(question)}],
                )
            record.update(parse_tool_result(response))
            usage = getattr(response, "usage", None)
            if usage:
                record["input_tokens"] = getattr(usage, "input_tokens", None)
                record["output_tokens"] = getattr(usage, "output_tokens", None)
        except Exception as error:  # preserve failures for audit and resume
            record["error"] = f"{type(error).__name__}: {str(error)[:500]}"

        async with ledger_lock:
            args.ledger.parent.mkdir(parents=True, exist_ok=True)
            with args.ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    tasks = []
    for pair in pairs.itertuples(index=False):
        for run in range(1, args.runs + 1):
            tasks.append(call_one(pair.bundle_id, pair.qid, pair.question, run))
    await asyncio.gather(*tasks)
    return records, pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact Anthropic model identifier.")
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=Path("local_data/anthropic_interviewer.parquet"),
    )
    parser.add_argument("--cohort", type=Path, default=Path("local_data/cohort.csv"))
    parser.add_argument("--questions", type=Path, default=Path("data/questions.csv"))
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("local_data/answerability_calls.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local_data/answerability_runs.csv"),
    )
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    return args


def main() -> None:
    args = parse_args()
    records, pairs = asyncio.run(run_calls(args))
    export_wide(records, pairs, args.runs, args.output)
    print(f"wrote {args.output} ({len(pairs)} pairs x {args.runs} runs)")


if __name__ == "__main__":
    main()
