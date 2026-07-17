"""Run the published full-context user proxy over transcript-question pairs.

This is a portable reference implementation, not a required inference harness.
It records an immutable run manifest and an append-only JSONL ledger so failed
or interrupted runs can resume without repeating successful calls.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts/user_proxy_system.txt"
PROMPT_VERSION = "historical-full-context-v1"
VALID_PROVIDERS = {"anthropic", "openai"}
OUTPUT_COLUMNS = [
    "bundle_id",
    "qid",
    "question",
    "provider",
    "model",
    "condition_id",
    "raw_response",
    "evidence",
    "reasoning",
    "response",
    "input_tokens",
    "output_tokens",
]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_system_prompt(template: str, transcript: str) -> str:
    if template.count("{{TRANSCRIPT}}") != 1:
        raise ValueError("user-proxy prompt must contain exactly one {{TRANSCRIPT}}")
    return template.replace("{{TRANSCRIPT}}", transcript.strip())


SECTION_PATTERNS = [
    (
        r"Step\s*1\)\s*(?:Evidence:?\s*)?",
        r"Step\s*2\)\s*(?:Reasoning:?\s*)?",
        r"Step\s*3\)\s*(?:Response:?\s*)?",
    ),
    (
        r"\*\*Step\s*1\s*[:\)]\s*Evidence\*\*\s*",
        r"\*\*Step\s*2\s*[:\)]\s*Reasoning\*\*\s*",
        r"\*\*Step\s*3\s*[:\)]\s*Response\*\*\s*",
    ),
    (
        r"\*\*Step\s*1[:\)]\*\*\s*",
        r"\*\*Step\s*2[:\)]\*\*\s*",
        r"\*\*Step\s*3[:\)]\*\*\s*",
    ),
    (
        r"\*\*Evidence\*?\*?:?\s*",
        r"\*\*Reasoning\*?\*?:?\s*",
        r"\*\*Response\*?\*?:?\s*",
    ),
    (r"^Evidence:\s*", r"^Reasoning:\s*", r"^Response:\s*"),
    (r"^1\)\s*", r"^2\)\s*", r"^3\)\s*"),
]


def _clean_section(text: str) -> str:
    return re.sub(r"[\s\*:]+$", "", re.sub(r"^[\s\*:]+", "", text)).strip()


def parse_proxy_response(raw: str) -> dict[str, str]:
    """Parse the historical Park-style three-section response leniently."""
    for evidence_pattern, reasoning_pattern, response_pattern in SECTION_PATTERNS:
        evidence = re.search(evidence_pattern, raw, re.IGNORECASE | re.MULTILINE)
        reasoning = re.search(reasoning_pattern, raw, re.IGNORECASE | re.MULTILINE)
        response = re.search(response_pattern, raw, re.IGNORECASE | re.MULTILINE)
        if not (evidence and reasoning and response):
            continue
        if not (evidence.end() <= reasoning.start() and reasoning.end() <= response.start()):
            continue
        return {
            "evidence": _clean_section(raw[evidence.end() : reasoning.start()]),
            "reasoning": _clean_section(raw[reasoning.end() : response.start()]),
            "response": _clean_section(raw[response.end() :]),
        }
    return {"evidence": "", "reasoning": "", "response": raw.strip()}


def build_pairs(
    transcripts: pd.DataFrame,
    cohort: pd.DataFrame,
    questions: pd.DataFrame,
    eligibility_path: Path | None,
) -> pd.DataFrame:
    required_transcripts = {"transcript_id", "text"}
    if not required_transcripts.issubset(transcripts.columns):
        raise ValueError("transcripts must contain transcript_id and text")
    if "transcript_id" not in cohort.columns:
        raise ValueError("cohort is missing transcript_id")
    if set(questions.columns) != {"qid", "question"}:
        raise ValueError("questions must contain exactly qid and question")
    if transcripts["transcript_id"].duplicated().any():
        raise ValueError("transcripts contain duplicate transcript IDs")
    if cohort["transcript_id"].duplicated().any():
        raise ValueError("cohort contains duplicate transcript IDs")
    if questions["qid"].duplicated().any():
        raise ValueError("questions contain duplicate qids")

    missing = sorted(set(cohort["transcript_id"]) - set(transcripts["transcript_id"]))
    if missing:
        raise ValueError(f"cohort IDs missing from transcripts: {missing[:5]}")
    pairs = cohort[["transcript_id"]].merge(questions, how="cross").rename(
        columns={"transcript_id": "bundle_id"}
    )
    if eligibility_path is None:
        return pairs

    eligibility = pd.read_csv(eligibility_path, dtype=str)
    required = {"bundle_id", "qid", "type"}
    if not required.issubset(eligibility.columns):
        raise ValueError(f"eligibility file is missing {sorted(required - set(eligibility.columns))}")
    eligibility = eligibility[["bundle_id", "qid", "type"]].copy()
    for column in eligibility.columns:
        eligibility[column] = eligibility[column].fillna("").str.strip()
    eligibility["type"] = eligibility["type"].str.upper()
    if eligibility.duplicated(["bundle_id", "qid"]).any():
        raise ValueError("eligibility contains duplicate pairs")
    invalid = sorted(set(eligibility["type"]) - {"A", "B", "C", "SKIP"})
    if invalid:
        raise ValueError(f"eligibility contains invalid types: {invalid}")
    pair_keys = set(map(tuple, pairs[["bundle_id", "qid"]].itertuples(index=False, name=None)))
    reference_keys = set(
        map(tuple, eligibility[["bundle_id", "qid"]].itertuples(index=False, name=None))
    )
    if pair_keys != reference_keys:
        raise ValueError("eligibility must match the cohort-question universe exactly")
    planned = pairs.merge(eligibility, on=["bundle_id", "qid"], validate="one_to_one")
    return planned.loc[planned["type"] != "SKIP", pairs.columns].reset_index(drop=True)


def ensure_manifest(args: argparse.Namespace, prompt: str) -> Path:
    path = args.ledger.with_suffix(".manifest.json")
    configuration = {
        "provider": args.provider,
        "model": args.model,
        "condition_id": args.condition_id,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "max_pairs": args.max_pairs,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": text_sha256(prompt),
        "transcripts_sha256": file_sha256(args.transcripts),
        "cohort_sha256": file_sha256(args.cohort),
        "questions_sha256": file_sha256(args.questions),
        "eligibility_sha256": file_sha256(args.eligibility) if args.eligibility else None,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise ValueError(f"run configuration differs from {path}; use a new ledger")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    return path


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def export_results(records: list[dict[str, Any]], pairs: pd.DataFrame, output: Path) -> None:
    successful: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record.get("error") or not record.get("raw_response"):
            continue
        key = (record["bundle_id"], record["qid"])
        if key in successful:
            raise ValueError(f"duplicate successful proxy result: {key}")
        successful[key] = record
    rows = []
    missing = []
    for pair in pairs.itertuples(index=False):
        key = (pair.bundle_id, pair.qid)
        if key not in successful:
            missing.append(key)
        else:
            rows.append({column: successful[key].get(column, "") for column in OUTPUT_COLUMNS})
    if missing:
        raise ValueError(f"cannot export: {len(missing)} proxy calls are incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output, index=False)


async def run_calls(args: argparse.Namespace) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    transcripts = pd.read_parquet(args.transcripts)
    cohort = pd.read_csv(args.cohort, dtype=str)
    questions = pd.read_csv(args.questions, dtype=str)
    pairs = build_pairs(transcripts, cohort, questions, args.eligibility)
    if args.max_pairs is not None:
        pairs = pairs.head(args.max_pairs).copy()
    ensure_manifest(args, prompt)

    text_by_id = transcripts.set_index("transcript_id")["text"].to_dict()
    records = read_ledger(args.ledger)
    done = {
        (record["bundle_id"], record["qid"])
        for record in records
        if record.get("raw_response") and not record.get("error")
    }
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    if args.provider == "anthropic":
        from anthropic import AsyncAnthropic

        client: Any = AsyncAnthropic()
    else:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

    async def call_one(bundle_id: str, qid: str, question: str) -> None:
        if (bundle_id, qid) in done:
            return
        rendered_prompt = render_system_prompt(prompt, text_by_id[bundle_id])
        record: dict[str, Any] = {
            "bundle_id": bundle_id,
            "qid": qid,
            "question": question,
            "provider": args.provider,
            "model": args.model,
            "condition_id": args.condition_id,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": text_sha256(prompt),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": "",
            "evidence": "",
            "reasoning": "",
            "response": "",
            "error": "",
        }
        try:
            async with semaphore:
                if args.provider == "anthropic":
                    result = await client.messages.create(
                        model=args.model,
                        max_tokens=args.max_tokens,
                        system=rendered_prompt,
                        messages=[{"role": "user", "content": question}],
                    )
                    raw = "".join(
                        block.text
                        for block in result.content
                        if getattr(block, "type", None) == "text"
                    )
                else:
                    request: dict[str, Any] = {
                        "model": args.model,
                        "input": [
                            {"role": "system", "content": rendered_prompt},
                            {"role": "user", "content": question},
                        ],
                        "max_output_tokens": args.max_tokens,
                    }
                    if args.reasoning_effort is not None:
                        request["reasoning"] = {"effort": args.reasoning_effort}
                    result = await client.responses.create(**request)
                    raw = result.output_text
                if not raw.strip():
                    raise ValueError("provider returned an empty response")
                record["raw_response"] = raw
                record.update(parse_proxy_response(raw))
                usage = getattr(result, "usage", None)
                if usage:
                    record["input_tokens"] = getattr(usage, "input_tokens", None)
                    record["output_tokens"] = getattr(usage, "output_tokens", None)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {str(error)[:500]}"
        async with lock:
            args.ledger.parent.mkdir(parents=True, exist_ok=True)
            with args.ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    await asyncio.gather(
        *(
            call_one(pair.bundle_id, pair.qid, pair.question)
            for pair in pairs.itertuples(index=False)
        )
    )
    return records, pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=sorted(VALID_PROVIDERS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition-id", default="question_only")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--transcripts", type=Path, default=Path("local_data/anthropic_interviewer.parquet")
    )
    parser.add_argument("--cohort", type=Path, default=Path("local_data/cohort.csv"))
    parser.add_argument("--questions", type=Path, default=Path("data/questions.csv"))
    parser.add_argument("--eligibility", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=Path("local_data/proxy_calls.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("local_data/proxy_answers.csv"))
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.provider != "openai" and args.reasoning_effort is not None:
        parser.error("--reasoning-effort is supported only for OpenAI")
    return args


def main() -> None:
    args = parse_args()
    if args.env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    records, pairs = asyncio.run(run_calls(args))
    export_results(records, pairs, args.output)
    print(f"wrote {args.output} ({len(pairs)} transcript-question pairs)")


if __name__ == "__main__":
    main()
