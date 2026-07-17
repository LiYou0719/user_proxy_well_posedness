"""Grade user-proxy responses against frozen researcher human references.

The runner implements only the published gold-based content check.  It does
not include legacy transcript-grader variants or private experiment modes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PATH = ROOT / "prompts/content_grader_system.txt"
USER_TEMPLATE_PATH = ROOT / "prompts/content_grader_user_template.txt"
SCHEMAS_PATH = ROOT / "prompts/content_grader_output_schemas.json"
TYPE_PROMPT_PATHS = {
    qtype: ROOT / f"prompts/content_grader_type_{qtype.lower()}.txt"
    for qtype in "ABC"
}
PROMPT_VERSION = "historical-gold-content-grader-v1"
VALID_PROVIDERS = {"anthropic", "openai"}
VALID_JUDGMENTS = {
    "A": {"correct", "partial", "wrong", "abstained"},
    "B": {"inferred_correctly", "abstained", "wrong"},
    "C": {"abstained_correctly", "non_abstained"},
}
PASS_JUDGMENT = {
    "A": "correct",
    "B": "inferred_correctly",
    "C": "abstained_correctly",
}
OUTPUT_COLUMNS = [
    "bundle_id",
    "qid",
    "type",
    "judgment",
    "passed",
    "rationale",
    "confidence",
    "provider",
    "model",
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


def load_prompt_artifacts() -> tuple[str, str, dict[str, str], dict[str, dict[str, Any]]]:
    system = SYSTEM_PATH.read_text(encoding="utf-8").strip()
    user_template = USER_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
    type_prompts = {
        qtype: path.read_text(encoding="utf-8").strip()
        for qtype, path in TYPE_PROMPT_PATHS.items()
    }
    schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))
    return system, user_template, type_prompts, schemas


def _provided(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return text or "[not provided]"


def render_user_prompt(template: str, row: dict[str, Any]) -> str:
    replacements = {
        "{{EVALUATION_QUESTION}}": _provided(row.get("question")),
        "{{TYPE}}": _provided(row.get("type")),
        "{{GOLD_ANSWER_OR_NOT_PROVIDED}}": _provided(row.get("gold_answer")),
        "{{GOLD_EVIDENCE_OR_NOT_PROVIDED}}": _provided(row.get("gold_evidence")),
        "{{ABSENCE_CHECK_OR_NOT_PROVIDED}}": _provided(row.get("absence_check")),
        "{{RESEARCHER_NOTES_OR_NOT_PROVIDED}}": _provided(row.get("notes")),
        "{{PROXY_EVIDENCE_OR_EMPTY}}": _provided(row.get("evidence")),
        "{{PROXY_REASONING_OR_EMPTY}}": _provided(row.get("reasoning")),
        "{{PROXY_RESPONSE_OR_EMPTY}}": _provided(row.get("response")),
    }
    rendered = template
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("content-grader user template contains an unresolved marker")
    return rendered


def validate_judgment(qtype: str, payload: dict[str, Any]) -> dict[str, Any]:
    judgment = str(payload.get("judgment", "")).strip()
    rationale = str(payload.get("rationale", "")).strip()
    confidence = str(payload.get("confidence", "")).strip().lower()
    if judgment not in VALID_JUDGMENTS[qtype]:
        raise ValueError(f"invalid Type {qtype} judgment: {judgment!r}")
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"invalid confidence: {confidence!r}")
    return {
        "judgment": judgment,
        "passed": judgment == PASS_JUDGMENT[qtype],
        "rationale": rationale,
        "confidence": confidence,
    }


def parse_anthropic_result(qtype: str, response: Any) -> dict[str, Any]:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return validate_judgment(qtype, dict(block.input))
    raise ValueError("response did not contain the grading tool result")


def parse_openai_result(qtype: str, response: Any) -> dict[str, Any]:
    if not getattr(response, "output_text", ""):
        raise ValueError("response did not contain structured output text")
    payload = json.loads(response.output_text)
    if not isinstance(payload, dict):
        raise ValueError("structured output was not an object")
    return validate_judgment(qtype, payload)


def openai_text_format(tool: dict[str, Any]) -> dict[str, Any]:
    schema = json.loads(json.dumps(tool["input_schema"]))
    schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "name": tool["name"],
        "description": tool["description"],
        "strict": True,
        "schema": schema,
    }


def build_grading_pairs(
    proxy: pd.DataFrame,
    human: pd.DataFrame,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    proxy_required = {"bundle_id", "qid", "question", "evidence", "reasoning", "response"}
    human_required = {
        "bundle_id",
        "qid",
        "question",
        "type",
        "gold_answer",
        "gold_evidence",
        "absence_check",
    }
    if not proxy_required.issubset(proxy.columns):
        raise ValueError(f"proxy output is missing {sorted(proxy_required - set(proxy.columns))}")
    if not human_required.issubset(human.columns):
        raise ValueError(f"human reference is missing {sorted(human_required - set(human.columns))}")
    if proxy.duplicated(["bundle_id", "qid"]).any():
        raise ValueError("proxy output contains duplicate pairs")
    if human.duplicated(["bundle_id", "qid"]).any():
        raise ValueError("human reference contains duplicate pairs")

    reference = human.copy()
    for column in human_required | {"skip_reason", "notes"}:
        if column not in reference.columns:
            reference[column] = ""
        reference[column] = reference[column].fillna("").astype(str).str.strip()
    reference["type"] = reference["type"].str.upper()
    if reference[["bundle_id", "qid", "question", "type"]].eq("").any().any():
        raise ValueError("human reference contains blank identifiers, questions, or types")
    invalid = sorted(set(reference["type"]) - {"A", "B", "C", "SKIP"})
    if invalid:
        raise ValueError(f"human reference contains invalid types: {invalid}")
    for row in reference.itertuples(index=False):
        if row.type in {"A", "B"} and not (
            row.gold_answer or row.gold_evidence or row.notes
        ):
            raise ValueError(
                f"{row.bundle_id}/{row.qid}: Type {row.type} requires at least "
                "one human reference field"
            )
        if row.type == "SKIP" and not row.skip_reason:
            raise ValueError(f"{row.bundle_id}/{row.qid}: skip requires skip_reason")
    reference = reference[reference["type"] != "SKIP"].copy()
    all_reference_keys = set(
        map(tuple, reference[["bundle_id", "qid"]].itertuples(index=False, name=None))
    )
    if max_pairs is not None:
        reference = reference.head(max_pairs).copy()

    expected_keys = set(
        map(tuple, reference[["bundle_id", "qid"]].itertuples(index=False, name=None))
    )
    proxy_keys = set(map(tuple, proxy[["bundle_id", "qid"]].itertuples(index=False, name=None)))
    if max_pairs is not None and proxy_keys == all_reference_keys:
        proxy = proxy[
            proxy.apply(lambda row: (row["bundle_id"], row["qid"]) in expected_keys, axis=1)
        ].copy()
        proxy_keys = expected_keys
    if expected_keys != proxy_keys:
        raise ValueError("proxy output pairs must match non-skip human-reference pairs exactly")

    proxy_fields = proxy[
        ["bundle_id", "qid", "question", "evidence", "reasoning", "response"]
    ].rename(columns={"question": "proxy_question"})
    for column in proxy_fields.columns:
        proxy_fields[column] = proxy_fields[column].fillna("").astype(str).str.strip()
    if proxy_fields[["bundle_id", "qid", "proxy_question", "response"]].eq("").any().any():
        raise ValueError("proxy output contains blank identifiers, questions, or responses")
    pairs = reference.merge(
        proxy_fields,
        on=["bundle_id", "qid"],
        validate="one_to_one",
    )
    mismatch = pairs[pairs["question"] != pairs["proxy_question"]]
    if not mismatch.empty:
        raise ValueError(f"proxy question differs from reference for {mismatch.iloc[0]['qid']}")
    return pairs.drop(columns="proxy_question")


def ensure_manifest(
    args: argparse.Namespace,
    system: str,
    user_template: str,
    type_prompts: dict[str, str],
) -> Path:
    path = args.ledger.with_suffix(".manifest.json")
    configuration = {
        "provider": args.provider,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "max_pairs": args.max_pairs,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": text_sha256(system),
        "user_template_sha256": text_sha256(user_template),
        "type_prompt_sha256": {key: text_sha256(value) for key, value in type_prompts.items()},
        "schemas_sha256": file_sha256(SCHEMAS_PATH),
        "proxy_answers_sha256": file_sha256(args.answers),
        "human_reference_sha256": file_sha256(args.human_reference),
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
        if record.get("error") or record.get("judgment") not in VALID_JUDGMENTS.get(record.get("type"), set()):
            continue
        key = (record["bundle_id"], record["qid"])
        if key in successful:
            raise ValueError(f"duplicate successful content grade: {key}")
        successful[key] = record
    rows, missing = [], []
    for pair in pairs.itertuples(index=False):
        key = (pair.bundle_id, pair.qid)
        if key not in successful:
            missing.append(key)
        else:
            rows.append({column: successful[key].get(column, "") for column in OUTPUT_COLUMNS})
    if missing:
        raise ValueError(f"cannot export: {len(missing)} content-grader calls are incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output, index=False)


async def run_calls(args: argparse.Namespace) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    system, user_template, type_prompts, tools = load_prompt_artifacts()
    proxy = pd.read_csv(args.answers, dtype=str, keep_default_na=False)
    human = pd.read_csv(args.human_reference, dtype=str, keep_default_na=False)
    pairs = build_grading_pairs(proxy, human, max_pairs=args.max_pairs)
    ensure_manifest(args, system, user_template, type_prompts)

    records = read_ledger(args.ledger)
    done = {
        (record["bundle_id"], record["qid"])
        for record in records
        if not record.get("error")
        and record.get("judgment") in VALID_JUDGMENTS.get(record.get("type"), set())
    }
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    if args.provider == "anthropic":
        from anthropic import AsyncAnthropic

        client: Any = AsyncAnthropic()
    else:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

    async def call_one(row: dict[str, Any]) -> None:
        key = (row["bundle_id"], row["qid"])
        if key in done:
            return
        qtype = row["type"]
        record: dict[str, Any] = {
            "bundle_id": row["bundle_id"],
            "qid": row["qid"],
            "type": qtype,
            "provider": args.provider,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "judgment": "",
            "passed": False,
            "rationale": "",
            "confidence": "",
            "error": "",
        }
        grader_system = system + "\n\n" + type_prompts[qtype]
        grader_user = render_user_prompt(user_template, row)
        try:
            async with semaphore:
                if args.provider == "anthropic":
                    result = await client.messages.create(
                        model=args.model,
                        max_tokens=args.max_tokens,
                        system=grader_system,
                        tools=[tools[qtype]],
                        tool_choice={"type": "tool", "name": tools[qtype]["name"]},
                        messages=[{"role": "user", "content": grader_user}],
                    )
                    record.update(parse_anthropic_result(qtype, result))
                else:
                    request: dict[str, Any] = {
                        "model": args.model,
                        "input": [
                            {"role": "system", "content": grader_system},
                            {"role": "user", "content": grader_user},
                        ],
                        "max_output_tokens": args.max_tokens,
                        "text": {"format": openai_text_format(tools[qtype])},
                    }
                    if args.reasoning_effort is not None:
                        request["reasoning"] = {"effort": args.reasoning_effort}
                    result = await client.responses.create(**request)
                    record.update(parse_openai_result(qtype, result))
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

    await asyncio.gather(*(call_one(row) for row in pairs.to_dict("records")))
    return records, pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=sorted(VALID_PROVIDERS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--answers", type=Path, default=Path("local_data/proxy_answers.csv"))
    parser.add_argument(
        "--human-reference", type=Path, default=Path("local_data/human_reference.csv")
    )
    parser.add_argument(
        "--ledger", type=Path, default=Path("local_data/content_grader_calls.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("local_data/content_grader_results.csv")
    )
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
