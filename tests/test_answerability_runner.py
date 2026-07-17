from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts.run_answerability import (
    apply_eligibility,
    apply_exclusions,
    build_pairs,
    ensure_run_manifest,
    export_wide,
    openai_text_format,
    parse_openai_result,
    parse_tool_result,
    render_transcript,
    run_calls,
)


class AnswerabilityRunnerTests(unittest.TestCase):
    def make_runner_args(
        self, root: Path, provider: str = "openai"
    ) -> argparse.Namespace:
        transcripts = root / "transcripts.parquet"
        cohort = root / "cohort.csv"
        questions = root / "questions.csv"
        pd.DataFrame(
            {"transcript_id": ["p1"], "text": ["User: I use AI for drafts."]}
        ).to_parquet(transcripts, index=False)
        pd.DataFrame({"transcript_id": ["p1"]}).to_csv(cohort, index=False)
        pd.DataFrame({"qid": ["Q01"], "question": ["What is AI used for?"]}).to_csv(
            questions, index=False
        )
        return argparse.Namespace(
            provider=provider,
            model=f"{provider}-model",
            runs=2,
            concurrency=2,
            max_tokens=100,
            reasoning_effort="minimal" if provider == "openai" else None,
            max_pairs=None,
            transcripts=transcripts,
            cohort=cohort,
            questions=questions,
            exclusions=None,
            eligibility=None,
            ledger=root / "calls.jsonl",
            output=root / "runs.csv",
        )

    def test_render_transcript_preserves_boundary(self) -> None:
        rendered = render_transcript("  User: example  ")
        self.assertIn("----- BEGIN TRANSCRIPT -----", rendered)
        self.assertIn("User: example", rendered)
        self.assertTrue(rendered.endswith("----- END TRANSCRIPT -----"))

    def test_parse_tool_result_validates_schema_values(self) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    input={
                        "type_inferred": "b",
                        "type_rationale": "Indirect evidence.",
                        "type_confidence": "medium",
                    },
                )
            ]
        )
        parsed = parse_tool_result(response)
        self.assertEqual(parsed["label"], "B")
        self.assertEqual(parsed["confidence"], "medium")

    def test_parse_openai_result_validates_structured_output(self) -> None:
        response = SimpleNamespace(
            output_text=json.dumps(
                {
                    "type_inferred": "c",
                    "type_rationale": "The topic is absent.",
                    "type_confidence": "high",
                }
            )
        )
        parsed = parse_openai_result(response)
        self.assertEqual(parsed["label"], "C")
        self.assertEqual(parsed["confidence"], "high")

    def test_openai_schema_is_strict(self) -> None:
        output_format = openai_text_format()
        self.assertTrue(output_format["strict"])
        self.assertFalse(output_format["schema"]["additionalProperties"])

    def test_build_pairs_crosses_cohort_and_questions(self) -> None:
        transcripts = pd.DataFrame(
            {"transcript_id": ["p1", "p2"], "text": ["a", "b"]}
        )
        cohort = pd.DataFrame({"transcript_id": ["p2"]})
        questions = pd.DataFrame(
            {"qid": ["Q01", "Q02"], "question": ["One?", "Two?"]}
        )
        pairs = build_pairs(transcripts, cohort, questions)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(set(pairs["bundle_id"]), {"p2"})

    def test_apply_exclusions_removes_only_requested_pairs(self) -> None:
        pairs = pd.DataFrame(
            {
                "bundle_id": ["p1", "p1", "p2"],
                "qid": ["Q01", "Q02", "Q01"],
                "question": ["One?", "Two?", "One?"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exclusions.csv"
            pd.DataFrame({"bundle_id": ["p1"], "qid": ["Q02"]}).to_csv(
                path, index=False
            )
            kept = apply_exclusions(pairs, path)
        self.assertEqual(
            kept[["bundle_id", "qid"]].to_dict("records"),
            [
                {"bundle_id": "p1", "qid": "Q01"},
                {"bundle_id": "p2", "qid": "Q01"},
            ],
        )

    def test_exclusions_reject_pairs_outside_run_universe(self) -> None:
        pairs = pd.DataFrame(
            {"bundle_id": ["p1"], "qid": ["Q01"], "question": ["One?"]}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exclusions.csv"
            pd.DataFrame({"bundle_id": ["typo"], "qid": ["Q01"]}).to_csv(
                path, index=False
            )
            with self.assertRaisesRegex(ValueError, "outside the run universe"):
                apply_exclusions(pairs, path)

    def test_eligibility_omits_only_skip_and_does_not_return_human_types(self) -> None:
        pairs = pd.DataFrame(
            {
                "bundle_id": ["p1", "p1", "p2"],
                "qid": ["Q01", "Q02", "Q01"],
                "question": ["One?", "Two?", "One?"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human_reference.csv"
            pd.DataFrame(
                {
                    "bundle_id": ["p1", "p1", "p2"],
                    "qid": ["Q01", "Q02", "Q01"],
                    "type": ["A", "skip", "C"],
                    "gold_answer": ["private", "", "private"],
                }
            ).to_csv(path, index=False)
            kept = apply_eligibility(pairs, path)
        self.assertEqual(list(kept.columns), list(pairs.columns))
        self.assertEqual(
            kept[["bundle_id", "qid"]].to_dict("records"),
            [
                {"bundle_id": "p1", "qid": "Q01"},
                {"bundle_id": "p2", "qid": "Q01"},
            ],
        )

    def test_eligibility_requires_exact_run_universe(self) -> None:
        pairs = pd.DataFrame(
            {
                "bundle_id": ["p1", "p1"],
                "qid": ["Q01", "Q02"],
                "question": ["One?", "Two?"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human_reference.csv"
            pd.DataFrame(
                {"bundle_id": ["p1"], "qid": ["Q01"], "type": ["A"]}
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "match the run universe exactly"):
                apply_eligibility(pairs, path)

    def test_eligibility_reproduces_historical_q03_denominator(self) -> None:
        pairs = pd.DataFrame(
            [
                {
                    "bundle_id": f"p{participant:02d}",
                    "qid": f"Q{question:02d}",
                    "question": "Example?",
                }
                for participant in range(50)
                for question in range(1, 24)
            ]
        )
        eligibility = pairs[["bundle_id", "qid"]].copy()
        eligibility["type"] = "A"
        eligibility.loc[
            (eligibility["qid"] == "Q03")
            & eligibility["bundle_id"].isin(
                [f"p{participant:02d}" for participant in range(7)]
            ),
            "type",
        ] = "skip"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human_reference.csv"
            eligibility.to_csv(path, index=False)
            kept = apply_eligibility(pairs, path)
        self.assertEqual(len(kept), 1143)
        self.assertEqual((kept["qid"] == "Q03").sum(), 43)

    def test_openai_runner_request_ledger_and_resume(self) -> None:
        class FakeResponses:
            def __init__(self, fail: bool = False) -> None:
                self.fail = fail
                self.calls: list[dict] = []

            async def create(self, **kwargs):
                self.calls.append(kwargs)
                if self.fail:
                    raise RuntimeError("temporary failure")
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "type_inferred": "A",
                            "type_rationale": "Direct statement.",
                            "type_confidence": "high",
                        }
                    ),
                    usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_runner_args(root)
            eligibility = root / "human_reference.csv"
            pd.DataFrame(
                {
                    "bundle_id": ["p1"],
                    "qid": ["Q01"],
                    "type": ["B"],
                    "gold_answer": ["must not enter the request"],
                }
            ).to_csv(eligibility, index=False)
            args.eligibility = eligibility

            failing = FakeResponses(fail=True)
            with patch(
                "openai.AsyncOpenAI",
                return_value=SimpleNamespace(responses=failing),
            ):
                first_records, pairs = asyncio.run(run_calls(args))
            self.assertEqual(len(failing.calls), 2)
            self.assertTrue(all(record["error"] for record in first_records))

            succeeding = FakeResponses()
            with patch(
                "openai.AsyncOpenAI",
                return_value=SimpleNamespace(responses=succeeding),
            ):
                records, pairs = asyncio.run(run_calls(args))
            self.assertEqual(len(succeeding.calls), 2)
            self.assertEqual(len(records), 4)
            request_text = json.dumps(succeeding.calls, ensure_ascii=False)
            self.assertNotIn("must not enter the request", request_text)
            self.assertEqual(
                [item["role"] for item in succeeding.calls[0]["input"]],
                ["system", "user"],
            )
            self.assertTrue(succeeding.calls[0]["text"]["format"]["strict"])
            export_wide(records, pairs, args.runs, args.output)

            resumed = FakeResponses()
            with patch(
                "openai.AsyncOpenAI",
                return_value=SimpleNamespace(responses=resumed),
            ):
                resumed_records, resumed_pairs = asyncio.run(run_calls(args))
            self.assertEqual(resumed.calls, [])
            self.assertEqual(len(resumed_records), 4)
            export_wide(resumed_records, resumed_pairs, args.runs, args.output)

    def test_anthropic_runner_preserves_historical_request_placement(self) -> None:
        class FakeMessages:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            type="tool_use",
                            input={
                                "type_inferred": "B",
                                "type_rationale": "Indirect statement.",
                                "type_confidence": "medium",
                            },
                        )
                    ],
                    usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_runner_args(root, provider="anthropic")
            messages = FakeMessages()
            with patch(
                "anthropic.AsyncAnthropic",
                return_value=SimpleNamespace(messages=messages),
            ):
                records, pairs = asyncio.run(run_calls(args))
            self.assertEqual(len(messages.calls), 2)
            request = messages.calls[0]
            self.assertEqual(
                request["tool_choice"]["name"], "submit_type_classification"
            )
            self.assertEqual(len(request["system"]), 2)
            self.assertIn("BEGIN TRANSCRIPT", request["system"][1]["text"])
            self.assertEqual(request["messages"][0]["role"], "user")
            export_wide(records, pairs, args.runs, args.output)

    def test_export_wide_requires_every_run(self) -> None:
        pairs = pd.DataFrame(
            {"bundle_id": ["p1"], "qid": ["Q01"], "question": ["One?"]}
        )
        records = [
            {"bundle_id": "p1", "qid": "Q01", "run": 1, "label": "A", "error": ""}
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs.csv"
            with self.assertRaisesRegex(ValueError, "1 requested calls are incomplete"):
                export_wide(records, pairs, runs=2, output_path=output)

    def test_export_wide_matches_layer_one_schema(self) -> None:
        pairs = pd.DataFrame(
            {"bundle_id": ["p1"], "qid": ["Q01"], "question": ["One?"]}
        )
        records = [
            {"bundle_id": "p1", "qid": "Q01", "run": 1, "label": "A", "error": ""},
            {"bundle_id": "p1", "qid": "Q01", "run": 2, "label": "C", "error": ""},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs.csv"
            export_wide(records, pairs, runs=2, output_path=output)
            exported = pd.read_csv(output)
        self.assertEqual(
            exported.to_dict("records"),
            [{"bundle_id": "p1", "qid": "Q01", "run01": "A", "run02": "C"}],
        )

    def test_manifest_rejects_changed_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for name in ["transcripts.parquet", "cohort.csv", "questions.csv"]:
                path = root / name
                path.write_text(name, encoding="utf-8")
                inputs.append(path)
            args = argparse.Namespace(
                provider="anthropic",
                model="model-a",
                runs=2,
                max_tokens=100,
                reasoning_effort=None,
                max_pairs=1,
                transcripts=inputs[0],
                cohort=inputs[1],
                questions=inputs[2],
                exclusions=None,
                eligibility=None,
                ledger=root / "calls.jsonl",
            )
            manifest = ensure_run_manifest(args, "prompt")
            self.assertEqual(json.loads(manifest.read_text())["provider"], "anthropic")
            self.assertEqual(json.loads(manifest.read_text())["model"], "model-a")
            args.provider = "openai"
            with self.assertRaisesRegex(ValueError, "run configuration differs"):
                ensure_run_manifest(args, "prompt")


if __name__ == "__main__":
    unittest.main()
