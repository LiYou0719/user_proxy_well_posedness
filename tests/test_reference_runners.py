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

from scripts.run_content_grader import (
    build_grading_pairs,
    openai_text_format,
    render_user_prompt,
    run_calls as run_grader_calls,
    validate_judgment,
)
from scripts.run_user_proxy import (
    build_pairs,
    parse_proxy_response,
    render_system_prompt,
    run_calls as run_proxy_calls,
)


class ReferenceRunnerTests(unittest.TestCase):
    def test_proxy_parser_extracts_historical_sections(self) -> None:
        parsed = parse_proxy_response(
            "Evidence: Direct quote.\n\nReasoning: It answers.\n\nResponse: Drafting."
        )
        self.assertEqual(parsed["evidence"], "Direct quote.")
        self.assertEqual(parsed["reasoning"], "It answers.")
        self.assertEqual(parsed["response"], "Drafting.")

    def test_proxy_eligibility_does_not_expose_private_columns(self) -> None:
        transcripts = pd.DataFrame({"transcript_id": ["p1"], "text": ["private transcript"]})
        cohort = pd.DataFrame({"transcript_id": ["p1"]})
        questions = pd.DataFrame({"qid": ["Q01"], "question": ["What task?"]})
        with tempfile.TemporaryDirectory() as directory:
            eligibility = Path(directory) / "human.csv"
            pd.DataFrame(
                {
                    "bundle_id": ["p1"],
                    "qid": ["Q01"],
                    "type": ["A"],
                    "gold_answer": ["SECRET GOLD"],
                }
            ).to_csv(eligibility, index=False)
            pairs = build_pairs(transcripts, cohort, questions, eligibility)
        self.assertEqual(list(pairs.columns), ["bundle_id", "qid", "question"])
        self.assertNotIn("SECRET GOLD", pairs.to_csv(index=False))

    def test_proxy_openai_request_contains_no_human_gold(self) -> None:
        class FakeResponses:
            def __init__(self) -> None:
                self.calls = []

            async def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    output_text="Evidence: Quote.\nReasoning: Direct.\nResponse: Drafting.",
                    usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcripts = root / "transcripts.parquet"
            cohort = root / "cohort.csv"
            questions = root / "questions.csv"
            eligibility = root / "human.csv"
            pd.DataFrame({"transcript_id": ["p1"], "text": ["User: I draft."]}).to_parquet(
                transcripts, index=False
            )
            pd.DataFrame({"transcript_id": ["p1"]}).to_csv(cohort, index=False)
            pd.DataFrame({"qid": ["Q01"], "question": ["What task?"]}).to_csv(
                questions, index=False
            )
            pd.DataFrame(
                {
                    "bundle_id": ["p1"],
                    "qid": ["Q01"],
                    "type": ["A"],
                    "gold_answer": ["SECRET GOLD"],
                }
            ).to_csv(eligibility, index=False)
            args = argparse.Namespace(
                provider="openai",
                model="test-model",
                condition_id="question_only",
                concurrency=1,
                max_tokens=100,
                reasoning_effort=None,
                max_pairs=None,
                transcripts=transcripts,
                cohort=cohort,
                questions=questions,
                eligibility=eligibility,
                ledger=root / "proxy.jsonl",
                output=root / "proxy.csv",
            )
            fake = FakeResponses()
            with patch("openai.AsyncOpenAI", return_value=SimpleNamespace(responses=fake)):
                asyncio.run(run_proxy_calls(args))
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn("SECRET GOLD", json.dumps(fake.calls[0]))

    def test_content_grader_contract_and_strict_schema(self) -> None:
        proxy = pd.DataFrame(
            {
                "bundle_id": ["p1"],
                "qid": ["Q01"],
                "question": ["What task?"],
                "evidence": ["Quote"],
                "reasoning": ["Direct"],
                "response": ["Drafting"],
            }
        )
        human = pd.DataFrame(
            {
                "bundle_id": ["p1"],
                "qid": ["Q01"],
                "question": ["What task?"],
                "type": ["A"],
                "gold_answer": ["Drafting"],
                "gold_evidence": ["Quote"],
                "absence_check": [""],
            }
        )
        pairs = build_grading_pairs(proxy, human)
        self.assertEqual(pairs.iloc[0]["type"], "A")
        result = validate_judgment(
            "A", {"judgment": "correct", "rationale": "Match", "confidence": "high"}
        )
        self.assertTrue(result["passed"])
        tool = {
            "name": "grade",
            "description": "Grade",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        self.assertTrue(openai_text_format(tool)["strict"])
        self.assertFalse(openai_text_format(tool)["schema"]["additionalProperties"])

    def test_content_grader_max_pairs_can_filter_full_proxy_output(self) -> None:
        proxy = pd.DataFrame(
            {
                "bundle_id": ["p1", "p1"],
                "qid": ["Q01", "Q02"],
                "question": ["One?", "Two?"],
                "evidence": ["Quote", ""],
                "reasoning": ["Direct", "Absent"],
                "response": ["Answer", "I do not have enough information."],
            }
        )
        human = pd.DataFrame(
            {
                "bundle_id": ["p1", "p1"],
                "qid": ["Q01", "Q02"],
                "question": ["One?", "Two?"],
                "type": ["A", "C"],
                "gold_answer": ["Answer", ""],
                "gold_evidence": ["Quote", ""],
                "absence_check": ["", "Topic absent"],
            }
        )
        pairs = build_grading_pairs(proxy, human, max_pairs=1)
        self.assertEqual(pairs[["bundle_id", "qid"]].to_dict("records"), [
            {"bundle_id": "p1", "qid": "Q01"}
        ])

    def test_content_prompt_renders_gold_only_at_grading_stage(self) -> None:
        template = (
            "{{EVALUATION_QUESTION}}|{{TYPE}}|{{GOLD_ANSWER_OR_NOT_PROVIDED}}|"
            "{{GOLD_EVIDENCE_OR_NOT_PROVIDED}}|{{ABSENCE_CHECK_OR_NOT_PROVIDED}}|"
            "{{RESEARCHER_NOTES_OR_NOT_PROVIDED}}|{{PROXY_EVIDENCE_OR_EMPTY}}|"
            "{{PROXY_REASONING_OR_EMPTY}}|{{PROXY_RESPONSE_OR_EMPTY}}"
        )
        rendered = render_user_prompt(
            template,
            {
                "question": "What task?",
                "type": "A",
                "gold_answer": "SECRET GOLD",
                "gold_evidence": "Quote",
                "evidence": "Proxy quote",
                "reasoning": "Proxy reasoning",
                "response": "Proxy answer",
            },
        )
        self.assertIn("SECRET GOLD", rendered)
        self.assertNotIn("{{", rendered)

    def test_content_grader_openai_request_and_ledger(self) -> None:
        class FakeResponses:
            def __init__(self) -> None:
                self.calls = []

            async def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "judgment": "correct",
                            "rationale": "The response matches the gold.",
                            "confidence": "high",
                        }
                    ),
                    usage=SimpleNamespace(input_tokens=20, output_tokens=8),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers = root / "proxy.csv"
            human = root / "human.csv"
            pd.DataFrame(
                {
                    "bundle_id": ["p1"],
                    "qid": ["Q01"],
                    "question": ["What task?"],
                    "evidence": ["Quote"],
                    "reasoning": ["Direct"],
                    "response": ["Drafting"],
                }
            ).to_csv(answers, index=False)
            pd.DataFrame(
                {
                    "bundle_id": ["p1"],
                    "qid": ["Q01"],
                    "question": ["What task?"],
                    "type": ["A"],
                    "gold_answer": ["Drafting"],
                    "gold_evidence": ["Quote"],
                    "absence_check": [""],
                }
            ).to_csv(human, index=False)
            args = argparse.Namespace(
                provider="openai",
                model="grader-model",
                concurrency=1,
                max_tokens=100,
                reasoning_effort=None,
                max_pairs=None,
                answers=answers,
                human_reference=human,
                ledger=root / "grades.jsonl",
                output=root / "grades.csv",
            )
            fake = FakeResponses()
            with patch("openai.AsyncOpenAI", return_value=SimpleNamespace(responses=fake)):
                records, pairs = asyncio.run(run_grader_calls(args))
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(records[-1]["judgment"], "correct")
            self.assertTrue(records[-1]["passed"])
            self.assertEqual(len(pairs), 1)
            self.assertIn("Drafting", json.dumps(fake.calls[0]))


if __name__ == "__main__":
    unittest.main()
