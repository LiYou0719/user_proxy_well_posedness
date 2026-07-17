from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.run_answerability import (
    build_pairs,
    ensure_run_manifest,
    export_wide,
    parse_tool_result,
    render_transcript,
)


class AnswerabilityRunnerTests(unittest.TestCase):
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
                model="model-a",
                runs=2,
                max_tokens=100,
                max_pairs=1,
                transcripts=inputs[0],
                cohort=inputs[1],
                questions=inputs[2],
                ledger=root / "calls.jsonl",
            )
            manifest = ensure_run_manifest(args, "prompt")
            self.assertEqual(json.loads(manifest.read_text())["model"], "model-a")
            args.model = "model-b"
            with self.assertRaisesRegex(ValueError, "run configuration differs"):
                ensure_run_manifest(args, "prompt")


if __name__ == "__main__":
    unittest.main()
