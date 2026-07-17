from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.generate_human_reference_templates import generate_templates
from scripts.merge_human_references import merge_references


class HumanReferenceIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = pd.DataFrame({"transcript_id": ["p1", "p2"]})
        self.questions = pd.DataFrame(
            {"qid": ["Q01", "Q02"], "question": ["First?", "Second?"]}
        )

    def test_generate_one_template_per_transcript_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "by_transcript"
            paths = generate_templates(self.cohort, self.questions, output)
            self.assertEqual([path.name for path in paths], ["p1.csv", "p2.csv"])
            first = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
            self.assertEqual(first["bundle_id"].tolist(), ["p1", "p1"])
            self.assertEqual(first["qid"].tolist(), ["Q01", "Q02"])
            with self.assertRaises(FileExistsError):
                generate_templates(self.cohort, self.questions, output)

    def test_merge_validates_and_preserves_cohort_question_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "by_transcript"
            paths = generate_templates(self.cohort, self.questions, output)
            for path in paths:
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                frame.loc[0, ["type", "gold_evidence"]] = ["A", "A quote"]
                frame.loc[1, ["type", "absence_check"]] = ["C", "Topic absent"]
                frame.to_csv(path, index=False)
            merged = merge_references(self.cohort, self.questions, output)
            self.assertEqual(len(merged), 4)
            self.assertEqual(
                merged[["bundle_id", "qid"]].to_dict("records"),
                [
                    {"bundle_id": "p1", "qid": "Q01"},
                    {"bundle_id": "p1", "qid": "Q02"},
                    {"bundle_id": "p2", "qid": "Q01"},
                    {"bundle_id": "p2", "qid": "Q02"},
                ],
            )

    def test_merge_rejects_unfinished_or_incomplete_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "by_transcript"
            paths = generate_templates(self.cohort, self.questions, output)
            for path in paths:
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                frame["type"] = "C"
                frame["absence_check"] = "Topic absent"
                frame.to_csv(path, index=False)
            first = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
            first.loc[0, "type"] = ""
            first.to_csv(paths[0], index=False)
            with self.assertRaisesRegex(ValueError, "unfinished type"):
                merge_references(self.cohort, self.questions, output)

    def test_merge_accepts_one_reference_field_and_minimal_type_c(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "by_transcript"
            paths = generate_templates(self.cohort, self.questions, output)
            for path in paths:
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                frame.loc[0, ["type", "notes"]] = ["B", "Reasonable inference"]
                frame.loc[1, "type"] = "C"
                frame.to_csv(path, index=False)
            merged = merge_references(self.cohort, self.questions, output)
            self.assertEqual(len(merged), 4)
            self.assertEqual(set(merged["type"]), {"B", "C"})

    def test_merge_rejects_type_a_or_b_without_any_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "by_transcript"
            paths = generate_templates(self.cohort, self.questions, output)
            for path in paths:
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                frame.loc[0, "type"] = "A"
                frame.loc[1, "type"] = "C"
                frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "at least one human reference"):
                merge_references(self.cohort, self.questions, output)


if __name__ == "__main__":
    unittest.main()
