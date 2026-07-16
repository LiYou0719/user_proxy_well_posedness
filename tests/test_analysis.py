from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze_question_suitability import (
    build_ranking,
    load_cells,
    load_human_pass_rates,
    load_questions,
)


class AnalysisTests(unittest.TestCase):
    def test_public_data_is_canonical_and_bounded(self) -> None:
        questions = load_questions()
        cells = load_cells(questions=questions)
        human = load_human_pass_rates(questions=questions)
        ranking = build_ranking(
            cells, questions, human, bootstrap_repetitions=100
        )

        self.assertEqual(len(questions), 23)
        self.assertEqual(len(ranking), 23)
        self.assertFalse(questions["qid"].str.startswith("A").any())
        for column in ["well_posedness", "ci_lo", "ci_hi", "human_pass_rate"]:
            self.assertTrue(ranking[column].between(0, 1).all())

    def test_a_and_b_collapse_to_answerable(self) -> None:
        questions = load_questions()
        qids = questions["qid"].tolist()
        rows = []
        for qid in qids:
            rows.append(
                {
                    "bundle_id": "participant-1",
                    "qid": qid,
                    "run01": "A",
                    "run02": "B",
                    "run03": "C",
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            cells = load_cells(path, questions)

        self.assertTrue((cells["p_answerable"] == 2 / 3).all())
        for value in cells["within_variance"]:
            self.assertAlmostEqual(value, 2 / 9)

    def test_invalid_label_is_rejected(self) -> None:
        questions = load_questions()
        rows = [
            {"bundle_id": "participant-1", "qid": qid, "run01": "C"}
            for qid in questions["qid"]
        ]
        rows[0]["run01"] = "UNKNOWN"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "invalid answerability labels"):
                load_cells(path, questions)

    def test_duplicate_pair_is_rejected(self) -> None:
        questions = load_questions()
        rows = [
            {"bundle_id": "participant-1", "qid": qid, "run01": "C"}
            for qid in questions["qid"]
        ]
        rows.append(rows[0].copy())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_cells(path, questions)


if __name__ == "__main__":
    unittest.main()
