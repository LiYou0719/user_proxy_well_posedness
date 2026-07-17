from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze_question_suitability import (
    build_well_posedness_summary,
    build_ranking,
    load_cells,
    load_human_summary,
    load_questions,
)


class AnalysisTests(unittest.TestCase):
    def test_public_data_is_canonical_and_bounded(self) -> None:
        questions = load_questions()
        cells = load_cells(questions=questions)
        human = load_human_summary(questions=questions)
        ranking = build_ranking(
            cells, questions, human, bootstrap_repetitions=100
        )

        self.assertEqual(len(questions), 23)
        self.assertEqual(len(ranking), 23)
        self.assertFalse(questions["qid"].str.startswith("A").any())
        participant_counts = ranking.set_index("qid")["n_participants"].to_dict()
        self.assertEqual(participant_counts["Q03"], 43)
        self.assertTrue(
            all(
                count == 50
                for qid, count in participant_counts.items()
                if qid != "Q03"
            )
        )
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

    def test_well_posedness_summary_does_not_require_human_results(self) -> None:
        questions = load_questions()
        cells = load_cells(questions=questions)
        summary = build_well_posedness_summary(
            cells, questions, bootstrap_repetitions=10
        )

        self.assertEqual(len(summary), 23)
        self.assertNotIn("human_pass_rate", summary.columns)
        for column in ["well_posedness", "ci_lo", "ci_hi", "pct_answerable"]:
            self.assertTrue(summary[column].between(0, 1).all())

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

    def test_missing_run_label_is_rejected(self) -> None:
        questions = load_questions()
        rows = [
            {
                "bundle_id": "participant-1",
                "qid": qid,
                "run01": "C",
                "run02": "C",
            }
            for qid in questions["qid"]
        ]
        rows[0]["run02"] = None

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing identifiers or labels"):
                load_cells(path, questions)

    def test_unexpected_question_is_rejected(self) -> None:
        questions = load_questions()
        rows = [
            {"bundle_id": "participant-1", "qid": qid, "run01": "C"}
            for qid in questions["qid"]
        ]
        rows[-1]["qid"] = "Q99"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "qids do not match"):
                load_cells(path, questions)

    def test_out_of_range_human_pass_rate_is_rejected(self) -> None:
        questions = load_questions()
        rates = pd.DataFrame(
            {
                "qid": questions["qid"],
                "n_passed": [1] * len(questions),
                "n_evaluated": [2] * len(questions),
                "human_pass_rate": [0.5] * len(questions),
            }
        )
        rates.loc[0, "human_pass_rate"] = 1.1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human.csv"
            rates.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                load_human_summary(path, questions)

    def test_human_and_answerability_denominators_must_match(self) -> None:
        questions = load_questions()
        cells = load_cells(questions=questions)
        human = load_human_summary(questions=questions)
        human.loc["Q03", "n_evaluated"] = 50

        with self.assertRaisesRegex(ValueError, "participant counts do not match"):
            build_ranking(cells, questions, human, bootstrap_repetitions=10)


if __name__ == "__main__":
    unittest.main()
