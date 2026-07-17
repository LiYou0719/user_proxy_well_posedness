from __future__ import annotations

import unittest

import pandas as pd

from scripts.aggregate_human_reference import aggregate_human_reference


class HumanReferenceTests(unittest.TestCase):
    def test_strict_type_aware_aggregation(self) -> None:
        rows = pd.DataFrame(
            [
                {"bundle_id": "p1", "qid": "Q01", "type": "A", "judgment": "correct"},
                {"bundle_id": "p2", "qid": "Q01", "type": "A", "judgment": "partial"},
                {
                    "bundle_id": "p3",
                    "qid": "Q01",
                    "type": "B",
                    "judgment": "inferred_correctly",
                },
                {
                    "bundle_id": "p4",
                    "qid": "Q01",
                    "type": "C",
                    "judgment": "abstained_correctly",
                },
                {"bundle_id": "p5", "qid": "Q01", "type": "skip", "judgment": ""},
            ]
        )
        summary = aggregate_human_reference(rows).iloc[0]
        self.assertEqual(summary["n_passed"], 3)
        self.assertEqual(summary["n_evaluated"], 4)
        self.assertEqual(summary["human_pass_rate"], 0.75)

    def test_duplicate_pair_is_rejected(self) -> None:
        rows = pd.DataFrame(
            [
                {"bundle_id": "p1", "qid": "Q01", "type": "A", "judgment": "correct"},
                {"bundle_id": "p1", "qid": "Q01", "type": "A", "judgment": "wrong"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_human_reference(rows)

    def test_type_incompatible_judgment_is_rejected(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "bundle_id": "p1",
                    "qid": "Q01",
                    "type": "C",
                    "judgment": "correct",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "invalid judgment"):
            aggregate_human_reference(rows)


if __name__ == "__main__":
    unittest.main()
