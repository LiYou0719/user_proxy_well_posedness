from __future__ import annotations

import unittest

import pandas as pd

from scripts.sample_cohort import sample_cohort


class ReplicationDataTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for split, size in {"workforce": 8, "creatives": 5, "scientists": 4}.items():
            rows.extend(
                {"transcript_id": f"{split}-{index:02d}", "split": split, "text": "x"}
                for index in range(size)
            )
        self.transcripts = pd.DataFrame(rows)

    def test_sampling_is_deterministic_and_stratified(self) -> None:
        counts = {"workforce": 4, "creatives": 2, "scientists": 2}
        first = sample_cohort(self.transcripts, counts, seed=123)
        second = sample_cohort(self.transcripts, counts, seed=123)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["split"].value_counts().to_dict(), counts)
        self.assertFalse(first["transcript_id"].duplicated().any())

    def test_sampling_changes_with_seed(self) -> None:
        counts = {"workforce": 4, "creatives": 2, "scientists": 2}
        first = sample_cohort(self.transcripts, counts, seed=123)
        second = sample_cohort(self.transcripts, counts, seed=456)

        self.assertNotEqual(first["transcript_id"].tolist(), second["transcript_id"].tolist())

    def test_oversampling_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requested 9"):
            sample_cohort(self.transcripts, {"workforce": 9}, seed=123)


if __name__ == "__main__":
    unittest.main()
