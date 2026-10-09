import unittest

from htx_fde_ai_assessment.date_reasoning import (
    compare_classifications_to_calendar,
    normalize_submission_date,
    validate_date_classifications,
)


class NormalizeSubmissionDateTests(unittest.TestCase):
    def test_normalizes_distribution_date(self) -> None:
        result = normalize_submission_date.invoke(
            {
                "date_text": "Distributed on Budget Day: 16 February 2024",
                "source_page": 1,
            }
        )
        self.assertEqual(result, "2024-02-16")

    def test_normalizes_estate_duty_date_with_source_sentence(self) -> None:
        result = normalize_submission_date.invoke(
            {
                "date_text": "Estate Duty does not apply after 15 February 2008.",
                "source_page": 36,
            }
        )
        self.assertEqual(result, "2008-02-15")

    def test_rejects_page_outside_assessment_scope(self) -> None:
        with self.assertRaises(ValueError):
            normalize_submission_date.invoke(
                {"date_text": "16 February 2024", "source_page": 5}
            )

    def test_rejects_text_without_a_date(self) -> None:
        with self.assertRaises(ValueError):
            normalize_submission_date.invoke(
                {"date_text": "No date is present", "source_page": 1}
            )


class DateClassificationValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalized_dates = [
            {
                "original_text": "Distributed on Budget Day: 16 February 2024",
                "source_page": 1,
                "normalized_date": "2024-02-16",
            },
            {
                "original_text": "Estate Duty does not apply after 15 February 2008.",
                "source_page": 36,
                "normalized_date": "2008-02-15",
            },
        ]

    def test_accepts_valid_classification_and_preserves_source_order(self) -> None:
        result = {
            "results": [
                {**self.normalized_dates[1], "status": "Expired"},
                {**self.normalized_dates[0], "status": "Upcoming"},
            ]
        }
        validated = validate_date_classifications(self.normalized_dates, result)
        self.assertEqual(
            [item["status"] for item in validated], ["Upcoming", "Expired"]
        )

    def test_rejects_classification_that_changes_normalized_date(self) -> None:
        result = {
            "results": [
                {**self.normalized_dates[0], "normalized_date": "2024-02-17", "status": "Upcoming"},
                {**self.normalized_dates[1], "status": "Expired"},
            ]
        }
        with self.assertRaises(ValueError):
            validate_date_classifications(self.normalized_dates, result)

    def test_compares_llm_status_with_calendar_status(self) -> None:
        classifications = [
            {**self.normalized_dates[0], "status": "Upcoming"},
            {**self.normalized_dates[1], "status": "Expired"},
        ]
        comparisons = compare_classifications_to_calendar(
            self.normalized_dates, classifications
        )
        self.assertEqual([item["matches"] for item in comparisons], [True, True])
        self.assertEqual(
            [item["date_comparison_status"] for item in comparisons],
            ["Upcoming", "Expired"],
        )

        classifications[0]["status"] = "Expired"
        comparisons = compare_classifications_to_calendar(
            self.normalized_dates, classifications
        )
        self.assertFalse(comparisons[0]["matches"])

    def test_reference_date_is_classified_as_ongoing(self) -> None:
        normalized_dates = [
            {
                "source_page": 1,
                "normalized_date": "2024-01-01",
            }
        ]
        classifications = [
            {
                **normalized_dates[0],
                "status": "Ongoing",
            }
        ]
        comparison = compare_classifications_to_calendar(
            normalized_dates, classifications
        )
        self.assertEqual(comparison[0]["date_comparison_status"], "Ongoing")
        self.assertTrue(comparison[0]["matches"])


if __name__ == "__main__":
    unittest.main()