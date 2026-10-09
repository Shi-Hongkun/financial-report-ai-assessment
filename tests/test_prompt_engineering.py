import unittest

from htx_fde_ai_assessment.pdf_extraction import extract_page_text_for_llm
from htx_fde_ai_assessment.prompt_engineering import (
    TASK2_OUTPUT_SCHEMA,
    TASK2_PAGE_LAYOUTS,
    build_task2_messages,
    validate_task2_field_result,
)


class CorporateIncomeTaxPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.page_texts = {
            5: (
                "Corporate Income Tax collections are revised to $28.4 billion, "
                "which is $4.1 billion (17.0%) higher than the Estimated FY2023 figure."
            ),
            6: "Operating Revenue details.",
            8: "Fiscal Position.",
            20: "Top-ups to Endowment and Trust Funds.",
        }

    def test_schema_uses_the_intended_fy2023_fields(self) -> None:
        self.assertIn("corporate_income_tax_2023", TASK2_OUTPUT_SCHEMA)
        self.assertIn("corporate_income_tax_yoy_change_2023", TASK2_OUTPUT_SCHEMA)
        self.assertNotIn("corporate_income_tax_2024", TASK2_OUTPUT_SCHEMA)
        self.assertNotIn("corporate_income_tax_yoy_change_2024", TASK2_OUTPUT_SCHEMA)

    def test_amount_prompt_asks_for_revised_fy2023(self) -> None:
        messages = build_task2_messages("corporate_income_tax_2023", self.page_texts)
        self.assertIn("Revised FY2023", messages[0].content)
        self.assertIn("PDF page 5", messages[1].content)

    def test_percentage_prompt_asks_for_the_percentage_difference(self) -> None:
        messages = build_task2_messages(
            "corporate_income_tax_yoy_change_2023", self.page_texts
        )
        self.assertIn("percentage difference", messages[0].content)
        self.assertIn("PDF page 5", messages[1].content)

    def test_system_prompt_follows_co_star_sections_in_order(self) -> None:
        content = build_task2_messages(
            "corporate_income_tax_2023", self.page_texts
        )[0].content
        sections = ["# CONTEXT", "# OBJECTIVE", "# STYLE", "# TONE", "# AUDIENCE", "# RESPONSE"]
        positions = [content.index(section) for section in sections]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("corporate_income_tax_2023", content.split("# RESPONSE")[1])

    def test_table_pages_use_markitdown_and_prose_pages_use_langchain(self) -> None:
        self.assertEqual(TASK2_PAGE_LAYOUTS[8], "table")
        self.assertEqual(TASK2_PAGE_LAYOUTS[5], "text")
        self.assertEqual(TASK2_PAGE_LAYOUTS[6], "text")
        with self.assertRaises(ValueError):
            extract_page_text_for_llm("unused.pdf", 1, "columns")  # type: ignore[arg-type]

    def test_validates_amount_and_percentage_as_floats(self) -> None:
        amount = validate_task2_field_result(
            "corporate_income_tax_2023",
            {
                "corporate_income_tax_2023": {
                    "value": 28.4,
                    "unit": "SGD billion",
                    "source_page": 5,
                    "evidence": "revised to $28.4 billion",
                    "status": "found",
                }
            },
        )
        percentage = validate_task2_field_result(
            "corporate_income_tax_yoy_change_2023",
            {
                "corporate_income_tax_yoy_change_2023": {
                    "value": 17,
                    "unit": "percent",
                    "source_page": 5,
                    "evidence": "(17.0%) higher than the Estimated FY2023 figure",
                    "status": "found",
                }
            },
        )
        self.assertEqual(amount["corporate_income_tax_2023"]["value"], 28.4)
        self.assertIsInstance(
            amount["corporate_income_tax_2023"]["value"], float
        )
        self.assertEqual(
            percentage["corporate_income_tax_yoy_change_2023"]["value"], 17.0
        )

    def test_rejects_source_page_outside_the_prompt_scope(self) -> None:
        with self.assertRaises(ValueError):
            validate_task2_field_result(
                "corporate_income_tax_2023",
                {
                    "corporate_income_tax_2023": {
                        "value": 28.4,
                        "unit": "SGD billion",
                        "source_page": 8,
                        "evidence": "revised to $28.4 billion",
                        "status": "found",
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()