import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from htx_fde_ai_assessment.pdf_extraction import extract_page_text_pymupdf


REFERENCE_DATE = "2024-01-01"
DATE_SOURCE_PAGES = (1, 36)
DATE_CLASSIFICATIONS = ("Expired", "Upcoming", "Ongoing")

_DATE_PATTERNS = (
    re.compile(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
)
_DATE_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%B %d, %Y", "%Y-%m-%d")
_SOURCE_PAGE_LABELS = {
    1: "document distribution date",
    36: "date relating to estate duty",
}


DATE_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Find exactly two dates in the supplied document excerpts: the document's "
            "distribution date from PDF page 1, and the date relating to Estate Duty "
            "from PDF page 36. For each, call the `normalize_submission_date` tool once. "
            "Pass the complete shortest source statement as `date_text`, not just the "
            "date token: include the `Distributed on Budget Day` label on page 1 and "
            "the full Estate Duty sentence on page 36. Include its 1-based PDF page as "
            "`source_page`. Do not use any other date on page 36. "
            "Do not return dates from memory or infer missing dates.",
        ),
        (
            "human",
            "PDF page 1:\n{page_1_text}\n\nPDF page 36:\n{page_36_text}",
        ),
    ]
)

DATE_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify each source date against the fixed reference date "
            "{reference_date}. Return exactly one JSON object with a `results` array. "
            "Each item must preserve `original_text`, `source_page`, and "
            "`normalized_date` from the input, and add `status` with exactly one of "
            "`Expired`, `Upcoming`, or `Ongoing`. A date before the reference date is "
            "Expired; a date after it is Upcoming. Use Ongoing only when the source "
            "describes a period active on the reference date, or a single date equal "
            "to the reference date. Do not treat a historical effective date as an "
            "ongoing period. Do not add explanations or alter the input values.",
        ),
        ("human", "Classify these normalized source dates:\n{normalized_dates}"),
    ]
)


@tool
def normalize_submission_date(date_text: str, source_page: int) -> str:
    """Normalize one source date to ISO format; only pages 1 and 36 are in scope."""
    if isinstance(source_page, bool) or source_page not in DATE_SOURCE_PAGES:
        raise ValueError(f"source_page must be one of {DATE_SOURCE_PAGES}")

    for pattern in _DATE_PATTERNS:
        match = pattern.search(date_text)
        if match is None:
            continue
        matched_date = match.group(0)
        for date_format in _DATE_FORMATS:
            try:
                return datetime.strptime(matched_date, date_format).date().isoformat()
            except ValueError:
                continue
        raise ValueError(f"Could not parse date in source text: {matched_date!r}")

    raise ValueError(f"No supported date found in source text: {date_text!r}")


def _normalized_source_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def extract_and_normalize_dates(
    llm: Any, page_texts: Mapping[int, str]
) -> list[dict[str, Any]]:
    """Use Gemini tool calling to extract and locally normalize the two source dates."""
    missing_pages = set(DATE_SOURCE_PAGES) - set(page_texts)
    if missing_pages:
        raise ValueError(f"Missing source pages: {sorted(missing_pages)}")

    response = (
        DATE_EXTRACTION_PROMPT
        | llm.bind_tools([normalize_submission_date], tool_choice="required")
    ).invoke(
        {
            "page_1_text": page_texts[1],
            "page_36_text": page_texts[36],
        }
    )
    tool_calls = getattr(response, "tool_calls", [])
    if len(tool_calls) != len(DATE_SOURCE_PAGES):
        raise ValueError(
            f"Expected {len(DATE_SOURCE_PAGES)} date tool calls; got {len(tool_calls)}"
        )

    results = []
    seen_pages = set()
    for tool_call in tool_calls:
        if tool_call.get("name") != normalize_submission_date.name:
            raise ValueError(f"Unexpected tool call: {tool_call.get('name')!r}")
        arguments = tool_call.get("args", {})
        source_page = arguments.get("source_page")
        date_text = arguments.get("date_text")
        if source_page not in _SOURCE_PAGE_LABELS or source_page in seen_pages:
            raise ValueError(f"Unexpected or repeated source page: {source_page!r}")
        if not isinstance(date_text, str) or not date_text.strip():
            raise ValueError("Tool call must include the exact source phrase as date_text")
        if _normalized_source_text(date_text) not in _normalized_source_text(
            page_texts[source_page]
        ):
            raise ValueError(f"date_text is not present on PDF page {source_page}")

        normalized_date = normalize_submission_date.invoke(arguments)
        date.fromisoformat(normalized_date)
        seen_pages.add(source_page)
        results.append(
            {
                "original_text": date_text,
                "source_page": source_page,
                "normalized_date": normalized_date,
            }
        )

    if seen_pages != set(DATE_SOURCE_PAGES):
        raise ValueError(f"Expected dates from pages {DATE_SOURCE_PAGES}; got {seen_pages}")
    return sorted(results, key=lambda result: result["source_page"])


def validate_date_classifications(
    normalized_dates: Sequence[Mapping[str, Any]], result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Validate that classification output preserves the source dates and pages."""
    if set(result) != {"results"} or not isinstance(result["results"], list):
        raise ValueError("Classification output must contain only a results array")
    classified = result["results"]
    if len(classified) != len(normalized_dates):
        raise ValueError("Classification result count does not match normalized dates")

    validated = []
    expected_by_page = {item["source_page"]: item for item in normalized_dates}
    for item in classified:
        if not isinstance(item, Mapping) or set(item) != {
            "original_text",
            "source_page",
            "normalized_date",
            "status",
        }:
            raise ValueError("Each classification must contain the required four fields")
        source_page = item["source_page"]
        expected = expected_by_page.get(source_page)
        if expected is None:
            raise ValueError(f"Unexpected source page in classification: {source_page!r}")
        if any(
            item[field] != expected[field]
            for field in ("original_text", "source_page", "normalized_date")
        ):
            raise ValueError(f"Classification altered source data for page {source_page}")
        if item["status"] not in DATE_CLASSIFICATIONS:
            raise ValueError(f"Unsupported classification status: {item['status']!r}")
        validated.append(dict(item))

    if {item["source_page"] for item in validated} != set(expected_by_page):
        raise ValueError("Each source page must appear exactly once in classification output")
    return sorted(validated, key=lambda item: item["source_page"])


def classify_normalized_dates(
    llm: Any,
    normalized_dates: Sequence[Mapping[str, Any]],
    reference_date: str = REFERENCE_DATE,
) -> list[dict[str, Any]]:
    """Ask the LLM to classify normalized dates and validate its structured result."""
    date.fromisoformat(reference_date)
    for item in normalized_dates:
        date.fromisoformat(item["normalized_date"])

    parsed_result = (
        DATE_CLASSIFICATION_PROMPT | llm | JsonOutputParser()
    ).invoke(
        {
            "reference_date": reference_date,
            "normalized_dates": list(normalized_dates),
        }
    )
    return validate_date_classifications(normalized_dates, parsed_result)


def compare_classifications_to_calendar(
    normalized_dates: Sequence[Mapping[str, Any]],
    classifications: Sequence[Mapping[str, Any]],
    reference_date: str = REFERENCE_DATE,
) -> list[dict[str, Any]]:
    """Compare each LLM label with a deterministic date-versus-reference result."""
    reference = date.fromisoformat(reference_date)
    expected_by_page = {item["source_page"]: item for item in normalized_dates}
    if len(expected_by_page) != len(normalized_dates):
        raise ValueError("Normalized dates must have unique source pages")
    if len(classifications) != len(normalized_dates):
        raise ValueError("Classification count does not match normalized dates")

    comparisons = []
    seen_pages = set()
    for item in classifications:
        source_page = item["source_page"]
        expected = expected_by_page.get(source_page)
        if expected is None or source_page in seen_pages:
            raise ValueError(f"Unexpected or repeated source page: {source_page!r}")
        if item["normalized_date"] != expected["normalized_date"]:
            raise ValueError(f"Normalized date differs for source page {source_page}")

        normalized_date = date.fromisoformat(expected["normalized_date"])
        if normalized_date < reference:
            comparison_status = "Expired"
        elif normalized_date > reference:
            comparison_status = "Upcoming"
        else:
            comparison_status = "Ongoing"

        llm_status = item["status"]
        if llm_status not in DATE_CLASSIFICATIONS:
            raise ValueError(f"Unsupported classification status: {llm_status!r}")
        comparisons.append(
            {
                "source_page": source_page,
                "normalized_date": expected["normalized_date"],
                "llm_status": llm_status,
                "date_comparison_status": comparison_status,
                "matches": llm_status == comparison_status,
            }
        )
        seen_pages.add(source_page)

    if seen_pages != set(expected_by_page):
        raise ValueError("Each normalized date must have one classification")
    return sorted(comparisons, key=lambda item: item["source_page"])


def run_part2_date_workflow(
    llm: Any, pdf_path: str
) -> dict[str, Any]:
    """Extract, normalize, and classify the two assessment dates from the source PDF."""
    page_texts = {
        page_number: extract_page_text_pymupdf(pdf_path, page_number)
        for page_number in DATE_SOURCE_PAGES
    }
    normalized_dates = extract_and_normalize_dates(llm, page_texts)
    classifications = classify_normalized_dates(llm, normalized_dates)
    return {
        "normalized_dates": [item["normalized_date"] for item in normalized_dates],
        "source_dates": normalized_dates,
        "classifications": classifications,
        "reference_date": REFERENCE_DATE,
    }