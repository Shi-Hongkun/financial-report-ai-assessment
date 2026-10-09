import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import pymupdf
import pdfplumber

PageLayout = Literal["table", "text"]
_LOADER_EXTRA_HINT = "Install the optional loaders with `uv sync --extra pdf-benchmark`."


def _validate_page_number(page_number: int, page_count: int) -> None:
    if not 1 <= page_number <= page_count:
        raise ValueError(
            f"page_number must be between 1 and {page_count}; got {page_number}"
        )


def extract_page_text_pdfplumber(pdf_path: str | Path, page_number: int) -> str:
    """Extract text from a 1-based PDF page with pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        _validate_page_number(page_number, len(pdf.pages))
        return pdf.pages[page_number - 1].extract_text() or ""


def extract_page_tables_pdfplumber(
    pdf_path: str | Path,
    page_number: int,
    table_settings: dict[str, str] | None = None,
) -> list[list[list[str | None]]]:
    """Extract tables from a 1-based PDF page with pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        _validate_page_number(page_number, len(pdf.pages))
        page = pdf.pages[page_number - 1]
        settings = table_settings or {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
        }
        return page.extract_tables(table_settings=settings) or []


def extract_page_text_pymupdf(pdf_path: str | Path, page_number: int) -> str:
    """Extract text from a 1-based PDF page with PyMuPDF."""
    with pymupdf.open(pdf_path) as document:
        _validate_page_number(page_number, len(document))
        return document[page_number - 1].get_text()


def extract_page_blocks_pymupdf(
    pdf_path: str | Path, page_number: int
) -> list[tuple[float, float, float, float, str, int, int]]:
    """Extract positioned text blocks from a 1-based PDF page with PyMuPDF."""
    with pymupdf.open(pdf_path) as document:
        _validate_page_number(page_number, len(document))
        return document[page_number - 1].get_text("blocks", sort=True)


def extract_page_tables_pymupdf(
    pdf_path: str | Path, page_number: int, strategy: str = "text"
) -> list[list[list[str | None]]]:
    """Extract detected tables from a 1-based PDF page with PyMuPDF."""
    with pymupdf.open(pdf_path) as document:
        _validate_page_number(page_number, len(document))
        table_finder = document[page_number - 1].find_tables(strategy=strategy)
        return [table.extract() for table in table_finder.tables]


def render_page_png_pymupdf(
    pdf_path: str | Path, page_number: int, dpi: int = 150
) -> bytes:
    """Render a 1-based PDF page to PNG bytes for display or vision-model input."""
    if dpi <= 0:
        raise ValueError(f"dpi must be positive; got {dpi}")

    with pymupdf.open(pdf_path) as document:
        _validate_page_number(page_number, len(document))
        page = document[page_number - 1]
        return page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")


@contextmanager
def _single_page_pdf(pdf_path: str | Path, page_number: int) -> Iterator[Path]:
    """Yield a temporary one-page PDF so whole-document loaders return only that page."""
    with pymupdf.open(pdf_path) as source:
        _validate_page_number(page_number, len(source))
        with tempfile.TemporaryDirectory(prefix="pdf_page_") as temporary_dir:
            page_path = Path(temporary_dir) / f"page_{page_number}.pdf"
            with pymupdf.open() as single_page:
                single_page.insert_pdf(
                    source, from_page=page_number - 1, to_page=page_number - 1
                )
                single_page.save(page_path)
            yield page_path


def extract_page_markdown_markitdown(pdf_path: str | Path, page_number: int) -> str:
    """Extract a 1-based page as Markdown with MarkItDown, which keeps table cells."""
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError(_LOADER_EXTRA_HINT) from exc

    with _single_page_pdf(pdf_path, page_number) as page_path:
        return MarkItDown().convert(str(page_path)).text_content


def extract_page_text_langchain(pdf_path: str | Path, page_number: int) -> str:
    """Extract a 1-based page with LangChain's PyPDFLoader, which suits prose and columns."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as exc:
        raise RuntimeError(_LOADER_EXTRA_HINT) from exc

    with _single_page_pdf(pdf_path, page_number) as page_path:
        documents = PyPDFLoader(str(page_path)).load()
    return "\n".join(document.page_content for document in documents)


def extract_page_text_for_llm(
    pdf_path: str | Path, page_number: int, layout: PageLayout
) -> str:
    """Pick the loader the Task 1 benchmark chose for this layout: tables or prose."""
    if layout == "table":
        return extract_page_markdown_markitdown(pdf_path, page_number)
    if layout == "text":
        return extract_page_text_langchain(pdf_path, page_number)
    raise ValueError(f"layout must be 'table' or 'text'; got {layout!r}")