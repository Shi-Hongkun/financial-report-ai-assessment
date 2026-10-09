# HTX FDE AI Assessment

## Configure and Run

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --extra pdf-benchmark
Copy-Item .env.example .env
# Add your key to GEMINI_API_KEY in .env
uv run jupyter lab
```

Run [notebooks/part1_part2.ipynb](notebooks/part1_part2.ipynb) for PDF extraction, prompt engineering, and date reasoning. Run [notebooks/part3.ipynb](notebooks/part3.ipynb) for the multi-agent supervisor; set `PART3_QUERY_TO_RUN` to `revenue_only`, `fund_only`, or `combined`. `GEMINI_MODEL` defaults to `gemini-3.8-flash`. Notebook runs make Gemini API requests and may incur charges. Keep `.env` private.

## Design and API

- [pdf_extraction.py](src/htx_fde_ai_assessment/pdf_extraction.py) provides page-level extraction with layout-aware loaders.
- [date_reasoning.py](src/htx_fde_ai_assessment/date_reasoning.py) combines Gemini date extraction/classification with local normalization and validation.
- [part3_supervisor.py](src/htx_fde_ai_assessment/part3_supervisor.py) builds a LangGraph supervisor that routes revenue and expenditure questions and validates page-grounded evidence and citations.
- [gemini_client.py](src/htx_fde_ai_assessment/gemini_client.py) loads local configuration and creates Gemini clients through its OpenAI-compatible API.

JupyterLab and ipykernel run the notebooks; LangChain handles prompts, tools, and structured output; LangGraph provides conditional agent routing. PyMuPDF and pdfplumber extract PDF text/layout, while python-dotenv loads local configuration. The optional `pdf-benchmark` extra adds Docling, MarkItDown, LangChain Community, and pypdf to compare PDF loaders.

## Project Docs

- [Problem statement](docs/problem_statement.md): assessment context and source files.
- [Plan and results](docs/plan.md): implementation status, findings, and design decisions.
