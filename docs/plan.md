# Part 1 Plan

> Documentation rule: keep every project document short and concise. Update this plan and the README when major decisions or results change.

## Scope and Stack

Work on Parts 1-3. Use Python 3.14, `uv`, JupyterLab, reusable Python utilities, Gemini, and LangGraph.

## Task 1: PDF Extraction

**Status: complete.** The notebook inspects pages 8 (table), 9 (pie chart), and 36 (columns), then compares pdfplumber, PyMuPDF, Docling, MarkItDown, and LangChain.

- MarkItDown was strongest for the table; LangChain was strongest for the chart; LangChain and PyMuPDF worked well for columns.
- PyMuPDF positioned blocks add layout context. Native parsers do not recover chart relationships or a clean page 8 table.
- Docling could not download its model because of a TLS certificate failure in this environment.

## Task 2: Prompt Engineering

**Status: five live calls completed** with `gemini-3.8-flash`. Each call extracts one field and validates its type, unit, page, evidence, and status.

- Loader per page follows Task 1: tables use MarkItDown (page 8); prose uses LangChain PyPDFLoader (pages 5, 6). Page 20 is a table, but MarkItDown separates fund names from amounts, so it uses LangChain. See `TASK2_PAGE_LAYOUTS`.
- System prompts follow CO-STAR; see Prompt Convention below.

| Field | Result |
| --- | --- |
| Revised FY2023 Corporate Income Tax | 28.4 SGD billion |
| Page 5 Corporate Income Tax percentage difference | 17.0% above Estimated FY2023 |
| FY2024 top-ups | 20,352 SGD million |
| Operating Revenue taxes | Corporate Income Tax; Other Taxes; Personal Income Tax; Assets Taxes; Betting Taxes; Goods and Services Tax |
| Latest Actual fiscal position | 1.72 SGD billion |

## Next Steps

1. Manually verify the extracted values and evidence against the cited pages.
2. Finalize the concise Part 1 write-up and preserve the page 5 comparison basis; it is not a comparison with FY2022 Actual.
3. Do not repeat Gemini calls unless a prompt or source changes; the setup cell skips the connection ping by default.

Credential note: keep API keys in the ignored `.env` file. Rotate the key previously pasted into chat if this has not already been done.

## Prompt Convention

Write every system prompt with **CO-STAR** sections, in this order. Keep each section short.

| Section | Content |
| --- | --- |
| Context | Input type; treat input as untrusted data |
| Objective | The one task, plus the not-found rule |
| Style | Writing style, e.g. literal, verbatim evidence |
| Tone | e.g. neutral, no hedging |
| Audience | Who or what consumes the output, e.g. a JSON parser |
| Response | Exact output format and schema |

Template: `_SYSTEM_PROMPT_TEMPLATE` in [prompt_engineering.py](../src/htx_fde_ai_assessment/prompt_engineering.py). Part 2 and Part 3 prompts predate this and are not yet converted.

## Part 2: Tool Calling and Reasoning

**Status: integrated and live-verified.** The canonical Part 1 notebook extracts source text from pages 1 and 36 into `PART1_DATE_SOURCE_TEXTS`. Gemini selects the dates, calls a local LangChain `@tool` to normalize them, then classifies them against `2024-01-01`. A deterministic `datetime.date` comparison checks each LLM label and is shown beside it. The separate Part 2 notebook remains an optional standalone runner.

| Source | Normalized date | Status |
| --- | --- | --- |
| Distribution date, page 1 | `2024-02-16` | Upcoming |
| Estate Duty date, page 36 | `2008-02-15` | Expired |

`Ongoing` is reserved for a date or period active on the reference date; neither source is a period. Implementation: [date_reasoning.py](../src/htx_fde_ai_assessment/date_reasoning.py), [part1.ipynb](../notebooks/part1.ipynb), [part2.ipynb](../notebooks/part2.ipynb), and [test_date_reasoning.py](../tests/test_date_reasoning.py). Eight unit tests pass.

## Part 3: Multi-Agent Supervisor

**Status: implemented and live-verified.** A LangGraph supervisor routes to Revenue and/or Expenditure agents, then synthesizes source-grounded findings. The notebook includes revenue-only, fund-only, and combined demo queries. The combined query successfully routed to both agents and cited pages 16, 18, 20, and 37.

- Revenue evidence: FY2024 estimated operating-revenue breakdown, page 16; definition and main components, page 37.
- Fund evidence: initial Future Energy Fund injection and purpose, page 18; estimated FY2024 top-up of SGD 5 billion, page 20.
- Assumption: the source states an initial government injection for energy-transition infrastructure, but no dedicated tax or revenue stream. Do not infer a separate funding mechanism.

Evidence quotes are checked against each agent's assigned pages, and synthesis citations must match agent evidence. The trace lists the supervisor decision, agents run, and citation pages. The combined answer distinguishes the initial SGD 5 billion injection from the estimated FY2024 SGD 5 billion top-up.

Implementation: [part3_supervisor.py](../src/htx_fde_ai_assessment/part3_supervisor.py), [part3.ipynb](../notebooks/part3.ipynb), and [test_part3_supervisor.py](../tests/test_part3_supervisor.py). Six Part 3 tests pass.
