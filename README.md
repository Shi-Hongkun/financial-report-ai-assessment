# HTX FDE AI Assessment

## Setup and Run

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --extra pdf-benchmark
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `.env`; keep the file local because it contains credentials. `GEMINI_MODEL` defaults to `gemini-3.8-flash`.

Open [notebooks/part1.ipynb](notebooks/part1.ipynb) for Parts 1 and 2. [notebooks/part2.ipynb](notebooks/part2.ipynb) is an optional standalone date runner. Part 3's LangGraph supervisor is in [notebooks/part3.ipynb](notebooks/part3.ipynb); set `PART3_QUERY_TO_RUN` to `revenue_only`, `fund_only`, or `combined`. Gemini API requests occur when you run the notebooks; the connection ping is off by default.

## Scope and Documentation

Current scope: Parts 1-3. Install dependencies with `uv sync --extra pdf-benchmark`. Keep all project documentation short and concise; update this README and [docs/plan.md](docs/plan.md) when major decisions or results change.
