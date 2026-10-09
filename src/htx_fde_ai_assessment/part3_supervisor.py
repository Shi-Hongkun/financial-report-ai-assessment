import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from htx_fde_ai_assessment.pdf_extraction import extract_page_text_pymupdf


AgentName = Literal["revenue", "expenditure"]
PART3_SOURCE_PAGES = (16, 18, 20, 37)
AGENT_SOURCE_PAGES: dict[str, tuple[int, ...]] = {
    "revenue": (16, 37),
    "expenditure": (18, 20),
}


class SupervisorState(TypedDict, total=False):
    query: str
    source_context: dict[int, str]
    selected_agents: list[AgentName]
    routing_reason: str
    revenue_report: dict[str, Any]
    expenditure_report: dict[str, Any]
    answer: str
    citations: list[dict[str, Any]]
    assumptions: list[str]
    trace: list[dict[str, Any]]


SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the supervisor for a government budget research team. Route the "
            "query to one or both specialized agents: `revenue` analyzes government "
            "revenue; `expenditure` analyzes government spending and funds. Select "
            "both when the query asks about both topics or requires a combined answer. "
            "Return exactly one JSON object with `agents` (a non-empty array containing "
            "only `revenue` and/or `expenditure`) and `reason` (a concise explanation). "
            "Do not answer the user query yourself.",
        ),
        ("human", "User query: {query}"),
    ]
)

REVENUE_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Revenue Agent. Identify the main government revenue streams "
            "from the supplied source pages. Use page 16 for FY2024 estimated values "
            "and page 37 for the document's definition of Operating Revenue. Distinguish "
            "tax revenue from fees and other receipts. Do not present estimates as actuals. "
            "Return exactly one JSON object with `findings` (concise prose) and `evidence` "
            "(a non-empty array of objects with integer `source_page` and a short exact "
            "`quote` copied from that page). Do not use outside knowledge.",
        ),
        (
            "human",
            "Question: {query}\n\nPDF page 16:\n{page_16_text}"
            "\n\nPDF page 37:\n{page_37_text}",
        ),
    ]
)

EXPENDITURE_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Expenditure Agent. Explain the Future Energy Fund budget "
            "using only the supplied PDF pages. Distinguish the Government's initial "
            "injection and purpose from the estimated FY2024 fund top-up. If the source "
            "does not identify a dedicated tax or recurring revenue stream, say so; do "
            "not infer one. Return exactly one JSON object with `findings` (concise prose) "
            "and `evidence` (a non-empty array of objects with integer `source_page` and "
            "a short exact `quote` copied from that page).",
        ),
        (
            "human",
            "Question: {query}\n\nPDF page 18:\n{page_18_text}"
            "\n\nPDF page 20:\n{page_20_text}",
        ),
    ]
)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Synthesize the selected agents' findings into a direct answer to the user. "
            "Use only their findings and supplied evidence. Cite claims by page number. "
            "Preserve whether numbers are estimated or initial injections. Do not claim "
            "a dedicated Future Energy Fund revenue source unless the evidence states one. "
            "Return exactly one JSON object with `answer` (string), `citations` (array of "
            "objects with integer `source_page` and exact `quote` copied from the supplied "
            "agent evidence), and `assumptions` (array of strings; use an empty array if "
            "none are needed).",
        ),
        (
            "human",
            "User query: {query}\n\nSupervisor routing reason: {routing_reason}"
            "\n\nAgent findings:\n{agent_reports}\n\nAvailable evidence:\n{evidence}",
        ),
    ]
)


def load_part3_source_context(pdf_path: str | Path) -> dict[int, str]:
    """Load only the assessment pages assigned to the Part 3 agents."""
    return {
        page_number: extract_page_text_pymupdf(pdf_path, page_number)
        for page_number in PART3_SOURCE_PAGES
    }


def validate_supervisor_decision(
    decision: Mapping[str, Any],
) -> tuple[list[AgentName], str]:
    """Validate supervisor routing before conditional graph dispatch."""
    if set(decision) != {"agents", "reason"}:
        raise ValueError("Supervisor decision must contain only agents and reason")
    agents = decision["agents"]
    reason = decision["reason"]
    if not isinstance(agents, list) or not agents:
        raise ValueError("Supervisor must select at least one agent")
    if any(agent not in AGENT_SOURCE_PAGES for agent in agents):
        raise ValueError(f"Supervisor selected an unknown agent: {agents!r}")
    if len(set(agents)) != len(agents):
        raise ValueError("Supervisor agent selection must not contain duplicates")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Supervisor must explain its routing decision")
    return agents, reason


def _normalized_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def validate_agent_report(
    agent_name: AgentName,
    report: Mapping[str, Any],
    source_context: Mapping[int, str],
) -> dict[str, Any]:
    """Check an agent's evidence quotes against its assigned source pages."""
    if set(report) != {"findings", "evidence"}:
        raise ValueError("Agent report must contain only findings and evidence")
    findings = report["findings"]
    evidence = report["evidence"]
    if not isinstance(findings, str) or not findings.strip():
        raise ValueError("Agent findings must be non-empty text")
    allowed_pages = set(AGENT_SOURCE_PAGES[agent_name])
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Agent report must include source evidence")

    validated_evidence = []
    for item in evidence:
        if not isinstance(item, Mapping) or set(item) != {"source_page", "quote"}:
            raise ValueError("Each evidence item must contain source_page and quote")
        source_page = item["source_page"]
        quote = item["quote"]
        if (
            isinstance(source_page, bool)
            or not isinstance(source_page, int)
            or source_page not in allowed_pages
        ):
            raise ValueError(f"{agent_name} used an unassigned source page")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("Evidence quote must be non-empty text")
        source_text = source_context.get(source_page)
        if source_text is None or _normalized_text(quote) not in _normalized_text(
            source_text
        ):
            raise ValueError(f"Evidence quote does not match source page {source_page}")
        validated_evidence.append({"source_page": source_page, "quote": quote})

    return {"findings": findings.strip(), "evidence": validated_evidence}


def validate_synthesis(
    synthesis: Mapping[str, Any],
    available_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Ensure synthesis cites only evidence returned by the agents."""
    if set(synthesis) != {"answer", "citations", "assumptions"}:
        raise ValueError("Synthesis must contain answer, citations, and assumptions")
    answer = synthesis["answer"]
    citations = synthesis["citations"]
    assumptions = synthesis["assumptions"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Synthesis answer must be non-empty text")
    if not isinstance(citations, list) or not citations:
        raise ValueError("Synthesis must cite agent evidence")
    if not isinstance(assumptions, list) or not all(
        isinstance(item, str) for item in assumptions
    ):
        raise ValueError("Synthesis assumptions must be a list of strings")

    evidence_set = {
        (item["source_page"], _normalized_text(item["quote"]))
        for item in available_evidence
    }
    validated_citations = []
    for citation in citations:
        if not isinstance(citation, Mapping) or set(citation) != {
            "source_page",
            "quote",
        }:
            raise ValueError("Each citation must contain source_page and quote")
        source_page = citation["source_page"]
        quote = citation["quote"]
        if not isinstance(source_page, int) or isinstance(source_page, bool):
            raise ValueError("Citation source_page must be an integer")
        if not isinstance(quote, str) or (
            source_page,
            _normalized_text(quote),
        ) not in evidence_set:
            raise ValueError("Synthesis citation must exactly match agent evidence")
        validated_citations.append({"source_page": source_page, "quote": quote})

    return {
        "answer": answer.strip(),
        "citations": validated_citations,
        "assumptions": assumptions,
    }


def build_part3_graph(llm: Any) -> Any:
    """Build the supervisor graph with sequential, conditional agent routing."""
    def supervisor_node(state: SupervisorState) -> dict[str, Any]:
        raw_decision = (SUPERVISOR_PROMPT | llm | JsonOutputParser()).invoke(
            {"query": state["query"]}
        )
        agents, reason = validate_supervisor_decision(raw_decision)
        trace = state.get("trace", []) + [
            {"node": "supervisor", "selected_agents": agents, "reason": reason}
        ]
        return {"selected_agents": agents, "routing_reason": reason, "trace": trace}

    def next_after_supervisor(state: SupervisorState) -> str:
        if "revenue" in state["selected_agents"]:
            return "revenue_agent"
        if "expenditure" in state["selected_agents"]:
            return "expenditure_agent"
        return "synthesize"

    def revenue_agent(state: SupervisorState) -> dict[str, Any]:
        context = state["source_context"]
        raw_report = (REVENUE_AGENT_PROMPT | llm | JsonOutputParser()).invoke(
            {
                "query": state["query"],
                "page_16_text": context[16],
                "page_37_text": context[37],
            }
        )
        report = validate_agent_report("revenue", raw_report, context)
        trace = state.get("trace", []) + [
            {
                "node": "revenue_agent",
                "status": "completed",
                "evidence_pages": sorted(
                    {item["source_page"] for item in report["evidence"]}
                ),
            }
        ]
        return {"revenue_report": report, "trace": trace}

    def next_after_revenue(state: SupervisorState) -> str:
        if "expenditure" in state["selected_agents"]:
            return "expenditure_agent"
        return "synthesize"

    def expenditure_agent(state: SupervisorState) -> dict[str, Any]:
        context = state["source_context"]
        raw_report = (EXPENDITURE_AGENT_PROMPT | llm | JsonOutputParser()).invoke(
            {
                "query": state["query"],
                "page_18_text": context[18],
                "page_20_text": context[20],
            }
        )
        report = validate_agent_report("expenditure", raw_report, context)
        trace = state.get("trace", []) + [
            {
                "node": "expenditure_agent",
                "status": "completed",
                "evidence_pages": sorted(
                    {item["source_page"] for item in report["evidence"]}
                ),
            }
        ]
        return {"expenditure_report": report, "trace": trace}

    def synthesize(state: SupervisorState) -> dict[str, Any]:
        reports = {
            name: state[f"{name}_report"]
            for name in state["selected_agents"]
        }
        available_evidence = [
            evidence
            for report in reports.values()
            for evidence in report["evidence"]
        ]
        raw_synthesis = (SYNTHESIS_PROMPT | llm | JsonOutputParser()).invoke(
            {
                "query": state["query"],
                "routing_reason": state["routing_reason"],
                "agent_reports": json.dumps(reports, ensure_ascii=True),
                "evidence": json.dumps(available_evidence, ensure_ascii=True),
            }
        )
        result = validate_synthesis(raw_synthesis, available_evidence)
        trace = state.get("trace", []) + [
            {
                "node": "synthesis",
                "status": "completed",
                "agents_used": state["selected_agents"],
                "citation_pages": sorted(
                    {item["source_page"] for item in result["citations"]}
                ),
            }
        ]
        return {**result, "trace": trace}

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("revenue_agent", revenue_agent)
    graph.add_node("expenditure_agent", expenditure_agent)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        next_after_supervisor,
        {
            "revenue_agent": "revenue_agent",
            "expenditure_agent": "expenditure_agent",
            "synthesize": "synthesize",
        },
    )
    graph.add_conditional_edges(
        "revenue_agent",
        next_after_revenue,
        {
            "expenditure_agent": "expenditure_agent",
            "synthesize": "synthesize",
        },
    )
    graph.add_edge("expenditure_agent", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_part3_query(
    graph: Any,
    query: str,
    source_context: Mapping[int, str],
) -> dict[str, Any]:
    """Run a query through the supervisor graph and return its full decision trace."""
    missing_pages = set(PART3_SOURCE_PAGES) - set(source_context)
    if missing_pages:
        raise ValueError(f"Missing Part 3 source pages: {sorted(missing_pages)}")
    if not query.strip():
        raise ValueError("query must not be empty")
    state = graph.invoke(
        {
            "query": query,
            "source_context": dict(source_context),
            "trace": [],
        }
    )
    return {
        "query": state["query"],
        "answer": state["answer"],
        "selected_agents": state["selected_agents"],
        "citations": state["citations"],
        "assumptions": state["assumptions"],
        "trace": state["trace"],
    }