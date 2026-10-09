import unittest
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from htx_fde_ai_assessment.part3_supervisor import (
    build_part3_graph,
    load_part3_source_context,
    run_part3_query,
    validate_agent_report,
    validate_supervisor_decision,
    validate_synthesis,
)


class SupervisorDecisionTests(unittest.TestCase):
    def test_accepts_combined_route(self) -> None:
        agents, reason = validate_supervisor_decision(
            {
                "agents": ["revenue", "expenditure"],
                "reason": "The query asks about revenue and a fund budget.",
            }
        )
        self.assertEqual(agents, ["revenue", "expenditure"])
        self.assertTrue(reason)

    def test_rejects_unknown_or_empty_route(self) -> None:
        with self.assertRaises(ValueError):
            validate_supervisor_decision({"agents": [], "reason": "No route"})
        with self.assertRaises(ValueError):
            validate_supervisor_decision(
                {"agents": ["finance"], "reason": "Unknown agent"}
            )


class EvidenceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_context = {
            16: "OPERATING REVENUE 108.64 Corporate Income Tax 28.03",
            18: "The Government will establish the Future Energy Fund with an initial injection of $5.0 billion.",
            20: "Future Energy Fund 5,000 Total 20,352",
            37: "The main components are Corporate Income Tax, Personal Income Tax, and Goods and Services Tax.",
        }

    def test_agent_evidence_must_quote_an_assigned_page(self) -> None:
        report = validate_agent_report(
            "expenditure",
            {
                "findings": "The fund receives an initial injection.",
                "evidence": [
                    {
                        "source_page": 18,
                        "quote": "initial injection of $5.0 billion",
                    }
                ],
            },
            self.source_context,
        )
        self.assertEqual(report["evidence"][0]["source_page"], 18)

    def test_agent_cannot_cite_another_agents_page_or_fabricate_a_quote(self) -> None:
        with self.assertRaises(ValueError):
            validate_agent_report(
                "expenditure",
                {
                    "findings": "Revenue streams fund the expenditure.",
                    "evidence": [{"source_page": 16, "quote": "108.64"}],
                },
                self.source_context,
            )
        with self.assertRaises(ValueError):
            validate_agent_report(
                "expenditure",
                {
                    "findings": "The fund receives a carbon tax allocation.",
                    "evidence": [
                        {"source_page": 18, "quote": "allocated from carbon tax"}
                    ],
                },
                self.source_context,
            )

    def test_synthesis_citations_must_match_agent_evidence(self) -> None:
        evidence = [
            {"source_page": 20, "quote": "Future Energy Fund 5,000"}
        ]
        validated = validate_synthesis(
            {
                "answer": "The estimated FY2024 top-up is SGD 5 billion.",
                "citations": [evidence[0]],
                "assumptions": ["The source does not state a dedicated revenue stream."],
            },
            evidence,
        )
        self.assertEqual(validated["citations"], evidence)

        invalid = {
            "answer": "Carbon tax funds the top-up.",
            "citations": [{"source_page": 20, "quote": "Carbon tax funding"}],
            "assumptions": [],
        }
        with self.assertRaises(ValueError):
            validate_synthesis(invalid, evidence)


class Part3SourceContextTests(unittest.TestCase):
    def test_loads_the_revenue_and_fund_source_pages(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        context = load_part3_source_context(
            project_root / "data/input/fy2024_analysis_of_revenue_and_expenditure.pdf"
        )
        self.assertEqual(set(context), {16, 18, 20, 37})
        self.assertIn("OPERATING REVENUE", context[16])
        self.assertIn("Future Energy Fund", context[18])
        self.assertIn("5,000", context[20])
        self.assertIn("main components", context[37])


class SupervisorGraphRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_context = {
            16: "OPERATING REVENUE 108.64 Corporate Income Tax 28.03",
            18: "The Government will establish the Future Energy Fund with an initial injection of $5.0 billion.",
            20: "Future Energy Fund 5,000 Total 20,352",
            37: "The main components are Corporate Income Tax, Personal Income Tax, and Goods and Services Tax.",
        }

        def fake_model(messages: list[Any]) -> AIMessage:
            if hasattr(messages, "to_messages"):
                messages = messages.to_messages()
            system_prompt = messages[0].content
            user_prompt = messages[-1].content
            if "supervisor for a government budget research team" in system_prompt:
                query = user_prompt.partition("User query: ")[2].casefold()
                agents = []
                if "revenue" in query:
                    agents.append("revenue")
                if "fund" in query or "spending" in query or "budget" in query:
                    agents.append("expenditure")
                if not agents:
                    agents = ["revenue"]
                result = {"agents": agents, "reason": "Route by subject."}
            elif "You are the Revenue Agent" in system_prompt:
                result = {
                    "findings": "The source lists tax and non-tax operating revenue.",
                    "evidence": [
                        {"source_page": 16, "quote": "OPERATING REVENUE 108.64"}
                    ],
                }
            elif "You are the Expenditure Agent" in system_prompt:
                result = {
                    "findings": "The fund receives an initial government injection.",
                    "evidence": [
                        {
                            "source_page": 18,
                            "quote": "initial injection of $5.0 billion",
                        }
                    ],
                }
            else:
                evidence_text = user_prompt.partition("Available evidence:\n")[2]
                evidence = json.loads(evidence_text)
                result = {
                    "answer": "The requested source-backed findings are summarized.",
                    "citations": evidence,
                    "assumptions": [],
                }
            return AIMessage(content=json.dumps(result))

        self.graph = build_part3_graph(RunnableLambda(fake_model))

    def test_graph_routes_each_demo_query_to_expected_agents(self) -> None:
        cases = (
            ("What are the key government revenue streams?", ["revenue"]),
            ("How is the Future Energy Fund budget supported?", ["expenditure"]),
            (
                "What are the key government revenue streams and how is the fund supported?",
                ["revenue", "expenditure"],
            ),
        )
        for query, expected_agents in cases:
            with self.subTest(query=query):
                result = run_part3_query(self.graph, query, self.source_context)
                self.assertEqual(result["selected_agents"], expected_agents)
                trace_nodes = [item["node"] for item in result["trace"]]
                self.assertEqual(trace_nodes[0], "supervisor")
                self.assertIn("synthesis", trace_nodes)
                self.assertEqual(
                    "revenue_agent" in trace_nodes, "revenue" in expected_agents
                )
                self.assertEqual(
                    "expenditure_agent" in trace_nodes,
                    "expenditure" in expected_agents,
                )


if __name__ == "__main__":
    unittest.main()