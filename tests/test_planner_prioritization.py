"""Regression tests for planner search-term prioritization, phrase focus, and abbreviation demotion.

Validates that:
- Meaningful multi-word phrases are prioritized first.
- Natural phrase variants and focused component words precede broad short aliases.
- Broad short abbreviations (e.g. 'ZM', 'LO', 'AI') are demoted to lower priority when a full phrase exists.
- Long awkward combinations (e.g. 'term <Concept> adopted') and generic framing words are eliminated.
- Single-entity acronyms (e.g. 'KXV') are NOT demoted when no multi-word concept is present.
- Existing generic behavior for questions like 'What does acting rationally mean?' is fully preserved.
- Uses arbitrary synthetic concepts exclusively ('Zyphora Matrix', 'Lumen orchestration', 'Zephron lattice').
"""

import pytest
import os
import fitz
from unittest.mock import MagicMock
from agent.planner import QuestionPlanner
from agent.controller import AgentController
from tools.document_tools import DocumentRegistry
from llm.provider import LLMProvider


def test_zyphora_matrix_phrase_prioritized_over_abbreviation_and_awkward_phrases(monkeypatch):
    """When a multi-word entity like 'Zyphora Matrix' is present, it must precede 'ZM' and strip awkward combinations."""
    mock_llm = MagicMock(spec=LLMProvider)
    # Simulate an LLM output producing awkward framing phrases and early abbreviations
    mock_llm.complete_json.return_value = {
        "question_type": "temporal",
        "concepts": ["Zyphora Matrix"],
        "search_terms": [
            "Zyphora Matrix",
            "term Zyphora Matrix adopted",
            "ZM",
            "zyphora matrix term",
            "coined"
        ],
        "requirements": ["Adoption year of Zyphora Matrix"],
        "anchor_entities": ["Zyphora Matrix"],
        "anchor_variants": {"Zyphora Matrix": ["ZM"]}
    }

    planner = QuestionPlanner(mock_llm)
    plan = planner.plan("When was the term Zyphora Matrix adopted?")

    search_terms = plan["search_terms"]

    # 1. Exact multi-word phrase must be the top priority
    assert search_terms[0] == "Zyphora Matrix"

    # 2. Awkward concatenated phrases and standalone framing words must be completely excluded
    assert "term Zyphora Matrix adopted" not in search_terms
    assert "zyphora matrix term" not in search_terms
    assert "coined" not in search_terms

    # 3. Broad abbreviation 'ZM' must be demoted to lower priority (after the full phrase and components)
    assert "ZM" in search_terms
    zm_index = search_terms.index("ZM")
    assert zm_index > 0, "Abbreviation 'ZM' must NOT be prioritized ahead of full phrase."

    # 4. Component words like 'Matrix' or 'Zyphora' should appear ahead of 'ZM'
    if "Matrix" in search_terms:
        assert search_terms.index("Matrix") < zm_index
    if "Zyphora" in search_terms:
        assert search_terms.index("Zyphora") < zm_index


def test_lumen_orchestration_prioritization_and_natural_variants(monkeypatch):
    """Multi-word concept 'Lumen orchestration' prioritizes full phrase, variants, components, then 'LO'."""
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.complete_json.return_value = {
        "question_type": "definition",
        "concepts": ["Lumen orchestration"],
        "search_terms": [
            "LO",
            "Lumen orchestration",
            "orchestration",
            "Lumen"
        ],
        "requirements": ["Definition of Lumen orchestration"],
        "anchor_entities": ["Lumen orchestration"],
        "anchor_variants": {"Lumen orchestration": ["orchestrating lumens", "LO"]}
    }

    planner = QuestionPlanner(mock_llm)
    plan = planner.plan("When was Lumen orchestration introduced?")

    search_terms = plan["search_terms"]

    # Full phrase must be #1, despite LLM having 'LO' first in raw_search_terms
    assert search_terms[0] == "Lumen orchestration"

    # Natural phrase variant 'orchestrating lumens' should be prioritized
    assert "orchestrating lumens" in search_terms

    # Short abbreviation 'LO' must be placed after the full phrase or dropped past top-5
    if "LO" in search_terms:
        assert search_terms.index("orchestrating lumens") < search_terms.index("LO")
        assert search_terms.index("LO") > search_terms.index("Lumen orchestration")


def test_zephron_lattice_generic_framing_exclusion(monkeypatch):
    """Question framing words like 'what', 'is', 'the' must be excluded and 'Zephron lattice' prioritized."""
    mock_llm = MagicMock(spec=LLMProvider)
    # LLM returns empty, forcing fallback and normalization
    mock_llm.complete_json.return_value = {}

    planner = QuestionPlanner(mock_llm)
    plan = planner.plan("What is the Zephron lattice?")

    search_terms = plan["search_terms"]
    assert len(search_terms) > 0
    assert search_terms[0] == "Zephron lattice"

    # None of the question function words may appear
    for term in search_terms:
        assert term.lower() not in {"what", "is", "the"}


def test_single_entity_short_acronym_not_demoted():
    """When a single-entity acronym is the only anchor (no multi-word concept), it remains Tier 1."""
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.complete_json.return_value = {
        "question_type": "definition",
        "concepts": ["KXV"],
        "search_terms": ["KXV", "protocol"],
        "requirements": ["KXV definition"],
        "anchor_entities": ["KXV"],
        "anchor_variants": {}
    }

    planner = QuestionPlanner(mock_llm)
    plan = planner.plan("What is KXV?")

    search_terms = plan["search_terms"]
    # KXV is the primary entity itself, so it must be 1st
    assert search_terms[0] == "KXV"


def test_acting_rationally_behavior_preserved():
    """Preserves successful search term extraction and prioritization for 'What does acting rationally mean?'."""
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.complete_json.return_value = {
        "question_type": "definition",
        "concepts": ["acting rationally"],
        "search_terms": ["acting rationally", "rational action"],
        "requirements": ["Definition of acting rationally"],
        "anchor_entities": ["acting rationally"],
        "anchor_variants": {"acting rationally": ["rational action", "act rationally"]}
    }

    planner = QuestionPlanner(mock_llm)
    plan = planner.plan("What does acting rationally mean?")

    search_terms = plan["search_terms"]
    assert search_terms[0] == "acting rationally"
    assert "rational action" in search_terms
    # Question words are excluded
    for t in search_terms:
        assert t.lower() not in {"what", "does", "mean"}


def test_end_to_end_controller_retrieval_efficiency_with_synthetic_document(tmp_path):
    """Verifies AgentController retrieves the specific multi-word phrase before noisy short aliases."""
    # Create a 2-page synthetic PDF:
    # Page 1: Contains frequent noise abbreviation 'ZM'
    # Page 2: Contains the target concept 'Zyphora Matrix was adopted in 2042.'
    pdf_path = os.path.join(tmp_path, "synthetic_zyphora.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "Overview and passing notes. The system operates under ZM protocol. See ZM standards.")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "Historical milestones. The Zyphora Matrix was adopted in 2042 by consensus.")

    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    doc_id = registry.register_pdf(pdf_path, title="synthetic_zyphora.pdf")

    # Mock LLM provider that simulates Gemini returning the multi-word concept and short alias
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.is_mock.return_value = True
    mock_llm.complete_json.return_value = {
        "question_type": "temporal",
        "concepts": ["Zyphora Matrix"],
        "search_terms": [
            "Zyphora Matrix",
            "term Zyphora Matrix adopted",
            "ZM",
            "Matrix"
        ],
        "requirements": ["Zyphora Matrix adoption year"],
        "anchor_entities": ["Zyphora Matrix"],
        "anchor_variants": {"Zyphora Matrix": ["ZM"]}
    }
    mock_llm.complete.return_value = "Based on retrieved evidence, the Zyphora Matrix was adopted in 2042 [Page 2]."

    controller = AgentController(llm_provider=mock_llm)
    state, final_answer = controller.answer_question(
        question="When was the term Zyphora Matrix adopted?",
        doc_id=doc_id
    )

    # Retrieval must search 'Zyphora Matrix' first
    executed_searches = [t["tool"] for t in state.trace]
    assert "search_keyword" in executed_searches

    first_search_trace = next(t for t in state.trace if t["tool"] == "search_keyword")
    assert first_search_trace["arguments"]["keyword"] == "Zyphora Matrix"

    # Page 2 must be fetched and supported
    assert 2 in state.fetched_pages
    assert state.answer_status == "SUPPORTED"
    assert "2042" in final_answer
    assert state.budget_used <= 6


# ==============================================================================
# 6 MANDATORY SYNTHETIC REGRESSION TESTS FOR RETRIEVAL-PLANNING IMPROVEMENTS
# ==============================================================================

def test_regression_1_concept_document_wording_differs_from_question_wording(tmp_path):
    """Scenario 1: Concept whose document wording differs from the question wording.
    
    The question asks about 'Vortex stabilization', but the document explains it as
    'damping vortex velocity'. The dynamically generated conceptual variant enables retrieval.
    """
    pdf_path = os.path.join(tmp_path, "synthetic_vortex.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "General fluid mechanics introduction. Thermal equilibrium overview.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "The damping vortex velocity mechanism operates to stabilize angular flux under high pressure.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    doc_id = registry.register_pdf(pdf_path, title="synthetic_vortex.pdf")

    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.is_mock.return_value = True
    # Planner dynamically generates the conceptual/technical variant
    mock_llm.complete_json.return_value = {
        "question_type": "definition",
        "concepts": ["Vortex stabilization"],
        "search_terms": [
            "Vortex stabilization",
            "damping vortex velocity",
            "angular flux"
        ],
        "requirements": ["Mechanism for vortex stabilization"],
        "anchor_entities": ["Vortex stabilization"],
        "anchor_variants": {"Vortex stabilization": ["damping vortex velocity"]}
    }
    mock_llm.complete.return_value = (
        "Based on retrieved evidence, the damping vortex velocity mechanism operates to stabilize angular flux [Page 2]."
    )

    controller = AgentController(llm_provider=mock_llm)
    state, final_answer = controller.answer_question(
        question="How does vortex stabilization work?",
        doc_id=doc_id
    )

    # Search for 'Vortex stabilization' returned 0 pages, but conceptual variant 'damping vortex velocity' matched Page 2
    assert 2 in state.fetched_pages
    assert state.answer_status == "SUPPORTED"
    assert "damping vortex velocity" in final_answer
    assert state.budget_used <= 6


def test_regression_2_multiple_candidate_pages_ranking_by_planner_evidence(tmp_path):
    """Scenario 2: Multiple candidate pages where one page is clearly better based on planner evidence.
    
    CandidatePageRanker scores Page 3 higher because it matches multiple search terms (co-occurrence)
    and covers the primary anchor entity, so Page 3 is fetched before unpromising candidate pages.
    """
    from agent.controller import CandidatePageRanker

    pdf_path = os.path.join(tmp_path, "synthetic_multicandidate.pdf")
    doc = fitz.open()
    # Page 1: Single broad match
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Mentioning protocol standards.")
    # Page 2: Distractor with broad word 'module'
    p2 = doc.new_page()
    p2.insert_text((50, 50), "A general module configuration.")
    # Page 3: Rich evidence matching both anchor 'Aethelgard Core' and technical variant 'containment field'
    p3 = doc.new_page()
    p3.insert_text((50, 50), "The Aethelgard Core operates a containment field to prevent plasma leakage.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    doc_id = registry.register_pdf(pdf_path, title="synthetic_multicandidate.pdf")

    # Verify ranker scoring directly with planner signals
    ranked = CandidatePageRanker.rank_pages(
        page_matches={
            2: [{"term": "module", "total_matches": 2}],
            3: [
                {"term": "Aethelgard Core", "total_matches": 1},
                {"term": "containment field", "total_matches": 1}
            ]
        },
        anchor_entities=["Aethelgard Core"],
        anchor_variants={"Aethelgard Core": ["containment field"]},
        search_terms=["Aethelgard Core", "containment field", "module"],
        heading_matches={},
        fetched_pages=[]
    )
    assert ranked[0][0] == 3, f"Page 3 must be ranked first due to multi-term co-occurrence and anchor coverage, got {ranked}"

    # Now verify end-to-end controller fetches Page 3 first
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.is_mock.return_value = True
    mock_llm.complete_json.return_value = {
        "question_type": "definition",
        "concepts": ["Aethelgard Core"],
        "search_terms": ["Aethelgard Core", "containment field", "module"],
        "requirements": ["Containment mechanism of Aethelgard Core"],
        "anchor_entities": ["Aethelgard Core"],
        "anchor_variants": {"Aethelgard Core": ["containment field"]}
    }
    mock_llm.complete.return_value = (
        "The Aethelgard Core operates a containment field to prevent plasma leakage [Page 3]."
    )

    controller = AgentController(llm_provider=mock_llm)
    state, final_answer = controller.answer_question(
        question="What is the purpose of the Aethelgard Core?",
        doc_id=doc_id
    )

    assert 3 in state.fetched_pages
    assert state.answer_status == "SUPPORTED"
    assert state.budget_used <= 6


def test_regression_3_redundant_search_term_suppression():
    """Scenario 3: Redundant search-term suppression.
    
    Verifies that case variations, hyphen/space variants, and simple plurals
    are suppressed to conserve document-tool budget.
    """
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.complete_json.return_value = {
        "question_type": "definition",
        "concepts": ["quantum lattice"],
        "search_terms": [
            "quantum lattice",
            "Quantum Lattice",
            "quantum-lattice",
            "quantum lattices",
            "lattice dynamics",
            "QL"
        ],
        "requirements": ["Definition of quantum lattice"],
        "anchor_entities": ["quantum lattice"],
        "anchor_variants": {"quantum lattice": ["quantum-lattice", "quantum lattices"]}
    }

    planner = QuestionPlanner(mock_llm)
    plan = planner.plan("What is a quantum lattice?")

    search_terms = plan["search_terms"]

    # Exactly one representation of 'quantum lattice' should exist
    ql_count = sum(
        1 for t in search_terms
        if t.lower().replace("-", " ") in {"quantum lattice", "quantum lattices"}
    )
    assert ql_count == 1, f"Redundant variants of quantum lattice were not suppressed: {search_terms}"

    # 'lattice dynamics' (conceptual variant) is retained
    assert any("lattice dynamics" in t.lower() for t in search_terms)


def test_regression_4_six_call_budget_remains_strictly_enforced(tmp_path):
    """Scenario 4: Six-call budget remains strictly enforced even with heavy candidate generation."""
    # Create a 10-page document where broad search matches all pages
    pdf_path = os.path.join(tmp_path, "synthetic_multipage.pdf")
    doc = fitz.open()
    for i in range(10):
        p = doc.new_page()
        p.insert_text((50, 50), f"Section {i+1}. The framework mentions component architecture and general details.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    doc_id = registry.register_pdf(pdf_path, title="synthetic_multipage.pdf")

    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.is_mock.return_value = True
    mock_llm.complete_json.return_value = {
        "question_type": "definition",
        "concepts": ["framework component"],
        "search_terms": ["framework component", "framework", "component", "architecture", "details"],
        "requirements": ["Definition of framework component"],
        "anchor_entities": ["framework component"],
        "anchor_variants": {}
    }
    mock_llm.complete.return_value = "The framework component is discussed in general details [Page 1]."

    controller = AgentController(llm_provider=mock_llm)
    state, final_answer = controller.answer_question(
        question="How does framework component function?",
        doc_id=doc_id
    )

    # Budget must never exceed 6 calls
    assert state.budget_used <= 6
    assert len(state.trace) == state.budget_used
    assert state.budget_used == 6


def test_regression_5_existing_acting_rationally_behavior_remains_supported(tmp_path):
    """Scenario 5: Existing 'acting rationally' behavior remains supported."""
    pdf_path = os.path.join(tmp_path, "synthetic_rational.pdf")
    doc = fitz.open()
    for i in range(3):
        p = doc.new_page()
        p.insert_text((50, 50), f"Introductory chapter {i+1}.")
    p4 = doc.new_page()
    p4.insert_text((50, 50), "Acting rationally means acting so as to achieve one's goals, given one's beliefs.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    doc_id = registry.register_pdf(pdf_path, title="synthetic_rational.pdf")

    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.is_mock.return_value = True
    mock_llm.complete_json.return_value = {
        "question_type": "definition",
        "concepts": ["acting rationally"],
        "search_terms": ["acting rationally", "rational action"],
        "requirements": ["Definition of acting rationally"],
        "anchor_entities": ["acting rationally"],
        "anchor_variants": {"acting rationally": ["rational action"]}
    }
    mock_llm.complete.return_value = (
        "Acting rationally means acting so as to achieve one's goals, given one's beliefs [Page 4]."
    )

    controller = AgentController(llm_provider=mock_llm)
    state, final_answer = controller.answer_question(
        question="What does acting rationally mean?",
        doc_id=doc_id
    )

    assert 4 in state.fetched_pages
    assert state.answer_status == "SUPPORTED"
    assert "achieve one's goals" in final_answer
    assert state.budget_used <= 6


def test_regression_6_existing_unavailable_information_remains_insufficient(tmp_path):
    """Scenario 6: Existing unavailable-information behavior remains INSUFFICIENT."""
    pdf_path = os.path.join(tmp_path, "synthetic_unrelated.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "This document covers introductory thermodynamics and heat transfer.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Carnot engine cycles and entropy principles.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    doc_id = registry.register_pdf(pdf_path, title="synthetic_unrelated.pdf")

    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.is_mock.return_value = True
    mock_llm.complete_json.return_value = {
        "question_type": "direct_fact",
        "concepts": ["population of Nexoria in 2099"],
        "search_terms": ["Nexoria population", "Nexoria", "2099"],
        "requirements": ["Population figure of Nexoria in 2099"],
        "anchor_entities": ["Nexoria"],
        "anchor_variants": {}
    }
    mock_llm.complete.return_value = "The document does not contain information regarding Nexoria."

    controller = AgentController(llm_provider=mock_llm)
    state, final_answer = controller.answer_question(
        question="What is the population of Nexoria in 2099?",
        doc_id=doc_id
    )

    assert state.answer_status == "INSUFFICIENT"
    assert state.budget_used <= 6
