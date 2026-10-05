"""Regression tests for planner search quality, explanatory grounding, and supersession/contradiction detection.

All tests utilize arbitrary, synthetic entities and concepts.
Zero domain-specific, course-specific, or test-PDF terms are present.
"""

import pytest
import os
import fitz
from agent.controller import AgentController
from tools.document_tools import DocumentRegistry
from core.answerability import AnswerabilityGate, AnswerabilityStatus


def test_arbitrary_concept_morphological_paraphrase_grounding(tmp_path):
    """Verifies that an arbitrary concept with morphological/paraphrased explanatory evidence
    is correctly grounded as SUPPORTED without question-function words polluting search."""
    pdf_path = os.path.join(tmp_path, "telemetry.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Section 1: Signal Dynamics.\n"
        "Dynamic modulation means adjusting frequency parameters in response to telemetry signals."
    )

    doc.set_toc([[1, "Signal Dynamics", 1]])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What does modulating dynamically mean?", doc_id)

    # 1. Budget constraint
    assert state.budget_used <= 6

    # 2. Planner search quality: 'mean' and 'does' must NOT be standalone search terms
    assert "mean" not in [t.lower() for t in state.search_terms]
    assert "does" not in [t.lower() for t in state.search_terms]

    # 3. Anchor entities should contain the cohesive concept phrase
    assert any("modulating dynamically" in a.lower() for a in state.anchor_entities)

    # 4. Explanatory grounding must recognize 'Dynamic modulation means...' via dynamic variants
    assert state.answer_status == AnswerabilityStatus.SUPPORTED
    assert len(state.evidence) > 0
    assert any(len(e.get("matched_anchors", [])) > 0 for e in state.evidence)


def test_bibliography_only_occurrence_remains_insufficient(tmp_path):
    """Verifies that an arbitrary synthetic entity occurring solely in a bibliography remains INSUFFICIENT."""
    pdf_path = os.path.join(tmp_path, "bib_only.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "References:\n"
        "[12] CryoLock-77 technical manual. In Proc. IEEE, 2021."
    )

    doc.set_toc([[1, "References", 1]])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is CryoLock-77?", doc_id)

    assert state.budget_used <= 6
    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    # Ensure no explanatory matched anchors were credited
    for item in state.evidence:
        assert len(item.get("matched_anchors", [])) == 0


def test_passing_mention_alone_remains_insufficient(tmp_path):
    """Verifies that an arbitrary synthetic entity occurring only in a passing mention remains INSUFFICIENT."""
    pdf_path = os.path.join(tmp_path, "passing.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Section 4: Comparison.\n"
        "Other alternative frameworks include HexaVolt-44, along with legacy systems."
    )

    doc.set_toc([[1, "Comparison", 1]])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is HexaVolt-44?", doc_id)

    assert state.budget_used <= 6
    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    for item in state.evidence:
        assert len(item.get("matched_anchors", [])) == 0


def test_explanatory_page_becomes_supported(tmp_path):
    """Verifies that an arbitrary synthetic entity with genuine explanatory text is SUPPORTED."""
    pdf_path = os.path.join(tmp_path, "explanatory.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Section 1: Power Units.\n"
        "HexaVolt-44 is a dual-rail power converter that regulates output to exactly 48V under fluctuating loads."
    )

    doc.set_toc([[1, "Power Units", 1]])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is HexaVolt-44?", doc_id)

    assert state.budget_used <= 6
    assert state.answer_status == AnswerabilityStatus.SUPPORTED
    assert any("HexaVolt-44" in e.get("matched_anchors", []) for e in state.evidence)


def test_two_entity_comparison_one_lacks_explanatory_yields_partially_supported(tmp_path):
    """Verifies that when comparing two synthetic entities and one lacks explanatory evidence,
    the gate returns PARTIALLY_SUPPORTED."""
    pdf_path = os.path.join(tmp_path, "comparison.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Section 1: Traffic.\n"
        "PrismScale-9 is an elastic load balancer that distributes traffic across heterogeneous worker pools."
    )

    p2 = doc.new_page()
    p2.insert_text(
        (50, 50),
        "Section 2: Citations.\n"
        "References:\n"
        "[4] OmniGrid-3 is cited in technical report 18."
    )

    doc.set_toc([
        [1, "Traffic", 1],
        [1, "Citations", 2]
    ])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is the difference between PrismScale-9 and OmniGrid-3?", doc_id)

    assert state.budget_used <= 6
    # PrismScale-9 is explained, OmniGrid-3 is bibliographic citation only
    assert state.answer_status == AnswerabilityStatus.PARTIALLY_SUPPORTED


def test_unrelated_pages_with_now_currently_do_not_become_conflicting(tmp_path):
    """Verifies that pages containing common temporal words like 'now' and 'currently'
    do NOT trigger false-positive contradiction or supersession."""
    pdf_path = os.path.join(tmp_path, "temporal_context.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Section 1: Architecture.\n"
        "TensorMesh provides a distributed matrix multiplication pipeline."
    )

    p2 = doc.new_page()
    p2.insert_text(
        (50, 50),
        "Section 2: Infrastructure.\n"
        "The cluster infrastructure is now running on European servers and currently processes 100TB daily."
    )

    doc.set_toc([
        [1, "Architecture", 1],
        [1, "Infrastructure", 2]
    ])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What does TensorMesh provide?", doc_id)

    assert state.budget_used <= 6
    # Must NOT have any contradictions detected
    assert len(state.contradictions) == 0
    assert state.answer_status != AnswerabilityStatus.CONFLICTING
    assert state.answer_status == AnswerabilityStatus.SUPPORTED


def test_genuine_synthetic_supersession_logic():
    """Unit test for AnswerabilityGate material conflict and supersession discrimination."""
    # Subtest 1: Genuine supersession with explicit revision language
    text_a = "Initial specification: The latency threshold for PulsarSync is 15ms."
    text_b = "Revision 3: The latency threshold for PulsarSync was updated to 35ms and supersedes earlier 15ms limit."
    res1, det1 = AnswerabilityGate.evaluate_supersession(text_a, 1, text_b, 2)
    assert res1 == "SUPERSEDED"
    assert "Page 2" in det1

    # Subtest 2: Genuine conflicting values without supersession language
    text_c = "The latency threshold for PulsarSync is 15ms."
    text_d = "The latency threshold for PulsarSync is 45ms."
    res2, _ = AnswerabilityGate.evaluate_supersession(text_c, 1, text_d, 2)
    assert res2 == "CONFLICTING"

    # Subtest 3: Contextual 'now' with no conflicting property
    text_e = "AI algorithms operate on state spaces."
    text_f = "AI is now more mature."
    res3, _ = AnswerabilityGate.evaluate_supersession(text_e, 3, text_f, 4)
    assert res3 == "NO_CONFLICT"
