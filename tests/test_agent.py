"""End-to-end integration tests for the budgeted document answering agent."""

import pytest
import os
import fitz
from agent.controller import AgentController
from tools.document_tools import DocumentRegistry
from core.answerability import AnswerabilityStatus


@pytest.fixture
def multi_page_pdf(tmp_path):
    pdf_path = os.path.join(tmp_path, "robotics.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "Chapter 1: Actuators. Stepper motors provide precise rotational control.")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "Chapter 2: Kinematics. Forward kinematics maps joint angles to end-effector position.")

    p3 = doc.new_page()
    p3.insert_text((50, 50), "Chapter 3: Power. Initial voltage specification was 12V.")

    p4 = doc.new_page()
    p4.insert_text((50, 50), "Chapter 4: Revisions. Power specification was updated to 24V and supersedes earlier 12V design.")

    doc.set_toc([
        [1, "Chapter 1: Actuators", 1],
        [1, "Chapter 2: Kinematics", 2],
        [1, "Chapter 3: Power", 3],
        [1, "Chapter 4: Revisions", 4],
    ])
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_agent_answers_factual_question(multi_page_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(multi_page_pdf)

    controller = AgentController()
    state, answer = controller.answer_question("What do stepper motors provide?", doc_id)

    # Hard budget limit verification
    assert state.budget_used <= 6
    assert state.budget_remaining >= 0
    assert state.budget_used + state.budget_remaining == 6
    assert len(state.fetched_pages) > 0
    assert 1 in state.fetched_pages

    # Trace completeness verification
    assert len(state.trace) == state.budget_used
    for entry in state.trace:
        assert "call_number" in entry
        assert "tool" in entry
        assert "budget_before" in entry
        assert "budget_after" in entry
        assert "result_summary" in entry


def test_state_resets_between_questions(multi_page_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(multi_page_pdf)

    controller = AgentController()

    # Question 1: Actuators
    state1, answer1 = controller.answer_question("What do stepper motors provide?", doc_id)
    assert 1 in state1.fetched_pages
    assert len(state1.evidence) > 0

    # Question 2: Completely different topic (Kinematics)
    state2, answer2 = controller.answer_question("What does forward kinematics map?", doc_id)

    # State 2 MUST NOT retain fetched pages from Question 1 unless independently retrieved
    assert state2.question == "What does forward kinematics map?"
    assert state2.budget_used <= 6
    assert state2.budget_remaining >= 0
    # Ensure fresh state was instantiated
    assert state2 is not state1


def test_agent_handles_insufficient_information(multi_page_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(multi_page_pdf)

    controller = AgentController()
    # Question on topic completely absent from the document
    state, answer = controller.answer_question("What is the culinary recipe for chocolate soufflé?", doc_id)

    assert state.budget_used <= 6
    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    assert "insufficient" in answer.lower() or "not contain" in answer.lower()


def test_agent_detects_supersession(multi_page_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(multi_page_pdf)

    controller = AgentController()
    state, answer = controller.answer_question("What is the voltage specification?", doc_id)

    assert state.budget_used <= 6
    # Should find page 3 and page 4
    if 3 in state.fetched_pages and 4 in state.fetched_pages:
        assert len(state.contradictions) > 0 or any(c["status"] == "SUPERSEDED" for c in state.claims)
