"""Generalization, Grounding, and Unseen-Document Validation Test Suite.

Proves that:
1. Anchor entities and substantive terms prevent false positives on distractor pages.
2. Short technical acronyms/identifiers (A*, BFS, DFS, AI, SQL) are preserved and matched.
3. Completely unseen synthetic entities (e.g., ZXQ-91, NovaCore) work identically without hardcoded logic.
4. Multi-entity comparison questions correctly identify partial vs complete evidence.
5. Generic filler words ('search', 'information') cannot trigger SUPPORTED.
6. The 6-call budget is strictly respected on unseen documents.
"""

import pytest
import os
import fitz
from agent.controller import AgentController
from tools.document_tools import DocumentRegistry
from core.answerability import AnswerabilityStatus


@pytest.fixture
def a_star_distractor_pdf(tmp_path):
    """PDF containing an introductory distractor page and a real A* page later."""
    pdf_path = os.path.join(tmp_path, "search_course.pdf")
    doc = fitz.open()

    # Page 1: Distractor containing words 'search', 'information', 'algorithm', but NO mention of A*
    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Chapter 1: Introduction to Artificial Intelligence. "
        "General computing systems process information and can execute search algorithms. "
        "The Turing Test evaluates whether a machine acts like a human."
    )

    # Page 2: The actual A* definition
    p2 = doc.new_page()
    p2.insert_text(
        (50, 50),
        "Chapter 2: Informed Search. "
        "A* is a best-first search algorithm that evaluates nodes using f(n) = g(n) + h(n). "
        "A* is optimal and complete when the heuristic h is admissible."
    )

    doc.set_toc([
        [1, "Introduction", 1],
        [1, "Informed Search", 2],
    ])
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def unseen_synthetic_pdf(tmp_path):
    """Completely unseen synthetic document with arbitrary fictional entities (ZXQ-91, NovaCore)."""
    pdf_path = os.path.join(tmp_path, "synthetic_specs.pdf")
    doc = fitz.open()

    # Page 1: Distractor containing generic scheduling words
    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Section 1: General Infrastructure. "
        "Distributed computing clusters coordinate background tasks across multiple nodes. "
        "System telemetry provides operational information."
    )

    # Page 2: Specific ZXQ-91 definition
    p2 = doc.new_page()
    p2.insert_text(
        (50, 50),
        "Section 2: High-Performance Schedulers. "
        "ZXQ-91 is a deterministic distributed scheduler designed for low-latency dispatch. "
        "It achieves sub-millisecond job allocation."
    )

    # Page 3: Specific NovaCore definition
    p3 = doc.new_page()
    p3.insert_text(
        (50, 50),
        "Section 3: Reactive Fabrics. "
        "NovaCore is an event-driven stream processor optimized for throughput. "
        "It buffers messages in ring memory."
    )

    doc.set_toc([
        [1, "General Infrastructure", 1],
        [1, "High-Performance Schedulers", 2],
        [1, "Reactive Fabrics", 3],
    ])
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def acronyms_pdf(tmp_path):
    """PDF covering short technical identifiers: BFS, DFS, AI, SQL."""
    pdf_path = os.path.join(tmp_path, "acronyms.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "BFS traverses tree levels using a FIFO queue to guarantee shortest path.")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "DFS explores tree depth using a LIFO stack to minimize memory footprint.")

    p3 = doc.new_page()
    p3.insert_text((50, 50), "AI encompasses systems that act rationally and solve computational problems.")

    p4 = doc.new_page()
    p4.insert_text((50, 50), "SQL is a declarative language for managing structured relational database records.")

    doc.save(pdf_path)
    doc.close()
    return pdf_path


# ==============================================================================
# 1. REGRESSION & DISTRACTOR TESTS (A*)
# ==============================================================================

def test_a_star_distractor_alone_yields_insufficient(tmp_path):
    """When only a distractor page mentioning 'search' and 'information' is retrieved, status must be INSUFFICIENT."""
    pdf_path = os.path.join(tmp_path, "distractor_only.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text(
        (50, 50),
        "Introduction to AI. General computing systems process information and execute search algorithms. "
        "The Turing Test evaluates whether a machine acts like a human."
    )
    doc.set_toc([[1, "A* Search Overview", 1]])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is A* search?", doc_id)

    # Must NOT be marked SUPPORTED because 'A*' is missing
    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    assert "A*" in state.anchor_entities
    assert len(state.evidence) > 0
    # The distractor page must not have matched anchors
    assert len(state.evidence[0]["matched_anchors"]) == 0
    assert len(state.evidence[0]["supports"]) == 0


def test_a_star_positive_with_target_page(a_star_distractor_pdf):
    """When the genuine A* definition page is retrieved, status must be SUPPORTED."""
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(a_star_distractor_pdf)

    controller = AgentController()
    state, answer = controller.answer_question("What is A* search?", doc_id)

    assert state.budget_used <= 6
    assert "A*" in state.anchor_entities
    assert state.answer_status == AnswerabilityStatus.SUPPORTED
    # Page 2 must be among fetched pages and have matched anchor
    assert 2 in state.fetched_pages
    supporting_items = [e for e in state.evidence if "A*" in e["matched_anchors"]]
    assert len(supporting_items) > 0


# ==============================================================================
# 2. SHORT ENTITIES TESTS (BFS, DFS, AI, SQL)
# ==============================================================================

def test_bfs_positive(acronyms_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(acronyms_pdf)

    controller = AgentController()
    state, answer = controller.answer_question("What is BFS?", doc_id)

    assert state.budget_used <= 6
    assert any("BFS" in a for a in state.anchor_entities)
    assert 1 in state.fetched_pages
    assert state.answer_status == AnswerabilityStatus.SUPPORTED


def test_dfs_positive(acronyms_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(acronyms_pdf)

    controller = AgentController()
    state, answer = controller.answer_question("What is DFS?", doc_id)

    assert state.budget_used <= 6
    assert any("DFS" in a for a in state.anchor_entities)
    assert 2 in state.fetched_pages
    assert state.answer_status == AnswerabilityStatus.SUPPORTED


def test_ai_positive(acronyms_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(acronyms_pdf)

    controller = AgentController()
    state, answer = controller.answer_question("What is AI?", doc_id)

    assert state.budget_used <= 6
    assert any("AI" in a for a in state.anchor_entities)
    assert 3 in state.fetched_pages
    assert state.answer_status == AnswerabilityStatus.SUPPORTED


def test_sql_positive(acronyms_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(acronyms_pdf)

    controller = AgentController()
    state, answer = controller.answer_question("What is SQL?", doc_id)

    assert state.budget_used <= 6
    assert any("SQL" in a for a in state.anchor_entities)
    assert 4 in state.fetched_pages
    assert state.answer_status == AnswerabilityStatus.SUPPORTED


# ==============================================================================
# 3. UNSEEN SYNTHETIC ENTITY TESTS (ZXQ-91, NovaCore)
# ==============================================================================

def test_unseen_entity_zxq91_positive(unseen_synthetic_pdf):
    """Completely synthetic entity ZXQ-91 must be identified and grounded generically."""
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(unseen_synthetic_pdf)

    controller = AgentController()
    state, answer = controller.answer_question("What is ZXQ-91?", doc_id)

    assert state.budget_used <= 6
    # Planner must have extracted ZXQ-91 dynamically
    assert any("ZXQ-91" in a for a in state.anchor_entities)
    assert 2 in state.fetched_pages
    assert state.answer_status == AnswerabilityStatus.SUPPORTED


def test_unseen_entity_distractor_negative(tmp_path):
    """Unseen entity queried against document with generic words but without entity yields INSUFFICIENT."""
    pdf_path = os.path.join(tmp_path, "distractor_infra.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text(
        (50, 50),
        "Distributed computing clusters coordinate background tasks across multiple nodes. "
        "System telemetry provides operational information."
    )
    doc.set_toc([[1, "ZXQ-91 Systems", 1]])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is ZXQ-91?", doc_id)

    assert state.budget_used <= 6
    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    assert len(state.evidence) > 0
    # No anchor matched
    assert len(state.evidence[0]["matched_anchors"]) == 0


def test_unseen_multi_entity_comparison(unseen_synthetic_pdf):
    """Comparison question between two synthetic entities (ZXQ-91 vs NovaCore)."""
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(unseen_synthetic_pdf)

    controller = AgentController()
    state, answer = controller.answer_question("What is the difference between ZXQ-91 and NovaCore?", doc_id)

    assert state.budget_used <= 6
    # Both anchors must be extracted dynamically
    anchors_lower = [a.lower() for a in state.anchor_entities]
    assert any("zxq-91" in a for a in anchors_lower)
    assert any("novacore" in a for a in anchors_lower)

    # If both pages were retrieved, status is SUPPORTED; if only one, PARTIALLY_SUPPORTED
    if 2 in state.fetched_pages and 3 in state.fetched_pages:
        assert state.answer_status == AnswerabilityStatus.SUPPORTED
    elif 2 in state.fetched_pages or 3 in state.fetched_pages:
        assert state.answer_status == AnswerabilityStatus.PARTIALLY_SUPPORTED


# ==============================================================================
# 4. GENERIC WORD DISTRACTOR PREVENTION TESTS
# ==============================================================================

def test_generic_filler_words_cannot_trigger_supported(tmp_path):
    """A document containing only words like 'information', 'search', 'details' must NOT produce SUPPORTED."""
    pdf_path = os.path.join(tmp_path, "fillers.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Search information details regarding general facts and questions.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is QuantumFlux-27?", doc_id)

    assert state.budget_used <= 6
    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT


# ==============================================================================
# 5. NATURAL LANGUAGE MULTI-WORD ENTITY TEST
# ==============================================================================

def test_natural_language_phrase_entity(tmp_path):
    """Natural language question with multi-word concept 'artificial intelligence'."""
    pdf_path = os.path.join(tmp_path, "history.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text(
        (50, 50),
        "The Dartmouth conference in 1956 was the historical event where the term "
        "artificial intelligence was officially adopted."
    )
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("When was the term artificial intelligence adopted?", doc_id)

    assert state.budget_used <= 6
    assert any("artificial intelligence" in a.lower() for a in state.anchor_entities)
    assert state.answer_status == AnswerabilityStatus.SUPPORTED


# ==============================================================================
# 6. PASSING-MENTION & BIBLIOGRAPHY REGRESSION TESTS (ARBITRARY SYNTHETIC ENTITIES)
# ==============================================================================

def test_bibliography_citation_alone_yields_insufficient(tmp_path):
    """When an entity appears strictly in a bibliography or references citation,
    it must NOT be marked SUPPORTED, and must yield INSUFFICIENT."""
    pdf_path = os.path.join(tmp_path, "bib_only.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text(
        (50, 50),
        "References:\n"
        "[12] KryoVex-9 protocol is cited in the literature as an external reference.\n"
        "[13] J. Smith, Tech Report 2021."
    )
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is KryoVex-9?", doc_id)

    assert state.budget_used <= 6
    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    assert len(state.evidence) > 0
    # Must be recorded as a passing/apparatus occurrence, NOT explanatory matched anchor
    assert "KryoVex-9" not in state.evidence[0]["matched_anchors"]
    assert "KryoVex-9" in state.evidence[0]["passing_anchors"]
    assert len(state.evidence[0]["supports"]) == 0


def test_passing_mention_alone_yields_insufficient(tmp_path):
    """When an entity appears strictly in a passing mention (listing without explanation),
    it must NOT be marked SUPPORTED, and must yield INSUFFICIENT."""
    pdf_path = os.path.join(tmp_path, "passing_only.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text(
        (50, 50),
        "Chapter 1: Related Work. Other alternative distributed tools include HelixMesh-7 and VectorNet."
    )
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("Explain HelixMesh-7.", doc_id)

    assert state.budget_used <= 6
    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    assert len(state.evidence) > 0
    # Mere listing must NOT be credited as explanatory
    assert "HelixMesh-7" not in state.evidence[0]["matched_anchors"]
    assert "HelixMesh-7" in state.evidence[0]["passing_anchors"]
    assert len(state.evidence[0]["supports"]) == 0


def test_bibliography_and_explanatory_page_yields_supported(tmp_path):
    """When a document contains both a bibliography citation and an actual explanatory page,
    the agent must not stop on the citation and must verify explanatory evidence on the target page."""
    pdf_path = os.path.join(tmp_path, "multi_evidence.pdf")
    doc = fitz.open()

    # Page 1: Bibliography citation
    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "References:\n"
        "[4] ChronosGate-3 specification is cited in tech report 2020."
    )

    # Page 2: Genuine explanatory evidence
    p2 = doc.new_page()
    p2.insert_text(
        (50, 50),
        "Section 3: Time Synchronization.\n"
        "ChronosGate-3 is an asynchronous consensus mechanism that coordinates distributed network clocks across validator nodes."
    )

    doc.set_toc([
        [1, "References", 1],
        [1, "Time Synchronization", 2],
    ])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is ChronosGate-3?", doc_id)

    assert state.budget_used <= 6
    assert 2 in state.fetched_pages
    assert state.answer_status == AnswerabilityStatus.SUPPORTED

    # Explanatory evidence was collected on Page 2
    explanatory_items = [e for e in state.evidence if "ChronosGate-3" in e["matched_anchors"]]
    assert len(explanatory_items) > 0
    assert any(e["page"] == 2 for e in explanatory_items)


def test_comparison_one_explanatory_one_bibliography_yields_partially_supported(tmp_path):
    """When comparing two synthetic entities where one is explained but the other is only
    in a bibliography citation, the status must be PARTIALLY_SUPPORTED."""
    pdf_path = os.path.join(tmp_path, "comparison_bib.pdf")
    doc = fitz.open()

    # Page 1: VeloSync-88 is explained
    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Section 1: Data Engines.\n"
        "VeloSync-88 is an in-memory transactional database engine that optimizes real-time write streams."
    )

    # Page 2: AeroQuant-12 is only cited in references
    p2 = doc.new_page()
    p2.insert_text(
        (50, 50),
        "References:\n"
        "[8] AeroQuant-12 was published in 2022. Literature cited in chapter 4."
    )

    doc.set_toc([
        [1, "Data Engines", 1],
        [1, "References", 2],
    ])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is the difference between VeloSync-88 and AeroQuant-12?", doc_id)

    assert state.budget_used <= 6
    # VeloSync-88 is explained, AeroQuant-12 is only in bibliography
    # Thus status must be PARTIALLY_SUPPORTED, NOT SUPPORTED
    assert state.answer_status == AnswerabilityStatus.PARTIALLY_SUPPORTED

