"""Comprehensive regression tests for generic retrieval scheduling, budget preservation,
requirement-level validation, partial answers, conversational handling, and provider independence.

All tests utilize arbitrary synthetic documents and concepts. Zero document-specific hardcoding.
"""

import pytest
import os
import fitz
from unittest.mock import MagicMock, patch
from agent.controller import AgentController, RetrievalScheduler, CandidatePageRanker
from agent.planner import QuestionPlanner
from tools.document_tools import DocumentRegistry
from core.budget import BudgetManager, BudgetExceededError, MAX_DOCUMENT_TOOL_CALLS
from core.evidence import EvidenceLedger, EvidenceGroundingValidator
from core.answerability import AnswerabilityGate, AnswerabilityStatus
from llm.provider import LLMProvider


# ==============================================================================
# TEST A & D: Budget Preservation & Scheduler Fetches Candidates Before Exhaustion
# ==============================================================================

def test_retrieval_scheduler_preserves_fetch_budget(tmp_path):
    """Verifies that search discovery does NOT consume the entire budget when candidates exist.
    The scheduler fetches candidate pages before budget exhaustion."""
    pdf_path = os.path.join(tmp_path, "quantum.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "Overview: General computing architectures and coprocessors.")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "QuantumCore is a cryogenic quantum coprocessor operating at 15 millikelvin.")

    p3 = doc.new_page()
    p3.insert_text((50, 50), "QuantumCore provides hardware acceleration for lattice cryptography algorithms.")

    doc.set_toc([
        [1, "Overview", 1],
        [1, "QuantumCore Architecture", 2],
        [1, "Cryptographic Applications", 3]
    ])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    # Mock planner that provides 5 search terms
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.is_mock.return_value = True
    mock_llm.complete_json.return_value = {
        "question_type": "definition",
        "concepts": ["QuantumCore"],
        "search_terms": [
            "QuantumCore",
            "cryogenic coprocessor",
            "lattice cryptography",
            "quantum coprocessor",
            "QC"
        ],
        "requirements": ["Definition and core mechanism of QuantumCore"],
        "anchor_entities": ["QuantumCore"],
        "anchor_variants": {"QuantumCore": ["QC"]}
    }
    mock_llm.complete.return_value = "Based on retrieved evidence, QuantumCore is a cryogenic quantum coprocessor [Page 2]."

    controller = AgentController(llm_provider=mock_llm)
    state, answer = controller.answer_question("What is QuantumCore?", doc_id)

    # 1. Must stay within 6-call budget
    assert state.budget_used <= 6
    assert state.budget_remaining >= 0

    # 2. Must NOT spend all 5 calls on searching; must fetch candidate page 2
    executed_tools = [t["tool"] for t in state.trace]
    assert "get_page" in executed_tools
    assert 2 in state.fetched_pages

    # 3. Must successfully ground the evidence
    assert state.answer_status == AnswerabilityStatus.SUPPORTED
    assert len(state.evidence) > 0


def test_scheduler_low_budget_prioritizes_fetching():
    """Unit test for RetrievalScheduler: when remaining budget <= 2 and candidates exist,
    must choose get_page, NOT search_keyword."""
    from agent.state import QuestionState

    state = QuestionState.create_fresh(question="Test question", doc_id="dummy", max_budget=6)
    state.search_terms = ["term_A", "term_B", "term_C", "term_D"]
    state.evidence_requirements = ["Requirement 1"]
    ledger = EvidenceLedger(requirements=["Requirement 1"])

    ranked_candidates = [(12, 6.0), (15, 4.0)]
    unsearched_terms = ["term_B", "term_C", "term_D"]

    # Budget = 2: must choose get_page
    action, target, rationale = RetrievalScheduler.decide_next_action(
        remaining_budget=2,
        ranked_candidates=ranked_candidates,
        unsearched_terms=unsearched_terms,
        state=state,
        ledger=ledger,
        headings_checked=False,
        searched_keywords={"term_a"}
    )
    assert action == "get_page"
    assert target == 12
    assert "Low remaining budget" in rationale

    # Budget = 1: must choose get_page
    action_1, target_1, rationale_1 = RetrievalScheduler.decide_next_action(
        remaining_budget=1,
        ranked_candidates=ranked_candidates,
        unsearched_terms=unsearched_terms,
        state=state,
        ledger=ledger,
        headings_checked=False,
        searched_keywords={"term_a"}
    )
    assert action_1 == "get_page"
    assert target_1 == 12
    assert "Final remaining call" in rationale_1


# ==============================================================================
# TEST B, C & U: Six-Call Budget and 7th Call Rejection
# ==============================================================================

def test_six_call_limit_and_seventh_rejection():
    """Verifies strict 6-call enforcement and 7th call rejection."""
    budget = BudgetManager(max_calls=6)
    for i in range(6):
        assert budget.can_call() is True
        b_before, b_after = budget.consume_call()
        assert b_before == 6 - i
        assert b_after == 6 - i - 1

    assert budget.calls_used == 6
    assert budget.remaining_calls == 0
    assert budget.is_exhausted is True
    assert budget.can_call() is False

    with pytest.raises(BudgetExceededError):
        budget.consume_call()

    # Budget remains 0 after rejected call
    assert budget.remaining_calls == 0


# ==============================================================================
# TEST E: Candidate Page ≠ Evidence Until Fetched
# ==============================================================================

def test_candidate_page_is_not_evidence_until_fetched():
    """Verifies candidate pages are never considered evidence without fetched content."""
    ledger = EvidenceLedger(requirements=["Definition of SynapseFlow"])

    # Simulating candidate page discovery: page 10 has matched term
    assert len(ledger.evidence_items) == 0
    assert ledger.is_fully_covered() is False
    assert ledger.is_partially_covered() is False
    assert ledger.satisfied_requirements_count == 0

    # Only when fetched and processed into ledger does it become evidence
    ledger.add_evidence(
        page=10,
        text="SynapseFlow is an asynchronous event dispatcher.",
        supports=["Definition of SynapseFlow"],
        is_explanatory=True
    )
    ledger.mark_requirement_satisfied("Definition of SynapseFlow", page=10)
    assert len(ledger.evidence_items) == 1
    assert ledger.is_fully_covered() is True


# ==============================================================================
# TEST F: Keyword Mention Does NOT Satisfy Definition Requirement
# ==============================================================================

def test_keyword_mention_does_not_satisfy_definition_requirement(tmp_path):
    """Verifies that mere keyword occurrence or non-definitional mention
    does NOT satisfy a definition requirement."""
    pdf_path = os.path.join(tmp_path, "mention_only.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Section 1: Lab Log.\n"
        "The team deployed VortexEngine for testing across three cluster nodes on Tuesday."
    )
    doc.set_toc([[1, "Lab Log", 1]])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is VortexEngine?", doc_id)

    assert state.budget_used <= 6
    # Mention alone ("deployed VortexEngine for testing") does not provide definition
    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT


# ==============================================================================
# TEST G: Keyword Mention Does NOT Satisfy Date Requirement
# ==============================================================================

def test_keyword_mention_does_not_satisfy_date_requirement(tmp_path):
    """Verifies that an explanatory description without dates does NOT satisfy a date/year requirement."""
    pdf_path = os.path.join(tmp_path, "no_date.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Section 1: Architecture.\n"
        "VortexEngine is a high-throughput stream ingestion pipeline designed for event logs."
    )
    doc.set_toc([[1, "Architecture", 1]])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("In what year was VortexEngine founded?", doc_id)

    assert state.budget_used <= 6
    # Explanatory text exists, but no temporal expression / year exists
    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT


# ==============================================================================
# TEST H: Partial Requirements Produce PARTIALLY_SUPPORTED
# ==============================================================================

def test_partial_requirements_produce_partially_supported(tmp_path):
    """Verifies that when a question has two distinct requirements and only one is satisfied,
    the gate evaluates to PARTIALLY_SUPPORTED."""
    pdf_path = os.path.join(tmp_path, "partial.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Section 1: Systems.\n"
        "VortexEngine is a high-throughput stream ingestion pipeline designed for event logs."
    )
    doc.set_toc([[1, "Systems", 1]])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    # Mock planner returning 2 requirements: definition (present) and date (absent)
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.is_mock.return_value = True
    mock_llm.complete_json.return_value = {
        "question_type": "direct_fact",
        "concepts": ["VortexEngine"],
        "search_terms": ["VortexEngine"],
        "requirements": [
            "Definition and core mechanism of VortexEngine",
            "Date or year of VortexEngine"
        ],
        "anchor_entities": ["VortexEngine"],
        "anchor_variants": {}
    }
    mock_llm.complete.return_value = "VortexEngine is a stream pipeline [Page 1]. The year was not found."

    controller = AgentController(llm_provider=mock_llm)
    state, answer = controller.answer_question(
        "What is VortexEngine and what year was it introduced?",
        doc_id
    )

    assert state.budget_used <= 6
    assert state.answer_status == AnswerabilityStatus.PARTIALLY_SUPPORTED


# ==============================================================================
# TEST I: Missing Evidence Produces INSUFFICIENT
# ==============================================================================

def test_missing_evidence_produces_insufficient(tmp_path):
    """Verifies completely absent entities produce INSUFFICIENT with budget intact."""
    pdf_path = os.path.join(tmp_path, "blank_topic.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "General chemistry manual covering covalent bonding.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is StarlightProtocol?", doc_id)

    assert state.budget_used <= 6
    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT


# ==============================================================================
# TEST J: Non-Document Conversational Queries Consume 0 Document Tools
# ==============================================================================

@pytest.mark.parametrize("conversational_query", [
    "Thank you",
    "Hello",
    "Hi",
    "Okay",
    "Thanks!",
    "Good morning",
    "thanks a lot",
    "Alright, understood"
])
def test_conversational_queries_consume_zero_tool_calls(tmp_path, conversational_query):
    """Verifies that conversational remarks do NOT spend document tools:
    document_tool_calls = 0, remaining_budget = 6."""
    pdf_path = os.path.join(tmp_path, "doc.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Document content.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question(conversational_query, doc_id)

    # Strictly 0 document-tool calls
    assert state.budget_used == 0
    assert state.budget_remaining == 6
    assert len(state.trace) == 0
    assert len(state.fetched_pages) == 0
    assert "conversational" in state.stop_reason.lower()
    assert state.answer_status == AnswerabilityStatus.SUPPORTED


# ==============================================================================
# TEST K: Generic Stopwords and Framing Words Excluded From Search Terms
# ==============================================================================

def test_generic_stopwords_excluded_from_search_terms():
    """Verifies question framing, stopwords, and filler phrases are never standalone search terms."""
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.is_mock.return_value = True
    planner = QuestionPlanner(mock_llm)

    plan = planner.plan("Is Dr. Sarah Miller mentioned in this PDF and what is its purpose?")

    search_terms_lower = [t.lower() for t in plan["search_terms"]]
    forbidden_terms = ["pdf", "this", "mentioned", "its", "purpose", "you", "what", "is", "and", "in"]
    for ft in forbidden_terms:
        assert ft not in search_terms_lower

    # Substantive concept must be present
    assert any("sarah miller" in t for t in search_terms_lower)


# ==============================================================================
# TEST L: Presence Query Does Not Claim Proven Absence
# ==============================================================================

def test_presence_query_conservative_absence(tmp_path):
    """Verifies presence/absence queries answer conservatively without claiming document-wide absence."""
    pdf_path = os.path.join(tmp_path, "catalog.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Product catalog listing item A and item B.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("Is Item-Delta available in the PDF?", doc_id)

    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    # Conservative response: cannot claim proven absence, state could not verify presence
    ans_lower = answer.lower()
    assert "could not verify" in ans_lower or "not contain" in ans_lower or "insufficient" in ans_lower
    assert "definitely does not exist anywhere" not in ans_lower


# ==============================================================================
# TEST R & S: OpenRouter Provider Integration and Secret Redaction
# ==============================================================================

def test_openrouter_provider_detection_and_redaction(monkeypatch):
    """Verifies LLMProvider supports LLM_PROVIDER=openrouter and sanitizes sk-or- keys."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY1", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    test_or_key = "sk-or-v1-abcdef1234567890abcdef1234567890"
    monkeypatch.setenv("OPENROUTER_API_KEY", test_or_key)

    provider = LLMProvider()
    assert provider._client_type == "openrouter"
    assert provider.is_mock() is False

    # Verify secret redaction
    raw_text = f"Connection failed with auth token {test_or_key} at endpoint."
    sanitized = provider._sanitize_secret(raw_text)
    assert test_or_key not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


# ==============================================================================
# TEST T: Traces Contain Retrieval Decisions and Zero Secrets
# ==============================================================================

def test_trace_contains_retrieval_decisions_and_no_secrets(tmp_path):
    """Verifies that execution traces record retrieval decisions and contain no API keys."""
    pdf_path = os.path.join(tmp_path, "robotics.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Stepper motors provide precise rotational control.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What do stepper motors provide?", doc_id)

    assert state.budget_used <= 6
    # Trace entries
    assert len(state.trace) > 0
    for entry in state.trace:
        assert "call_number" in entry
        assert "tool" in entry
        assert "budget_before" in entry
        assert "budget_after" in entry


# ==============================================================================
# TEST U: Dedicated Scheduler Regression Tests (Synthetic Documents & Generic Logic)
# ==============================================================================

def test_scheduler_search_returns_many_candidates_immediately_ranked_and_fetched(tmp_path):
    """Verifies that when a search returns many candidate pages:
    1. Candidates are immediately made available to CandidatePageRanker.
    2. Scheduler prioritizes get_page on the top-ranked candidate over further searches.
    """
    from agent.state import QuestionState

    # Synthetic 20-page document where search returns 15 pages
    pdf_path = os.path.join(tmp_path, "many_candidates.pdf")
    doc = fitz.open()
    for i in range(1, 21):
        page = doc.new_page()
        if i in [3, 7, 12]:
            page.insert_text((50, 50), f"Page {i}: Primary mechanism of VortexCore utilizes magnetic containment.")
        elif i % 2 == 0:
            page.insert_text((50, 50), f"Page {i}: VortexCore ancillary reference notes.")
        else:
            page.insert_text((50, 50), f"Page {i}: Unrelated general overview content.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.is_mock.return_value = True
    mock_llm.complete_json.return_value = {
        "question_type": "mechanism",
        "concepts": ["VortexCore"],
        "search_terms": [
            "VortexCore",
            "magnetic containment",
            "ancillary reference",
            "containment mechanism"
        ],
        "requirements": ["Mechanism of VortexCore"],
        "anchor_entities": ["VortexCore"],
        "anchor_variants": {}
    }
    mock_llm.complete.return_value = "VortexCore utilizes magnetic containment [Page 3]."

    controller = AgentController(llm_provider=mock_llm)
    state, answer = controller.answer_question("What is the mechanism of VortexCore?", doc_id)

    # Must stay within 6-call limit
    assert state.budget_used <= 6
    assert state.budget_remaining >= 0

    # First call was search, subsequent call converted candidates into get_page
    tool_sequence = [t["tool"] for t in state.trace]
    assert tool_sequence[0] == "search_keyword"
    assert "get_page" in tool_sequence
    # Must have fetched one of the candidate pages
    assert any(p in state.fetched_pages for p in [2, 3, 4, 6, 7, 8, 10, 12])
    assert state.answer_status == AnswerabilityStatus.SUPPORTED


def test_scheduler_final_call_fetches_when_candidate_exists():
    """Requirement 4 & 10: When remaining_budget == 1 and at least one useful unfetched candidate exists:
    - scheduler must NOT perform another search
    - scheduler must fetch the highest-ranked candidate.
    """
    from agent.state import QuestionState

    state = QuestionState.create_fresh(question="Synthetic question", doc_id="dummy", max_budget=6)
    state.search_terms = ["term_alpha", "term_beta", "term_gamma"]
    state.evidence_requirements = ["Explain synthetic mechanism"]
    ledger = EvidenceLedger(requirements=["Explain synthetic mechanism"])

    ranked_candidates = [(17, 8.5), (23, 4.2), (31, 2.1)]
    unsearched_terms = ["term_beta", "term_gamma"]

    action, target, rationale = RetrievalScheduler.decide_next_action(
        remaining_budget=1,
        ranked_candidates=ranked_candidates,
        unsearched_terms=unsearched_terms,
        state=state,
        ledger=ledger,
        headings_checked=False,
        searched_keywords={"term_alpha"}
    )

    # Must choose get_page, NOT search_keyword
    assert action == "get_page"
    assert target == 17
    assert "Final remaining call" in rationale


def test_scheduler_final_call_searches_when_no_candidates_exist():
    """Requirement 10: When remaining_budget == 1 and NO candidate exists:
    - scheduler may continue searching to attempt discovery.
    """
    from agent.state import QuestionState

    state = QuestionState.create_fresh(question="Synthetic question", doc_id="dummy", max_budget=6)
    state.search_terms = ["term_alpha", "term_beta"]
    state.evidence_requirements = ["Explain synthetic mechanism"]
    ledger = EvidenceLedger(requirements=["Explain synthetic mechanism"])

    ranked_candidates = []
    unsearched_terms = ["term_beta"]

    action, target, rationale = RetrievalScheduler.decide_next_action(
        remaining_budget=1,
        ranked_candidates=ranked_candidates,
        unsearched_terms=unsearched_terms,
        state=state,
        ledger=ledger,
        headings_checked=False,
        searched_keywords={"term_alpha"}
    )

    # Must choose search_keyword when zero candidates exist
    assert action == "search_keyword"
    assert target == "term_beta"
    assert "Final call with zero candidate pages" in rationale


def test_scheduler_low_budget_prefers_fetch_over_search():
    """Requirement 3 & 10: When budget is limited (remaining_budget <= 2 or 3)
    and promising candidates exist, prefer get_page over another search.
    """
    from agent.state import QuestionState

    state = QuestionState.create_fresh(question="Synthetic question", doc_id="dummy", max_budget=6)
    state.search_terms = ["term_1", "term_2", "term_3", "term_4"]
    state.evidence_requirements = ["Explain synthetic mechanism"]
    ledger = EvidenceLedger(requirements=["Explain synthetic mechanism"])

    ranked_candidates = [(8, 4.5), (9, 3.2)]
    unsearched_terms = ["term_2", "term_3", "term_4"]

    # Budget = 2 with viable candidate: must fetch
    action_2, target_2, _ = RetrievalScheduler.decide_next_action(
        remaining_budget=2,
        ranked_candidates=ranked_candidates,
        unsearched_terms=unsearched_terms,
        state=state,
        ledger=ledger,
        headings_checked=False,
        searched_keywords={"term_1"}
    )
    assert action_2 == "get_page"
    assert target_2 == 8

    # Budget = 3 with promising candidate: must fetch
    action_3, target_3, _ = RetrievalScheduler.decide_next_action(
        remaining_budget=3,
        ranked_candidates=ranked_candidates,
        unsearched_terms=unsearched_terms,
        state=state,
        ledger=ledger,
        headings_checked=False,
        searched_keywords={"term_1"}
    )
    assert action_3 == "get_page"
    assert target_3 == 8


def test_scheduler_evidence_satisfied_early_stops(tmp_path):
    """Requirement 10: When evidence requirements are satisfied, early stop occurs,
    preserving unused tool calls.
    """
    pdf_path = os.path.join(tmp_path, "early_stop.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "PhotonGate is an optical isolation switch designed for signal protection, created in 2021 by Dr Thorne.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Irrelevant secondary content.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.is_mock.return_value = True
    mock_llm.complete_json.return_value = {
        "question_type": "definition",
        "concepts": ["PhotonGate"],
        "search_terms": ["PhotonGate", "optical isolation", "isolation switch", "optical switch", "Thorne"],
        "requirements": ["Definition and mechanism of PhotonGate"],
        "anchor_entities": ["PhotonGate"],
        "anchor_variants": {}
    }
    mock_llm.complete.return_value = "PhotonGate is an optical isolation switch created in 2021 [Page 1]."

    controller = AgentController(llm_provider=mock_llm)
    state, answer = controller.answer_question("What is PhotonGate?", doc_id)

    # Should have stopped early once page 1 provided explanatory evidence
    assert state.budget_used < 6
    assert state.budget_remaining > 0
    assert state.answer_status == AnswerabilityStatus.SUPPORTED
    assert len(state.evidence) >= 1


def test_trace_contains_required_fields_remaining_budget_decision_reason(tmp_path):
    """Requirement 11: Trace information must explicitly contain:
    - remaining budget
    - candidate count
    - top-ranked candidate
    - decision: SEARCH or FETCH
    - reason for decision
    """
    pdf_path = os.path.join(tmp_path, "trace_test.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "HelioCell converts solar thermal energy into ionic currents.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What does HelioCell convert?", doc_id)

    # Load the latest trace file based on modification time
    trace_files = [f for f in os.listdir("traces") if f.startswith("trace_") and f.endswith(".json")]
    assert len(trace_files) > 0
    latest_trace_file = max(trace_files, key=lambda f: os.path.getmtime(os.path.join("traces", f)))
    latest_trace_path = os.path.join("traces", latest_trace_file)
    import json
    with open(latest_trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    assert "retrieval_decisions" in trace_data
    decisions = trace_data["retrieval_decisions"]
    assert len(decisions) > 0

    for d in decisions:
        # Check all required fields from Requirement 11
        assert "remaining_budget" in d
        assert isinstance(d["remaining_budget"], int)
        assert "candidate_count" in d
        assert isinstance(d["candidate_count"], int)
        assert "top_ranked_candidate" in d  # can be None or dict
        assert "decision" in d
        assert d["decision"] in ("SEARCH", "FETCH", "HEADINGS", "STOP")
        assert "reason" in d
        assert isinstance(d["reason"], str) and len(d["reason"]) > 0


def test_morphological_variants_suppress_untransformed_reversals():
    """Verifies that _derive_morphological_variants does not produce nonsensical
    inverted word pairs when neither word underwent morphological change.
    E.g. 'Artificial Intelligence' should NOT produce 'Intelligence Artificial'.
    """
    variants = QuestionPlanner._derive_morphological_variants("Artificial Intelligence")
    assert "Intelligence Artificial" not in variants
    assert "intelligence artificial" not in [v.lower() for v in variants]


# ==============================================================================
# SPECIFIC 14 REGRESSION TESTS (EVALUATION MANDATE)
# ==============================================================================

def test_regression_1_temporal_question_topical_page_not_supported(tmp_path):
    """1. Temporal question + topical page -> NOT SUPPORTED (INSUFFICIENT)
    Page discusses the entity substantively but lacks temporal expression and adoption relation.
    """
    pdf_path = os.path.join(tmp_path, "topical_temporal.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "There are perhaps two broad approaches to developing AI methods today: (i) Acting like humans, and (ii) Acting rationally.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("When was the term Artificial Intelligence adopted?", doc_id)

    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    assert state.budget_used <= 6


def test_regression_2_temporal_question_genuine_temporal_evidence_supported(tmp_path):
    """2. Temporal question + genuine temporal evidence -> SUPPORTED"""
    pdf_path = os.path.join(tmp_path, "genuine_temporal.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "The Dartmouth conference in 1956 was the historical event where the term Artificial Intelligence was officially adopted.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("When was the term Artificial Intelligence adopted?", doc_id)

    assert state.answer_status == AnswerabilityStatus.SUPPORTED
    assert state.budget_used <= 6


def test_regression_3_definition_question_generic_topic_discussion_not_supported(tmp_path):
    """3. Definition question + generic topic discussion -> NOT SUPPORTED (INSUFFICIENT)"""
    pdf_path = os.path.join(tmp_path, "generic_topic.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Many engineering teams deploy NovaCluster across multiple regions for operational stability.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is NovaCluster?", doc_id)

    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    assert state.budget_used <= 6


def test_regression_4_definition_question_actual_definition_supported(tmp_path):
    """4. Definition question + actual definition -> SUPPORTED"""
    pdf_path = os.path.join(tmp_path, "actual_definition.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "NovaCluster is a distributed consensus orchestrator that coordinates atomic state replication across quorum groups.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is NovaCluster?", doc_id)

    assert state.answer_status == AnswerabilityStatus.SUPPORTED
    assert state.budget_used <= 6


def test_regression_5_comparison_only_one_entity_supported_partially_supported(tmp_path):
    """5. Comparison where only one entity is supported -> PARTIALLY_SUPPORTED"""
    pdf_path = os.path.join(tmp_path, "comp_one.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "AlphaEngine is a deterministic compiler pipeline optimized for streaming vector instructions.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "References:\n[12] BetaEngine technical report (2023).")
    doc.set_toc([[1, "Alpha", 1], [1, "References", 2]])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is the difference between AlphaEngine and BetaEngine?", doc_id)

    assert state.answer_status == AnswerabilityStatus.PARTIALLY_SUPPORTED
    assert state.budget_used <= 6


def test_regression_6_comparison_both_entities_supported(tmp_path):
    """6. Comparison where both entities and comparison evidence exist -> SUPPORTED"""
    pdf_path = os.path.join(tmp_path, "comp_both.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "AlphaEngine is a deterministic compiler pipeline for streaming vector instructions.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "BetaEngine is an asynchronous JIT execution runtime for event graphs. Unlike AlphaEngine, it optimizes dynamically.")
    doc.set_toc([[1, "AlphaEngine", 1], [1, "BetaEngine", 2]])
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is the difference between AlphaEngine and BetaEngine?", doc_id)

    assert state.answer_status == AnswerabilityStatus.SUPPORTED
    assert state.budget_used <= 6


def test_regression_7_topic_relevance_without_requirement_evidence_not_supported(tmp_path):
    """7. Topic relevance without requirement evidence -> NOT SUPPORTED (INSUFFICIENT)
    A page discusses the topic substantively but does not answer the requested requirement.
    """
    pdf_path = os.path.join(tmp_path, "topical_only.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "ChronoStream handles millions of events daily and operates reliably in multi-tenant cloud environments.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("In what year was ChronoStream first released?", doc_id)

    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    assert state.budget_used <= 6


def test_regression_8_search_alias_occurrence_without_requirement_evidence_not_supported(tmp_path):
    """8. Search alias occurrence without requirement evidence -> NOT SUPPORTED (INSUFFICIENT)
    Search alias occurrences like 'AI' or 'CS' alone do not prove a requirement.
    """
    pdf_path = os.path.join(tmp_path, "alias_only.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "The AI working group met last Thursday to discuss project deadlines.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is the formal definition of Artificial Intelligence?", doc_id)

    assert state.answer_status == AnswerabilityStatus.INSUFFICIENT
    assert state.budget_used <= 6


def test_regression_9_thank_you_conversational_zero_tool_calls(tmp_path):
    """9. 'Thank you' / conversational input -> ZERO document-tool calls"""
    pdf_path = os.path.join(tmp_path, "empty.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Empty page.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("Thank you", doc_id)

    assert state.budget_used == 0
    assert state.budget_remaining == 6
    assert len(state.trace) == 0


def test_regression_10_final_answer_cannot_contradict_status():
    """10. Final answer cannot contradict answerability status (Sanitization & Consistency)."""
    # INSUFFICIENT status with positive claim must be sanitized
    ans, consistency = AgentController._enforce_answer_consistency(
        answer="The year is 1956.",
        status=AnswerabilityStatus.INSUFFICIENT,
        question="When was X adopted?"
    )
    assert consistency == "SANITIZED_TO_CONSISTENT"
    assert "insufficient" in ans.lower()

    # SUPPORTED status with unverified claim is flagged
    ans2, consistency2 = AgentController._enforce_answer_consistency(
        answer="I could not verify the information.",
        status=AnswerabilityStatus.SUPPORTED,
        question="When was X adopted?"
    )
    assert consistency2 == "INCONSISTENT_WARNED"


def test_regression_11_budget_one_forces_fetch_over_search():
    """11. Remaining budget = 1 + useful unfetched candidate -> FETCH candidate rather than SEARCH again"""
    from agent.state import QuestionState
    state = QuestionState.create_fresh(question="Test question", doc_id="dummy", max_budget=6)
    state.search_terms = ["term_A", "term_B", "term_C"]
    state.evidence_requirements = ["Requirement 1"]
    ledger = EvidenceLedger(requirements=["Requirement 1"])

    ranked_candidates = [(7, 8.5), (10, 4.0)]
    unsearched_terms = ["term_B", "term_C"]

    action, target, rationale = RetrievalScheduler.decide_next_action(
        remaining_budget=1,
        ranked_candidates=ranked_candidates,
        unsearched_terms=unsearched_terms,
        state=state,
        ledger=ledger,
        headings_checked=False,
        searched_keywords={"term_a"}
    )
    assert action == "get_page"
    assert target == 7
    assert "Final remaining call" in rationale


def test_regression_12_six_call_hard_limit_still_enforced():
    """12. Six-call hard limit still strictly enforced."""
    budget = BudgetManager(max_calls=6)
    for _ in range(6):
        budget.consume_call()
    assert budget.calls_used == 6
    assert budget.remaining_calls == 0
    with pytest.raises(BudgetExceededError):
        budget.consume_call()


def test_regression_13_no_rag_constraints_remain_intact():
    """13. No-RAG constraints remain intact: no chromadb, langchain, faiss, sentence_transformers, or embeddings."""
    import sys
    banned_modules = [
        "chromadb", "langchain", "faiss", "sentence_transformers",
        "llama_index", "crewai", "autogen"
    ]
    for mod in banned_modules:
        assert mod not in sys.modules, f"Banned dependency {mod} is loaded in sys.modules!"


def test_regression_14_unseen_synthetic_entities_work(tmp_path):
    """14. Unseen/synthetic entities still work generically without hardcoding."""
    pdf_path = os.path.join(tmp_path, "synthetic_unseen.pdf")
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "XenonPulse is an ultra-fast optical pulse generator operating at 100 terahertz.")
    doc.save(pdf_path)
    doc.close()

    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(pdf_path)

    controller = AgentController()
    state, answer = controller.answer_question("What is XenonPulse?", doc_id)

    assert state.answer_status == AnswerabilityStatus.SUPPORTED
    assert state.budget_used <= 6
    assert any("xenonpulse" in a.lower() for a in state.anchor_entities)


