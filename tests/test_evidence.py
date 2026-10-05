"""Unit tests for Evidence Ledger, Requirement Coverage, and Answerability Decisions."""

import pytest
from core.evidence import EvidenceLedger
from core.answerability import AnswerabilityGate, AnswerabilityStatus


def test_evidence_ledger_item_addition():
    ledger = EvidenceLedger(requirements=["heuristic definition", "cost handling"])
    
    item = ledger.add_evidence(
        page=22,
        section="3.2 A* search",
        text="A* uses f(n) = g(n) + h(n).",
        supports=["heuristic definition"]
    )
    assert item.page == 22
    assert item.section == "3.2 A* search"
    assert "heuristic definition" in item.supports
    assert len(ledger.evidence_items) == 1


def test_requirement_coverage_tracking():
    requirements = ["best-first behavior", "A* behavior", "optimality condition"]
    ledger = EvidenceLedger(requirements=requirements)

    assert ledger.is_fully_covered() is False
    assert ledger.is_partially_covered() is False
    assert len(ledger.get_missing_requirements()) == 3

    # Satisfy one requirement
    ledger.mark_requirement_satisfied("best-first behavior", page=10)
    assert ledger.is_partially_covered() is True
    assert ledger.is_fully_covered() is False
    assert ledger.satisfied_requirements_count == 1

    # Satisfy remaining
    ledger.mark_requirement_satisfied("A* behavior", page=11)
    ledger.mark_requirement_satisfied("optimality condition", page=12)
    assert ledger.is_fully_covered() is True
    assert ledger.satisfied_requirements_count == 3
    assert len(ledger.get_missing_requirements()) == 0


def test_claim_statuses_and_contradiction_recording():
    ledger = EvidenceLedger()
    c1 = ledger.add_claim("The maximum timeout is 30 seconds", sources=[5], status="SUPPORTED")
    assert c1.status == "SUPPORTED"

    # Add contradiction
    ledger.add_contradiction(
        claim_1="The maximum timeout is 30 seconds",
        page_1=5,
        claim_2="The maximum timeout is 60 seconds",
        page_2=12,
        resolution="CONFLICTING",
        details="Discrepancy in timeout specification without revision note."
    )
    assert ledger.has_conflicts() is True


def test_supersession_logic():
    # Test 1: Page 12 explicitly supersedes Page 5 with 'updated' and 'supersedes'
    text_p5 = "Initial policy: The refund window is 14 days."
    text_p12 = "Revision 2.0: The refund window is updated to 30 days and supersedes all previous policy versions."
    
    res, details = AnswerabilityGate.evaluate_supersession(text_p5, 5, text_p12, 12)
    assert res == "SUPERSEDED"
    assert "Page 12" in details

    # Test 2: Two conflicting numbers without any temporal/revision keywords
    text_a = "Maximum team capacity is 10 engineers."
    text_b = "Maximum team capacity is 25 engineers."
    res2, _ = AnswerabilityGate.evaluate_supersession(text_a, 1, text_b, 8)
    # Higher page number MUST NOT automatically supersede
    assert res2 == "CONFLICTING"


def test_answerability_gate_decisions():
    # 1. INSUFFICIENT when empty
    empty_ledger = EvidenceLedger(["req1", "req2"])
    status, _ = AnswerabilityGate.evaluate(empty_ledger)
    assert status == AnswerabilityStatus.INSUFFICIENT

    # 2. PARTIALLY_SUPPORTED when only 1 of 2 requirements met
    partial_ledger = EvidenceLedger(["req1", "req2"])
    partial_ledger.add_evidence(page=1, text="Text for req1", supports=["req1"])
    partial_ledger.mark_requirement_satisfied("req1", page=1)
    status, _ = AnswerabilityGate.evaluate(partial_ledger)
    assert status == AnswerabilityStatus.PARTIALLY_SUPPORTED

    # 3. SUPPORTED when all requirements met
    full_ledger = EvidenceLedger(["req1", "req2"])
    full_ledger.add_evidence(page=1, text="Text for req1", supports=["req1"])
    full_ledger.add_evidence(page=2, text="Text for req2", supports=["req2"])
    full_ledger.mark_requirement_satisfied("req1", page=1)
    full_ledger.mark_requirement_satisfied("req2", page=2)
    status, _ = AnswerabilityGate.evaluate(full_ledger)
    assert status == AnswerabilityStatus.SUPPORTED

    # 4. CONFLICTING when unresolved contradiction present
    conflict_ledger = EvidenceLedger(["req1"])
    conflict_ledger.add_evidence(page=1, text="Claim A")
    conflict_ledger.mark_requirement_satisfied("req1", page=1)
    conflict_ledger.add_contradiction("Claim A", 1, "Claim B", 2, "CONFLICTING", "Unresolved disagreement")
    status, _ = AnswerabilityGate.evaluate(conflict_ledger)
    assert status == AnswerabilityStatus.CONFLICTING
