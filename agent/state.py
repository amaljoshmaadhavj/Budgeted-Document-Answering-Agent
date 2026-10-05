"""QuestionState structure for tracking the lifecycle of answering a single question.

A fresh QuestionState is instantiated for every user question.
Page content is never retained across questions.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class QuestionState:
    """Encapsulates the complete state of an agent answering a single document question."""
    question: str
    doc_id: str
    question_type: str = "direct_fact"
    concepts: List[str] = field(default_factory=list)
    search_terms: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    anchor_entities: List[str] = field(default_factory=list)
    candidate_pages: List[int] = field(default_factory=list)
    fetched_pages: List[int] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    answer_status: str = "INSUFFICIENT"  # SUPPORTED, PARTIALLY_SUPPORTED, CONFLICTING, INSUFFICIENT
    budget_used: int = 0
    budget_remaining: int = 6
    stop_reason: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create_fresh(cls, question: str, doc_id: str, max_budget: int = 6) -> "QuestionState":
        """Factory method ensuring a clean, unpolluted state for a new question."""
        return cls(
            question=question,
            doc_id=doc_id,
            budget_used=0,
            budget_remaining=max_budget,
            answer_status="INSUFFICIENT"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
