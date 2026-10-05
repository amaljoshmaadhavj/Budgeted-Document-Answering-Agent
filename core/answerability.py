"""Answerability Gate and Contradiction/Supersession Resolution.

Enforces deterministic decision rules:
- SUPPORTED: Evidence adequately establishes the requested answer.
- PARTIALLY_SUPPORTED: Only part of the requested answer is established.
- CONFLICTING: Retrieved evidence genuinely conflicts and cannot safely be resolved.
- INSUFFICIENT: The document does not provide enough evidence.
"""

from typing import List, Dict, Any, Tuple, Optional
import re
from core.evidence import EvidenceLedger, ContradictionItem, ClaimItem


class AnswerabilityStatus:
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT = "INSUFFICIENT"


class AnswerabilityGate:
    """Evaluates evidence coverage, contradictions, and supersessions to decide answerability."""

    # Temporal & revision keywords that indicate supersession rather than unresolvable conflict
    SUPERSEDING_TERMS = [
        r"\bsuperseded\b",
        r"\breplaced\b",
        r"\bupdated\b",
        r"\bchanged\b",
        r"\bnow\b",
        r"\bcurrently\b",
        r"\blater\b",
        r"\brevised\b",
        r"\bamended\b",
        r"\bno longer\b",
        r"\bpreviously\b",
        r"\bformerly\b",
    ]

    @classmethod
    def evaluate_supersession(cls, text_a: str, page_a: int, text_b: str, page_b: int) -> Tuple[str, str]:
        """Examines two conflicting excerpts to determine if one explicitly supersedes the other.
        
        DOES NOT assume higher page number automatically supersedes lower page number.
        Requires explicit revision or temporal language.
        
        Returns:
            Tuple of (resolution: 'SUPERSEDED' | 'CONFLICTING' | 'UNRESOLVED', details: str)
        """
        combined = f"{text_a} {text_b}".lower()
        
        # Check for explicit supersession phrases in text_b referring to text_a or vice versa
        has_revision_lang = any(re.search(term, combined) for term in cls.SUPERSEDING_TERMS)
        
        if not has_revision_lang:
            return "CONFLICTING", "Direct factual conflict with no revision or temporal qualification."

        # Check if text_b explicitly updates or replaces text_a
        text_b_lower = text_b.lower()
        text_a_lower = text_a.lower()

        if any(re.search(term, text_b_lower) for term in [r"supersedes", r"replaces", r"updated to", r"now", r"currently", r"effective"]):
            return "SUPERSEDED", f"Page {page_b} contains explicit updating/superseding language qualifying Page {page_a}."
        elif any(re.search(term, text_a_lower) for term in [r"supersedes", r"replaces", r"updated to", r"now", r"currently"]):
            return "SUPERSEDED", f"Page {page_a} contains explicit updating/superseding language qualifying Page {page_b}."

        return "CONFLICTING", "Revision terms detected but causal supersession relation is ambiguous."

    @classmethod
    def evaluate(cls, ledger: EvidenceLedger, anchor_entities: Optional[List[str]] = None) -> Tuple[str, str]:
        """Evaluates the EvidenceLedger to determine the final answerability status.
        
        Enforces that SUPPORTED requires both requirement satisfaction and verified anchor entities.
        
        Returns:
            Tuple of (status, rationale)
        """
        # 1. Check for genuine unresolved contradictions
        has_unresolved_conflict = False
        conflict_details = []

        for contradiction in ledger.contradictions:
            if contradiction.resolution in ("CONFLICTING", "UNRESOLVED"):
                has_unresolved_conflict = True
                conflict_details.append(
                    f"Conflict between p.{contradiction.page_1} ('{contradiction.claim_1}') and p.{contradiction.page_2} ('{contradiction.claim_2}'): {contradiction.details}"
                )

        for claim in ledger.claims:
            if claim.status in ("CONTRADICTED", "UNRESOLVED"):
                has_unresolved_conflict = True
                conflict_details.append(f"Unresolved claim: '{claim.claim}' ({claim.reason or 'Conflicted'})")

        if has_unresolved_conflict:
            return (
                AnswerabilityStatus.CONFLICTING,
                f"Contradictory information found in document: {'; '.join(conflict_details)}"
            )

        # 2. Check for zero evidence or empty ledger
        if not ledger.evidence_items:
            return (
                AnswerabilityStatus.INSUFFICIENT,
                "No relevant document pages or evidence could be retrieved."
            )

        # 3. Check Anchor Entity Verification (if anchor entities defined)
        clean_anchors = [a.strip() for a in (anchor_entities or []) if a and a.strip()]
        matched_anchors = ledger.get_all_matched_anchors()
        covered_anchors = [
            a for a in clean_anchors
            if any(a.lower() == m.lower() for m in matched_anchors)
        ]

        if clean_anchors and len(covered_anchors) == 0:
            passing = ledger.get_all_passing_anchors()
            if any(a.lower() in [p.lower() for p in passing] for a in clean_anchors):
                return (
                    AnswerabilityStatus.INSUFFICIENT,
                    f"Anchor entities {clean_anchors} only appear as passing mentions or bibliographic citations without explanatory evidence."
                )
            return (
                AnswerabilityStatus.INSUFFICIENT,
                f"None of the required anchor entities {clean_anchors} were verified in retrieved evidence."
            )

        all_anchors_covered = (len(covered_anchors) == len(clean_anchors)) if clean_anchors else True

        # 4. Check requirement coverage
        total_reqs = ledger.total_requirements
        satisfied_reqs = ledger.satisfied_requirements_count

        if total_reqs == 0:
            if len(ledger.evidence_items) > 0 and (not clean_anchors or len(covered_anchors) > 0):
                return AnswerabilityStatus.SUPPORTED, "Relevant evidence retrieved."
            return AnswerabilityStatus.INSUFFICIENT, "No evidence found."

        if satisfied_reqs == total_reqs and satisfied_reqs > 0:
            if clean_anchors and not all_anchors_covered:
                missing_anchors = [a for a in clean_anchors if a.lower() not in [c.lower() for c in covered_anchors]]
                return (
                    AnswerabilityStatus.PARTIALLY_SUPPORTED,
                    f"Partially established: missing anchor entity evidence for {missing_anchors}."
                )
            return (
                AnswerabilityStatus.SUPPORTED,
                f"All {total_reqs} evidence requirements and anchor entities fully established by retrieved pages."
            )
        elif satisfied_reqs > 0:
            missing = ledger.get_missing_requirements()
            return (
                AnswerabilityStatus.PARTIALLY_SUPPORTED,
                f"Partially established: {satisfied_reqs}/{total_reqs} requirements met. Missing: {missing}."
            )
        else:
            passing = ledger.get_all_passing_anchors()
            if clean_anchors and any(a.lower() in [p.lower() for p in passing] for a in clean_anchors):
                return (
                    AnswerabilityStatus.INSUFFICIENT,
                    f"Anchor entities {clean_anchors} only appeared in citations/passing mentions; explanatory evidence is insufficient."
                )
            return (
                AnswerabilityStatus.INSUFFICIENT,
                f"None of the {total_reqs} question requirements were established by the retrieved pages."
            )

