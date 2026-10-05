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

    GENERIC_DISCOURSE_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "to", "in",
        "for", "with", "on", "at", "by", "from", "as", "it", "this", "that", "these",
        "those", "now", "currently", "later", "then", "also", "here", "there", "more",
        "most", "all", "some", "any", "each", "both", "few", "other", "such", "than",
        "too", "very", "can", "could", "will", "would", "shall", "should", "may", "might",
        "must", "have", "has", "had", "do", "does", "did", "chapter", "section", "page",
        "document", "text", "figure", "table", "details", "information", "discussion"
    }

    # Explicit revision keywords indicating supersession rather than unresolvable conflict
    SUPERSEDING_TERMS = [
        r"\bsuperseded\b",
        r"\bsupersedes\b",
        r"\breplaced\b",
        r"\breplaces\b",
        r"\bupdated\s+to\b",
        r"\brevised\s+to\b",
        r"\bamended\s+to\b",
        r"\bno\s+longer\b",
        r"\bpreviously\b",
        r"\bformerly\b",
    ]

    @classmethod
    def check_material_conflict(cls, text_a: str, text_b: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """Checks whether text_a and text_b contain mutually inconsistent claims or an explicit replacement relationship.
        
        Requires:
        1. Shared subject/concept
        2. Evidence about the same claim/property
        3. Materially different statements or values
        
        Returns:
            Tuple of (is_conflict: bool, property_name: Optional[str], val_a: Optional[str], val_b: Optional[str])
        """
        # 1. Explicit replacement / supersession syntax referencing a subject or earlier value
        explicit_replacement_patterns = [
            r"\b(?:supersedes|superseded)\b",
            r"\breplaces\s+(?:earlier|previous|all\s+previous|[A-Za-z0-9_-]+)\b",
            r"\bupdated\s+(?:from\s+[^.!?]+\s+)?to\s+[^.!?]+\s+and\s+(?:supersedes|replaces)\b",
            r"\b(?:previously|formerly|initially)\s+([^.!?]+?),\s*(?:but\s+)?(?:is\s+)?(?:now|currently|updated)\s+([^.!?]+)",
            r"\bno\s+longer\s+([^.!?]+?),\s*(?:but\s+)?(?:now|instead)\s+([^.!?]+)",
        ]
        for p in explicit_replacement_patterns:
            if re.search(p, text_b, re.IGNORECASE) or re.search(p, text_a, re.IGNORECASE):
                words_a = set(re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', text_a.lower()))
                words_b = set(re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', text_b.lower()))
                substantive_shared = [w for w in words_a.intersection(words_b) if w not in cls.GENERIC_DISCOURSE_WORDS]
                if substantive_shared:
                    return True, " ".join(substantive_shared[:2]), None, None

        # 2. Quantitative / numeric property conflict
        # Pattern: <property words> (is|was|are|were|set to|updated to|of) <number> <unit/noun>
        val_pattern = r'(\b[a-zA-Z\s-]{3,30}?\b)\s*(?:is|was|are|were|has\s+been|set\s+to|updated\s+to|of)\s*([\$£€¥]?\d+(?:\.\d+)?\s*(?:[a-zA-Z/%]+|\b))'
        matches_a = re.findall(val_pattern, text_a, re.IGNORECASE)
        matches_b = re.findall(val_pattern, text_b, re.IGNORECASE)

        for prop_a, val_a in matches_a:
            prop_words_a = set([w.lower().strip() for w in prop_a.split() if w.lower().strip() not in cls.GENERIC_DISCOURSE_WORDS and len(w) >= 3])
            if not prop_words_a:
                continue
            for prop_b, val_b in matches_b:
                prop_words_b = set([w.lower().strip() for w in prop_b.split() if w.lower().strip() not in cls.GENERIC_DISCOURSE_WORDS and len(w) >= 3])
                if prop_words_a.intersection(prop_words_b):
                    val_a_clean = val_a.strip().lower()
                    val_b_clean = val_b.strip().lower()
                    if val_a_clean != val_b_clean:
                        shared_prop = " ".join(sorted(list(prop_words_a.intersection(prop_words_b))))
                        return True, shared_prop, val_a_clean, val_b_clean

        # 3. Polar antonymous states about the same subject
        polar_pairs = [
            (r"\boptimal\b", r"\b(?:suboptimal|not\s+optimal)\b"),
            (r"\bcomplete\b", r"\b(?:incomplete|not\s+complete)\b"),
            (r"\bsupported\b", r"\b(?:unsupported|not\s+supported)\b"),
            (r"\b(?:permitted|allowed)\b", r"\b(?:prohibited|forbidden|disallowed|not\s+allowed)\b"),
            (r"\benabled\b", r"\bdisabled\b"),
            (r"\bvalid\b", r"\binvalid\b"),
            (r"\bdeterministic\b", r"\b(?:non-deterministic|nondeterministic|stochastic)\b"),
        ]
        words_a = set(re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', text_a.lower()))
        words_b = set(re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', text_b.lower()))
        substantive_shared = [w for w in words_a.intersection(words_b) if w not in cls.GENERIC_DISCOURSE_WORDS]
        if substantive_shared:
            for pos_pat, neg_pat in polar_pairs:
                a_pos = bool(re.search(pos_pat, text_a, re.IGNORECASE))
                a_neg = bool(re.search(neg_pat, text_a, re.IGNORECASE))
                b_pos = bool(re.search(pos_pat, text_b, re.IGNORECASE))
                b_neg = bool(re.search(neg_pat, text_b, re.IGNORECASE))
                if (a_pos and b_neg) or (a_neg and b_pos):
                    return True, " ".join(substantive_shared[:2]), "positive", "negative"

        return False, None, None, None

    @classmethod
    def evaluate_supersession(cls, text_a: str, page_a: int, text_b: str, page_b: int) -> Tuple[str, str]:
        """Examines two excerpts to determine if they establish a genuine factual contradiction
        or an explicit supersession.
        
        Requires:
        - shared subject/concept
        - evidence about the same claim/property
        - materially different statements/values
        - actual revision/supersession language or replacement relationship
        
        Ordinary contextual phrases such as 'AI is now more mature' or 'We now discuss'
        do NOT establish a contradiction.
        
        Returns:
            Tuple of (resolution: 'SUPERSEDED' | 'CONFLICTING' | 'NO_CONFLICT', details: str)
        """
        is_conflict, claim_prop, val_a, val_b = cls.check_material_conflict(text_a, text_b)
        
        if not is_conflict:
            return "NO_CONFLICT", "No contradictory claims or conflicting properties detected."

        text_b_lower = text_b.lower()
        text_a_lower = text_a.lower()

        superseding_patterns = [
            r"\bsupersedes\b",
            r"\bsuperseded\b",
            r"\breplaces\b",
            r"\breplaced\b",
            r"\bupdated\s+to\b",
            r"\brevised\s+to\b",
            r"\bamended\s+to\b",
            r"\bno\s+longer\b",
        ]

        b_supersedes = any(re.search(p, text_b_lower) for p in superseding_patterns)
        a_supersedes = any(re.search(p, text_a_lower) for p in superseding_patterns)

        if b_supersedes and not a_supersedes:
            return "SUPERSEDED", f"Page {page_b} contains explicit updating/superseding language qualifying Page {page_a}."
        elif a_supersedes and not b_supersedes:
            return "SUPERSEDED", f"Page {page_a} contains explicit updating/superseding language qualifying Page {page_b}."
        else:
            return "CONFLICTING", f"Direct factual conflict regarding {claim_prop or 'specification'} between Page {page_a} and Page {page_b}."

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

