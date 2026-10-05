"""Evidence Ledger module for tracking extracted document evidence, claims, contradictions, and coverage."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict


class EvidenceGroundingValidator:
    """Validates whether document text contains substantive explanatory evidence
    or mere anchor occurrence / passing mention / bibliographic apparatus citation.
    
    Operates strictly via generic linguistic and document structure patterns.
    Contains zero domain keywords, PDF-specific logic, or page numbers.
    """

    APPARATUS_HEADINGS = [
        r"^\s*(?:[0-9IVXLCDM]+\.?\s+)?(?:references|bibliography|works\s+cited|literature\s+cited|citations|author\s+index|subject\s+index)\b",
    ]

    CITATION_ENTRY_PATTERNS = [
        r"^\s*\[\d+\]",
        r"^\s*\(\d+\)\s+[A-Z]",
        r"^\s*\d+\.\s+[A-Z][a-z]+(?:,\s*[A-Z]\.|\s+[A-Z]\.)",
        r"\b(?:pp\.\s*\d+[-–]\d+|doi:\s*10\.\d+|isbn[:\s]|in\s+proc\.|proceedings\s+of|journal\s+of|tech(?:nical)?\s+report)\b",
    ]

    PASSING_MENTION_PATTERNS = [
        # Passive citation: "is cited in", "was mentioned in", "has been referenced in", "is listed in"
        r"\b(?:is|are|was|were|has\s+been|have\s+been)\s+(?:cited|mentioned|referenced|listed|cataloged|indexed|included\s+as\s+a\s+baseline)\b",
        # Citation pointer: "see [1]", "refer to Section 3", "cited in [12]", "as discussed in [4]"
        r"\b(?:see|refer\s+to|cited\s+in|mentioned\s+in|discussed\s+in|as\s+in)\s+(?:section|chapter|appendix|table|figure|\[|\()",
        # "et al." citation style
        r"\bet\s+al\.\b",
        # Bare listing / enumeration
        r"\b(?:other\s+(?:systems|tools|algorithms|methods|approaches|protocols|architectures|models|frameworks)\s+include)\b",
        r"\b(?:along\s+with|compared\s+(?:with|to)|together\s+with|in\s+addition\s+to)\s+[^.!?]*\b",
    ]

    EXPLANATORY_PREDICATE_PATTERNS = [
        # Copular / Definitional / Value assignment
        # "is an algorithm", "was $5 million", "is 12V", "is optimal", "is complete", "was updated to"
        r"\b(?:is|are|was|were)\s+(?:an?|the|[a-zA-Z-]+|[\$£€¥]?\d+)\b",
        r"\b(?:is|are|was|were)\s+(?:defined|known|characterized|classified|described|considered)\s+(?:as|to\s+be)\b",
        r"\b(?:refers\s+to|denotes|represents|stands\s+for|means|signifies|entails|involves|constitutes|embodies|serves\s+as|acts\s+as)\b",
        r"\b(?:consists\s+of|composed\s+of|characterized\s+by)\b",
        # Functional / Operational / Behavioral predicates
        r"\b(?:operates?|operat(?:ed|ing)|computes?|comput(?:ed|ing)|provides?|provid(?:ed|ing)|enables?|enabl(?:ed|ing)|uses?|us(?:ed|ing)|implements?|implement(?:ed|ing)|performs?|perform(?:ed|ing)|evaluates?|evaluat(?:ed|ing)|determines?|determin(?:ed|ing)|handles?|handl(?:ed|ing)|achieves?|achiev(?:ed|ing)|works?|work(?:ed|ing)|allows?|allow(?:ed|ing)|executes?|execut(?:ed|ing)|transforms?|transform(?:ed|ing)|produces?|produc(?:ed|ing)|requires?|requir(?:ed|ing)|optimizes?|optimiz(?:ed|ing)|coordinates?|coordinat(?:ed|ing)|processes|process(?:ed|ing)|manages?|manag(?:ed|ing)|generates?|generat(?:ed|ing)|stores?|stor(?:ed|ing)|transfers?|transferr(?:ed|ing)|traverses?|travers(?:ed|ing)|dispatches?|dispatch(?:ed|ing)|buffers?|buffer(?:ed|ing)|measures?|measur(?:ed|ing)|structures?|structur(?:ed|ing)|facilitates?|facilitat(?:ed|ing)|calculates?|calculat(?:ed|ing)|acts?|act(?:ed|ing)|solves?|solv(?:ed|ing)|encompasses?|maps?|mapp(?:ed|ing)|supersedes?|supersed(?:ed|ing))\b",
        # Teleological / Design / Intent
        r"\b(?:designed\s+(?:to|for)|intended\s+to|built\s+to|developed\s+to|created\s+to|used\s+(?:to|for)|serves\s+to|aims\s+to)\b",
        # Historical / Adoption
        r"\b(?:was\s+(?:the\s+)?(?:historical\s+)?event|adopted|introduced|proposed\s+by|established\s+in)\b",
    ]

    GENERIC_STOPWORDS = {
        "a", "an", "the", "and", "or", "of", "to", "in", "for", "with", "on", "at",
        "by", "from", "as", "is", "are", "was", "were", "it", "this", "that", "these",
        "those", "can", "could", "will", "would", "shall", "should", "may", "might",
        "must", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "not", "but", "if", "then", "else", "when", "where", "why", "how", "all",
        "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "than", "too", "very", "information", "details", "also", "into", "over",
        "after", "before", "between", "under", "above", "re", "see"
    }

    @classmethod
    def anchor_matches_text(cls, text: str, anchor: str) -> bool:
        """Robust generic word/symbol boundary matching for entity identifiers."""
        a_clean = anchor.strip()
        if not a_clean:
            return False
        a_lower = a_clean.lower()
        text_lower = text.lower()
        if a_clean.isalnum():
            pattern = r'\b' + re.escape(a_lower) + r'\b'
            return bool(re.search(pattern, text_lower))
        else:
            pattern = r'(?:^|[^\w])' + re.escape(a_lower) + r'(?:$|[^\w])'
            return bool(re.search(pattern, text_lower) or a_lower in text_lower)

    @classmethod
    def split_into_context_segments(cls, text: str) -> List[Tuple[str, bool]]:
        """Splits page text into sentences/statements while tracking whether each segment
        belongs to an apparatus/bibliographic section or standalone citation entry.
        
        Returns list of (segment_text, in_apparatus).
        """
        lines = text.splitlines()
        segments: List[Tuple[str, bool]] = []
        in_apparatus_section = False

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if this line marks the start of an apparatus section (References, Bibliography, Index, etc.)
            if any(re.search(p, line_str, re.IGNORECASE) for p in cls.APPARATUS_HEADINGS):
                in_apparatus_section = True

            is_citation_line = in_apparatus_section or any(
                re.search(p, line_str, re.IGNORECASE) for p in cls.CITATION_ENTRY_PATTERNS
            )

            # Split line into sentences if it contains sentence boundaries
            sub_sentences = re.split(r'(?<=[.!?])\s+', line_str)
            for s in sub_sentences:
                s_clean = s.strip()
                if s_clean:
                    segments.append((s_clean, is_citation_line))

        return segments

    @classmethod
    def evaluate_anchor_occurrence(cls, text: str, anchor: str) -> Dict[str, Any]:
        """Evaluates whether an anchor occurrence on a page is explanatory evidence
        or mere occurrence (bibliography citation / passing mention)."""
        if not cls.anchor_matches_text(text, anchor):
            return {
                "has_occurrence": False,
                "is_explanatory": False,
                "is_passing": False,
                "explanatory_contexts": [],
                "passing_contexts": []
            }

        segments = cls.split_into_context_segments(text)
        explanatory_contexts = []
        passing_contexts = []

        anchor_lower = anchor.lower().strip()
        anchor_tokens = set([w.strip("?,.:;\"'()[]{}").lower() for w in anchor.split()])

        for idx, (seg, in_apparatus) in enumerate(segments):
            if not cls.anchor_matches_text(seg, anchor):
                continue

            # Case A: If segment is in apparatus or citation entry, it is bibliographic passing mention
            if in_apparatus:
                passing_contexts.append(seg)
                continue

            seg_lower = seg.lower()

            # Case B: Check for explicit passing-mention disqualifiers
            has_passing_pattern = any(
                re.search(p, seg_lower) for p in cls.PASSING_MENTION_PATTERNS
            )

            # Check for multi-sentence window if this sentence is an introduction (e.g., "Consider X. It is a ...")
            eval_seg = seg
            eval_seg_lower = seg_lower
            if idx + 1 < len(segments):
                next_seg, next_in_app = segments[idx + 1]
                if not next_in_app and re.match(
                    r'^(?:it|this|these|the\s+(?:system|algorithm|protocol|model|framework|mechanism|scheduler|engine|architecture|tool))\b',
                    next_seg.strip(),
                    re.IGNORECASE
                ):
                    eval_seg = f"{seg} {next_seg}"
                    eval_seg_lower = eval_seg.lower()

            # Check explanatory predicates
            has_explanatory_predicate = any(
                re.search(p, eval_seg_lower) for p in cls.EXPLANATORY_PREDICATE_PATTERNS
            )

            # Extract substantive content tokens beyond the anchor itself
            words = [w.strip("?,.:;\"'()[]{}").lower() for w in eval_seg.split()]
            substantive_tokens = [
                w for w in words
                if w not in cls.GENERIC_STOPWORDS
                and w not in anchor_tokens
                and len(w) >= 1
            ]

            # Explanatory evidence requires:
            # 1. No disqualifying passing pattern
            # 2. Presence of an explanatory predicate
            # 3. Substantive descriptive or value content beyond the entity name
            if not has_passing_pattern and has_explanatory_predicate and len(substantive_tokens) >= 1:
                explanatory_contexts.append(eval_seg)
            else:
                passing_contexts.append(eval_seg)

        is_explanatory = len(explanatory_contexts) > 0
        is_passing = (not is_explanatory) and (len(passing_contexts) > 0)

        return {
            "has_occurrence": True,
            "is_explanatory": is_explanatory,
            "is_passing": is_passing,
            "explanatory_contexts": explanatory_contexts,
            "passing_contexts": passing_contexts
        }

    @classmethod
    def evaluate_anchor_with_variants(
        cls,
        text: str,
        anchor: str,
        variants: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluates whether an anchor or any of its dynamic variants/aliases establishes
        explanatory evidence or mere occurrence (passing mention / citation).
        
        Prioritizes primary anchor evaluation, then checks each dynamic variant.
        Grounding is established if any variant achieves genuine explanatory grounding.
        """
        a_clean = anchor.strip()
        candidate_forms = [a_clean]
        if variants:
            for v in variants:
                v_clean = str(v).strip()
                if v_clean and v_clean.lower() not in [x.lower() for x in candidate_forms]:
                    candidate_forms.append(v_clean)

        explanatory_matches: List[str] = []
        passing_matches: List[str] = []
        all_explanatory_contexts: List[str] = []
        all_passing_contexts: List[str] = []

        for cand in candidate_forms:
            res = cls.evaluate_anchor_occurrence(text, cand)
            if res["is_explanatory"]:
                explanatory_matches.append(cand)
                all_explanatory_contexts.extend(res["explanatory_contexts"])
            elif res["has_occurrence"]:
                passing_matches.append(cand)
                all_passing_contexts.extend(res["passing_contexts"])

        if explanatory_matches:
            return {
                "has_occurrence": True,
                "is_explanatory": True,
                "is_passing": False,
                "matched_form": explanatory_matches[0],
                "explanatory_contexts": all_explanatory_contexts,
                "passing_contexts": all_passing_contexts
            }
        elif passing_matches:
            return {
                "has_occurrence": True,
                "is_explanatory": False,
                "is_passing": True,
                "matched_form": passing_matches[0],
                "explanatory_contexts": [],
                "passing_contexts": all_passing_contexts
            }
        else:
            return {
                "has_occurrence": False,
                "is_explanatory": False,
                "is_passing": False,
                "matched_form": None,
                "explanatory_contexts": [],
                "passing_contexts": []
            }

    @classmethod
    def has_explanatory_requirement_evidence(cls, text: str, substantive_words: List[str]) -> bool:
        """Determines if substantive requirement keywords are satisfied in an explanatory context
        rather than exclusively within apparatus or citation lines."""
        if not substantive_words:
            return True

        segments = cls.split_into_context_segments(text)
        for seg, in_apparatus in segments:
            if in_apparatus:
                continue
            seg_lower = seg.lower()
            if any(w.lower() in seg_lower for w in substantive_words):
                # Ensure the line is not a passing citation pointer
                if not any(re.search(p, seg_lower) for p in cls.PASSING_MENTION_PATTERNS):
                    return True
        return False


@dataclass
class EvidenceItem:
    """A distinct piece of factual evidence extracted from a specific document page."""
    page: int
    section: str
    text: str
    supports: List[str] = field(default_factory=list)
    matched_anchors: List[str] = field(default_factory=list)
    passing_anchors: List[str] = field(default_factory=list)
    is_explanatory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimItem:
    """A factual claim derived from evidence with a verified status."""
    claim: str
    sources: List[int]
    status: str = "SUPPORTED"  # SUPPORTED, CONTRADICTED, SUPERSEDED, UNRESOLVED
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContradictionItem:
    """Records a conflict or supersession between statements across pages or sections."""
    claim_1: str
    page_1: int
    claim_2: str
    page_2: int
    resolution: str  # SUPERSEDED, CONFLICTING, UNRESOLVED
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RequirementCoverage:
    """Tracks whether a question requirement has supporting evidence."""
    requirement: str
    satisfied: bool = False
    pages: List[int] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceLedger:
    """Deterministic ledger for logging evidence items, claims, coverage, and contradictions."""

    def __init__(self, requirements: Optional[List[str]] = None):
        self.evidence_items: List[EvidenceItem] = []
        self.claims: List[ClaimItem] = []
        self.contradictions: List[ContradictionItem] = []
        self.requirements: Dict[str, RequirementCoverage] = {}

        if requirements:
            self.set_requirements(requirements)

    def set_requirements(self, requirements: List[str]):
        """Initialize or update requirements to track."""
        for req in requirements:
            clean_req = req.strip()
            if clean_req and clean_req not in self.requirements:
                self.requirements[clean_req] = RequirementCoverage(requirement=clean_req)

    def add_evidence(
        self,
        page: int,
        text: str,
        section: str = "",
        supports: Optional[List[str]] = None,
        matched_anchors: Optional[List[str]] = None,
        passing_anchors: Optional[List[str]] = None,
        is_explanatory: bool = False
    ) -> EvidenceItem:
        item = EvidenceItem(
            page=page,
            section=section,
            text=text,
            supports=supports or [],
            matched_anchors=matched_anchors or [],
            passing_anchors=passing_anchors or [],
            is_explanatory=is_explanatory
        )
        self.evidence_items.append(item)
        return item

    def get_all_matched_anchors(self) -> List[str]:
        """Returns all distinct anchor entities verified with explanatory evidence across all evidence items."""
        anchors: List[str] = []
        for e in self.evidence_items:
            for a in e.matched_anchors:
                if a.lower() not in [x.lower() for x in anchors]:
                    anchors.append(a)
        return anchors

    def get_all_passing_anchors(self) -> List[str]:
        """Returns all anchor entities that only appeared as passing mentions or citations."""
        anchors: List[str] = []
        for e in self.evidence_items:
            for a in e.passing_anchors:
                if a.lower() not in [x.lower() for x in anchors]:
                    anchors.append(a)
        return anchors

    def is_anchor_covered(self, anchor: str) -> bool:
        """Determines whether a specific anchor entity was verified with explanatory evidence."""
        matched = self.get_all_matched_anchors()
        return any(anchor.lower() == m.lower() for m in matched)

    def add_claim(
        self,
        claim: str,
        sources: List[int],
        status: str = "SUPPORTED",
        reason: Optional[str] = None
    ) -> ClaimItem:
        item = ClaimItem(
            claim=claim,
            sources=sorted(list(set(sources))),
            status=status,
            reason=reason
        )
        self.claims.append(item)
        return item

    def update_claim_status(self, claim_text: str, new_status: str, reason: Optional[str] = None):
        for item in self.claims:
            if item.claim.lower() == claim_text.lower():
                item.status = new_status
                if reason:
                    item.reason = reason

    def add_contradiction(
        self,
        claim_1: str,
        page_1: int,
        claim_2: str,
        page_2: int,
        resolution: str,
        details: str
    ) -> ContradictionItem:
        item = ContradictionItem(
            claim_1=claim_1,
            page_1=page_1,
            claim_2=claim_2,
            page_2=page_2,
            resolution=resolution,
            details=details
        )
        self.contradictions.append(item)
        return item

    def mark_requirement_satisfied(self, requirement: str, page: int, notes: str = ""):
        # Match requirement case-insensitively or by substring
        matched = False
        for req_key, coverage in self.requirements.items():
            if requirement.lower() in req_key.lower() or req_key.lower() in requirement.lower():
                coverage.satisfied = True
                if page not in coverage.pages:
                    coverage.pages.append(page)
                if notes:
                    coverage.notes = notes
                matched = True

        if not matched:
            self.requirements[requirement] = RequirementCoverage(
                requirement=requirement,
                satisfied=True,
                pages=[page],
                notes=notes
            )

    @property
    def total_requirements(self) -> int:
        return len(self.requirements)

    @property
    def satisfied_requirements_count(self) -> int:
        return sum(1 for c in self.requirements.values() if c.satisfied)

    def is_fully_covered(self) -> bool:
        if not self.requirements:
            return len(self.evidence_items) > 0
        return all(c.satisfied for c in self.requirements.values())

    def is_partially_covered(self) -> bool:
        if not self.requirements:
            return False
        satisfied = sum(1 for c in self.requirements.values() if c.satisfied)
        return 0 < satisfied < len(self.requirements)

    def get_missing_requirements(self) -> List[str]:
        return [req for req, cov in self.requirements.items() if not cov.satisfied]

    def has_conflicts(self) -> bool:
        for c in self.contradictions:
            if c.resolution in ("CONFLICTING", "UNRESOLVED"):
                return True
        for cl in self.claims:
            if cl.status in ("CONTRADICTED", "UNRESOLVED"):
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence": [e.to_dict() for e in self.evidence_items],
            "claims": [c.to_dict() for c in self.claims],
            "contradictions": [k.to_dict() for k in self.contradictions],
            "requirements": {k: v.to_dict() for k, v in self.requirements.items()},
            "is_fully_covered": self.is_fully_covered(),
            "coverage_ratio": (
                self.satisfied_requirements_count / self.total_requirements
                if self.total_requirements > 0 else 0.0
            ),
        }
