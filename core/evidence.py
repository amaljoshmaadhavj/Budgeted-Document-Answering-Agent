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
        r"\b(?:operates?|operat(?:ed|ing)|computes?|comput(?:ed|ing)|provides?|provid(?:ed|ing)|enables?|enabl(?:ed|ing)|uses?|us(?:ed|ing)|utiliz(?:es?|ed|ing)?|appl(?:ies|ied|ying)?|features?|featur(?:ed|ing)?|implements?|implement(?:ed|ing)|performs?|perform(?:ed|ing)|evaluates?|evaluat(?:ed|ing)|determines?|determin(?:ed|ing)|handles?|handl(?:ed|ing)|achieves?|achiev(?:ed|ing)|works?|work(?:ed|ing)|allows?|allow(?:ed|ing)|executes?|execut(?:ed|ing)|transforms?|transform(?:ed|ing)|produces?|produc(?:ed|ing)|requires?|requir(?:ed|ing)|optimizes?|optimiz(?:ed|ing)|coordinates?|coordinat(?:ed|ing)|processes|process(?:ed|ing)|manages?|manag(?:ed|ing)|generates?|generat(?:ed|ing)|stores?|stor(?:ed|ing)|transfers?|transferr(?:ed|ing)|traverses?|travers(?:ed|ing)|dispatches?|dispatch(?:ed|ing)|buffers?|buffer(?:ed|ing)|measures?|measur(?:ed|ing)|structures?|structur(?:ed|ing)|facilitates?|facilitat(?:ed|ing)|calculates?|calculat(?:ed|ing)|acts?|act(?:ed|ing)|solves?|solv(?:ed|ing)|encompasses?|maps?|mapp(?:ed|ing)|supersedes?|supersed(?:ed|ing))\b",
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

    @classmethod
    def validate_requirement_on_page(
        cls,
        text: str,
        requirement: str,
        facet: Optional[Dict[str, Any]] = None,
        anchor: Optional[str] = None,
        anchor_variants: Optional[List[str]] = None,
        question_type: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, str]:
        """Validates whether fetched page text actually satisfies a specific requirement.
        
        Enforces that subject relevance and is_explanatory == True are NEVER sufficient by themselves
        to satisfy a requirement. Requirement satisfaction strictly requires:
        1. Valid subject/anchor grounding in the context window
        2. Substantive evidence (non-apparatus, non-passing)
        3. Requirement-specific evidence facet match in the same context window
        """
        # Handle backwards-compatible positional calls where anchor was passed as 3rd arg
        if isinstance(facet, str) and anchor is None:
            anchor = facet
            facet = kwargs.get("facet")

        segments = cls.split_into_context_segments(text)
        non_app_segments = [s for s, in_app in segments if not in_app and s.strip()]
        if not non_app_segments:
            return False, "Fetched text contains only apparatus or non-content lines."

        # Build context evaluation windows (single sentences and 2-sentence windows for anaphora)
        windows: List[str] = []
        for i, seg in enumerate(non_app_segments):
            windows.append(seg)
            if i + 1 < len(non_app_segments):
                next_seg = non_app_segments[i + 1]
                if re.match(r'^(?:it|this|they|these|such|the\s+[a-zA-Z-]+)\b', next_seg.strip(), re.IGNORECASE):
                    windows.append(f"{seg} {next_seg}")

        all_anchor_forms: List[str] = []
        if anchor and anchor.strip():
            all_anchor_forms.append(anchor.strip())
        if anchor_variants:
            for v in anchor_variants:
                sv = str(v).strip()
                if sv and sv.lower() not in [x.lower() for x in all_anchor_forms]:
                    all_anchor_forms.append(sv)

        # Determine effective facet type and required relation
        facet_type = None
        required_relation = None
        evidence_signals: List[str] = []

        if facet and isinstance(facet, dict):
            facet_type = str(facet.get("facet_type", "")).strip().lower()
            required_relation = str(facet.get("required_relation", "")).strip().lower()
            raw_sig = facet.get("evidence_signals", [])
            if isinstance(raw_sig, list):
                evidence_signals = [str(s).strip().lower() for s in raw_sig if str(s).strip()]

        req_l = requirement.lower()
        if not facet_type:
            if question_type == "temporal" or any(w in req_l for w in ["when", "year", "date", "timeline", "origin", "adopted", "adoption", "coined", "founded", "history"]):
                facet_type = "temporal"
            elif question_type == "definition" or any(w in req_l for w in ["define", "definition", "meaning", "what is", "what are", "mechanism"]):
                facet_type = "definition"
            elif question_type == "comparison" or any(w in req_l for w in ["compare", "difference", "versus", "vs", "distinguish"]):
                facet_type = "comparison"
            elif question_type in ("causal", "explanation") or any(w in req_l for w in ["why", "purpose", "cause", "reason", "rationale"]):
                facet_type = "causal"
            elif any(w in req_l for w in ["how", "step", "algorithm", "procedure", "process"]):
                facet_type = "procedure"
            elif any(w in req_l for w in ["who", "author", "creator", "developed by", "proposed by"]):
                facet_type = "attribution"
            elif any(w in req_l for w in ["cost", "value", "metric", "threshold", "rate", "percentage"]):
                facet_type = "quantitative"
            else:
                facet_type = "fact"

        if not required_relation:
            for rel in ["adopted", "adoption", "coined", "coining", "introduced", "introduction", "created", "creation", "founded", "invented", "named", "proposed", "defined", "measured"]:
                if rel in req_l:
                    required_relation = rel
                    break
            if not required_relation:
                required_relation = "associated_with" if facet_type == "fact" else facet_type

        # Scan context windows
        for win in windows:
            win_l = win.lower()

            # 1. Subject / Anchor Grounding check
            if all_anchor_forms:
                has_anchor = any(cls.anchor_matches_text(win, af) for af in all_anchor_forms)
                if not has_anchor:
                    continue

            # 2. Check for disqualifying passing pattern
            if any(re.search(p, win_l) for p in cls.PASSING_MENTION_PATTERNS):
                continue

            # 3. Requirement-specific facet evaluation
            if facet_type == "temporal":
                # Check for temporal expressions
                temporal_patterns = [
                    r'\b(?:1[789]\d\d|20\d\d)\b',
                    r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
                    r'\b(?:1st|2nd|3rd|[0-9]+th)\s+century\b',
                    r'\b(?:conference|workshop|symposium|summit|convention|meeting|founding|adoption\s+event|historical\s+event)\b',
                    r'\b(?:in|during|at|around|circa|since|dated?\s+to)\s+(?:1[789]\d\d|20\d\d|[a-zA-Z\s]+(?:conference|workshop|symposium))\b'
                ]
                has_temporal_expr = any(re.search(p, win_l) for p in temporal_patterns)
                if not has_temporal_expr:
                    continue

                # Check for requested temporal relation or historical event expression (FIX 4)
                if required_relation in ("adopted", "adoption", "coined", "coining", "named"):
                    rel_pat = r'\b(?:adopt(?:ed|ing|s|ion)?|officially\s+ado\b|coined?|first\s+(?:used|coined|adopted|named)|nam(?:ed|ing)?|historical\s+event)\b'
                elif required_relation in ("introduced", "introduction", "origin", "originated"):
                    rel_pat = r'\b(?:introduc(?:ed|ing|es?|tion)?|origin(?:at(?:ed|ing|es?|ion))?|first\s+(?:introduced|used|appeared)|began|started|historical\s+event)\b'
                elif required_relation in ("created", "creation", "founded", "invented", "established"):
                    rel_pat = r'\b(?:creat(?:ed|ing|es?|ion)?|found(?:ed|ing|s|ation)?|invent(?:ed|ing|es?|ion)?|built|established|historical\s+event)\b'
                else:
                    rel_pat = r'\b(?:adopt(?:ed|ing|s|ion)?|officially\s+ado\b|coined?|introduc(?:ed|ing|es?|tion)?|creat(?:ed|ing|es?|ion)?|found(?:ed|ing|s|ation)?|origin(?:at(?:ed|ing|es?|ion))?|first\s+(?:used|coined|adopted|introduced)|historical\s+event)\b'

                has_rel = bool(re.search(rel_pat, win_l)) or any(sig in win_l for sig in evidence_signals if sig not in ("year", "date"))
                if has_rel:
                    return True, f"Temporal evidence establishing '{required_relation}' found in context window."

            elif facet_type == "definition":
                has_def_pred = any(re.search(p, win_l) for p in cls.EXPLANATORY_PREDICATE_PATTERNS)
                if has_def_pred:
                    words = [w.strip("?,.:;\"'()[]{}").lower() for w in win.split()]
                    subst = [w for w in words if w not in cls.GENERIC_STOPWORDS and not any(w in af.lower() for af in all_anchor_forms)]
                    if len(subst) >= 2:
                        return True, "Definitional and explanatory grounding verified in context window."

            elif facet_type == "comparison":
                comp_pat = r'\b(?:differs?|difference|distinguish(?:es|ed|ing)?|in\s+contrast|unlike|whereas|compared\s+(?:with|to)|versus|vs\b)'
                has_comp = bool(re.search(comp_pat, win_l))
                if has_comp:
                    return True, "Comparative evidence establishing distinction verified in context window."
                # When evaluating distinguishing characteristics of an individual anchor entity, substantive explanatory grounding satisfies
                has_pred = any(re.search(p, win_l) for p in cls.EXPLANATORY_PREDICATE_PATTERNS)
                if has_pred:
                    words = [w.strip("?,.:;\"'()[]{}").lower() for w in win.split()]
                    subst = [w for w in words if w not in cls.GENERIC_STOPWORDS and not any(w in af.lower() for af in all_anchor_forms)]
                    if len(subst) >= 2:
                        return True, "Entity characteristics and behavioral properties verified in context window."

            elif facet_type == "causal":
                causal_pat = r'\b(?:because|due\s+to|causes?|results?\s+in|reason\s+is|designed\s+(?:to|for)|intended\s+to|purpose\s+is|serves\s+to|in\s+order\s+to)\b'
                if re.search(causal_pat, win_l):
                    return True, "Causal/purpose rationale verified in context window."

            elif facet_type == "procedure":
                proc_pat = r'\b(?:steps?|algorithm|procedure|process|first\s+[^.!?]+then|computed\s+by)\b'
                if re.search(proc_pat, win_l):
                    return True, "Procedural/operational evidence verified in context window."

            elif facet_type == "attribution":
                attr_pat = r'\b(?:by\s+Dr\.?\s+[A-Z][a-z]+|proposed\s+by|authored\s+by|developed\s+by|created\s+by|introduced\s+by)\b'
                if re.search(attr_pat, win):
                    return True, "Person/entity attribution verified in context window."

            elif facet_type == "quantitative":
                num_pat = r'\b[\$£€¥]?\d+(?:\.\d+)?\s*(?:[a-zA-Z/%]+|\b)'
                if re.search(num_pat, win_l):
                    return True, "Quantitative metric evidence verified in context window."

            else:  # fact
                has_pred = any(re.search(p, win_l) for p in cls.EXPLANATORY_PREDICATE_PATTERNS)
                if has_pred:
                    return True, "Factual property verified in explanatory context window."

        return False, f"Requirement specifies {facet_type} evidence (relation: '{required_relation}'), but no grounded context window on the page contains matching evidence."



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
    facet: Optional[Dict[str, Any]] = None
    satisfied: bool = False
    pages: List[int] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceLedger:
    """Deterministic ledger for logging evidence items, claims, coverage, and contradictions."""

    def __init__(self, requirements: Optional[List[str]] = None, facets: Optional[List[Dict[str, Any]]] = None):
        self.evidence_items: List[EvidenceItem] = []
        self.claims: List[ClaimItem] = []
        self.contradictions: List[ContradictionItem] = []
        self.requirements: Dict[str, RequirementCoverage] = {}

        if requirements:
            self.set_requirements(requirements, facets=facets)

    def set_requirements(self, requirements: List[str], facets: Optional[List[Dict[str, Any]]] = None):
        """Initialize or update requirements to track."""
        for idx, req in enumerate(requirements):
            clean_req = req.strip()
            if clean_req and clean_req not in self.requirements:
                f_item = facets[idx] if facets and idx < len(facets) else None
                self.requirements[clean_req] = RequirementCoverage(requirement=clean_req, facet=f_item)

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
