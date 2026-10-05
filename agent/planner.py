"""Question Analyzer and Planner module."""

import re
from typing import Dict, Any, List, Optional
from agent.prompts import PLANNER_SYSTEM_PROMPT
from llm.provider import LLMProvider


class QuestionPlanner:
    """Analyzes questions prior to retrieval to construct a structured search strategy."""

    VALID_QUESTION_TYPES = {
        "direct_fact",
        "definition",
        "explanation",
        "comparison",
        "multi_page_synthesis",
        "temporal",
        "contradiction_sensitive",
        "likely_absence",
    }

    QUESTION_FUNCTION_WORDS = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "does", "do", "did", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "can", "could", "will", "would", "shall", "should",
        "may", "might", "must",
        "mean", "means", "meaning", "meant",
        "explain", "explains", "explained", "explanation",
        "define", "defines", "defined", "definition",
        "describe", "describes", "described", "description",
        "discuss", "discussion", "overview", "detail", "details",
        "information", "regarding", "about", "tell", "show", "give", "state",
        "the", "a", "an", "difference", "compare", "between", "versus", "vs",
        "provide", "provides", "provided", "answer", "question",
        "and", "or", "but", "if", "of", "to", "in", "for", "with", "on", "at",
        "by", "from", "as", "into", "through", "during", "before", "after", "above", "below",
        "term", "terms", "phrase", "phrases", "word", "words", "concept", "concepts",
        "called", "named", "names", "known", "coined", "adopted", "introduced", "invented", "originated",
        "year", "date", "time", "history",
        "differ", "differs", "differed", "differing", "distinguish", "distinguishes", "distinguished", "comparison"
    }

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def plan(self, question: str) -> Dict[str, Any]:
        """Analyzes the question and produces structured requirements and search terms."""
        user_prompt = f"Analyze the following question for document retrieval:\nQuestion: {question.strip()}"
        
        try:
            plan_data = self.llm.complete_json(PLANNER_SYSTEM_PROMPT, user_prompt)
        except Exception:
            plan_data = {}

        return self._normalize_plan(question, plan_data)

    @classmethod
    def _derive_morphological_variants(cls, phrase: str) -> List[str]:
        """Dynamically generates generic morphological and grammatical variants for an arbitrary phrase.
        
        Applies standard English inflectional/derivational transformations without domain-specific logic:
        - -ly adverbs <-> base adjectives (e.g. rationally <-> rational, dynamically <-> dynamic)
        - -ing gerunds/participles <-> base verbs, 3rd person, and -ion nouns (e.g. acting <-> act, action)
        - phrase inversion (e.g. 'acting rationally' <-> 'act rationally', 'rational action', 'rational behavior')
        """
        words = [w.strip("?,.:;\"'()[]{}") for w in phrase.split()]
        if not words:
            return []

        word_variants: List[List[str]] = []
        for w in words:
            w_lower = w.lower()
            derived: List[str] = []
            # Adverb -> Adjective
            if w_lower.endswith("ically") and len(w_lower) > 6:
                derived.append(w_lower[:-4])  # e.g. dynamically -> dynamic
                derived.append(w_lower[:-2])  # dynamical
            elif w_lower.endswith("ly") and len(w_lower) > 3:
                base_adj = w_lower[:-2]
                if base_adj.endswith("i"):
                    base_adj = base_adj[:-1] + "y"
                derived.append(base_adj)
            # Gerund -> Verb / Noun
            if w_lower.endswith("ing") and len(w_lower) > 4:
                base_v = w_lower[:-3]
                if base_v.endswith("t"):
                    derived.append(base_v + "ion")  # e.g. act -> action
                elif base_v.endswith("at"):
                    derived.append(base_v + "ion")
                else:
                    derived.append(base_v + "ion")
                    derived.append(base_v + "ation")
                derived.append(base_v + "e")  # modulate, operate
                derived.append(base_v)
                derived.append(base_v + "s")
            # Noun (-tion/-sion) -> Verb
            if (w_lower.endswith("tion") or w_lower.endswith("sion")) and len(w_lower) > 5:
                derived.append(w_lower[:-3])
                derived.append(w_lower[:-4] + "te")
            # Plural / 3rd-person -> Base
            if w_lower.endswith("s") and len(w_lower) > 3 and not w_lower.endswith("ss"):
                derived.append(w_lower[:-1])

            combined_w = derived + [w]
            clean_v = []
            for v in combined_w:
                if v and v.lower() not in [x.lower() for x in clean_v]:
                    clean_v.append(v)
            word_variants.append(clean_v)

        results: List[str] = []
        # If multi-word phrase, generate combinations
        if len(words) == 2:
            w0_vars = word_variants[0]
            w1_vars = word_variants[1]
            
            # Prioritize inverted forms (adjective + noun / adverb + verb)
            # E.g. "modulating dynamically" -> "dynamic modulation", "acting rationally" -> "rational action"
            for v1 in w1_vars:
                for v0 in w0_vars:
                    inv = f"{v1} {v0}"
                    if inv.lower() != phrase.lower() and inv.lower() not in [r.lower() for r in results]:
                        results.append(inv)

            # Also include direct order variations
            for v0 in w0_vars:
                for v1 in w1_vars:
                    comb = f"{v0} {v1}"
                    if comb.lower() != phrase.lower() and comb.lower() not in [r.lower() for r in results]:
                        results.append(comb)
        else:
            for w_v in word_variants:
                for v in w_v:
                    if v.lower() != phrase.lower() and v.lower() not in [r.lower() for r in results]:
                        results.append(v)

        return results[:16]

    @classmethod
    def _clean_candidate_term(cls, term: str) -> Optional[str]:
        """Cleans a candidate search term by stripping surrounding punctuation and framing words.
        
        Preserves technical notation like f(n), g(n), A* while eliminating awkward combinations
        and standalone framing words.
        """
        t = str(term).strip(" \t\n")
        # Strip outer quotes, periods, commas, exclamation marks, question marks
        t = re.sub(r'^[?,:;"\'!]+|[?,:;"\'!]+$', '', t).strip()
        if not t:
            return None

        words = t.split()
        if not words:
            return None

        # Strip leading and trailing function words
        while words and re.sub(r'[?,.:;"\'!]', '', words[0]).lower() in cls.QUESTION_FUNCTION_WORDS:
            words.pop(0)
        while words and re.sub(r'[?,.:;"\'!]', '', words[-1]).lower() in cls.QUESTION_FUNCTION_WORDS:
            words.pop()

        if not words:
            return None

        # Reject awkward phrases with internal function words if longer than 3 words
        if len(words) > 3 and any(re.sub(r'[?,.:;"\'!]', '', w).lower() in cls.QUESTION_FUNCTION_WORDS for w in words):
            return None

        cleaned = " ".join(words).strip()
        cleaned_no_punct = re.sub(r'[?,.:;"\'!]', '', cleaned).lower()
        if cleaned_no_punct in cls.QUESTION_FUNCTION_WORDS or len(cleaned) < 2:
            return None

        return cleaned

    @classmethod
    def _prioritize_search_terms(
        cls,
        candidate_terms: List[str],
        anchor_entities: List[str],
        phrases: List[str],
        natural_variants: Optional[List[str]] = None,
        morphological_variants: Optional[List[str]] = None
    ) -> List[str]:
        """Prioritizes search terms to maximize retrieval efficiency under a hard tool-call budget.
        
        5-Level Ranking Order:
        1. Exact meaningful concept phrase from question / anchors
        2. Natural terminology variants (morphological / grammatical variants)
        3. Conceptual / technical variants (mechanisms, formal notations like f(n), domain variants)
        4. Focused component keywords (substantive words from the concept)
        5. Broad abbreviations or short acronyms (placed LAST)
        
        Suppresses redundant/synonymous variants to conserve tool calls.
        """
        multi_word_anchors = [a.strip() for a in anchor_entities if len(a.strip().split()) >= 2]
        multi_word_phrases = [p.strip() for p in phrases if len(p.strip().split()) >= 2]
        all_multi_words = multi_word_anchors + multi_word_phrases
        has_multi_word = len(all_multi_words) > 0

        concept_initials = set()
        for mc in all_multi_words:
            words = [w for w in mc.split() if w.lower() not in cls.QUESTION_FUNCTION_WORDS]
            if len(words) >= 2:
                concept_initials.add("".join(w[0] for w in words).lower())

        tier1: List[str] = []  # Exact primary entity / multi-word phrase
        tier2: List[str] = []  # Natural terminology variants (morphological)
        tier3: List[str] = []  # Conceptual / technical variants
        tier4: List[str] = []  # Focused component words / content keywords
        tier5: List[str] = []  # Broad short abbreviations / aliases

        existing_all: List[str] = []

        def is_redundant(term_to_check: str) -> bool:
            norm = re.sub(r'[\s\-_]+', ' ', term_to_check.lower())
            for ex in existing_all:
                ex_norm = re.sub(r'[\s\-_]+', ' ', ex.lower())
                if norm == ex_norm:
                    return True
                if norm.endswith("s") and norm[:-1] == ex_norm:
                    return True
                if ex_norm.endswith("s") and ex_norm[:-1] == norm:
                    return True
            return False

        def add_term_to_tier(term: str, target_tier: List[str]):
            clean = term.strip()
            if not clean or is_redundant(clean):
                return
            existing_all.append(clean)
            target_tier.append(clean)

        natural_lower = set(m.lower() for m in (natural_variants or morphological_variants or []))

        for raw_t in candidate_terms:
            t = cls._clean_candidate_term(raw_t)
            if not t:
                continue

            t_lower = t.lower()
            words = t.split()
            word_count = len(words)

            if has_multi_word:
                is_exact_multi = any(t_lower == m.lower() for m in all_multi_words)
                is_natural_var = t_lower in natural_lower and not is_exact_multi

                # Check if this term is a broad short abbreviation / alias
                is_abbreviation = (
                    word_count == 1
                    and (
                        len(t) <= 3
                        or (t.isupper() and len(t) <= 4)
                        or t_lower in concept_initials
                    )
                    and not re.search(r'[\(\)\*]', t)
                )

                # Check if this is a technical / conceptual variant (e.g. formula f(n) or multi-word concept)
                is_technical_or_conceptual = (
                    bool(re.search(r'[\(\)\*]', t))
                    or (word_count >= 2 and not is_exact_multi and not is_natural_var)
                )

                if is_exact_multi:
                    add_term_to_tier(t, tier1)
                elif is_abbreviation:
                    add_term_to_tier(t, tier5)
                elif is_natural_var:
                    add_term_to_tier(t, tier2)
                elif is_technical_or_conceptual:
                    add_term_to_tier(t, tier3)
                else:
                    # Single substantive content word (component keyword)
                    add_term_to_tier(t, tier4)
            else:
                # No multi-word concept: single-word entity is primary
                is_primary_anchor = any(t_lower == a.lower() for a in anchor_entities)
                if is_primary_anchor:
                    add_term_to_tier(t, tier1)
                elif t_lower in natural_lower:
                    add_term_to_tier(t, tier2)
                elif word_count >= 2 or re.search(r'[\(\)\*]', t):
                    add_term_to_tier(t, tier3)
                else:
                    add_term_to_tier(t, tier4)

        return tier1 + tier2 + tier3 + tier4 + tier5

    def _normalize_plan(self, question: str, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures all expected keys, types, and high search-quality constraints are enforced."""
        q_type = plan_data.get("question_type", "direct_fact")
        if q_type not in self.VALID_QUESTION_TYPES:
            q_type = "direct_fact"

        concepts = plan_data.get("concepts", [])
        if not isinstance(concepts, list):
            concepts = [str(concepts)]

        raw_search_terms = plan_data.get("search_terms", [])
        if not isinstance(raw_search_terms, list):
            raw_search_terms = [str(raw_search_terms)]

        raw_requirements = plan_data.get("requirements", [])
        if not isinstance(raw_requirements, list):
            raw_requirements = [str(raw_requirements)]

        raw_anchors = plan_data.get("anchor_entities", [])
        if not isinstance(raw_anchors, list):
            raw_anchors = [str(raw_anchors)]

        raw_variants = plan_data.get("anchor_variants", {})
        if not isinstance(raw_variants, dict):
            raw_variants = {}

        # Extract substantive multi-word concept phrases from the question
        q_words = [w.strip("?,.:;\"'()[]{}") for w in question.split()]
        substantive_tokens = [w for w in q_words if w.lower() not in self.QUESTION_FUNCTION_WORDS and len(w) >= 2]

        phrases: List[str] = []
        curr_p: List[str] = []
        for w in q_words:
            if w.lower() not in self.QUESTION_FUNCTION_WORDS and len(w) >= 2:
                curr_p.append(w)
            else:
                if len(curr_p) >= 2:
                    phrases.append(" ".join(curr_p))
                curr_p = []
        if len(curr_p) >= 2:
            phrases.append(" ".join(curr_p))

        # 1. Clean and focus anchor entities
        clean_anchors: List[str] = []
        for a in raw_anchors:
            sa = str(a).strip()
            # Reject anchors that are pure question-function words
            if not sa or sa.lower() in self.QUESTION_FUNCTION_WORDS:
                continue
            if sa.lower() not in [x.lower() for x in clean_anchors]:
                clean_anchors.append(sa)

        # If raw_anchors contained separate fragmented words of a multi-word phrase from the question,
        # unify them into the single cohesive concept phrase.
        for p in phrases:
            p_words = [w.lower() for w in p.split()]
            matching_raw_words = [a for a in clean_anchors if a.lower() in p_words and len(a.split()) == 1]
            if len(matching_raw_words) >= 2 or (matching_raw_words and len(matching_raw_words) == len(p_words)):
                new_anchors = [a for a in clean_anchors if a not in matching_raw_words]
                if p not in new_anchors:
                    new_anchors.append(p)
                clean_anchors = new_anchors

        if not clean_anchors:
            clean_anchors = phrases[:2] if phrases else (substantive_tokens[:2] if substantive_tokens else [question.strip()])

        # 2. Build and augment anchor variants
        clean_variants: Dict[str, List[str]] = {}
        for anchor in clean_anchors:
            v_list: List[str] = []
            # Check LLM provided variants
            llm_v = raw_variants.get(anchor, [])
            if isinstance(llm_v, list):
                for lv in llm_v:
                    slv = str(lv).strip()
                    if slv and slv.lower() != anchor.lower() and slv.lower() not in [x.lower() for x in v_list]:
                        v_list.append(slv)
            # Add dynamic morphological variants
            morph_v = self._derive_morphological_variants(anchor)
            for mv in morph_v:
                if mv.lower() != anchor.lower() and mv.lower() not in [x.lower() for x in v_list]:
                    v_list.append(mv)
            clean_variants[anchor] = v_list

        # 3. Clean and prioritize search terms
        raw_candidates: List[str] = []

        # (a) Exact anchors and substantive phrases from question
        for a in clean_anchors:
            raw_candidates.append(a)
        for p in phrases:
            raw_candidates.append(p)

        # (b) Natural variants provided by LLM
        for a in clean_anchors:
            llm_v = raw_variants.get(a, [])
            if isinstance(llm_v, list):
                for lv in llm_v:
                    raw_candidates.append(str(lv).strip())

        # (c) Raw search terms from planner LLM
        for t in raw_search_terms:
            raw_candidates.append(t)

        # (d) Dynamic programmatic morphological variants
        for a in clean_anchors:
            for mv in self._derive_morphological_variants(a):
                raw_candidates.append(mv)

        # (e) Focused component content keywords from multi-word anchors (e.g. "Intelligence", "Artificial")
        for a in clean_anchors:
            if len(a.split()) >= 2:
                comp_words = [w.strip("?,.:;\"'()") for w in a.split()]
                for cw in reversed(comp_words):
                    if len(cw) >= 3 and cw.lower() not in self.QUESTION_FUNCTION_WORDS:
                        raw_candidates.append(cw)

        # (f) Substantive tokens from question
        for tok in substantive_tokens:
            if len(tok) >= 3 and tok.lower() not in self.QUESTION_FUNCTION_WORDS:
                raw_candidates.append(tok)

        # Collect all natural variants across anchors for Tier 2 classification
        all_natural_variants: List[str] = []
        for a in clean_anchors:
            llm_v = raw_variants.get(a, [])
            if isinstance(llm_v, list):
                for lv in llm_v:
                    slv = str(lv).strip()
                    if slv:
                        all_natural_variants.append(slv)
            all_natural_variants.extend(self._derive_morphological_variants(a))

        clean_terms = self._prioritize_search_terms(
            candidate_terms=raw_candidates,
            anchor_entities=clean_anchors,
            phrases=phrases,
            natural_variants=all_natural_variants
        )

        if not clean_terms:
            clean_terms = [question.strip()]

        # 4. Clean requirements
        clean_reqs: List[str] = []
        for r in raw_requirements:
            sr = str(r).strip()
            if not sr:
                continue
            # Filter out requirements that are purely about question words like "Information regarding mean"
            r_words = [w.lower().strip("?,.:;\"'()") for w in sr.split()]
            sub_r_words = [w for w in r_words if w not in self.QUESTION_FUNCTION_WORDS and len(w) >= 2]
            if not sub_r_words:
                continue
            clean_reqs.append(sr)

        if not clean_reqs:
            for a in clean_anchors:
                if q_type == "definition":
                    clean_reqs.append(f"Definition and explanation of {a}")
                elif q_type == "comparison":
                    clean_reqs.append(f"Characteristics and behavior of {a}")
                else:
                    clean_reqs.append(f"Information regarding {a}")

        return {
            "question_type": q_type,
            "concepts": [str(c).strip() for c in concepts if str(c).strip()],
            "search_terms": clean_terms[:5],
            "requirements": clean_reqs[:4],
            "anchor_entities": clean_anchors[:3],
            "anchor_variants": clean_variants
        }
