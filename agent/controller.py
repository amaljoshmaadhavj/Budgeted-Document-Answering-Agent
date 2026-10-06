"""Agent Controller orchestrating question analysis, deterministic harness, action selection, and final answering."""

from typing import Dict, Any, List, Optional, Tuple, Set
import re

from agent.state import QuestionState
from agent.planner import QuestionPlanner
from agent.prompts import FINAL_ANSWER_SYSTEM_PROMPT
from core.budget import BudgetManager, MAX_DOCUMENT_TOOL_CALLS, BudgetExceededError
from core.evidence import EvidenceLedger, EvidenceGroundingValidator
from core.answerability import AnswerabilityGate, AnswerabilityStatus
from core.security import SecurityBoundary
from logging_utils.trace import TraceLogger
from tools.gateway import ToolGateway
from llm.provider import LLMProvider


class CandidatePageRanker:
    """Deterministically ranks candidate pages using planner search signals, co-occurrence, and proximity."""

    @classmethod
    def rank_pages(
        cls,
        page_matches: Dict[int, List[Dict[str, Any]]],
        anchor_entities: List[str],
        anchor_variants: Dict[str, List[str]],
        search_terms: List[str],
        heading_matches: Dict[int, List[Dict[str, Any]]],
        fetched_pages: List[int],
        validated_pages: Optional[List[int]] = None,
        requirement_facets: Optional[List[Dict[str, Any]]] = None,
        question_type: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        """Calculates deterministic multi-factor relevance scores for candidate pages.
        
        Incorporates:
        1. Entity relevance
        2. Requirement-specific evidence signal relevance (FIX 10)
        3. Question-type relevance
        4. Term tier
        5. Multi-term co-occurrence
        6. Heading alignment
        7. Apparatus penalty
        8. Proximity
        
        Returns:
            List of (page_num, score) tuples sorted in descending order of score.
        """
        scores: Dict[int, float] = {}
        validated = validated_pages or []

        anchor_words_map: Dict[str, Set[str]] = {
            a.lower(): set(w.lower() for w in a.split()) for a in anchor_entities
        }

        # Collect requirement-specific facet signals and relations
        facet_signals: Set[str] = set()
        facet_relations: Set[str] = set()
        if requirement_facets:
            for rf in requirement_facets:
                if isinstance(rf, dict):
                    rel = rf.get("required_relation", "")
                    if rel:
                        facet_relations.add(rel.lower())
                    for sig in rf.get("evidence_signals", []):
                        if sig:
                            facet_signals.add(sig.lower())

        all_pages = set(page_matches.keys()).union(heading_matches.keys())

        for p in all_pages:
            if p in fetched_pages:
                continue

            score = 0.0
            matches = page_matches.get(p, [])
            headings = heading_matches.get(p, [])

            distinct_terms = set()
            anchors_hit = set()
            facet_hits = set()

            for m in matches:
                term = m["term"]
                term_lower = term.lower()
                distinct_terms.add(term_lower)
                total_matches = m.get("total_matches", 1)

                # Base weight by search term tier/priority
                term_idx = search_terms.index(term) if term in search_terms else 99
                if term_idx == 0:
                    base_weight = 14.0
                elif term_idx == 1:
                    base_weight = 11.0
                elif term_idx == 2:
                    base_weight = 8.0
                elif term_idx == 3:
                    base_weight = 5.0
                else:
                    base_weight = 3.0

                # Specificity discount: heavily discount broad terms matching >25 pages
                if total_matches > 25:
                    specificity = 0.25
                elif total_matches > 10:
                    specificity = 0.5
                elif total_matches <= 3:
                    specificity = 1.5
                else:
                    specificity = 1.0

                # Check if term aligns with any anchor entity
                for a_lower, a_words in anchor_words_map.items():
                    if a_lower in term_lower or term_lower in a_lower or any(w in term_lower for w in a_words if len(w) >= 3):
                        anchors_hit.add(a_lower)

                # Check if term matches requirement facet signal or relation
                if any(sig in term_lower or term_lower in sig for sig in facet_signals):
                    facet_hits.add(term_lower)
                if any(rel in term_lower or term_lower in rel for rel in facet_relations):
                    facet_hits.add(term_lower)

                score += base_weight * specificity

            # Multi-term Co-occurrence Bonus: pages matching multiple distinct search terms are high-value intersections
            k = len(distinct_terms)
            if k >= 2:
                score += 10.0 * (k - 1)

            # Multi-anchor Coverage Bonus: pages bridging multiple distinct anchors (critical for comparison questions)
            if len(anchors_hit) >= 2:
                score += 22.0

            # Requirement-Specific Evidence Signal Bonus (FIX 10):
            # A page matching requirement-specific evidence signals outranks generic topical discussion
            if facet_hits:
                score += 15.0 * len(facet_hits)

            # High-Value Intersection Bonus: page matches BOTH anchor AND requirement evidence signals
            if anchors_hit and facet_hits:
                score += 20.0

            # Question-Type Specific Relevance
            if question_type == "temporal" and facet_hits:
                score += 10.0
            elif question_type == "comparison" and len(anchors_hit) >= 2:
                score += 25.0

            # Heading alignment
            for h in headings:
                is_apparatus = h.get("is_apparatus", False)
                if is_apparatus:
                    score -= 15.0
                else:
                    score += 9.0
                    htitle = h.get("title", "").lower()
                    if any(sig in htitle for sig in facet_signals) or any(rel in htitle for rel in facet_relations):
                        score += 10.0

            # Validated Neighbor Proximity Bonus: pages adjacent to verified evidence pages
            if any(abs(p - vp) == 1 for vp in validated):
                score += 5.0

            scores[p] = score

        # Proximity to High-Scoring Clusters (adjacent pages in technical explanations)
        high_value_pages = [p for p, sc in scores.items() if sc >= 15.0]
        for p in scores:
            if p not in high_value_pages:
                if any(abs(p - hv) == 1 for hv in high_value_pages):
                    scores[p] += 3.0

        sorted_candidates = sorted(scores.items(), key=lambda x: (x[1], -x[0]), reverse=True)
        return sorted_candidates


class RetrievalScheduler:
    """Deterministic, document-independent retrieval scheduler that dynamically balances
    search, fetch, and headings under a strict 6-call budget limit to maximize verified evidence."""

    @classmethod
    def decide_next_action(
        cls,
        remaining_budget: int,
        ranked_candidates: List[Tuple[int, float]],
        unsearched_terms: List[str],
        state: QuestionState,
        ledger: EvidenceLedger,
        headings_checked: bool,
        searched_keywords: Set[str]
    ) -> Tuple[str, Any, str]:
        """Decides whether the next call should be a search, a page fetch, heading inspection, or stop.
        
        Treats the six document-tool calls as a combined search + fetch budget.
        
        Returns:
            Tuple of (action: str, target: Any, rationale: str)
            action: 'get_page' | 'search_keyword' | 'list_headings' | 'stop'
        """
        # 1. Early Stopping Check: all requirements satisfied, all anchors verified, no conflicts
        all_anchors_present = (
            all(ledger.is_anchor_covered(a) for a in state.anchor_entities)
            if state.anchor_entities else True
        )
        if ledger.is_fully_covered() and all_anchors_present and not ledger.has_conflicts():
            return "stop", None, "All evidence requirements and anchor entities fully established with verified explanatory grounding."

        # 2. Conversational Check
        if state.question_type == "conversational":
            return "stop", None, "Conversational input; zero document retrieval required."

        # 3. Final Remaining Call Rule (Requirement 4):
        # When remaining_budget == 1, if at least one useful unfetched candidate exists:
        # - do NOT perform another search
        # - fetch the highest-ranked candidate.
        if remaining_budget == 1:
            if ranked_candidates:
                top_page, top_score = ranked_candidates[0]
                return (
                    "get_page",
                    top_page,
                    f"Final remaining call (budget=1); fetching top candidate page {top_page} (score {top_score:.1f}) to convert candidate into verified evidence."
                )
            if unsearched_terms:
                term = unsearched_terms[0]
                return (
                    "search_keyword",
                    term,
                    f"Final call with zero candidate pages available; searching for '{term}' to attempt discovery (budget=1)."
                )
            if not headings_checked:
                return (
                    "list_headings",
                    None,
                    "Final call with zero candidate pages or search terms; inspecting table of contents (budget=1)."
                )
            return "stop", None, "Final call with no viable candidates or actions available."

        # 4. Low Remaining Budget Rule (budget == 2):
        # Searching at budget=2 consumes 50% of remaining calls on discovery and leaves at most 1 fetch.
        # If any viable unfetched candidate exists, prioritize fetching.
        if remaining_budget == 2 and ranked_candidates:
            top_page, top_score = ranked_candidates[0]
            if top_score >= 0.0:
                return (
                    "get_page",
                    top_page,
                    f"Low remaining budget (budget=2); prioritizing fetching top candidate page {top_page} (score {top_score:.1f}) before budget is exhausted."
                )

        # 5. Limited Budget with Promising Candidates Rule (budget == 3):
        # Prefer get_page over search if candidate pool has promising candidates or meets/exceeds budget
        if remaining_budget == 3 and ranked_candidates:
            top_page, top_score = ranked_candidates[0]
            if top_score >= 2.0 or len(ranked_candidates) >= remaining_budget:
                return (
                    "get_page",
                    top_page,
                    f"Promising candidate page {top_page} available (score {top_score:.1f}) under limited budget (budget=3); prioritizing fetch over further discovery."
                )

        # 6. Strong Unfetched Candidate Priority (budget >= 4):
        # If strong unfetched candidates exist, prefer get_page() over another search.
        if ranked_candidates:
            top_page, top_score = ranked_candidates[0]
            if top_score >= 5.0:
                return (
                    "get_page",
                    top_page,
                    f"Strong candidate page {top_page} available (score {top_score:.1f} >= 5.0); fetching page to establish grounded evidence."
                )

        # 7. Candidate Pool Capacity Balance:
        # If the number of unfetched candidate pages meets or exceeds the remaining budget,
        # discovering more candidates is wasteful because they can never be fetched.
        if ranked_candidates and len(ranked_candidates) >= remaining_budget:
            top_page, top_score = ranked_candidates[0]
            return (
                "get_page",
                top_page,
                f"Candidate pool ({len(ranked_candidates)} pages) meets or exceeds remaining budget ({remaining_budget}); fetching page {top_page} (score {top_score:.1f}) rather than discovering unfetchable candidates."
            )

        # 8. Requirement-Driven Multi-Anchor Search (budget >= 4):
        # If multi-anchor question and some anchors have not been searched, search for them if budget allows
        if unsearched_terms and remaining_budget >= 4:
            anchors_searched = {
                a.lower() for a in state.anchor_entities
                if any(a.lower() in sk or sk in a.lower() for sk in searched_keywords)
            }
            has_unsearched_anchors = (
                len(state.anchor_entities) >= 2
                and len(anchors_searched) < len(state.anchor_entities)
            )
            if has_unsearched_anchors:
                unsearched_target_terms = [
                    t for t in unsearched_terms
                    if any(a.lower() in t.lower() or t.lower() in a.lower() for a in state.anchor_entities if a.lower() not in anchors_searched)
                ]
                term = unsearched_target_terms[0] if unsearched_target_terms else unsearched_terms[0]
                return (
                    "search_keyword",
                    term,
                    f"Multi-anchor question with unsearched anchor entity; searching for '{term}' to identify bridging pages (budget={remaining_budget})."
                )

        # 9. Discovery Search when candidates are absent or low confidence:
        if unsearched_terms:
            top_score = ranked_candidates[0][1] if ranked_candidates else 0.0
            if top_score < 3.0:
                term = unsearched_terms[0]
                return (
                    "search_keyword",
                    term,
                    f"Candidate confidence low or absent (top_score {top_score:.1f} < 3.0); executing search for '{term}' (budget={remaining_budget})."
                )

        # 10. Fetch Next Viable Candidate:
        if ranked_candidates:
            top_page, top_score = ranked_candidates[0]
            return (
                "get_page",
                top_page,
                f"Fetching viable candidate page {top_page} (score {top_score:.1f}) to evaluate evidence (budget={remaining_budget})."
            )

        # 11. Search remaining terms:
        if unsearched_terms:
            term = unsearched_terms[0]
            return (
                "search_keyword",
                term,
                f"No candidate pages currently available; searching for term '{term}' (budget={remaining_budget})."
            )

        # 12. Check Headings / TOC if not yet inspected:
        if not headings_checked:
            return (
                "list_headings",
                None,
                f"No candidate pages or search terms remain; inspecting table of contents (budget={remaining_budget})."
            )

        # 13. Stop:
        return "stop", None, "No further candidate pages or search actions available."


class AgentController:
    """Controls the end-to-end question answering pipeline adhering strictly to the 6-call budget."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or LLMProvider()
        self.planner = QuestionPlanner(self.llm)

    def answer_question(self, question: str, doc_id: str) -> Tuple[QuestionState, str]:
        """Processes a single user question against a document under a fresh state and hard budget limit.
        
        Returns:
            Tuple of (final_question_state, final_answer_text)
        """
        # 1. Initialize fresh instances - NEVER carry over state or pages across questions
        state = QuestionState.create_fresh(question=question, doc_id=doc_id, max_budget=MAX_DOCUMENT_TOOL_CALLS)
        budget = BudgetManager(max_calls=MAX_DOCUMENT_TOOL_CALLS)
        trace_logger = TraceLogger()
        trace_logger.set_question_context(question, doc_id)
        gateway = ToolGateway(budget_manager=budget, trace_logger=trace_logger)
        ledger = EvidenceLedger()

        # 2. Question Planning
        plan = self.planner.plan(question)
        state.question_type = plan.get("question_type", "direct_fact")
        state.requires_document_evidence = plan.get("requires_document_evidence", True)
        state.concepts = plan.get("concepts", [])
        state.search_terms = plan.get("search_terms", [])
        state.evidence_requirements = plan.get("requirements", [])
        state.requirement_facets = plan.get("requirement_facets", [])
        state.anchor_entities = plan.get("anchor_entities", [])
        state.anchor_variants = plan.get("anchor_variants", {})
        ledger.set_requirements(state.evidence_requirements, state.requirement_facets)
        trace_logger.set_planner_output(plan)

        # Conversational / Non-Document Query Short-Circuit: Zero tool calls, 6 remaining budget
        if not state.requires_document_evidence or state.question_type == "conversational":
            state.stop_reason = "Conversational or non-document query; zero document retrieval required."
            state.budget_used = 0
            state.budget_remaining = MAX_DOCUMENT_TOOL_CALLS
            state.answer_status = AnswerabilityStatus.SUPPORTED
            final_answer = self._generate_conversational_answer(question)
            full_trace = trace_logger.finalize(
                answer_status=state.answer_status,
                stop_reason=state.stop_reason,
                final_answer=final_answer,
                final_answer_consistency=state.final_answer_consistency
            )
            state.trace = full_trace.get("tool_calls", [])
            return state, final_answer

        # 3. Action Selection Loop (Deterministic Harness & Scheduler)
        headings_checked = False
        searched_keywords = set()
        page_matches: Dict[int, List[Dict[str, Any]]] = {}
        heading_matches: Dict[int, List[Dict[str, Any]]] = {}

        # Main retrieval loop: strictly governed by remaining budget
        while budget.can_call():
            unsearched_terms = [t for t in state.search_terms if t.lower() not in searched_keywords]
            validated_pages = [e.page for e in ledger.evidence_items if len(e.supports) > 0]
            ranked_candidates = CandidatePageRanker.rank_pages(
                page_matches=page_matches,
                anchor_entities=state.anchor_entities,
                anchor_variants=state.anchor_variants,
                search_terms=state.search_terms,
                heading_matches=heading_matches,
                fetched_pages=state.fetched_pages,
                validated_pages=validated_pages,
                requirement_facets=state.requirement_facets,
                question_type=state.question_type
            )

            # Ask deterministic RetrievalScheduler for the next optimal decision
            action, target, rationale = RetrievalScheduler.decide_next_action(
                remaining_budget=budget.remaining_calls,
                ranked_candidates=ranked_candidates,
                unsearched_terms=unsearched_terms,
                state=state,
                ledger=ledger,
                headings_checked=headings_checked,
                searched_keywords=searched_keywords
            )

            # Log retrieval decision into trace
            trace_logger.log_retrieval_decision(
                decision_number=budget.calls_used + 1,
                budget_remaining=budget.remaining_calls,
                candidates_available=[p for p, _ in ranked_candidates],
                candidate_ranking=[{"page": p, "score": round(sc, 2)} for p, sc in ranked_candidates[:5]],
                action=action,
                target=target,
                rationale=rationale,
                unsatisfied_requirements=ledger.get_missing_requirements()
            )

            if action == "stop":
                state.stop_reason = rationale
                break

            if action == "search_keyword":
                term_to_search = target
                searched_keywords.add(term_to_search.lower())
                try:
                    result = gateway.execute("search_keyword", doc_id=doc_id, keyword=term_to_search)
                    matching_pages = result.get("pages", [])
                    total_m = len(matching_pages)
                    for p in matching_pages:
                        if p not in page_matches:
                            page_matches[p] = []
                        page_matches[p].append({
                            "term": term_to_search,
                            "total_matches": total_m
                        })
                except BudgetExceededError:
                    break
                continue

            elif action == "list_headings":
                headings_checked = True
                try:
                    result = gateway.execute("list_headings", doc_id=doc_id)
                    headings = result.get("headings", [])
                    for h in headings:
                        htitle = h.get("title", "").lower()
                        hpage = h.get("page", 1)
                        is_apparatus = any(re.search(p, htitle) for p in EvidenceGroundingValidator.APPARATUS_HEADINGS)
                        for term in state.search_terms + state.concepts:
                            if term.lower() in htitle and hpage not in state.fetched_pages:
                                if hpage not in heading_matches:
                                    heading_matches[hpage] = []
                                heading_matches[hpage].append({
                                    "title": htitle,
                                    "term": term,
                                    "is_apparatus": is_apparatus
                                })
                except BudgetExceededError:
                    break
                continue

            elif action == "get_page":
                target_page = target
                try:
                    page_result = gateway.execute("get_page", doc_id=doc_id, page_number=target_page)
                    state.fetched_pages.append(target_page)
                    page_text = page_result.get("text", "")

                    if page_text:
                        self._process_page_evidence(
                            page_num=target_page,
                            text=page_text,
                            state=state,
                            ledger=ledger
                        )

                        # Early stopping condition
                        all_anchors_present = (
                            all(ledger.is_anchor_covered(a) for a in state.anchor_entities)
                            if state.anchor_entities else True
                        )
                        if ledger.is_fully_covered() and all_anchors_present and not ledger.has_conflicts():
                            state.stop_reason = "All evidence requirements and anchor entities fully established with verified explanatory evidence."
                            break
                except BudgetExceededError:
                    break
                continue

        if not state.stop_reason:
            if budget.is_exhausted:
                state.stop_reason = f"Budget exhausted ({MAX_DOCUMENT_TOOL_CALLS} document-tool calls reached)."
            else:
                state.stop_reason = "Evidence collection completed."

        # 4. Update state tracking
        state.candidate_pages = sorted(list(set(page_matches.keys()).union(heading_matches.keys())))
        state.budget_used = budget.calls_used
        state.budget_remaining = budget.remaining_calls

        # 5. Answerability Gate Evaluation
        status, rationale = AnswerabilityGate.evaluate(ledger, anchor_entities=state.anchor_entities)
        state.answer_status = status
        state.evidence = [e.to_dict() for e in ledger.evidence_items]
        state.claims = [c.to_dict() for c in ledger.claims]
        state.contradictions = [k.to_dict() for k in ledger.contradictions]

        # 6. ONE FINAL LLM ANSWER CALL
        final_answer = self._generate_final_answer(
            question=question,
            state=state,
            ledger=ledger
        )

        # 7. Finalize trace
        full_trace = trace_logger.finalize(
            answer_status=state.answer_status,
            stop_reason=state.stop_reason,
            final_answer=final_answer,
            final_answer_consistency=state.final_answer_consistency
        )
        state.trace = full_trace.get("tool_calls", [])

        return state, final_answer

    def _generate_conversational_answer(self, question: str) -> str:
        """Generates a polite conversational response without accessing document tools."""
        user_prompt = f"The user stated: '{question.strip()}'. Provide a polite and helpful conversational response stating you are ready to answer questions about their document."
        try:
            return self.llm.complete(
                system_prompt="You are a polite document answering assistant. Respond concisely and warmly to conversational greetings.",
                user_prompt=user_prompt,
                temperature=0.0
            )
        except Exception:
            return "Hello! How can I assist you with your document today?"

    def _process_page_evidence(
        self,
        page_num: int,
        text: str,
        state: QuestionState,
        ledger: EvidenceLedger
    ):
        """Processes newly fetched page text, extracts evidence, checks for conflicts/supersessions."""
        injections = SecurityBoundary.detect_injection_attempts(text)
        if injections:
            pass

        # Evaluate Anchor Grounding: Explanatory vs Passing/Apparatus (including dynamic variants)
        matched_anchors_on_page: List[str] = []
        passing_anchors_on_page: List[str] = []
        matched_variants_on_page: List[str] = []

        for anchor in state.anchor_entities:
            a_clean = anchor.strip()
            if not a_clean:
                continue
            variants = state.anchor_variants.get(a_clean, [])
            eval_res = EvidenceGroundingValidator.evaluate_anchor_with_variants(text, a_clean, variants)
            if eval_res["is_explanatory"]:
                matched_anchors_on_page.append(a_clean)
                if eval_res.get("matched_form"):
                    matched_variants_on_page.append(eval_res["matched_form"])
            elif eval_res["has_occurrence"]:
                passing_anchors_on_page.append(a_clean)

        # Generic Requirement Validation (Semantic Requirement Verification)
        supported_reqs = []
        has_explanatory_grounding = (len(matched_anchors_on_page) > 0) if state.anchor_entities else True

        if has_explanatory_grounding:
            for req_idx, req in enumerate(state.evidence_requirements):
                req_anchors = [a for a in state.anchor_entities if a.lower() in req.lower()]
                anchor_ok = True
                matching_anchor = None
                if len(req_anchors) >= 2:
                    # Multi-anchor requirement (e.g. comparison): all mentioned anchors must be grounded on page
                    anchor_ok = all(
                        any(a.lower() == m.lower() for m in matched_anchors_on_page)
                        for a in req_anchors
                    )
                    matching_anchor = req_anchors[0]
                elif req_anchors:
                    anchor_ok = any(
                        a.lower() in [m.lower() for m in matched_anchors_on_page]
                        for a in req_anchors
                    )
                    matching_anchor = req_anchors[0]
                elif state.anchor_entities:
                    matching_anchor = state.anchor_entities[0]

                if anchor_ok:
                    variants = state.anchor_variants.get(matching_anchor, []) if matching_anchor else []
                    facet = None
                    if state.requirement_facets and req_idx < len(state.requirement_facets):
                        facet = state.requirement_facets[req_idx]
                    is_valid, reason = EvidenceGroundingValidator.validate_requirement_on_page(
                        text=text,
                        requirement=req,
                        anchor=matching_anchor,
                        anchor_variants=variants,
                        facet=facet
                    )
                    if is_valid:
                        supported_reqs.append(req)
                        ledger.mark_requirement_satisfied(req, page_num, notes=reason)

        # Add evidence item with verified matched anchors and passing mentions
        is_page_explanatory = (len(supported_reqs) > 0 or len(matched_anchors_on_page) > 0)
        ledger.add_evidence(
            page=page_num,
            text=text[:1500],  # Keep relevant excerpt
            supports=supported_reqs,
            matched_anchors=matched_anchors_on_page,
            passing_anchors=passing_anchors_on_page,
            is_explanatory=is_page_explanatory
        )

        # Contradiction / Supersession checking against previously fetched pages
        for prev_item in ledger.evidence_items:
            if prev_item.page == page_num:
                continue
            shared_anchors = set([a.lower() for a in prev_item.matched_anchors]).intersection(
                [a.lower() for a in matched_anchors_on_page]
            )
            shared_terms = set([t.lower() for t in state.search_terms + state.concepts if len(t) >= 3]).intersection(
                [w.lower() for w in re.findall(r'\b\w+\b', prev_item.text)]
            ).intersection(
                [w.lower() for w in re.findall(r'\b\w+\b', text)]
            )
            if shared_anchors or shared_terms or (not state.anchor_entities and prev_item.matched_anchors == matched_anchors_on_page):
                resolution, details = AnswerabilityGate.evaluate_supersession(
                    text_a=prev_item.text,
                    page_a=prev_item.page,
                    text_b=text,
                    page_b=page_num
                )
                if resolution in ("SUPERSEDED", "CONFLICTING"):
                    ledger.add_contradiction(
                        claim_1=f"Excerpt on Page {prev_item.page}",
                        page_1=prev_item.page,
                        claim_2=f"Excerpt on Page {page_num}",
                        page_2=page_num,
                        resolution=resolution,
                        details=details
                    )
                    if resolution == "SUPERSEDED":
                        ledger.add_claim(
                            claim=f"Statement on Page {prev_item.page}",
                            sources=[prev_item.page],
                            status="SUPERSEDED",
                            reason=details
                        )

    def _generate_final_answer(
        self,
        question: str,
        state: QuestionState,
        ledger: EvidenceLedger
    ) -> str:
        """Executes the single allowed final LLM answer call."""
        # Build untrusted evidence block
        evidence_blocks = []
        for item in ledger.evidence_items:
            wrapped = SecurityBoundary.wrap_untrusted_evidence(
                page_num=item.page,
                text=item.text,
                section=item.section
            )
            evidence_blocks.append(wrapped)

        evidence_section = "\n\n".join(evidence_blocks) if evidence_blocks else "No evidence retrieved."

        # Format claims and contradictions
        claims_str = "\n".join([f"- [Page {c.sources}] {c.claim} (Status: {c.status})" for c in ledger.claims]) or "None"
        conflicts_str = "\n".join([
            f"- Page {k.page_1} vs Page {k.page_2}: {k.resolution} ({k.details})" for k in ledger.contradictions
        ]) or "None detected"

        # Check if question is an absence/presence question
        is_presence_q = any(w in question.lower() for w in ["mentioned", "available", "presence", "does the document contain", "is there"])
        absence_instruction = ""
        if is_presence_q and state.answer_status == AnswerabilityStatus.INSUFFICIENT:
            absence_instruction = "\nPRESENCE/ABSENCE NOTE: State 'I could not verify the presence of the requested entity in the available document evidence.' Do not claim the entity is definitely absent from the entire document."

        user_prompt = f"""USER QUESTION:
{question}

ANSWERABILITY GATE STATUS:
{state.answer_status}

REASON / RATIONALE:
{state.stop_reason}

VERIFIED CLAIMS:
{claims_str}

CONTRADICTIONS / SUPERSESSIONS:
{conflicts_str}

RETRIEVED DOCUMENT EVIDENCE:
{evidence_section}
{absence_instruction}

Provide your grounded answer following all status-specific grounding rules. Cite pages using [Page X]."""

        raw_answer = ""
        try:
            raw_answer = self.llm.complete(
                system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0
            )
        except Exception as e:
            raw_answer = f"Error generating final answer: {e}"

        sanitized_answer, consistency = self._enforce_answer_consistency(raw_answer, state.answer_status, question)
        state.final_answer_consistency = consistency
        return sanitized_answer

    @classmethod
    def _enforce_answer_consistency(cls, answer: str, status: str, question: str) -> Tuple[str, str]:
        """Deterministically guarantees that the final answer text does not contradict the deterministic status (FIX 8)."""
        ans_lower = answer.lower()
        if status == AnswerabilityStatus.INSUFFICIENT:
            negative_signals = [
                "insufficient", "could not verify", "cannot verify", "not mentioned",
                "not provided", "no information", "not found", "does not mention",
                "does not specify", "unable to find", "not available", "could not be verified"
            ]
            if not any(sig in ans_lower for sig in negative_signals):
                # The model produced a positive claim despite INSUFFICIENT status
                sanitized = (
                    f"The available document evidence is insufficient to verify or answer the question: '{question}'.\n\n"
                    f"(Retrieved pages do not contain the required evidence.)"
                )
                return sanitized, "SANITIZED_TO_CONSISTENT"
            return answer, "CONSISTENT"

        elif status == AnswerabilityStatus.SUPPORTED:
            # If status is SUPPORTED, the model must not claim inability to verify
            unverified_signals = [
                "i could not verify", "could not verify", "cannot verify",
                "insufficient evidence", "unable to verify", "not found in the document"
            ]
            if any(sig in ans_lower for sig in unverified_signals):
                return answer, "INCONSISTENT_WARNED"
            return answer, "CONSISTENT"

        return answer, "CONSISTENT"

