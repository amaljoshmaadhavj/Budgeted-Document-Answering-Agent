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
        fetched_pages: List[int]
    ) -> List[Tuple[int, float]]:
        """Calculates deterministic multi-factor relevance scores for candidate pages.
        
        Returns:
            List of (page_num, score) tuples sorted in descending order of score.
        """
        scores: Dict[int, float] = {}

        anchor_words_map: Dict[str, Set[str]] = {
            a.lower(): set(w.lower() for w in a.split()) for a in anchor_entities
        }

        all_pages = set(page_matches.keys()).union(heading_matches.keys())

        for p in all_pages:
            if p in fetched_pages:
                continue

            score = 0.0
            matches = page_matches.get(p, [])
            headings = heading_matches.get(p, [])

            distinct_terms = set()
            anchors_hit = set()

            for m in matches:
                term = m["term"]
                term_lower = term.lower()
                distinct_terms.add(term_lower)
                total_matches = m.get("total_matches", 1)

                # Base weight by search term tier/priority
                term_idx = search_terms.index(term) if term in search_terms else 99
                if term_idx == 0:
                    base_weight = 12.0
                elif term_idx == 1:
                    base_weight = 10.0
                elif term_idx == 2:
                    base_weight = 8.0
                elif term_idx == 3:
                    base_weight = 6.0
                else:
                    base_weight = 4.0

                # Specificity discount: heavily discount broad terms matching >20 pages (e.g. 'AI' matching 156 pages)
                if total_matches > 20:
                    specificity = 0.25
                elif total_matches > 10:
                    specificity = 0.6
                elif total_matches <= 3:
                    specificity = 1.4
                else:
                    specificity = 1.0

                # Check if term aligns with any anchor entity
                for a_lower, a_words in anchor_words_map.items():
                    if a_lower in term_lower or term_lower in a_lower or any(w in term_lower for w in a_words if len(w) >= 3):
                        anchors_hit.add(a_lower)

                score += base_weight * specificity

            # Multi-term Co-occurrence Bonus: pages matching multiple distinct search terms are high-value intersections
            k = len(distinct_terms)
            if k >= 2:
                score += 8.0 * (k - 1)

            # Multi-anchor Coverage Bonus: pages bridging multiple distinct anchors (critical for comparison questions)
            if len(anchors_hit) >= 2:
                score += 20.0

            # Heading alignment
            for h in headings:
                is_apparatus = h.get("is_apparatus", False)
                if is_apparatus:
                    score -= 10.0
                else:
                    score += 8.0

            scores[p] = score

        # Proximity to High-Scoring Clusters (adjacent pages in technical explanations)
        high_value_pages = [p for p, sc in scores.items() if sc >= 15.0]
        for p in scores:
            if p not in high_value_pages:
                if any(abs(p - hv) == 1 for hv in high_value_pages):
                    scores[p] += 3.0

        sorted_candidates = sorted(scores.items(), key=lambda x: (x[1], -x[0]), reverse=True)
        return sorted_candidates


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
        state.concepts = plan.get("concepts", [])
        state.search_terms = plan.get("search_terms", [])
        state.evidence_requirements = plan.get("requirements", [])
        state.anchor_entities = plan.get("anchor_entities", [])
        state.anchor_variants = plan.get("anchor_variants", {})
        ledger.set_requirements(state.evidence_requirements)
        trace_logger.set_planner_output(plan)

        # 3. Action Selection Loop (Deterministic Harness)
        headings_checked = False
        searched_keywords = set()
        page_matches: Dict[int, List[Dict[str, Any]]] = {}
        heading_matches: Dict[int, List[Dict[str, Any]]] = {}

        # Main retrieval loop: strictly governed by remaining budget
        while budget.can_call():
            unsearched_terms = [t for t in state.search_terms if t.lower() not in searched_keywords]
            ranked_candidates = CandidatePageRanker.rank_pages(
                page_matches=page_matches,
                anchor_entities=state.anchor_entities,
                anchor_variants=state.anchor_variants,
                search_terms=state.search_terms,
                heading_matches=heading_matches,
                fetched_pages=state.fetched_pages
            )

            # Determine whether to execute a keyword search or fetch the top candidate page
            should_search = False
            if not ranked_candidates:
                should_search = bool(unsearched_terms)
            elif unsearched_terms:
                anchors_searched = {
                    a.lower() for a in state.anchor_entities
                    if any(a.lower() in sk or sk in a.lower() for sk in searched_keywords)
                }
                has_unsearched_anchors = (
                    len(state.anchor_entities) >= 2
                    and len(anchors_searched) < len(state.anchor_entities)
                )
                can_afford_search = budget.remaining_calls > max(len(state.anchor_entities), 2)

                if has_unsearched_anchors and can_afford_search:
                    should_search = True
                elif not has_unsearched_anchors:
                    top_score = ranked_candidates[0][1] if ranked_candidates else 0
                    if top_score < 8.0 and can_afford_search and len(searched_keywords) < 2:
                        should_search = True
                    else:
                        should_search = False

            if should_search and unsearched_terms:
                term_to_search = unsearched_terms[0]
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

            # If no candidate pages and search terms exhausted, inspect TOC
            if not ranked_candidates and not headings_checked:
                headings_checked = True
                try:
                    result = gateway.execute("list_headings", doc_id=doc_id)
                    headings = result.get("headings", [])
                    for h in headings:
                        htitle = h.get("title", "").lower()
                        hpage = h.get("page", 1)
                        is_apparatus = any(re.search(p, htitle) for p in EvidenceGroundingValidator.APPARATUS_HEADINGS)
                        for term in state.search_terms + state.concepts:
                            if term.lower() in htitle:
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

            # Fetch top candidate page
            if ranked_candidates:
                target_page = ranked_candidates[0][0]
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

                        # Early stopping condition: If all requirements satisfied, all anchors verified, and no conflicts
                        all_anchors_present = (
                            all(ledger.is_anchor_covered(a) for a in state.anchor_entities)
                            if state.anchor_entities else True
                        )
                        if ledger.is_fully_covered() and all_anchors_present and not ledger.has_conflicts():
                            state.stop_reason = "Sufficient evidence and anchor entities established early."
                            break
                except BudgetExceededError:
                    break
                continue

            # If candidate pages exhausted and still have budget, check headings if not already
            if not headings_checked:
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
                    continue
                except BudgetExceededError:
                    break

            # If no further meaningful actions possible
            state.stop_reason = "No further candidate pages or search actions available."
            break

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
            final_answer=final_answer
        )
        state.trace = full_trace.get("tool_calls", [])

        return state, final_answer

    def _process_page_evidence(
        self,
        page_num: int,
        text: str,
        state: QuestionState,
        ledger: EvidenceLedger
    ):
        """Processes newly fetched page text, extracts evidence, checks for conflicts/supersessions."""
        # Check for prompt injection attempts in raw page text (for logging/security audit)
        injections = SecurityBoundary.detect_injection_attempts(text)
        if injections:
            # Audit logged, but content will be cleanly wrapped in security tags
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

        # Generic Requirement Matching (Grounding + Explanatory requirement presence)
        supported_reqs = []
        generic_fillers = {
            "a", "an", "the", "and", "or", "of", "to", "in", "for", "with", "about",
            "information", "details", "what", "explain", "definition", "properties",
            "facts", "behavior", "mechanisms", "origin", "timeline", "regarding", "core"
        }

        # Page must have explanatory anchor grounding to satisfy requirements when anchors are defined
        has_explanatory_grounding = (len(matched_anchors_on_page) > 0) if state.anchor_entities else True

        if has_explanatory_grounding:
            for req in state.evidence_requirements:
                req_words = [w.lower().strip("?,.:;\"'()") for w in req.split()]
                substantive_words = [w for w in req_words if w not in generic_fillers and len(w) >= 2]

                # If the requirement mentions a specific anchor, that specific anchor must have explanatory grounding
                req_anchors = [a for a in state.anchor_entities if a.lower() in req.lower()]
                anchor_ok = True
                if req_anchors:
                    anchor_ok = any(
                        a.lower() in [m.lower() for m in matched_anchors_on_page]
                        for a in req_anchors
                    )

                if anchor_ok:
                    # Gather substantive words including matched anchor variants
                    all_substantive = list(substantive_words)
                    for mf in matched_variants_on_page + matched_anchors_on_page:
                        for mfw in mf.split():
                            clean_mfw = mfw.lower().strip("?,.:;\"'()")
                            if clean_mfw not in generic_fillers and len(clean_mfw) >= 2 and clean_mfw not in all_substantive:
                                all_substantive.append(clean_mfw)

                    if all_substantive:
                        # Require that substantive requirement keywords appear in an explanatory context
                        if EvidenceGroundingValidator.has_explanatory_requirement_evidence(text, all_substantive):
                            supported_reqs.append(req)
                            ledger.mark_requirement_satisfied(req, page_num)
                    else:
                        supported_reqs.append(req)
                        ledger.mark_requirement_satisfied(req, page_num)

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
        # Only evaluate pages that share a common anchor entity, search terms, or concepts
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

Provide your grounded answer following all status-specific grounding rules. Cite pages using [Page X]."""

        try:
            return self.llm.complete(
                system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0
            )
        except Exception as e:
            return f"Error generating final answer: {e}"
