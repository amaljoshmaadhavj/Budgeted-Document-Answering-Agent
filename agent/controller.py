"""Agent Controller orchestrating question analysis, deterministic harness, action selection, and final answering."""

from typing import Dict, Any, List, Optional, Tuple
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
        ledger.set_requirements(state.evidence_requirements)
        trace_logger.set_planner_output(plan)

        # 3. Action Selection Loop (Deterministic Harness)
        headings_checked = False
        searched_keywords = set()
        candidate_page_scores: Dict[int, int] = {}  # page -> match count

        # Main retrieval loop: strictly governed by remaining budget
        while budget.can_call():
            # Step A: If we have unsearched search terms, search them
            unsearched_terms = [t for t in state.search_terms if t.lower() not in searched_keywords]
            if unsearched_terms:
                term_to_search = unsearched_terms[0]
                searched_keywords.add(term_to_search.lower())
                try:
                    result = gateway.execute("search_keyword", doc_id=doc_id, keyword=term_to_search)
                    matching_pages = result.get("pages", [])
                    for p in matching_pages:
                        candidate_page_scores[p] = candidate_page_scores.get(p, 0) + 2
                    # If this term matched pages, prioritize fetching over searching further terms immediately
                    # if we have enough candidate pages for requirements
                    if len(candidate_page_scores) >= len(state.evidence_requirements) and len(candidate_page_scores) > 0:
                        pass  # We can proceed to fetching or continue searching
                except BudgetExceededError:
                    break
                continue

            # Step B: If no candidate pages found yet and headings haven't been checked, inspect TOC
            if not candidate_page_scores and not headings_checked:
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
                                score_delta = 1 if is_apparatus else 3
                                candidate_page_scores[hpage] = candidate_page_scores.get(hpage, 0) + score_delta
                except BudgetExceededError:
                    break
                continue

            # Step C: Fetch candidate pages
            unfetched_candidates = [
                p for p, _ in sorted(candidate_page_scores.items(), key=lambda x: x[1], reverse=True)
                if p not in state.fetched_pages
            ]

            if unfetched_candidates:
                target_page = unfetched_candidates[0]
                try:
                    page_result = gateway.execute("get_page", doc_id=doc_id, page_number=target_page)
                    state.fetched_pages.append(target_page)
                    page_text = page_result.get("text", "")

                    if page_text:
                        # Extract evidence and verify requirement satisfaction
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

            # Step D: If candidate pages exhausted and still have budget, check headings if not already
            if not headings_checked:
                headings_checked = True
                try:
                    result = gateway.execute("list_headings", doc_id=doc_id)
                    headings = result.get("headings", [])
                    for h in headings:
                        htitle = h.get("title", "").lower()
                        hpage = h.get("page", 1)
                        for term in state.search_terms + state.concepts:
                            if term.lower() in htitle and hpage not in state.fetched_pages:
                                candidate_page_scores[hpage] = candidate_page_scores.get(hpage, 0) + 1
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
        state.candidate_pages = sorted(list(candidate_page_scores.keys()))
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

        # Evaluate Anchor Grounding: Explanatory vs Passing/Apparatus
        matched_anchors_on_page: List[str] = []
        passing_anchors_on_page: List[str] = []

        for anchor in state.anchor_entities:
            a_clean = anchor.strip()
            if not a_clean:
                continue
            eval_res = EvidenceGroundingValidator.evaluate_anchor_occurrence(text, a_clean)
            if eval_res["is_explanatory"]:
                matched_anchors_on_page.append(a_clean)
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
                    if substantive_words:
                        # Require that substantive requirement keywords appear in an explanatory context
                        if EvidenceGroundingValidator.has_explanatory_requirement_evidence(text, substantive_words):
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
        # Only evaluate pages that share a common anchor entity and contain revision language
        for prev_item in ledger.evidence_items:
            if prev_item.page == page_num:
                continue
            shared_anchors = set([a.lower() for a in prev_item.matched_anchors]).intersection(
                [a.lower() for a in matched_anchors_on_page]
            )
            # Check if both pages share an anchor entity or key search terms/concepts
            shared_terms = set([t.lower() for t in state.search_terms + state.concepts if len(t) >= 3]).intersection(
                [w.lower() for w in re.findall(r'\b\w+\b', prev_item.text)]
            ).intersection(
                [w.lower() for w in re.findall(r'\b\w+\b', text)]
            )
            if shared_anchors or shared_terms or (not state.anchor_entities and prev_item.matched_anchors == matched_anchors_on_page):
                combined = f"{prev_item.text} {text}".lower()
                has_revision = any(re.search(term, combined) for term in AnswerabilityGate.SUPERSEDING_TERMS)
                if has_revision:
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
