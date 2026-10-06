"""Trace logger for document tool calls and agent reasoning workflow."""

from typing import Dict, Any, List, Optional
import json
import os
import time
import uuid


class ToolCallTrace:
    """Represents a single document tool execution trace entry."""

    def __init__(
        self,
        call_number: int,
        tool: str,
        arguments: Dict[str, Any],
        budget_before: int,
        budget_after: int,
        success: bool,
        result_summary: str,
        error: Optional[str] = None
    ):
        self.call_number = call_number
        self.tool = tool
        self.arguments = arguments
        self.budget_before = budget_before
        self.budget_after = budget_after
        self.success = success
        self.result_summary = result_summary
        self.error = error
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "call_number": self.call_number,
            "tool": self.tool,
            "arguments": self.arguments,
            "budget_before": self.budget_before,
            "budget_after": self.budget_after,
            "success": self.success,
            "result_summary": self.result_summary,
            "timestamp": self.timestamp,
        }
        if self.error:
            data["error"] = self.error
        return data


class TraceLogger:
    """Manages recording, formatting, and saving traces for agent executions."""

    def __init__(self, trace_dir: str = "traces"):
        self.trace_dir = trace_dir
        self.trace_id = str(uuid.uuid4())[:8]
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.question: str = ""
        self.doc_id: str = ""
        self.planner_output: Dict[str, Any] = {}
        self.tool_calls: List[ToolCallTrace] = []
        self.retrieval_decisions: List[Dict[str, Any]] = []
        self.answer_status: str = "INSUFFICIENT"
        self.stop_reason: str = ""
        self.final_answer: str = ""
        self.final_answer_consistency: str = "CONSISTENT"

        os.makedirs(self.trace_dir, exist_ok=True)

    def set_question_context(self, question: str, doc_id: str):
        self.question = question
        self.doc_id = doc_id

    def set_planner_output(self, planner_output: Dict[str, Any]):
        self.planner_output = planner_output

    def log_retrieval_decision(
        self,
        decision_number: int,
        budget_remaining: int,
        candidates_available: List[int],
        candidate_ranking: List[Dict[str, Any]],
        action: str,
        target: Any,
        rationale: str,
        unsatisfied_requirements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        # Standardize decision label: SEARCH, FETCH, HEADINGS, STOP
        action_map = {
            "search_keyword": "SEARCH",
            "get_page": "FETCH",
            "list_headings": "HEADINGS",
            "stop": "STOP"
        }
        decision = action_map.get(action, action.upper())
        candidate_count = len(candidates_available)
        top_ranked_candidate = candidate_ranking[0] if candidate_ranking else None

        decision_entry = {
            "decision_number": decision_number,
            "remaining_budget": budget_remaining,
            "budget_remaining": budget_remaining,
            "candidate_count": candidate_count,
            "candidates_available": candidates_available,
            "top_ranked_candidate": top_ranked_candidate,
            "candidate_ranking": candidate_ranking,
            "decision": decision,
            "action": action,
            "target": target,
            "reason": rationale,
            "rationale": rationale,
            "unsatisfied_requirements": unsatisfied_requirements or [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.retrieval_decisions.append(decision_entry)
        return decision_entry

    def log_tool_call(
        self,
        call_number: int,
        tool: str,
        arguments: Dict[str, Any],
        budget_before: int,
        budget_after: int,
        success: bool,
        result_summary: str,
        error: Optional[str] = None
    ) -> ToolCallTrace:
        trace_entry = ToolCallTrace(
            call_number=call_number,
            tool=tool,
            arguments=arguments,
            budget_before=budget_before,
            budget_after=budget_after,
            success=success,
            result_summary=result_summary,
            error=error
        )
        self.tool_calls.append(trace_entry)
        return trace_entry

    def finalize(
        self,
        answer_status: str,
        stop_reason: str,
        final_answer: str,
        final_answer_consistency: str = "CONSISTENT"
    ) -> Dict[str, Any]:
        self.answer_status = answer_status
        self.stop_reason = stop_reason
        self.final_answer = final_answer
        self.final_answer_consistency = final_answer_consistency

        full_trace = self.to_dict()
        file_path = os.path.join(self.trace_dir, f"trace_{self.trace_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(full_trace, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to write trace file: {e}")

        return full_trace

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.start_time,
            "question": self.question,
            "doc_id": self.doc_id,
            "question_type": self.planner_output.get("question_type", "direct_fact"),
            "requires_document_evidence": self.planner_output.get("requires_document_evidence", True),
            "requirement_facets": self.planner_output.get("requirement_facets", []),
            "planner_output": self.planner_output,
            "retrieval_decisions": self.retrieval_decisions,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "total_tool_calls": len(self.tool_calls),
            "answer_status": self.answer_status,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "final_answer_consistency": self.final_answer_consistency
        }
