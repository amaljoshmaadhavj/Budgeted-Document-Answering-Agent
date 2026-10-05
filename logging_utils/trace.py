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
        self.answer_status: str = "INSUFFICIENT"
        self.stop_reason: str = ""
        self.final_answer: str = ""

        os.makedirs(self.trace_dir, exist_ok=True)

    def set_question_context(self, question: str, doc_id: str):
        self.question = question
        self.doc_id = doc_id

    def set_planner_output(self, planner_output: Dict[str, Any]):
        self.planner_output = planner_output

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

    def finalize(self, answer_status: str, stop_reason: str, final_answer: str) -> Dict[str, Any]:
        self.answer_status = answer_status
        self.stop_reason = stop_reason
        self.final_answer = final_answer

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
            "planner_output": self.planner_output,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "total_tool_calls": len(self.tool_calls),
            "answer_status": self.answer_status,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
        }
