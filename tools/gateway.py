"""Tool Gateway enforcing the hard limit of 6 document-tool calls per question.

Every document tool invocation MUST pass through this gateway.
The gateway logs execution details to the trace logger and enforces budget boundaries.
"""

from typing import Dict, Any, Optional
from core.budget import BudgetManager, BudgetExceededError
from logging_utils.trace import TraceLogger
from tools.document_tools import (
    list_documents,
    list_headings,
    search_keyword,
    get_page,
)


class ToolGateway:
    """Deterministic gateway regulating access to document tools with strict budget tracking."""


    ALLOWED_TOOLS = {
        "list_documents": list_documents,
        "list_headings": list_headings,
        "search_keyword": search_keyword,
        "get_page": get_page,
    }

    def __init__(self, budget_manager: BudgetManager, trace_logger: Optional[TraceLogger] = None):
        self.budget_manager = budget_manager
        self.trace_logger = trace_logger or TraceLogger()

    def set_trace_logger(self, trace_logger: TraceLogger):
        self.trace_logger = trace_logger

    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a document tool under budget governance.
        
        Args:
            tool_name: One of 'list_documents', 'list_headings', 'search_keyword', 'get_page'.
            **kwargs: Arguments to the tool.
            
        Returns:
            Dict containing the tool result or an error dict.
            
        Raises:
            BudgetExceededError: When remaining_calls == 0.
            ValueError: If tool_name is not an allowed document tool.
        """
        if tool_name not in self.ALLOWED_TOOLS:
            raise ValueError(
                f"Unauthorized tool '{tool_name}'. Allowed tools: {list(self.ALLOWED_TOOLS.keys())}"
            )

        if not self.budget_manager.can_call():
            err_msg = f"Budget exceeded: maximum {self.budget_manager.max_calls} document-tool calls reached."
            if self.trace_logger:
                self.trace_logger.log_tool_call(
                    call_number=self.budget_manager.calls_used + 1,
                    tool=tool_name,
                    arguments=kwargs,
                    budget_before=0,
                    budget_after=0,
                    success=False,
                    result_summary="Call rejected (budget exhausted)",
                    error=err_msg
                )
            raise BudgetExceededError(err_msg)

        # Consume budget
        budget_before, budget_after = self.budget_manager.consume_call()
        call_num = self.budget_manager.calls_used
        func = self.ALLOWED_TOOLS[tool_name]

        try:
            result = func(**kwargs)
            summary = self._summarize_result(tool_name, kwargs, result)
            has_error = isinstance(result, dict) and "error" in result
            self.trace_logger.log_tool_call(
                call_number=call_num,
                tool=tool_name,
                arguments=kwargs,
                budget_before=budget_before,
                budget_after=budget_after,
                success=not has_error,
                result_summary=summary,
                error=result.get("error") if has_error else None
            )
            return result
        except Exception as e:
            err_msg = str(e)
            self.trace_logger.log_tool_call(
                call_number=call_num,
                tool=tool_name,
                arguments=kwargs,
                budget_before=budget_before,
                budget_after=budget_after,
                success=False,
                result_summary=f"Execution error: {err_msg}",
                error=err_msg
            )
            raise

    def _summarize_result(self, tool_name: str, args: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Create a concise, non-leaking summary for the trace log."""
        if not isinstance(result, dict):
            return "Non-dictionary response received"

        if "error" in result:
            return f"Error: {result['error']}"

        if tool_name == "list_documents":
            docs = result.get("documents", [])
            return f"Found {len(docs)} documents"
        elif tool_name == "list_headings":
            headings = result.get("headings", [])
            return f"Found {len(headings)} headings in TOC"
        elif tool_name == "search_keyword":
            pages = result.get("pages", [])
            kw = args.get("keyword", "")
            return f"Found {len(pages)} matching page(s) for '{kw}': {pages}"
        elif tool_name == "get_page":
            page = result.get("page", args.get("page_number"))
            text = result.get("text", "")
            char_count = len(text)
            return f"Retrieved page {page} ({char_count} characters)"
        return "Tool executed successfully"
