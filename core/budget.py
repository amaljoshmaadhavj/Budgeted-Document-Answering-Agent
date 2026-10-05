"""Budget management for document tool invocations.

Enforces the strict competition constraint:
MAX_DOCUMENT_TOOL_CALLS = 6.
Never allow remaining_calls < 0.
"""

from typing import Tuple


MAX_DOCUMENT_TOOL_CALLS: int = 6


class BudgetExceededError(Exception):
    """Raised when an attempt is made to call a document tool after budget is exhausted."""
    pass


class BudgetManager:
    """Manages and enforces the 6-call budget for document tools."""

    def __init__(self, max_calls: int = MAX_DOCUMENT_TOOL_CALLS):
        self.max_calls = max_calls
        self.calls_used = 0

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.calls_used)

    @property
    def is_exhausted(self) -> bool:
        return self.remaining_calls <= 0

    def can_call(self) -> bool:
        return self.remaining_calls > 0

    def consume_call(self) -> Tuple[int, int]:
        """Consumes one call from the budget.
        
        Returns:
            Tuple of (budget_before, budget_after)
            
        Raises:
            BudgetExceededError if budget is exhausted (remaining_calls <= 0).
        """
        if self.is_exhausted:
            raise BudgetExceededError(
                f"Budget exceeded: maximum {self.max_calls} document-tool calls reached."
            )

        budget_before = self.remaining_calls
        self.calls_used += 1
        budget_after = self.remaining_calls
        return budget_before, budget_after

    def reset(self):
        """Resets the budget for a new question."""
        self.calls_used = 0
