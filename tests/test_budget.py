"""Unit tests for BudgetManager and ToolGateway budget enforcement."""

import pytest
import os
import fitz
from core.budget import BudgetManager, BudgetExceededError, MAX_DOCUMENT_TOOL_CALLS
from tools.gateway import ToolGateway
from tools.document_tools import DocumentRegistry
from logging_utils.trace import TraceLogger


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = os.path.join(tmp_path, "sample.pdf")
    doc = fitz.open()
    for i in range(1, 10):
        page = doc.new_page()
        page.insert_text((50, 50), f"This is page {i} content about testing.")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_budget_manager_initialization():
    bm = BudgetManager(max_calls=6)
    assert bm.max_calls == 6
    assert bm.calls_used == 0
    assert bm.remaining_calls == 6
    assert bm.can_call() is True
    assert bm.is_exhausted is False


def test_budget_consumption():
    bm = BudgetManager(max_calls=6)
    before, after = bm.consume_call()
    assert before == 6
    assert after == 5
    assert bm.calls_used == 1
    assert bm.remaining_calls == 5


def test_budget_hard_limit_rejection():
    bm = BudgetManager(max_calls=6)
    for i in range(6):
        assert bm.can_call() is True
        bm.consume_call()

    assert bm.calls_used == 6
    assert bm.remaining_calls == 0
    assert bm.is_exhausted is True
    assert bm.can_call() is False

    # 7th call must be rejected
    with pytest.raises(BudgetExceededError) as exc_info:
        bm.consume_call()
    assert "Budget exceeded" in str(exc_info.value)
    # Remaining calls must never be negative
    assert bm.remaining_calls >= 0


def test_gateway_enforces_budget_and_blocks_seventh_call(sample_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(sample_pdf)

    bm = BudgetManager(max_calls=6)
    trace_logger = TraceLogger()
    gateway = ToolGateway(budget_manager=bm, trace_logger=trace_logger)

    # Make exactly 6 valid calls
    for i in range(1, 7):
        res = gateway.execute("get_page", doc_id=doc_id, page_number=i)
        assert res.get("page") == i
        assert bm.calls_used == i

    assert bm.remaining_calls == 0

    # 7th call MUST be rejected by Gateway
    with pytest.raises(BudgetExceededError):
        gateway.execute("get_page", doc_id=doc_id, page_number=7)

    # Verify trace logger captured the rejected 7th call
    assert len(trace_logger.tool_calls) == 7
    last_call = trace_logger.tool_calls[-1]
    assert last_call.success is False
    assert "rejected" in last_call.result_summary.lower()


def test_budget_reset():
    bm = BudgetManager(max_calls=6)
    for _ in range(4):
        bm.consume_call()
    assert bm.calls_used == 4
    bm.reset()
    assert bm.calls_used == 0
    assert bm.remaining_calls == 6
