"""Unit tests for the 4 document tools and access boundary enforcement."""

import pytest
import os
import fitz
from tools.document_tools import (
    DocumentRegistry,
    list_documents,
    list_headings,
    search_keyword,
    get_page,
)
from core.budget import BudgetManager
from tools.gateway import ToolGateway


@pytest.fixture
def structured_pdf(tmp_path):
    pdf_path = os.path.join(tmp_path, "ai_search.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "Chapter 1: Search Basics. Best-first search evaluates nodes.")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "Chapter 2: Heuristic Methods. A* search uses evaluation function f(n) = g(n) + h(n).")

    p3 = doc.new_page()
    p3.insert_text((50, 50), "Chapter 3: Optimality. A* is admissible when heuristic h is optimistic.")

    doc.set_toc([
        [1, "1. Search Basics", 1],
        [1, "2. Heuristic Methods", 2],
        [1, "3. Optimality", 3],
    ])
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_list_documents(structured_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(structured_pdf, title="AI Search Book")

    res = list_documents()
    assert "documents" in res
    docs = res["documents"]
    assert len(docs) == 1
    doc = docs[0]
    assert doc["doc_id"] == doc_id
    assert doc["title"] == "AI Search Book"
    assert doc["page_count"] == 3
    # Critical constraint: list_documents must NEVER return document text
    assert "text" not in doc
    assert "content" not in doc


def test_list_headings(structured_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(structured_pdf)

    res = list_headings(doc_id)
    assert res["doc_id"] == doc_id
    headings = res["headings"]
    assert len(headings) == 3
    assert headings[0] == {"title": "1. Search Basics", "page": 1}
    assert headings[1] == {"title": "2. Heuristic Methods", "page": 2}
    assert headings[2] == {"title": "3. Optimality", "page": 3}
    # Heading response must not contain document body text
    for h in headings:
        assert "text" not in h


def test_search_keyword_returns_only_page_numbers(structured_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(structured_pdf)

    # Search for term present on page 2 and 3
    res = search_keyword(doc_id, "heuristic")
    assert res["keyword"] == "heuristic"
    assert res["pages"] == [2, 3]

    # Search for term present only on page 1
    res2 = search_keyword(doc_id, "Best-first")
    assert res2["pages"] == [1]

    # Search for absent term
    res3 = search_keyword(doc_id, "quantum")
    assert res3["pages"] == []

    # CRITICAL CONSTRAINT: search_keyword MUST NOT return snippets or text
    assert "snippet" not in res
    assert "snippets" not in res
    assert "text" not in res
    assert "content" not in res


def test_get_page_returns_single_page(structured_pdf):
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(structured_pdf)

    res = get_page(doc_id, 2)
    assert res["doc_id"] == doc_id
    assert res["page"] == 2
    assert "A* search uses evaluation function" in res["text"]
    assert "Chapter 1: Search Basics" not in res["text"]

    # Bounds checking
    err_res = get_page(doc_id, 99)
    assert "error" in err_res


def test_page_caching_cannot_bypass_budget(structured_pdf):
    """Re-fetching the same page through the gateway still consumes a tool call."""
    registry = DocumentRegistry.get_instance()
    registry.clear()
    doc_id = registry.register_pdf(structured_pdf)

    bm = BudgetManager(max_calls=6)
    gw = ToolGateway(budget_manager=bm)

    # Call get_page(1) twice
    gw.execute("get_page", doc_id=doc_id, page_number=1)
    assert bm.calls_used == 1

    gw.execute("get_page", doc_id=doc_id, page_number=1)
    assert bm.calls_used == 2
    assert bm.remaining_calls == 4
