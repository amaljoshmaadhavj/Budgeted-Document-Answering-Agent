"""Document tools implementation.

CRITICAL ARCHITECTURAL BOUNDARY:
- PyMuPDF (fitz) is used ONLY internally inside this module.
- The agent NEVER receives raw PDF paths, bytes, fitz document/page objects,
  full text corpora, or precomputed indexes.
- All page reads and keyword searches are performed lazily on demand.
- NO prefetching, NO caching intended to evade budgets.
"""

from typing import Dict, Any, List, Optional
import os
import fitz  # PyMuPDF


class DocumentRegistry:
    """Internal registry holding file paths and basic metadata for uploaded documents.
    
    This registry is private to the tools module and is NEVER exposed directly to the agent.
    """
    _instance: Optional["DocumentRegistry"] = None

    def __init__(self):
        # Maps doc_id -> {"file_path": str, "title": str, "page_count": int}
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._doc_counter = 0

    @classmethod
    def get_instance(cls) -> "DocumentRegistry":
        if cls._instance is None:
            cls._instance = DocumentRegistry()
        return cls._instance

    def register_pdf(self, file_path: str, title: Optional[str] = None) -> str:
        """Register a PDF file lazily. Only inspects page count without extracting text."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        self._doc_counter += 1
        doc_id = f"doc_{self._doc_counter}"
        doc_title = title or os.path.basename(file_path)

        with fitz.open(file_path) as doc:
            page_count = len(doc)

        self._documents[doc_id] = {
            "file_path": file_path,
            "title": doc_title,
            "page_count": page_count,
        }
        return doc_id

    def get_doc_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._documents.get(doc_id)

    def clear(self):
        """Clear all registered documents."""
        self._documents.clear()
        self._doc_counter = 0


# ==============================================================================
# ALLOWED DOCUMENT TOOLS (EXACTLY 4 INTERFACES)
# ==============================================================================

def list_documents() -> Dict[str, Any]:
    """Returns only document metadata for all registered documents.
    
    Example response:
    {
      "documents": [
        {
          "doc_id": "doc_1",
          "title": "example.pdf",
          "page_count": 204
        }
      ]
    }
    """
    registry = DocumentRegistry.get_instance()
    docs = []
    for doc_id, info in registry._documents.items():
        docs.append({
            "doc_id": doc_id,
            "title": info["title"],
            "page_count": info["page_count"]
        })
    return {"documents": docs}


def list_headings(doc_id: str) -> Dict[str, Any]:
    """Returns only headings/table of contents and page references for a document.
    
    Example response:
    {
      "doc_id": "doc_1",
      "headings": [
        {
          "title": "1.1 Introduction",
          "page": 1
        }
      ]
    }
    """
    registry = DocumentRegistry.get_instance()
    info = registry.get_doc_info(doc_id)
    if not info:
        return {"error": f"Document '{doc_id}' not found.", "doc_id": doc_id, "headings": []}

    file_path = info["file_path"]
    headings: List[Dict[str, Any]] = []

    with fitz.open(file_path) as doc:
        toc = doc.get_toc()  # returns list of [level, title, page, ...]
        for item in toc:
            if len(item) >= 3:
                headings.append({
                    "title": str(item[1]).strip(),
                    "page": int(item[2])
                })

    return {
        "doc_id": doc_id,
        "headings": headings
    }


def search_keyword(doc_id: str, keyword: str) -> Dict[str, Any]:
    """Returns only matching page numbers for a keyword in a document.
    
    DOES NOT RETURN SNIPPETS OR TEXT. ONLY PAGE NUMBERS.
    
    Example response:
    {
      "keyword": "heuristic",
      "pages": [21, 22, 23]
    }
    """
    if not keyword or not keyword.strip():
        return {"keyword": keyword, "pages": []}

    clean_kw = keyword.strip()
    registry = DocumentRegistry.get_instance()
    info = registry.get_doc_info(doc_id)
    if not info:
        return {"error": f"Document '{doc_id}' not found.", "keyword": clean_kw, "pages": []}

    file_path = info["file_path"]
    matching_pages: List[int] = []

    # Perform lazy page-by-page search
    with fitz.open(file_path) as doc:
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            # PyMuPDF search_for matches text case-insensitively or according to flags
            # Case-insensitive by default in fitz or search_for flags
            rects = page.search_for(clean_kw)
            if rects:
                # 1-indexed page number
                matching_pages.append(page_idx + 1)

    return {
        "keyword": clean_kw,
        "pages": matching_pages
    }


def get_page(doc_id: str, page_number: int) -> Dict[str, Any]:
    """Returns exactly one page of text.
    
    No ranges. Extracted lazily on demand.
    
    Example response:
    {
      "doc_id": "doc_1",
      "page": 22,
      "text": "..."
    }
    """
    registry = DocumentRegistry.get_instance()
    info = registry.get_doc_info(doc_id)
    if not info:
        return {"error": f"Document '{doc_id}' not found.", "doc_id": doc_id, "page": page_number, "text": ""}

    page_count = info["page_count"]
    if page_number < 1 or page_number > page_count:
        return {
            "error": f"Page number {page_number} out of bounds (1 to {page_count}).",
            "doc_id": doc_id,
            "page": page_number,
            "text": ""
        }

    file_path = info["file_path"]
    with fitz.open(file_path) as doc:
        page = doc.load_page(page_number - 1)
        text = page.get_text("text")

    return {
        "doc_id": doc_id,
        "page": page_number,
        "text": text
    }
