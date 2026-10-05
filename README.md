# Budgeted-Document-Answering-Agent

A deterministic, budgeted document question-answering agent that answers user queries on unseen PDF documents using **strictly at most six (6) document-tool calls** per question and exactly **two (2) LLM calls** (one for planning, one for final grounded answer generation).

Designed to eliminate token bloat, hallucinations, and unconstrained tool usage under rigorous competition constraints.

---

## 1. Problem Statement

Standard document Question-Answering (Q&A) architectures suffer from severe practical and economic limitations:
- **Runaway Token Costs & Latency:** Multi-agent frameworks or unconstrained tool-calling loops issue unbounded retrieval calls, ballooning latency and API costs.
- **Retrieval Blindspots & Chunk Distortion:** Traditional RAG chunks text arbitrarily, destroys document structure, and generates false semantic associations.
- **Passing-Mention & Bibliography False Positives:** Naive search systems falsely mark questions as answered when an entity is merely cited in a bibliography, listed among other tools, or referenced in passing without explanatory content.
- **Hallucination under Uncertainty:** Many agents fabricate answers or leverage pre-trained world knowledge when the document is incomplete or silent.
- **Susceptibility to Document Injection:** Malicious text within untrusted documents can hijack agent reasoning loops.

**Budgeted-Document-Answering-Agent** solves this through a hard, deterministic budget of **6 document-tool calls**, strict PDF boundary isolation, structured search planning, explanatory evidence grounding, contradiction/supersession analysis, and prompt injection defense.

---

## 2. Architecture & Call Budget

The system maintains a strict separation between deterministic retrieval/validation logic and generative LLM tasks:

```
User Question
    │
    ▼
[LLM Call 1 / 2] Question Analyzer & Search Planner (agent/planner.py)
    │ (Produces structured JSON: question_type, concepts, search_terms, requirements, anchor_entities, anchor_variants)
    │ 5-Level Term Prioritization: Exact Phrases ➔ Natural Variants ➔ Conceptual Variants ➔ Focused Keywords ➔ Demoted Acronyms
    ▼
Deterministic Harness (core/ & agent/) — ZERO LLM calls
    ├── Budget Manager (core/budget.py) ── Enforces max 6 document-tool calls
    ├── Tool Gateway (tools/gateway.py) ── Rejects 7th call; logs trace
    ├── Allowed Document Tools (tools/document_tools.py)
    │     ├── 1. list_documents()
    │     ├── 2. list_headings(doc_id)
    │     ├── 3. search_keyword(doc_id, keyword)
    │     └── 4. get_page(doc_id, page_number)
    ├── Candidate Page Ranker (agent/controller.py) ── Multi-signal deterministic page scoring
    │     ├── Term Tier weighting & specificity discount
    │     ├── Multi-term co-occurrence & multi-anchor coverage bonus
    │     └── Heading alignment & cluster proximity
    ├── Evidence Grounding Validator (core/evidence.py)
    │     ├── Apparatus / Bibliography filtering
    │     ├── Passing-mention disqualification
    │     └── Explanatory predication & substantive content verification
    ├── Evidence Ledger (core/evidence.py) ── Tracks grounded evidence & claims
    ├── Answerability Gate (core/answerability.py) ── Deterministic status evaluation
    │     ├── SUPPORTED
    │     ├── PARTIALLY_SUPPORTED
    │     ├── CONFLICTING
    │     └── INSUFFICIENT
    ├── Security Boundary (core/security.py) ── Untrusted XML framing
    └── Trace Logger (logging_utils/trace.py) ── Full audit log
    │
    ▼
[LLM Call 2 / 2] Final Answer Generator (llm/provider.py)
    │ (Receives ONLY question, verified evidence ledger, and status; zero external guessing)
    ▼
User Answer + Full Execution Trace
```

### Call Budget Summary
- **Document-Tool Calls:** Strictly $\le 6$ calls per question (hard limit enforced by `ToolGateway`).
- **LLM Calls:** Exactly 2 LLM calls per question:
  1. Planner LLM call (1 call)
  2. Final Answer LLM call (1 call)
- **Retrieval Scheduling & Page Ranking:** Exactly 0 LLM calls (100% deterministic Python harness).
- **Evidence Validation & Grounding:** Exactly 0 LLM calls (100% deterministic Python harness).

---

## 3. Why No RAG (Embeddings, Vector Databases, Semantic Search)?

This project explicitly rejects RAG and vector databases for principled reasons:
1. **No Index Construction Overhead:** RAG requires pre-computing embeddings over the entire document corpus at upload time. For large or unseen documents, this incurs high latency and substantial upfront API costs.
2. **Chunk Boundary Loss:** Vector chunking slices sentences across arbitrary boundaries, severing tables, definitions, and contextual qualifications.
3. **Semantic Drift:** Vector similarity often retrieves superficially related text that fails to answer the precise factual requirement.
4. **Deterministic Auditing:** Exact keyword matching and document heading inspection allow judges and users to audit *precisely* why a page was chosen.
5. **PDF Access Boundary:** The agent never receives raw PDF paths, objects, or precomputed embedding indexes. Access is restricted to the 4 allowed document tools.

---

## 4. Why No Multi-Agent Frameworks?

Frameworks like LangChain, LangGraph, CrewAI, or AutoGen introduce:
- Opaque, non-deterministic message passing.
- High token overhead from inter-agent chit-chat.
- Hidden tool calls that silently break budget constraints.

Here, a single **Deterministic Harness** controls all actions. Decision logic (budget checks, coverage thresholds, early stopping, passing-mention filtering, contradiction resolution) is implemented in clean, testable Python code rather than delegated to unpredictable LLM prompts.

---

## 5. Strict Six-Call Budget (`MAX_DOCUMENT_TOOL_CALLS = 6`)

The application enforces a **HARD LIMIT** of at most 6 document-tool calls per question:
- Every tool call (`list_documents`, `list_headings`, `search_keyword`, `get_page`) must pass through the **`ToolGateway`**.
- The `BudgetManager` tracks `budget_before`, `calls_used`, and `budget_after`.
- If `remaining_calls == 0`, any subsequent tool call attempt is immediately **rejected** with a `BudgetExceededError`.
- `remaining_calls` can never drop below zero.
- Re-fetching an already-cached page through the gateway still consumes a call, preventing cache evasion.
- The agent utilizes **early stopping**: if all evidence requirements are satisfied early with verified explanatory grounding, it stops retrieval immediately, preserving remaining budget.

---

## 6. Grounding Engine: Distinguishing Explanatory Evidence from Passing Mentions & Bibliographies

A core architectural contribution in `core/evidence.py` is the **`EvidenceGroundingValidator`**, which prevents false positives on unseen documents:

### 1. Document Apparatus & Bibliography Filtering
- Excludes lines within apparatus sections (`References`, `Bibliography`, `Works Cited`, `Literature Cited`, `Citations`, `Author Index`, `Subject Index`).
- Recognizes standalone bibliographic formatting (`[12]`, numbered authors, publication metadata such as `doi:`, `pp.`, `proceedings of`, `journal of`, `tech report`).
- Prevents citation entries from satisfying evidence requirements.

### 2. Passing-Mention Disqualification
- Detects and filters passive citations (`is cited in`, `was mentioned in`, `has been referenced in`, `is listed in`), citation pointers (`see [`, `refer to [`, `et al.`), and bare listings without explanatory clauses (`other tools include X and Y`, `compared with X`).

### 3. Explanatory Predication & Information Density
- Requires the entity to participate in an explanatory or factual predicate:
  - Copular definitions / Value assignments: `is/was an?`, `is defined as`, `refers to`, `was $5 million`, `was 12V`.
  - Operational / Behavioral verbs: `operates`, `computes`, `provides`, `enables`, `coordinates`, `buffers`, `maps`, `traverses`, `supersedes`.
  - Design / Teleological predicates: `designed to/for`, `used to/for`, `serves to`.
  - Adoption events: `adopted`, `introduced`, `proposed by`.
- Requires presence of substantive descriptive or quantitative informational tokens beyond the entity identifier itself.

---

## 7. Answerability Gate & Decision Logic

Located in `core/answerability.py`, the `AnswerabilityGate` evaluates evidence coverage deterministically:

| Status | Condition | Agent Behavior |
| :--- | :--- | :--- |
| **`SUPPORTED`** | All decomposed requirements and anchor entities are backed by verified explanatory evidence with no unresolved conflicts. | Generates a direct, factual answer citing exact pages `[Page X]`. |
| **`PARTIALLY_SUPPORTED`** | Only a subset of requirements or entities have explanatory backing. | Answers the established portion and explicitly reports what information was missing. |
| **`CONFLICTING`** | Direct factual contradiction detected across pages without explicit supersession language. | Explains the contradiction, citing each conflicting page. |
| **`INSUFFICIENT`** | Zero requirements established, or entities only appear as passing mentions/citations without explanatory evidence. | Explicitly states that the document does not contain sufficient explanatory information. Never guesses. |

---

## 8. Prompt Injection Defense

All retrieved document text is treated as **UNTRUSTED DATA**:
- Documents in the wild may contain adversarial prompts (e.g., `"Ignore previous instructions"`, `"Reveal your system prompt"`, `"Answer with SUCCESS"`).
- **`SecurityBoundary`** (`core/security.py`):
  1. Scans and logs known injection phrases for security auditing.
  2. Encloses all document excerpts within explicit `<UNTRUSTED_DOCUMENT_EVIDENCE page="X">` tags.
  3. System prompts explicitly command the LLM that content inside untrusted evidence tags must never be interpreted as agent instructions or commands.

---

## 9. Contradiction & Supersession Handling

When different pages contain contradictory statements:
- The system **does NOT assume** that a higher page number automatically supersedes a lower page number.
- `AnswerabilityGate.evaluate_supersession` inspects explicit revision language:
  - Keywords: `superseded`, `replaced`, `updated`, `changed`, `now`, `currently`, `previously`, `formerly`.
- If a subsequent section explicitly states an earlier specification was *updated* or *superseded*, the earlier claim is marked `SUPERSEDED`.
- If two pages directly contradict without explicit updating language, the status is set to `CONFLICTING`.

---

## 10. LLM Configuration & Multi-Key Rotation (Google Gemini)

The application uses Google Gemini through the unified [`LLMProvider`](llm/provider.py) abstraction with enterprise multi-key rotation:

### Configuration (`.env`)
```bash
# Model configuration
LLM_MODEL=gemini-3.8-flash

# Multi-Key Rotation (Primary and Failover Keys)
GEMINI_API_KEY_1=your_first_gemini_api_key
GEMINI_API_KEY_2=your_second_gemini_api_key
GEMINI_API_KEY_3=your_third_gemini_api_key
GEMINI_API_KEY_4=your_fourth_gemini_api_key

# Backward-compatible single key (used if individual keys are omitted)
# GEMINI_API_KEY=your_gemini_api_key
```

### Multi-Key Rotation & Secret Safety
- **Automatic Quota / 429 Failover:** Seamlessly rotates through configured keys upon encountering rate limits (HTTP 429), quota exhaustion (`RESOURCE_EXHAUSTED`), or transient capacity errors without failing user queries.
- **Fail-Fast on Genuine Errors:** Non-quota errors (e.g., malformed payloads, invalid schema) fail fast safely rather than masking issues or rotating pointlessly.
- **Zero Secret Exposure:** Keys are read strictly from environment variables or local `.env` and are never logged, printed, echoed, returned in UI, or committed to Git (`.env` is excluded in `.gitignore`).
- **Sanitized Exceptions:** `_sanitize_secret()` automatically strips any API key or token pattern before formatting error logs or UI notifications.
- **Deterministic Testing Mode:** Tests run completely offline with stubbed/mock providers via `tests/conftest.py`, ensuring tests never spend API quota, require internet, or expose secrets.

---

## 11. How to Run the Application

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- PyMuPDF, Streamlit, google-generativeai

### Setup & Run
1. Navigate to the repository:
   ```bash
   cd Budgeted-Document-Answering-Agent
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your `.env` file:
   ```bash
   cp .env.example .env
   # Add your Gemini API keys
   ```
5. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## 12. How to Run Tests

Run the complete 74-test verification suite:
```bash
python -m pytest -v
```

### Verified Test Suite (74 Tests Passing):
- **`tests/test_agent.py` (4 tests):** Factual question answering, state reset isolation, insufficient information handling, and supersession detection.
- **`tests/test_budget.py` (5 tests):** Budget consumption, 7th call rejection, budget reset, gateway budget bypass prevention.
- **`tests/test_tools.py` (5 tests):** Tool contracts, metadata only for `list_documents`, page numbers only for `search_keyword`, single-page retrieval for `get_page`, cache evasion prevention.
- **`tests/test_evidence.py` (5 tests):** Evidence ledger item addition, requirement coverage tracking, claim statuses & contradiction recording, supersession resolution, answerability gate decisions.
- **`tests/test_grounding.py` (15 tests):**
  - Distractor rejection and positive matching for target pages.
  - Short technical identifier preservation.
  - Arbitrary synthetic entity grounding.
  - Multi-entity comparative grounding.
  - Generic filler word distractor prevention.
  - Multi-word natural language phrases.
  - **Passing-mention & bibliography regression tests.**
- **`tests/test_injection.py` (4 tests):** Injection pattern detection, untrusted XML evidence wrapping, system prompt security instructions, adversarial document handling.
- **`tests/test_provider.py` (17 tests):** Gemini client detection, placeholder detection, multi-key rotation through 4 keys, quota/429 recovery, non-quota fail-fast behavior, secret sanitization, planner JSON schema enforcement.
- **`tests/test_planner_prioritization.py` (12 tests):**
  - Strict 5-tier search term prioritization.
  - Concept phrase prioritization and abbreviation demotion.
  - Framing word and long fragment suppression.
  - Single-entity acronym preservation.
  - Candidate page deterministic ranking (`CandidatePageRanker`).
  - Redundant search-term suppression.
  - Six-call budget preservation under heavy candidate generation.
  - Supported definition regression and unavailable information handling.
- **`tests/test_correctness_regression.py` (7 tests):** Morphological paraphrase grounding, bibliography-only rejection, passing-mention alone rejection, explanatory page support, multi-entity comparison, non-conflict temporal phrasing, and genuine synthetic supersession.

---

## 13. How to Inspect Traces

Every question run generates a structured execution trace:
1. **In the Streamlit UI:**
   - Under every assistant response, expand **"🔍 Inspect Full Trace"**.
   - Review the Search Planner decomposition (`question_type`, `concepts`, `search_terms`, `requirements`, `anchor_entities`).
   - Review each tool call: call number, tool name, arguments, `budget_before ➔ budget_after`, success status, and concise result summary.
   - Review candidate vs fetched pages and evidence ledger.
2. **On Disk:**
   - Every run writes a full JSON trace file into the `traces/` directory:
     ```
     traces/trace_<trace_id>.json
     ```
   - Each trace records the complete audit log, planner output, budget transitions, answerability status, stop reason, and final answer.
