"""Streamlit Web Application for Budgeted-Document-Answering-Agent.

Enforces:
- Strict 6-call document-tool budget per question.
- Full trace inspection.
- Answerability status indicators.
- Lazy PDF handling (no bulk prefetching or semantic indexing).
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from tools.document_tools import DocumentRegistry, list_documents
from agent.controller import AgentController
from llm.provider import LLMProvider
from core.budget import MAX_DOCUMENT_TOOL_CALLS

st.set_page_config(
    page_title="Budgeted Document Answering Agent",
    page_icon="📄",
    layout="wide"
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None
if "current_doc_info" not in st.session_state:
    st.session_state.current_doc_info = None

# Sidebar: PDF Upload & System Settings
with st.sidebar:
    st.title("📄 Document Portal")
    st.markdown("Upload an unseen PDF. The agent operates under a **hard budget of 6 document-tool calls** per question.")

    # API Key Configuration
    with st.expander("⚙️ LLM Configuration", expanded=False):
        user_gemini_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        selected_model = st.text_input("Model Name", value=os.getenv("LLM_MODEL", "gemini-3.8-flash"))
        
        if user_gemini_key:
            os.environ["GEMINI_API_KEY"] = user_gemini_key
        if selected_model:
            os.environ["LLM_MODEL"] = selected_model

    # File Uploader
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        # Check if this is a newly uploaded file or different from session
        file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("uploaded_sig") != file_signature:
            # Save uploaded PDF to a temporary file
            upload_dir = os.path.join(tempfile.gettempdir(), "budgeted_agent_uploads")
            os.makedirs(upload_dir, exist_ok=True)
            saved_path = os.path.join(upload_dir, uploaded_file.name)
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Register PDF lazily
            registry = DocumentRegistry.get_instance()
            doc_id = registry.register_pdf(saved_path, title=uploaded_file.name)
            info = registry.get_doc_info(doc_id)

            st.session_state.current_doc_id = doc_id
            st.session_state.current_doc_info = info
            st.session_state.uploaded_sig = file_signature
            st.session_state.chat_history = []  # Reset chat for new document
            st.success(f"Registered {uploaded_file.name} as `{doc_id}`")

    # Document Metadata Display
    if st.session_state.current_doc_info:
        info = st.session_state.current_doc_info
        st.divider()
        st.subheader("📋 Document Metadata")
        st.markdown(f"**Doc ID:** `{st.session_state.current_doc_id}`")
        st.markdown(f"**Title:** {info['title']}")
        st.markdown(f"**Total Pages:** {info['page_count']}")

        st.divider()
        st.subheader("🛡️ Budget Governance")
        st.info(f"**Max Document-Tool Calls:** {MAX_DOCUMENT_TOOL_CALLS}\n\n**Allowed Tools:** `list_documents`, `list_headings`, `search_keyword`, `get_page`")

        if st.button("Clear Document & History", use_container_width=True):
            DocumentRegistry.get_instance().clear()
            st.session_state.current_doc_id = None
            st.session_state.current_doc_info = None
            st.session_state.uploaded_sig = None
            st.session_state.chat_history = []
            st.rerun()

# Main Application Area
st.title("⚖️ Budgeted Document Answering Agent")
st.caption("Deterministic tool orchestration • Strict 6-call budget • No RAG/embeddings • Prompt injection defense")

if not st.session_state.current_doc_id:
    st.info("👈 Please upload a PDF document in the sidebar to begin.")
    st.stop()

# Display Chat History
for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(item["question"])

    with st.chat_message("assistant"):
        # Status header badge
        status = item["answer_status"]
        budget_used = item["budget_used"]
        badge_color = {
            "SUPPORTED": "🟢",
            "PARTIALLY_SUPPORTED": "🟡",
            "CONFLICTING": "🔴",
            "INSUFFICIENT": "⚪"
        }.get(status, "⚪")

        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.markdown(f"**Status:** {badge_color} `{status}`")
        with col2:
            st.markdown(f"**Budget:** `{budget_used} / {MAX_DOCUMENT_TOOL_CALLS} calls used`")
        with col3:
            st.markdown(f"**Stop Reason:** *{item.get('stop_reason', '')}*")

        st.markdown(item["answer"])

        # Trace Expander
        with st.expander(f"🔍 Inspect Full Trace (Call Log: {len(item.get('trace', []))} tool executions)"):
            st.markdown("#### 1. Search Planner Output")
            st.json(item.get("planner_output", {}))

            st.markdown("#### 2. Tool Execution Log")
            trace_calls = item.get("trace", [])
            if trace_calls:
                for call in trace_calls:
                    success_icon = "✅" if call.get("success") else "❌"
                    st.markdown(
                        f"**Call {call.get('call_number')}:** {success_icon} `{call.get('tool')}` "
                        f"(Budget: {call.get('budget_before')} ➔ {call.get('budget_after')})"
                    )
                    st.markdown(f"- **Arguments:** `{call.get('arguments')}`")
                    st.markdown(f"- **Result Summary:** {call.get('result_summary')}")
                    if call.get("error"):
                        st.error(f"Error: {call.get('error')}")
                    st.markdown("---")
            else:
                st.write("No document tools were called.")

            st.markdown("#### 3. Evidence Ledger Summary")
            st.json({
                "candidate_pages": item.get("candidate_pages", []),
                "fetched_pages": item.get("fetched_pages", []),
                "evidence_count": len(item.get("evidence", [])),
                "contradictions": item.get("contradictions", [])
            })

# Question Input
if user_question := st.chat_input("Ask a question about the document..."):
    # Display user question immediately
    with st.chat_message("user"):
        st.markdown(user_question)

    # Process via Agent Controller
    with st.chat_message("assistant"):
        with st.spinner("Analyzing question and querying document under budget limit..."):
            provider = LLMProvider()
            controller = AgentController(llm_provider=provider)
            state, answer = controller.answer_question(
                question=user_question,
                doc_id=st.session_state.current_doc_id
            )

        # Status badge
        badge_color = {
            "SUPPORTED": "🟢",
            "PARTIALLY_SUPPORTED": "🟡",
            "CONFLICTING": "🔴",
            "INSUFFICIENT": "⚪"
        }.get(state.answer_status, "⚪")

        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.markdown(f"**Status:** {badge_color} `{state.answer_status}`")
        with col2:
            st.markdown(f"**Budget:** `{state.budget_used} / {MAX_DOCUMENT_TOOL_CALLS} calls used`")
        with col3:
            st.markdown(f"**Stop Reason:** *{state.stop_reason}*")

        st.markdown(answer)

        # Show trace expander
        with st.expander(f"🔍 Inspect Full Trace (Call Log: {len(state.trace)} tool executions)"):
            st.markdown("#### 1. Search Planner Output")
            st.json({
                "question_type": state.question_type,
                "concepts": state.concepts,
                "search_terms": state.search_terms,
                "requirements": state.evidence_requirements
            })

            st.markdown("#### 2. Tool Execution Log")
            if state.trace:
                for call in state.trace:
                    success_icon = "✅" if call.get("success") else "❌"
                    st.markdown(
                        f"**Call {call.get('call_number')}:** {success_icon} `{call.get('tool')}` "
                        f"(Budget: {call.get('budget_before')} ➔ {call.get('budget_after')})"
                    )
                    st.markdown(f"- **Arguments:** `{call.get('arguments')}`")
                    st.markdown(f"- **Result Summary:** {call.get('result_summary')}")
                    if call.get("error"):
                        st.error(f"Error: {call.get('error')}")
                    st.markdown("---")
            else:
                st.write("No document tools were called.")

            st.markdown("#### 3. Evidence Ledger Summary")
            st.json({
                "candidate_pages": state.candidate_pages,
                "fetched_pages": state.fetched_pages,
                "evidence": state.evidence,
                "claims": state.claims,
                "contradictions": state.contradictions
            })

        # Append to session history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer,
            "answer_status": state.answer_status,
            "budget_used": state.budget_used,
            "stop_reason": state.stop_reason,
            "candidate_pages": state.candidate_pages,
            "fetched_pages": state.fetched_pages,
            "evidence": state.evidence,
            "claims": state.claims,
            "contradictions": state.contradictions,
            "planner_output": {
                "question_type": state.question_type,
                "concepts": state.concepts,
                "search_terms": state.search_terms,
                "requirements": state.evidence_requirements
            },
            "trace": state.trace
        })
