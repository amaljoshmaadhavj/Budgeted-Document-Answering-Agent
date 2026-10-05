"""Question Analyzer and Planner module."""

from typing import Dict, Any, List
from agent.prompts import PLANNER_SYSTEM_PROMPT
from llm.provider import LLMProvider


class QuestionPlanner:
    """Analyzes questions prior to retrieval to construct a structured search strategy."""

    VALID_QUESTION_TYPES = {
        "direct_fact",
        "definition",
        "explanation",
        "comparison",
        "multi_page_synthesis",
        "temporal",
        "contradiction_sensitive",
        "likely_absence",
    }

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def plan(self, question: str) -> Dict[str, Any]:
        """Analyzes the question and produces structured requirements and search terms."""
        user_prompt = f"Analyze the following question for document retrieval:\nQuestion: {question.strip()}"
        
        try:
            plan_data = self.llm.complete_json(PLANNER_SYSTEM_PROMPT, user_prompt)
        except Exception:
            plan_data = {}

        return self._normalize_plan(question, plan_data)

    def _normalize_plan(self, question: str, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures all expected keys and types are present in the plan."""
        q_type = plan_data.get("question_type", "direct_fact")
        if q_type not in self.VALID_QUESTION_TYPES:
            q_type = "direct_fact"

        concepts = plan_data.get("concepts", [])
        if not isinstance(concepts, list):
            concepts = [str(concepts)]

        search_terms = plan_data.get("search_terms", [])
        if not isinstance(search_terms, list):
            search_terms = [str(search_terms)]

        requirements = plan_data.get("requirements", [])
        if not isinstance(requirements, list):
            requirements = [str(requirements)]

        anchor_entities = plan_data.get("anchor_entities", [])
        if not isinstance(anchor_entities, list):
            anchor_entities = [str(anchor_entities)]

        # Fallback if empty
        if not search_terms:
            words = [w.strip("?,.:;\"'()") for w in question.split() if len(w) >= 2]
            stopwords = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an", "explain", "compare", "define", "describe", "between", "difference"}
            search_terms = [w for w in words if w.lower() not in stopwords][:3]
            if not search_terms:
                search_terms = [question.strip()]

        if not requirements:
            requirements = [f"Information regarding {t}" for t in search_terms]

        # Clean search terms: deduplicate, strip whitespace, remove empty
        clean_terms = []
        for t in search_terms:
            st = str(t).strip()
            if st and st.lower() not in [x.lower() for x in clean_terms]:
                clean_terms.append(st)

        clean_anchors = []
        for a in anchor_entities:
            sa = str(a).strip()
            if sa and sa.lower() not in [x.lower() for x in clean_anchors]:
                clean_anchors.append(sa)

        if not clean_anchors and clean_terms:
            clean_anchors = clean_terms[:2]

        return {
            "question_type": q_type,
            "concepts": [str(c).strip() for c in concepts if str(c).strip()],
            "search_terms": clean_terms[:5],  # Keep focused
            "requirements": [str(r).strip() for r in requirements if str(r).strip()],
            "anchor_entities": clean_anchors
        }
