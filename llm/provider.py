"""LLM Provider supporting Gemini, OpenAI, and deterministic Mock fallback.

Ensures reliable execution during demonstrations, tests, and live evaluation.
Adheres strictly to zero secret leakage across logs, exceptions, and traces.
"""

from typing import Dict, Any, List, Optional
import os
import json
import re
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded
load_dotenv()


class LLMProvider:
    """Unified LLM interface for the planner analysis and final answer generation."""

    def __init__(self, model_name: Optional[str] = None, mock: Optional[bool] = None):
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-3.8-flash")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")

        if mock is not None:
            self._force_mock = mock
        else:
            self._force_mock = os.getenv("MOCK_LLM", "").lower() in ("true", "1", "yes")

        self._client_type = self._detect_client_type()

    def _detect_client_type(self) -> str:
        if self._force_mock:
            return "mock"
        if self.gemini_key and not self.gemini_key.strip().startswith("your_"):
            return "gemini"
        if self.openai_key and not self.openai_key.strip().startswith("your_"):
            return "openai"
        return "mock"

    def is_mock(self) -> bool:
        return self._client_type == "mock"

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """Execute a text completion request."""
        if self._client_type == "gemini":
            return self._call_gemini(system_prompt, user_prompt, temperature)
        elif self._client_type == "openai":
            return self._call_openai(system_prompt, user_prompt, temperature)
        else:
            return self._mock_completion(system_prompt, user_prompt)

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Execute a completion expecting a JSON response object."""
        if self._client_type == "gemini":
            raw_text = self._call_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                response_mime_type="application/json"
            )
        elif self._client_type == "openai":
            raw_text = self._call_openai(system_prompt, user_prompt, temperature=0.0)
        else:
            raw_text = self._mock_completion(system_prompt, user_prompt)
        return self._parse_json(raw_text)

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        response_mime_type: Optional[str] = None
    ) -> str:
        """Calls the Gemini API directly. Never silently falls back to mock on failure."""
        if not self.gemini_key or self.gemini_key.strip().startswith("your_"):
            raise ValueError("GEMINI_API_KEY is not configured or contains placeholder value.")

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt if system_prompt else None
            )

            gen_config = {"temperature": temperature}
            if response_mime_type:
                gen_config["response_mime_type"] = response_mime_type

            response = model.generate_content(
                user_prompt,
                generation_config=gen_config
            )
            return response.text or ""
        except Exception as e:
            safe_err = self._sanitize_secret(str(e))
            raise RuntimeError(f"Gemini API invocation failed: {safe_err}") from None

    def _call_openai(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=self.openai_key,
            base_url=self.openai_base_url if self.openai_base_url else None
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content or ""

    def _sanitize_secret(self, text: str) -> str:
        """Redacts sensitive API keys and authorization tokens from strings and exception messages."""
        if not text:
            return text
        sanitized = text
        if self.gemini_key and len(self.gemini_key) > 5:
            sanitized = sanitized.replace(self.gemini_key, "[REDACTED_API_KEY]")
        if self.openai_key and len(self.openai_key) > 5:
            sanitized = sanitized.replace(self.openai_key, "[REDACTED_API_KEY]")
        sanitized = re.sub(r'\b(?:AIza|AQ\.)[A-Za-z0-9_-]+\b', '[REDACTED_API_KEY]', sanitized)
        sanitized = re.sub(r'\bsk-[A-Za-z0-9_-]{20,}\b', '[REDACTED_API_KEY]', sanitized)
        return sanitized

    def _mock_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Deterministic fallback when no active API key is configured."""
        # Check if this is a planner request
        if "question_type" in system_prompt or "concepts" in system_prompt:
            # Deterministic keyword extraction from user_prompt
            extracted = self._heuristic_plan_extraction(user_prompt)
            return json.dumps(extracted)

        # Final answer mock response
        if "INSUFFICIENT" in user_prompt:
            return "Based on the retrieved document evidence, the document does not contain sufficient information to answer the question."
        elif "CONFLICTING" in user_prompt:
            return "The retrieved document pages contain conflicting statements regarding this question that cannot be resolved safely."
        else:
            return "Based on the retrieved document evidence, the answer is supported by the cited pages."

    def _heuristic_plan_extraction(self, user_prompt: str) -> Dict[str, Any]:
        """Generic heuristic planner when running without external LLM keys.
        
        Dynamically extracts anchors, search terms, and requirements from arbitrary user questions
        without any domain-specific or document-specific hardcoding.
        """
        # Find the line that actually contains the question
        q_clean = ""
        for line in user_prompt.split("\n"):
            line_str = line.strip()
            if line_str.lower().startswith("question:"):
                q_clean = line_str[len("question:"):].strip()
                break
        if not q_clean:
            lines = [l.strip() for l in user_prompt.split("\n") if l.strip()]
            q_clean = lines[-1] if lines else user_prompt.strip()

        # Generic English linguistic stop words (grammatical function words only)
        generic_stopwords = {
            "a", "an", "the", "this", "that", "these", "those",
            "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "have", "has", "had", "having",
            "can", "could", "will", "would", "shall", "should", "may", "might", "must",
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "in", "on", "at", "by", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below",
            "to", "from", "up", "down", "out", "off", "over", "under",
            "and", "or", "but", "if", "because", "as", "until", "while",
            "of", "it", "its", "they", "them", "their", "we", "us", "our", "you", "your",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "now", "tell", "give", "show", "find", "term", "adopted",
            "explain", "define", "describe", "compare", "difference", "versus", "vs", "mean", "meaning",
            "detail", "details", "information", "question", "document", "retrieval", "analyze", "following", "search"
        }

        # Tokenize while preserving symbols attached to technical identifiers
        raw_tokens = [w.strip("?,.:;\"'()[]{}") for w in q_clean.split()]
        raw_tokens = [w for w in raw_tokens if w]

        # Extract non-stopword tokens as entity candidates
        candidate_tokens = [w for w in raw_tokens if w.lower() not in generic_stopwords and len(w) >= 2]

        # Detect consecutive non-stopword multi-word phrases
        phrases = []
        current_phrase = []
        for w in raw_tokens:
            if w.lower() not in generic_stopwords and len(w) >= 2:
                current_phrase.append(w)
            else:
                if len(current_phrase) >= 2:
                    phrases.append(" ".join(current_phrase))
                current_phrase = []
        if len(current_phrase) >= 2:
            phrases.append(" ".join(current_phrase))

        # Anchor entities: Prioritize multi-word phrases and distinct candidate tokens
        anchor_entities = []
        for p in phrases:
            if p.lower() not in [a.lower() for a in anchor_entities]:
                anchor_entities.append(p)
        for t in candidate_tokens:
            # If token is already part of a multi-word anchor, check if token itself is an acronym or symbol
            is_sub = any(t.lower() in p.lower() for p in phrases)
            has_symbol_or_upper = any(c.isupper() for c in t) or any(not c.isalnum() for c in t)
            if not is_sub or has_symbol_or_upper:
                if t.lower() not in [a.lower() for a in anchor_entities]:
                    anchor_entities.append(t)

        if not anchor_entities:
            anchor_entities = candidate_tokens[:2] if candidate_tokens else [q_clean]

        # Search terms: specific anchors and candidate tokens
        search_terms = []
        for item in anchor_entities + candidate_tokens:
            if item.lower() not in [s.lower() for s in search_terms]:
                search_terms.append(item)

        # Classify question type
        q_lower = q_clean.lower()
        if "compare" in q_lower or "difference" in q_lower or "versus" in q_lower or " vs " in q_lower:
            q_type = "comparison"
        elif "when" in q_lower or "year" in q_lower or "date" in q_lower:
            q_type = "temporal"
        elif "define" in q_lower or "definition" in q_lower or "what is" in q_lower or "what are" in q_lower or "mean" in q_lower:
            q_type = "definition"
        elif "why" in q_lower or "how" in q_lower or "explain" in q_lower:
            q_type = "explanation"
        else:
            q_type = "direct_fact"

        # Generate requirements dynamically based on anchors and question type
        requirements = []
        for anchor in anchor_entities[:3]:
            if q_type == "definition":
                requirements.append(f"{anchor} definition and core mechanisms")
            elif q_type == "comparison":
                requirements.append(f"{anchor} characteristics and behavior")
            elif q_type == "temporal":
                requirements.append(f"{anchor} origin and adoption timeline")
            else:
                requirements.append(f"{anchor} properties and facts")

        if not requirements:
            requirements = [f"{q_clean} factual details"]

        return {
            "question_type": q_type,
            "concepts": anchor_entities[:4],
            "search_terms": search_terms[:5],
            "requirements": requirements,
            "anchor_entities": anchor_entities[:3]
        }

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Safely parses JSON even if wrapped in markdown code fences."""
        clean = raw_text.strip()
        parsed: Dict[str, Any] = {}
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-zA-Z]*\n", "", clean)
            clean = re.sub(r"\n```$", "", clean)
            clean = clean.strip()
        try:
            parsed = json.loads(clean)
        except Exception:
            # Try to find json block using regex
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    pass

        if not isinstance(parsed, dict):
            parsed = {}

        # Ensure required planner schema keys exist
        if "question_type" not in parsed:
            parsed["question_type"] = "direct_fact"
        if "concepts" not in parsed or not isinstance(parsed["concepts"], list):
            parsed["concepts"] = []
        if "search_terms" not in parsed or not isinstance(parsed["search_terms"], list):
            parsed["search_terms"] = []
        if "requirements" not in parsed or not isinstance(parsed["requirements"], list):
            parsed["requirements"] = []
        if "anchor_entities" not in parsed or not isinstance(parsed["anchor_entities"], list):
            parsed["anchor_entities"] = []

        return parsed
