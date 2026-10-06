"""LLM Provider supporting Gemini, OpenAI, and deterministic Mock fallback.

Ensures reliable execution during demonstrations, tests, and live evaluation.
Adheres strictly to zero secret leakage across logs, exceptions, and traces.
"""

from typing import Dict, Any, List, Optional
import os
import json
import re
import logging
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded
load_dotenv()

logger = logging.getLogger(__name__)


class LLMProvider:
    """Unified LLM interface for the planner analysis and final answer generation."""

    def __init__(self, model_name: Optional[str] = None, mock: Optional[bool] = None):
        self._gemini_keys: List[str] = self._load_gemini_keys()
        self._current_key_index: int = 0
        self.gemini_key: Optional[str] = self._gemini_keys[0] if self._gemini_keys else None
        self.model_name: str = (
            model_name
            or os.getenv("GEMINI_FLASH_MODEL")
            or os.getenv("LLM_MODEL")
            or "gemini-3.1-flash-lite-preview"
        )
        self.openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")
        self.openrouter_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
        self.llm_provider_env: str = os.getenv("LLM_PROVIDER", "").lower()

        if mock is not None:
            self._force_mock = mock
        else:
            self._force_mock = os.getenv("MOCK_LLM", "").lower() in ("true", "1", "yes")

        self._client_type = self._detect_client_type()

    def _load_gemini_keys(self) -> List[str]:
        """Loads all configured Gemini API keys in priority order.
        
        Reads GEMINI_API_KEY1 through GEMINI_API_KEY4, ignoring empty or placeholder values.
        Falls back cleanly to legacy GEMINI_API_KEY or GOOGLE_API_KEY if no numbered keys are configured.
        """
        keys: List[str] = []
        for var_name in ["GEMINI_API_KEY1", "GEMINI_API_KEY2", "GEMINI_API_KEY3", "GEMINI_API_KEY4"]:
            val = os.getenv(var_name)
            if val and val.strip() and not val.strip().startswith("your_"):
                keys.append(val.strip())

        if not keys:
            legacy = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if legacy and legacy.strip() and not legacy.strip().startswith("your_"):
                keys.append(legacy.strip())

        return keys

    def _detect_client_type(self) -> str:
        if self._force_mock:
            return "mock"
        if self.llm_provider_env == "openrouter":
            if self.openrouter_key and not self.openrouter_key.strip().startswith("your_"):
                return "openrouter"
            if self.openai_key and not self.openai_key.strip().startswith("your_"):
                return "openrouter"
        if self.llm_provider_env == "gemini" and self._gemini_keys:
            return "gemini"
        if self._gemini_keys:
            return "gemini"
        if self.openrouter_key and not self.openrouter_key.strip().startswith("your_"):
            return "openrouter"
        if self.openai_key and not self.openai_key.strip().startswith("your_"):
            return "openai"
        return "mock"

    def is_mock(self) -> bool:
        return self._client_type == "mock"

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """Execute a text completion request."""
        if self._client_type == "gemini":
            return self._call_gemini(system_prompt, user_prompt, temperature)
        elif self._client_type in ("openai", "openrouter"):
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
        elif self._client_type in ("openai", "openrouter"):
            raw_text = self._call_openai(system_prompt, user_prompt, temperature=0.0)
        else:
            raw_text = self._mock_completion(system_prompt, user_prompt)
        return self._parse_json(raw_text)

    def _is_quota_error(self, exc: Exception) -> bool:
        """Identifies whether an exception is specifically caused by quota exhaustion or rate limits.
        
        Inspects structured status codes and SDK exception types first, avoiding rotation on
        auth (401/403), client errors (400), internal server errors (500), or programming bugs.
        """
        code = getattr(exc, "code", None)
        status_code = getattr(exc, "status_code", None)
        http_status = getattr(exc, "http_status", None)

        if callable(code):
            try:
                code = code()
            except Exception:
                code = None

        status_values = [c for c in (code, status_code, http_status) if c is not None]
        for s in status_values:
            if s == 429 or str(s) == "429":
                return True
            if isinstance(s, int) and s in (400, 401, 403, 404, 500, 502, 503, 504):
                return False

        cls_name = exc.__class__.__name__
        if cls_name in ("ResourceExhausted", "TooManyRequests"):
            return True
        if cls_name in ("InvalidArgument", "Unauthenticated", "PermissionDenied", "NotFound", "InternalServerError"):
            return False

        grpc_status = getattr(exc, "grpc_status_code", None)
        if grpc_status is not None:
            grpc_str = str(grpc_status)
            if "RESOURCE_EXHAUSTED" in grpc_str:
                return True
            if any(non_quota in grpc_str for non_quota in ("INVALID_ARGUMENT", "UNAUTHENTICATED", "PERMISSION_DENIED")):
                return False

        msg = str(exc).lower()

        exclusion_patterns = [
            "api_key_invalid", "invalid_api_key", "api key not valid",
            "permission_denied", "unauthenticated", "forbidden", "401", "403",
            "invalid_argument", "bad request", "jsondecodeerror", "validation error",
            "malformed"
        ]
        if any(pat in msg for pat in exclusion_patterns):
            return False

        quota_indicators = [
            "429",
            "resource_exhausted",
            "resourceexhausted",
            "quota exceeded",
            "quota_exceeded",
            "rate limit",
            "ratelimit",
            "too many requests",
            "exceeded your current quota",
            "queries per minute",
            "queries per day",
            "requests per minute",
            "requests per day"
        ]
        if any(q in msg for q in quota_indicators):
            return True

        return False

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        response_mime_type: Optional[str] = None
    ) -> str:
        """Calls the Gemini API with multi-key quota rotation.
        
        Never silently falls back to mock on failure.
        """
        if not self._gemini_keys:
            raise ValueError("GEMINI_API_KEY is not configured or contains placeholder value.")

        total_keys = len(self._gemini_keys)
        start_idx = self._current_key_index
        last_error: Optional[Exception] = None

        for attempt in range(total_keys):
            key_idx = (start_idx + attempt) % total_keys
            api_key = self._gemini_keys[key_idx]

            logger.info("Gemini request attempt: key_index=%d", key_idx + 1)

            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
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

                # Update active key index to the successfully responding key
                self._current_key_index = key_idx
                self.gemini_key = api_key
                return response.text or ""

            except Exception as e:
                last_error = e
                if self._is_quota_error(e) and attempt < total_keys - 1:
                    next_idx = (key_idx + 1) % total_keys
                    logger.warning(
                        "Gemini quota error: rotating to key_index=%d",
                        next_idx + 1
                    )
                    continue
                else:
                    safe_err = self._sanitize_secret(str(e))
                    raise RuntimeError(f"Gemini API invocation failed: {safe_err}") from None

        safe_err = self._sanitize_secret(str(last_error)) if last_error else "Quota exhausted across all configured keys."
        raise RuntimeError(f"Gemini API invocation failed: All configured Gemini keys exhausted due to quota/rate limits: {safe_err}")

    def _call_openai(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        from openai import OpenAI
        api_key = self.openrouter_key if (self._client_type == "openrouter" and self.openrouter_key) else self.openai_key
        base_url = self.openai_base_url
        if not base_url and self._client_type == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None
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
        for key in self._gemini_keys:
            if key and len(key) > 5:
                sanitized = sanitized.replace(key, "[REDACTED_API_KEY]")
        if self.gemini_key and len(self.gemini_key) > 5:
            sanitized = sanitized.replace(self.gemini_key, "[REDACTED_API_KEY]")
        if self.openai_key and len(self.openai_key) > 5:
            sanitized = sanitized.replace(self.openai_key, "[REDACTED_API_KEY]")
        if self.openrouter_key and len(self.openrouter_key) > 5:
            sanitized = sanitized.replace(self.openrouter_key, "[REDACTED_API_KEY]")
        sanitized = re.sub(r'\b(?:AIza|AQ\.)[A-Za-z0-9_-]+\b', '[REDACTED_API_KEY]', sanitized)
        sanitized = re.sub(r'\bsk-[A-Za-z0-9_-]{20,}\b', '[REDACTED_API_KEY]', sanitized)
        sanitized = re.sub(r'\bsk-or-[A-Za-z0-9_-]{20,}\b', '[REDACTED_API_KEY]', sanitized)
        return sanitized

    def _mock_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Deterministic fallback when no active API key is configured."""
        # Check if this is a planner request
        if "question_type" in system_prompt or "concepts" in system_prompt:
            # Deterministic keyword extraction from user_prompt
            extracted = self._heuristic_plan_extraction(user_prompt)
            return json.dumps(extracted)

        # Conversational mock response
        if "conversational" in user_prompt.lower() or "conversational input" in user_prompt.lower():
            return "Hello! I am your budgeted document assistant. How can I assist you with your document today?"

        # Final answer mock response
        if "INSUFFICIENT" in user_prompt:
            u_low = user_prompt.lower()
            if any(term in u_low for term in ["mentioned", "available", "presence", "does the document contain", "is there"]):
                return "I could not verify the presence of the requested entity in the available document evidence."
            return "Based on the retrieved document evidence, the document does not contain sufficient information to answer the question."
        elif "PARTIALLY_SUPPORTED" in user_prompt:
            return "Based on the retrieved document evidence, the supported portion is verified by the cited pages; however, certain requested requirements could not be verified in the document."
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

        # Check conversational queries
        words_lower = [w.lower().strip("?,.:;\"'()[]{}") for w in q_clean.split()]
        conversational_tokens = {
            "hello", "hi", "hey", "greetings", "good", "morning", "afternoon", "evening",
            "thank", "thanks", "you", "very", "much", "lot", "appreciated", "cheers",
            "ok", "okay", "alright", "sure", "cool", "fine", "got", "it", "understood",
            "bye", "goodbye", "see", "later", "please", "welcome", "yes", "no", "yep", "nope",
            "a", "an", "the", "so", "there", "to", "for", "all"
        }
        if words_lower and all(w in conversational_tokens for w in words_lower):
            return {
                "question_type": "conversational",
                "concepts": [],
                "search_terms": [],
                "requirements": [],
                "anchor_entities": [],
                "anchor_variants": {}
            }

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
            "detail", "details", "information", "question", "document", "retrieval", "analyze", "following", "search",
            "pdf", "mentioned", "available", "presence", "contain", "contains"
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
                requirements.append(f"{anchor} distinguishing characteristics and behavior")
            elif q_type == "temporal":
                requirements.append(f"Date or year of {anchor}")
            elif "why" in q_lower or "purpose" in q_lower:
                requirements.append(f"Purpose and operational rationale of {anchor}")
            else:
                requirements.append(f"{anchor} functional properties and verified facts")

        if not requirements:
            requirements = [f"{q_clean} factual details"]

        return {
            "question_type": q_type,
            "concepts": anchor_entities[:4],
            "search_terms": search_terms[:5],
            "requirements": requirements,
            "anchor_entities": anchor_entities[:3],
            "anchor_variants": {}
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
        if "anchor_variants" not in parsed or not isinstance(parsed["anchor_variants"], dict):
            parsed["anchor_variants"] = {}

        return parsed
