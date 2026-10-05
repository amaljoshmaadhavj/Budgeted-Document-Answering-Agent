"""Unit tests for LLMProvider Gemini integration, secret sanitization, and fallback prevention."""

import pytest
import os
from unittest.mock import MagicMock, patch
from llm.provider import LLMProvider


def test_provider_detects_gemini_client(monkeypatch):
    """When a valid Gemini key is configured, client type must be 'gemini'."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_test_gemini_key_12345")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")

    provider = LLMProvider()
    assert provider._client_type == "gemini"
    assert provider.is_mock() is False
    assert provider.model_name == "gemini-3.8-flash"


def test_provider_detects_placeholder_as_mock(monkeypatch):
    """Placeholder keys must not be treated as live Gemini configuration."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")

    provider = LLMProvider()
    assert provider._client_type == "mock"
    assert provider.is_mock() is True


def test_provider_error_handling_does_not_silently_fallback(monkeypatch):
    """When Gemini API call fails, provider MUST raise RuntimeError, NOT silently fall back to mock."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    test_key = "dummy_secret_key_abcdef_98765"
    monkeypatch.setenv("GEMINI_API_KEY", test_key)

    provider = LLMProvider()
    assert provider._client_type == "gemini"

    # Mock google.generativeai to raise an exception containing the secret
    with patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_instance = MagicMock()
        mock_instance.generate_content.side_effect = Exception(f"Failed with key {test_key} connection timeout")
        mock_model_cls.return_value = mock_instance

        with pytest.raises(RuntimeError) as exc_info:
            provider.complete("System prompt", "User prompt")

        error_message = str(exc_info.value)
        # Must raise explicit error
        assert "Gemini API invocation failed" in error_message
        # MUST NEVER expose the actual secret in the error
        assert test_key not in error_message
        assert "[REDACTED_API_KEY]" in error_message


def test_secret_sanitization_patterns(monkeypatch):
    """Verifies that _sanitize_secret redacts API keys matching various patterns."""
    provider = LLMProvider(mock=True)
    provider.gemini_key = "dummy_custom_token_XYZ999"
    
    raw_error = "Error from API: key dummy_custom_token_XYZ999 was rejected by upstream."
    sanitized = provider._sanitize_secret(raw_error)
    
    assert "dummy_custom_token_XYZ999" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


def test_complete_json_enforces_required_schema(monkeypatch):
    """JSON output must strictly enforce question_type, concepts, search_terms, requirements, anchor_entities."""
    provider = LLMProvider(mock=True)
    
    # In mock mode, complete_json produces valid schema
    plan = provider.complete_json(
        system_prompt="question_type concepts search_terms requirements anchor_entities",
        user_prompt="Question: What is KryoVex-9?"
    )
    
    required_keys = ["question_type", "concepts", "search_terms", "requirements", "anchor_entities"]
    for k in required_keys:
        assert k in plan
        assert isinstance(plan[k], (list, str))
