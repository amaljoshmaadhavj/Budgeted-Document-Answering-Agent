"""Unit tests for LLMProvider Gemini integration, multi-key rotation, secret sanitization, and fallback prevention."""

import pytest
import os
import json
from unittest.mock import MagicMock, patch
from llm.provider import LLMProvider


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    """Isolate provider unit tests from live environment secrets and models."""
    for var in [
        "GEMINI_API_KEY",
        "GEMINI_API_KEY1",
        "GEMINI_API_KEY2",
        "GEMINI_API_KEY3",
        "GEMINI_API_KEY4",
        "GOOGLE_API_KEY",
        "GEMINI_FLASH_MODEL",
        "GEMINI_PRO_MODEL",
    ]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_gemini():
    """Context fixture to intercept google.generativeai configuration and generation calls."""
    configured_keys = []

    def fake_configure(api_key=None, **kwargs):
        configured_keys.append(api_key)

    with patch("google.generativeai.configure", side_effect=fake_configure) as m_cfg, \
         patch("google.generativeai.GenerativeModel") as m_model_cls:

        mock_model_instance = MagicMock()
        m_model_cls.return_value = mock_model_instance
        yield {
            "configured_keys": configured_keys,
            "mock_model": mock_model_instance,
            "mock_configure": m_cfg,
            "mock_model_cls": m_model_cls,
        }


# ==============================================================================
# EXISTING BASE TESTS (PRESERVED)
# ==============================================================================

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


# ==============================================================================
# REQUIRED ROTATION TESTS (TEST 1 - TEST 12)
# ==============================================================================

def test_1_one_configured_key_succeeds(monkeypatch, mock_gemini):
    """TEST 1: One configured key succeeds. Exactly one attempt."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY1", "TEST_KEY_1")

    mock_resp = MagicMock()
    mock_resp.text = "Answer from Key 1"
    mock_gemini["mock_model"].generate_content.return_value = mock_resp

    provider = LLMProvider()
    res = provider.complete("System instruction", "User query")

    assert res == "Answer from Key 1"
    assert mock_gemini["mock_model"].generate_content.call_count == 1
    assert mock_gemini["configured_keys"] == ["TEST_KEY_1"]


def test_2_four_configured_keys_key1_succeeds(monkeypatch, mock_gemini):
    """TEST 2: Four configured keys, key 1 succeeds. Key 2/3/4 are not used."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY1", "TEST_KEY_1")
    monkeypatch.setenv("GEMINI_API_KEY2", "TEST_KEY_2")
    monkeypatch.setenv("GEMINI_API_KEY3", "TEST_KEY_3")
    monkeypatch.setenv("GEMINI_API_KEY4", "TEST_KEY_4")

    mock_resp = MagicMock()
    mock_resp.text = "Answer from Key 1"
    mock_gemini["mock_model"].generate_content.return_value = mock_resp

    provider = LLMProvider()
    res = provider.complete("System instruction", "User query")

    assert res == "Answer from Key 1"
    assert mock_gemini["mock_model"].generate_content.call_count == 1
    assert mock_gemini["configured_keys"] == ["TEST_KEY_1"]


def test_3_key1_returns_429_key2_succeeds(monkeypatch, mock_gemini):
    """TEST 3: Key 1 returns simulated 429. Key 2 succeeds. Exactly two attempts."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY1", "TEST_KEY_1")
    monkeypatch.setenv("GEMINI_API_KEY2", "TEST_KEY_2")
    monkeypatch.setenv("GEMINI_API_KEY3", "TEST_KEY_3")
    monkeypatch.setenv("GEMINI_API_KEY4", "TEST_KEY_4")

    mock_resp = MagicMock()
    mock_resp.text = "Answer from Key 2"
    mock_gemini["mock_model"].generate_content.side_effect = [
        Exception("429 Resource has been exhausted (quota exceeded)"),
        mock_resp,
    ]

    provider = LLMProvider()
    res = provider.complete("System instruction", "User query")

    assert res == "Answer from Key 2"
    assert mock_gemini["mock_model"].generate_content.call_count == 2
    assert mock_gemini["configured_keys"] == ["TEST_KEY_1", "TEST_KEY_2"]


def test_4_keys1_and_2_quota_errors_key3_succeeds(monkeypatch, mock_gemini):
    """TEST 4: Keys 1 and 2 return simulated quota errors. Key 3 succeeds. Exactly three attempts."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY1", "TEST_KEY_1")
    monkeypatch.setenv("GEMINI_API_KEY2", "TEST_KEY_2")
    monkeypatch.setenv("GEMINI_API_KEY3", "TEST_KEY_3")
    monkeypatch.setenv("GEMINI_API_KEY4", "TEST_KEY_4")

    mock_resp = MagicMock()
    mock_resp.text = "Answer from Key 3"

    class StructuredQuotaError(Exception):
        code = 429

    mock_gemini["mock_model"].generate_content.side_effect = [
        StructuredQuotaError("ResourceExhausted quota exceeded"),
        Exception("RESOURCE_EXHAUSTED: rate limit exceeded"),
        mock_resp,
    ]

    provider = LLMProvider()
    res = provider.complete("System instruction", "User query")

    assert res == "Answer from Key 3"
    assert mock_gemini["mock_model"].generate_content.call_count == 3
    assert mock_gemini["configured_keys"] == ["TEST_KEY_1", "TEST_KEY_2", "TEST_KEY_3"]


def test_5_keys1_2_3_quota_errors_key4_succeeds(monkeypatch, mock_gemini):
    """TEST 5: Keys 1, 2, 3 return quota errors. Key 4 succeeds. Exactly four attempts."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY1", "TEST_KEY_1")
    monkeypatch.setenv("GEMINI_API_KEY2", "TEST_KEY_2")
    monkeypatch.setenv("GEMINI_API_KEY3", "TEST_KEY_3")
    monkeypatch.setenv("GEMINI_API_KEY4", "TEST_KEY_4")

    mock_resp = MagicMock()
    mock_resp.text = "Answer from Key 4"
    mock_gemini["mock_model"].generate_content.side_effect = [
        Exception("429 Too Many Requests: requests per minute exceeded"),
        Exception("RESOURCE_EXHAUSTED"),
        Exception("Quota exceeded for quota metric 'Queries'"),
        mock_resp,
    ]

    provider = LLMProvider()
    res = provider.complete("System instruction", "User query")

    assert res == "Answer from Key 4"
    assert mock_gemini["mock_model"].generate_content.call_count == 4
    assert mock_gemini["configured_keys"] == ["TEST_KEY_1", "TEST_KEY_2", "TEST_KEY_3", "TEST_KEY_4"]


def test_6_all_four_return_quota_or_429(monkeypatch, mock_gemini):
    """TEST 6: All four return quota/429. Exactly four attempts, clear final error, no 5th attempt."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY1", "TEST_KEY_1")
    monkeypatch.setenv("GEMINI_API_KEY2", "TEST_KEY_2")
    monkeypatch.setenv("GEMINI_API_KEY3", "TEST_KEY_3")
    monkeypatch.setenv("GEMINI_API_KEY4", "TEST_KEY_4")

    mock_gemini["mock_model"].generate_content.side_effect = [
        Exception("429 Resource has been exhausted"),
        Exception("429 Resource has been exhausted"),
        Exception("RESOURCE_EXHAUSTED"),
        Exception("Quota exceeded"),
    ]

    provider = LLMProvider()
    with pytest.raises(RuntimeError) as exc_info:
        provider.complete("System instruction", "User query")

    assert mock_gemini["mock_model"].generate_content.call_count == 4
    assert len(mock_gemini["configured_keys"]) == 4
    assert mock_gemini["configured_keys"] == ["TEST_KEY_1", "TEST_KEY_2", "TEST_KEY_3", "TEST_KEY_4"]
    assert "Gemini API invocation failed" in str(exc_info.value)


def test_7_key1_returns_non_quota_error(monkeypatch, mock_gemini):
    """TEST 7: Key 1 returns a non-quota error. Exactly one attempt, NO rotation to key 2."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY1", "TEST_KEY_1")
    monkeypatch.setenv("GEMINI_API_KEY2", "TEST_KEY_2")
    monkeypatch.setenv("GEMINI_API_KEY3", "TEST_KEY_3")
    monkeypatch.setenv("GEMINI_API_KEY4", "TEST_KEY_4")

    class ClientBadRequestError(Exception):
        status_code = 400

    mock_gemini["mock_model"].generate_content.side_effect = ClientBadRequestError("400 Bad Request: Invalid argument supplied")

    provider = LLMProvider()
    with pytest.raises(RuntimeError) as exc_info:
        provider.complete("System instruction", "User query")

    assert mock_gemini["mock_model"].generate_content.call_count == 1
    assert mock_gemini["configured_keys"] == ["TEST_KEY_1"]
    assert "Gemini API invocation failed" in str(exc_info.value)


def test_8_only_keys_1_and_2_configured(monkeypatch, mock_gemini):
    """TEST 8: Only keys 1 and 2 are configured. Rotation stops at key 2, no attempt on key 3/4."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY1", "TEST_KEY_1")
    monkeypatch.setenv("GEMINI_API_KEY2", "TEST_KEY_2")

    mock_gemini["mock_model"].generate_content.side_effect = [
        Exception("429 Resource has been exhausted"),
        Exception("RESOURCE_EXHAUSTED: rate limit exceeded"),
    ]

    provider = LLMProvider()
    with pytest.raises(RuntimeError) as exc_info:
        provider.complete("System instruction", "User query")

    assert mock_gemini["mock_model"].generate_content.call_count == 2
    assert mock_gemini["configured_keys"] == ["TEST_KEY_1", "TEST_KEY_2"]
    assert "TEST_KEY_3" not in mock_gemini["configured_keys"]
    assert "TEST_KEY_4" not in mock_gemini["configured_keys"]


def test_9_no_gemini_keys_configured(monkeypatch):
    """TEST 9: No Gemini keys configured. Clear configuration/provider error, no secret exposure."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    provider = LLMProvider()
    assert provider._client_type == "mock"
    assert provider.is_mock() is True

    # If direct internal call attempted without keys, raises clear ValueError with no secret exposure
    with pytest.raises(ValueError) as exc_info:
        provider._call_gemini("System", "User")
    assert "not configured" in str(exc_info.value)


def test_10_backward_compatibility_single_key(monkeypatch, mock_gemini):
    """TEST 10: Backward compatibility with existing single-key configuration."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "TEST_LEGACY_KEY")

    mock_resp = MagicMock()
    mock_resp.text = "Legacy single key response"
    mock_gemini["mock_model"].generate_content.return_value = mock_resp

    provider = LLMProvider()
    assert provider._client_type == "gemini"
    assert provider.gemini_key == "TEST_LEGACY_KEY"
    assert provider._gemini_keys == ["TEST_LEGACY_KEY"]

    res = provider.complete("System", "User")
    assert res == "Legacy single key response"
    assert mock_gemini["mock_model"].generate_content.call_count == 1
    assert mock_gemini["configured_keys"] == ["TEST_LEGACY_KEY"]


def test_11_secret_sanitization(monkeypatch, mock_gemini):
    """TEST 11: Secret sanitization. Verify fake values such as TEST_KEY_1 do not appear in errors."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY1", "TEST_KEY_1")
    monkeypatch.setenv("GEMINI_API_KEY2", "TEST_KEY_2")

    mock_gemini["mock_model"].generate_content.side_effect = Exception("Auth failed with credential TEST_KEY_1 in request header")

    provider = LLMProvider()
    with pytest.raises(RuntimeError) as exc_info:
        provider.complete("System", "User")

    error_msg = str(exc_info.value)
    assert "TEST_KEY_1" not in error_msg
    assert "[REDACTED_API_KEY]" in error_msg

    raw_debug = "Headers: Authorization: Bearer TEST_KEY_2 and fallback TEST_KEY_1"
    sanitized = provider._sanitize_secret(raw_debug)
    assert "TEST_KEY_1" not in sanitized
    assert "TEST_KEY_2" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


def test_12_planner_and_final_answer_interface_with_rotation(monkeypatch, mock_gemini):
    """TEST 12: Verify existing planner and final-answer provider behavior still works through same interface."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY1", "TEST_KEY_1")
    monkeypatch.setenv("GEMINI_API_KEY2", "TEST_KEY_2")

    planner_json = json.dumps({
        "question_type": "definition",
        "concepts": ["modulating dynamically"],
        "search_terms": ["modulating dynamically"],
        "requirements": ["modulating dynamically definition"],
        "anchor_entities": ["modulating dynamically"],
        "anchor_variants": {"modulating dynamically": ["dynamic modulation"]}
    })

    mock_planner_resp = MagicMock()
    mock_planner_resp.text = f"```json\n{planner_json}\n```"

    # Planner call: Key 1 hits 429, rotates to Key 2 which succeeds with JSON
    mock_gemini["mock_model"].generate_content.side_effect = [
        Exception("429 Resource has been exhausted"),
        mock_planner_resp,
    ]

    provider = LLMProvider()
    plan = provider.complete_json(
        system_prompt="question_type concepts search_terms requirements anchor_entities anchor_variants",
        user_prompt="Analyze the question: What does modulating dynamically mean?"
    )

    assert plan["question_type"] == "definition"
    assert plan["concepts"] == ["modulating dynamically"]
    assert plan["search_terms"] == ["modulating dynamically"]
    assert plan["requirements"] == ["modulating dynamically definition"]
    assert plan["anchor_entities"] == ["modulating dynamically"]
    assert plan["anchor_variants"] == {"modulating dynamically": ["dynamic modulation"]}
    assert mock_gemini["mock_model"].generate_content.call_count == 2
    assert mock_gemini["configured_keys"] == ["TEST_KEY_1", "TEST_KEY_2"]

    # Final answer call continues cleanly using the provider interface
    mock_final_resp = MagicMock()
    mock_final_resp.text = "Based on retrieved evidence, modulating dynamically is defined on page 5."
    mock_gemini["mock_model"].generate_content.side_effect = None
    mock_gemini["mock_model"].generate_content.return_value = mock_final_resp

    final_ans = provider.complete(
        system_prompt="Generate grounded answer",
        user_prompt="Provide the final answer for the user."
    )
    assert "modulating dynamically is defined" in final_ans
