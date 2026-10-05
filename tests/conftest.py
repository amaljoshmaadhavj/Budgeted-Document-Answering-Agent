"""Test configuration and fixtures.

Ensures the test suite runs deterministically with mocks/stubs,
avoiding external network calls, API quota usage, or secret exposure in test output.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def mock_llm_environment(monkeypatch):
    """Enforces mock LLM execution for deterministic testing without external API calls."""
    monkeypatch.setenv("MOCK_LLM", "true")
