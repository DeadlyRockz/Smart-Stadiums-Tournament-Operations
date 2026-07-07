"""Shared pytest fixtures for the AccessMate test suite.

Provides a reset TestClient, a sample venue id, and a lightweight fake
google-genai client so tests can exercise the live path without a network or
a real API key. The offline path uses the real deterministic engine.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app, rate_limiter

VENUE_ID = "new-york-new-jersey"


@pytest.fixture
def venue_id() -> str:
    """A stable, real venue id (MetLife Stadium — hosts the final)."""
    return VENUE_ID


@pytest.fixture
def client():
    """A TestClient with the rate limiter reset before and after the test."""
    rate_limiter.reset()
    with TestClient(app) as test_client:
        yield test_client
    rate_limiter.reset()


# --- Fake google-genai client -------------------------------------------------

class _FakeCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _FakeCandidate:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    """Mimics a google-genai GenerateContentResponse for tests."""

    def __init__(self, *, function_calls=None, text=None, model_turn=None):
        self.function_calls = function_calls or []
        self.text = text
        self.candidates = [_FakeCandidate(model_turn)] if model_turn else []


class FakeGeminiClient:
    """Returns a scripted sequence of FakeResponse objects."""

    def __init__(self, responses):
        self._responses = list(responses)

        class _Models:
            def __init__(self, outer):
                self._outer = outer

            def generate_content(self, *, model, contents, config):
                return self._outer._responses.pop(0)

        self.models = _Models(self)


@pytest.fixture
def make_function_call():
    """Factory to build a fake model function-call for the scripted responses."""
    return _FakeCall


@pytest.fixture
def patch_gemini(monkeypatch):
    """Install a FakeGeminiClient built from the given scripted responses."""

    def _install(responses):
        client = FakeGeminiClient(responses)
        monkeypatch.setattr("app.assistant.genai.Client", lambda *a, **k: client)
        return client

    return _install
