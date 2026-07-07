"""Tests for the Gemini assistant core (app.assistant).

The google-genai client is fully mocked (monkeypatched ``genai.Client``); no
network is ever touched. Covers the manual function-calling loop shape, the
blocked-response guard, and every offline-fallback trigger.
"""

import pytest

from app import assistant
from google.genai import errors, types

VENUE = "new-york-new-jersey"


class _FakeCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _FakeCandidate:
    def __init__(self, content):
        self.content = content


class _FakeResponse:
    def __init__(self, *, function_calls=None, text=None, model_turn=None):
        self.function_calls = function_calls or []
        self.text = text
        # Sentinel model turn that must be appended VERBATIM by the loop.
        self.candidates = [_FakeCandidate(model_turn)] if model_turn else []


class _FakeClient:
    """Records the ``contents`` snapshot of each call and returns a script."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.contents_snapshots = []

        class _Models:
            def __init__(self, outer):
                self._outer = outer

            def generate_content(self, *, model, contents, config):
                # Snapshot the list as-passed (the loop mutates it in place).
                self._outer.contents_snapshots.append(list(contents))
                return self._outer._responses.pop(0)

        self.models = _Models(self)


def _patch_client(monkeypatch, client):
    monkeypatch.setattr("app.assistant.genai.Client", lambda *a, **k: client)


def _with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_function_call_roundtrip_appends_model_turn_and_one_user_content(monkeypatch):
    _with_key(monkeypatch)
    model_turn = types.Content(role="model", parts=[types.Part(text="thinking")])
    first = _FakeResponse(
        function_calls=[
            _FakeCall("find_accessible_services", {"venue_id": VENUE, "need": "mobility"}),
            _FakeCall("get_live_status", {"venue_id": VENUE}),
        ],
        model_turn=model_turn,
    )
    final = _FakeResponse(text="MetLife has wheelchair access on all levels.")
    client = _FakeClient([first, final])
    _patch_client(monkeypatch, client)

    reply = assistant.answer(
        "wheelchair access at MetLife?", profile={"venue_id": VENUE, "language": "en"}
    )

    assert reply.mode == "live"
    assert reply.text == "MetLife has wheelchair access on all levels."
    # Both parallel calls recorded, in order.
    assert reply.tool_calls_made == ["find_accessible_services", "get_live_status"]

    # The second request's contents: [user_turn, model_turn(verbatim), func_responses].
    second = client.contents_snapshots[1]
    assert second[1] is model_turn, "model turn must be appended verbatim (same object)"
    func_content = second[2]
    assert func_content.role == "user"
    # ALL function responses live in ONE user Content (parallel-call requirement).
    assert len(func_content.parts) == 2


def test_plain_text_response_returns_text(monkeypatch):
    _with_key(monkeypatch)
    client = _FakeClient([_FakeResponse(text="The opening match is on 2026-06-11.")])
    _patch_client(monkeypatch, client)

    reply = assistant.answer("when is the opening match?", profile={"language": "en"})

    assert reply.mode == "live"
    assert reply.text == "The opening match is on 2026-06-11."
    assert reply.tool_calls_made == []
    assert len(client.contents_snapshots) == 1  # no tool loop iteration


def test_blocked_none_text_returns_polite_decline(monkeypatch):
    _with_key(monkeypatch)
    client = _FakeClient([_FakeResponse(text=None)])  # blocked / SAFETY
    _patch_client(monkeypatch, client)

    reply = assistant.answer("something blocked", profile={"language": "es"})

    assert reply.mode == "live"
    assert reply.text == assistant._DECLINE["es"]  # localized decline


def test_client_error_401_falls_back_to_offline(monkeypatch):
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content(*, model, contents, config):
                raise errors.ClientError(401, {"error": {"message": "bad key"}})

    _patch_client(monkeypatch, _RaisingClient())

    reply = assistant.answer(
        "wheelchair access?", profile={"venue_id": VENUE, "language": "en"}
    )

    assert reply.mode == "offline"
    assert reply.text  # offline engine produced a real answer


def test_model_not_found_404_falls_back_to_offline(monkeypatch):
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content(*, model, contents, config):
                # e.g. the configured model id is not available to this key.
                raise errors.ClientError(404, {"error": {"message": "model not found"}})

    _patch_client(monkeypatch, _RaisingClient())
    reply = assistant.answer("hello", profile={"language": "en"})
    assert reply.mode == "offline"
    assert reply.text  # degrades instead of surfacing a 500


def test_rate_limit_429_falls_back_to_offline(monkeypatch):
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content(*, model, contents, config):
                raise errors.ClientError(429, {"error": {"message": "rate limited"}})

    _patch_client(monkeypatch, _RaisingClient())
    reply = assistant.answer("hello", profile={"language": "en"})
    assert reply.mode == "offline"


def test_server_error_falls_back_to_offline(monkeypatch):
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content(*, model, contents, config):
                raise errors.ServerError(503, {"error": {"message": "unavailable"}})

    _patch_client(monkeypatch, _RaisingClient())
    reply = assistant.answer("hello", profile={"language": "en"})
    assert reply.mode == "offline"


def test_non_fallback_client_error_is_reraised(monkeypatch):
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content(*, model, contents, config):
                raise errors.ClientError(400, {"error": {"message": "our bug"}})

    _patch_client(monkeypatch, _RaisingClient())
    with pytest.raises(errors.ClientError):
        assistant.answer("hello", profile={"language": "en"})


def test_no_api_key_goes_straight_to_offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # If the live path were taken, this would blow up; it must not be called.
    def _boom(*a, **k):
        raise AssertionError("Client must not be constructed without a key")

    monkeypatch.setattr("app.assistant.genai.Client", _boom)

    reply = assistant.answer(
        "quietest gate for a sensory-sensitive kid?",
        profile={"venue_id": VENUE, "language": "en", "needs": ["sensory"]},
    )
    assert reply.mode == "offline"
    assert reply.text


def test_history_round_trips_as_alternating_turns(monkeypatch):
    _with_key(monkeypatch)
    client = _FakeClient([_FakeResponse(text="ok")])
    _patch_client(monkeypatch, client)

    assistant.answer(
        "and the nearest restroom?",
        profile={"venue_id": VENUE, "language": "en"},
        history=[
            {"role": "user", "text": "wheelchair access?"},
            {"role": "assistant", "text": "Yes, on all levels."},
        ],
    )

    contents = client.contents_snapshots[0]
    # 2 history turns + 1 current user turn.
    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[1].role == "model"  # assistant turn mapped to "model"
    assert contents[2].role == "user"
