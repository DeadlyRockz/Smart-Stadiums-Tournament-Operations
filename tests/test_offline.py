"""Tests for app.offline: deterministic keyword-routed fallback assistant."""

from app.offline import offline_answer


def profile(language="en", venue_id=None, needs=None):
    return {"language": language, "needs": needs or [], "venue_id": venue_id}


# ------------------------------------------------------------- accessibility

def test_english_accessibility_question_mentions_facilities():
    answer = offline_answer(
        "Is there wheelchair access for my seat?",
        profile("en", "new-york-new-jersey"),
    )
    assert "Accessibility at MetLife Stadium" in answer
    assert "Accessible gates" in answer
    assert "MetLife Gate" in answer
    assert "wheelchair" in answer.lower()


def test_sensory_question_routes_to_sensory_fields():
    answer = offline_answer(
        "Where is the quietest place for my autistic son?",
        profile("en", "los-angeles"),
    )
    assert "Sensory support" in answer
    assert "sensory" in answer.lower()


def test_unverified_venue_gets_caveat():
    answer = offline_answer("Is there a ramp?", profile("en", "dallas"))
    assert "not yet verified" in answer


# ---------------------------------------------------------------- languages

def test_spanish_nursing_room_question_answered_in_spanish():
    answer = offline_answer(
        "¿Dónde está el área de lactancia?", profile("es", "mexico-city")
    )
    assert "lactancia" in answer
    assert "Ubicación" in answer
    assert "Nursing room near Puerta 1" in answer
    # Real Spanish, not prefixed English boilerplate.
    assert not answer.startswith("Yes")


def test_french_question_answered_in_french():
    answer = offline_answer(
        "Où sont les toilettes accessibles ?", profile("fr", "toronto")
    )
    assert "Accessibilité" in answer
    assert "Toilettes accessibles" in answer
    assert "vérifiées" in answer  # unverified caveat, in French


def test_unknown_language_code_falls_back_to_english():
    answer = offline_answer("Is there wheelchair access?", profile("de", "dallas"))
    assert "Accessibility at AT&T Stadium" in answer


# --------------------------------------------------------------- navigation

def test_navigation_question_mentions_a_gate():
    answer = offline_answer(
        "Which gate should I use to get in?", profile("en", "seattle")
    )
    assert "Recommended entrance" in answer
    assert "Northwest Gate" in answer or "Southeast Gate" in answer


# ------------------------------------------------------------------ no venue

def test_no_venue_asks_user_to_pick_one_with_examples():
    answer = offline_answer("Is there a sensory room?", profile("en", None))
    assert "choose a stadium" in answer
    assert "Estadio Azteca" in answer
    assert "MetLife Stadium" in answer


def test_no_venue_prompt_is_localized():
    answer = offline_answer(
        "¿Hay rampa para silla de ruedas?", profile("es", None)
    )
    assert "elija primero un estadio" in answer


def test_unknown_venue_id_treated_as_no_venue():
    answer = offline_answer("Is there a ramp?", profile("en", "atlantis"))
    assert "choose a stadium" in answer


# ------------------------------------------------------------- other intents

def test_greeting():
    answer = offline_answer("Hello!", profile("en", None))
    assert "AccessMate" in answer


def test_fallback_help_for_unmatched_message():
    answer = offline_answer("asdfghjkl", profile("en", "dallas"))
    assert "I can help" in answer


def test_schedule_final_date():
    answer = offline_answer(
        "When is the final?", profile("en", "new-york-new-jersey")
    )
    assert "2026-07-19" in answer


def test_schedule_works_without_venue_in_french():
    answer = offline_answer("Quand a lieu la finale ?", profile("fr", None))
    assert "2026-06-11" in answer
    assert "2026-07-19" in answer
    assert "finale" in answer


def test_food_water_spanish():
    answer = offline_answer("¿Dónde hay agua?", profile("es", "guadalajara"))
    assert "Agua en Estadio Akron" in answer


# ------------------------------------------------------------- determinism

def test_offline_answers_are_deterministic():
    cases = [
        ("Is there wheelchair access?", profile("en", "dallas")),
        ("¿Dónde está el área de lactancia?", profile("es", "mexico-city")),
        ("Hello!", profile("en", None)),
    ]
    for message, prof in cases:
        assert offline_answer(message, prof) == offline_answer(message, prof)


def test_no_emoji_in_answers():
    answer = offline_answer(
        "Is there wheelchair access?", profile("en", "dallas")
    )
    assert all(ord(ch) < 0x2600 for ch in answer)
