# Plan 01 — AccessMate: Accessible Stadium Copilot for FIFA World Cup 2026

**Challenge:** [Challenge 4] Smart Stadiums & Tournament Operations — build a GenAI-enabled solution that enhances stadium operations / tournament experience.

**Chosen vertical:** **Accessibility** (primary), with **multilingual assistance** and **real-time decision support** as supporting capabilities.
**Persona:** Fans with accessibility needs (mobility, low vision, hearing, sensory) attending FIFA World Cup 2026 matches.

**Why this vertical:** "Accessibility – inclusive and usable design" is an explicit scoring criterion for the hackathon, so an accessibility-first product aligns the problem statement with the rubric twice over. Multilingual support comes nearly free from the LLM and covers the 3-host-country (US/Canada/Mexico) audience.

**Product in one sentence:** A chat copilot where a fan declares their language and access needs once, then asks anything — "quietest gate for a sensory-sensitive kid?", "wheelchair route from Gate C to section 214?", "¿dónde está el área de lactancia?" — and gets answers grounded in a structured venue dataset via Gemini function calling, with context-aware logic (needs profile → different routes/answers) and a graceful **offline deterministic mode** when no API key is present (so evaluators can run it without credentials).

**Tech stack (decided):**
- Python 3.12+ (3.14.3 available locally), virtualenv
- FastAPI + Uvicorn (API layer), Pydantic (validation)
- `google-genai` official Python SDK (Gemini API via AI Studio key) — model **`gemini-3.5-flash`** (stable GA, available on the free tier, supports function calling)
- pytest + httpx `TestClient` (testing; Gemini client mocked)
- Vanilla accessible HTML/CSS/JS frontend (no build step — keeps repo small and reviewable)
- No database — static JSON dataset + simulated live-ops feed (assumption documented in README)

**Hard submission constraints (from challenge brief `.claude.md`):**
- Public GitHub repo, **single branch**, **< 10 MB**, max 1 attempt
- README must explain: chosen vertical, approach/logic, how it works, assumptions
- Evaluation: Code Quality, Security, Efficiency, Testing, Accessibility, Problem Statement Alignment

---

## Phase 0 — Documentation Discovery (COMPLETED by orchestrator; findings below)

Executing sessions must NOT re-derive API usage from memory. Use only the snippets below (verified 2026-07-07 against ai.google.dev docs, the googleapis/python-genai SDK reference, and the SDK's `errors.py` source). If more Gemini API detail is needed in a later phase, WebFetch https://googleapis.github.io/python-genai/ or https://ai.google.dev/gemini-api/docs — do not guess.

### 0.1 Allowed APIs (google-genai Python SDK — the current SDK, NOT the deprecated `google-generativeai`)

**Install & client init** (source: ai.google.dev/gemini-api/docs/quickstart + SDK reference):
```python
# pip install -U google-genai
from google import genai
client = genai.Client()  # auto-reads GEMINI_API_KEY (or GOOGLE_API_KEY — which wins if both set); never hardcode
```

**generate_content with system instruction** (source: SDK reference § GenerateContentConfig):
```python
from google.genai import types

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=contents,                        # list[types.Content]
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,     # frozen string — no timestamps/user data interpolated (stable prefix aids implicit caching)
        tools=[TOOLS],
        max_output_tokens=2048,               # check finish_reason == "MAX_TOKENS" for truncation
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # we run the loop ourselves
    ),
)
```
Do NOT set `temperature` (Gemini 3.x official guidance: keep the 1.0 default) and do NOT set any thinking config (`thinking_level`/`thinking_budget`) — omit both entirely.

**Tool definition shape** (source: ai.google.dev/gemini-api/docs/function-calling):
```python
find_services = types.FunctionDeclaration(
    name="find_accessible_services",
    description="Look up accessibility services at a venue. Call this whenever the user asks about wheelchair access, sensory rooms, assistive listening, elevators, or accessible seating/toilets.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "venue_id": {"type": "string", "description": "Venue id from the dataset, e.g. 'new-york-new-jersey'"},
            "need": {"type": "string", "enum": ["mobility", "vision", "hearing", "sensory", "general"]},
        },
        "required": ["venue_id"],
    },
)
TOOLS = types.Tool(function_declarations=[find_services, get_venue_info_fn, get_live_status_fn, plan_visit_fn])
```
Descriptions must be prescriptive about *when* to call the tool.

**Manual function-calling loop** (source: SDK reference — copy this shape, do not invent):
```python
contents = [types.Content(role="user", parts=[types.Part(text=user_turn)])]
for _ in range(8):                                        # iteration cap — prevents runaway loops
    response = client.models.generate_content(model=MODEL, contents=contents, config=CONFIG)
    calls = response.function_calls or []
    if not calls:
        break
    contents.append(response.candidates[0].content)        # append the model turn VERBATIM — thought signatures must survive
    contents.append(types.Content(role="user", parts=[     # ALL function responses in ONE user Content
        types.Part.from_function_response(
            name=c.name,
            response={"result": execute_tool(c.name, dict(c.args))},
        )
        for c in calls
    ]))
final_text = response.text                                 # may be None — guard below
```
Failed tools still return a `from_function_response` part carrying an error payload — never dropped.

**Error handling chain** (source: SDK `google/genai/errors.py` — most-specific first):
```python
from google.genai import errors

try:
    ...
except errors.ClientError as e:        # 4xx — e.code: 401/403 bad key, 429 free-tier rate limit, 400 bad request
    if e.code in (401, 403, 429):
        return offline_answer(...)     # graceful degrade — evaluators may have no key / burst past free-tier RPM
    raise
except errors.ServerError:             # 5xx
    return offline_answer(...)
except (errors.APIError, ConnectionError):
    return offline_answer(...)         # no network → offline mode
```

**Blocked/empty-response guard** (source: ai.google.dev safety-settings docs): `response.text` returns `None` when the prompt is blocked (empty `candidates`, `prompt_feedback.block_reason` set), when output is blocked (`finish_reason == "SAFETY"`), or on a function-call-only turn. Always None-check `response.text`; return a polite language-appropriate decline when blocked.

**Multi-turn:** chat history round-trips through the client as plain text turns; rebuild `contents` per request (alternating user/model text parts), letting the loop above add function-call turns within a single request. (`client.chats.create` exists but holds server-format state in process memory — the stateless rebuild fits the API layer better.)

### 0.2 Anti-patterns (these error or silently break on the Gemini API — MUST NOT appear)

| Forbidden | Why | Use instead |
|---|---|---|
| `import google.generativeai` / `genai.configure(...)` / `GenerativeModel(...)` | deprecated legacy SDK (unmaintained since 2025-11-30) | `from google import genai; genai.Client()` |
| `gemini-2.0-*`, `gemini-1.5-*` model ids | shut down (2.0 retired 2026-06-01) → errors | exact `gemini-3.5-flash` |
| `gemini-3.1-pro-preview` or other Pro models | not available on free-tier AI Studio keys | `gemini-3.5-flash` |
| setting `temperature` (esp. < 1.0) on Gemini 3.x | official guidance: risks looping/degraded output | omit — default 1.0 |
| `thinking_level` and/or `thinking_budget` | mixing them 400s; unneeded here | omit thinking config entirely |
| stripping/rebuilding the model's function-call `Content` before replying | breaks thought-signature validation on 3.x | append `response.candidates[0].content` verbatim |
| assuming `response.text` is always a str | it is `None` on blocked or function-call-only turns | None-check; read `response.function_calls` |
| splitting function responses across multiple `Content`s | breaks parallel function calling | one user `Content` with all `from_function_response` parts |
| hardcoded API key anywhere in repo | security criterion | env var + `.env.example`, `.gitignore` |

### 0.3 Environment facts (verified 2026-07-07)

- `gh` CLI authenticated as **DeadlyRockz** (active account), scopes include `repo` → repo creation from CLI works
- Python 3.14.3, Node v24.12.0 available
- Working dir is NOT yet a git repo; path contains spaces/`&` (quote all paths in shell commands)
- Live Gemini verification is conditional: only run live-API smoke tests if `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set; otherwise verify offline mode only
- Free-tier limits for `gemini-3.5-flash` are roughly 10 requests/min (unofficial — the official page defers to the AI Studio dashboard): expect 429s under bursts; the assistant treats 429 as an offline-fallback trigger, and the app's own rate limiter should sit at/below this ceiling

### 0.4 Venue dataset (from web research subagent — sources: Wikipedia 2026 FIFA World Cup venue table cross-checked against ussoccerplayers.com; official venue sites for accessibility)

**Tournament metadata** (verified): 16 venues — 11 USA, 3 Mexico, 2 Canada. Opening match **2026-06-11** at Estadio Azteca/Banorte, Mexico City (tournament is in progress as of plan date). Final **2026-07-19** at MetLife Stadium, East Rutherford NJ.

```json
{
  "tournament": {
    "openingMatch": { "date": "2026-06-11", "venueId": "mexico-city" },
    "final": { "date": "2026-07-19", "venueId": "new-york-new-jersey" }
  },
  "venues": [
    { "id": "mexico-city", "name": "Estadio Azteca", "commercialName": "Estadio Banorte", "fifaName": "Estadio Ciudad de Mexico (Mexico City Stadium)", "city": "Mexico City", "country": "Mexico", "capacity": 80824 },
    { "id": "new-york-new-jersey", "name": "MetLife Stadium", "commercialName": "MetLife Stadium", "fifaName": "New York New Jersey Stadium", "city": "East Rutherford, New Jersey (New York/New Jersey)", "country": "USA", "capacity": 80663 },
    { "id": "dallas", "name": "AT&T Stadium", "commercialName": "AT&T Stadium", "fifaName": "Dallas Stadium", "city": "Arlington, Texas (Dallas)", "country": "USA", "capacity": 70649 },
    { "id": "los-angeles", "name": "SoFi Stadium", "commercialName": "SoFi Stadium", "fifaName": "Los Angeles Stadium", "city": "Inglewood, California (Los Angeles)", "country": "USA", "capacity": 70492 },
    { "id": "kansas-city", "name": "Arrowhead Stadium", "commercialName": "GEHA Field at Arrowhead Stadium", "fifaName": "Kansas City Stadium", "city": "Kansas City, Missouri", "country": "USA", "capacity": 69045 },
    { "id": "san-francisco", "name": "Levi's Stadium", "commercialName": "Levi's Stadium", "fifaName": "San Francisco Bay Area Stadium", "city": "Santa Clara, California (SF Bay Area)", "country": "USA", "capacity": 68827 },
    { "id": "houston", "name": "NRG Stadium", "commercialName": "NRG Stadium", "fifaName": "Houston Stadium", "city": "Houston, Texas", "country": "USA", "capacity": 68777 },
    { "id": "philadelphia", "name": "Lincoln Financial Field", "commercialName": "Lincoln Financial Field", "fifaName": "Philadelphia Stadium", "city": "Philadelphia, Pennsylvania", "country": "USA", "capacity": 68324 },
    { "id": "atlanta", "name": "Mercedes-Benz Stadium", "commercialName": "Mercedes-Benz Stadium", "fifaName": "Atlanta Stadium", "city": "Atlanta, Georgia", "country": "USA", "capacity": 68239 },
    { "id": "seattle", "name": "Lumen Field", "commercialName": "Lumen Field", "fifaName": "Seattle Stadium", "city": "Seattle, Washington", "country": "USA", "capacity": 66925 },
    { "id": "miami", "name": "Hard Rock Stadium", "commercialName": "Hard Rock Stadium", "fifaName": "Miami Stadium", "city": "Miami Gardens, Florida (Miami)", "country": "USA", "capacity": 64478 },
    { "id": "boston", "name": "Gillette Stadium", "commercialName": "Gillette Stadium", "fifaName": "Boston Stadium", "city": "Foxborough, Massachusetts (Boston)", "country": "USA", "capacity": 64146 },
    { "id": "vancouver", "name": "BC Place", "commercialName": "BC Place", "fifaName": "BC Place Vancouver", "city": "Vancouver, British Columbia", "country": "Canada", "capacity": 52497 },
    { "id": "monterrey", "name": "Estadio BBVA", "commercialName": "Estadio BBVA", "fifaName": "Estadio Monterrey", "city": "Guadalupe, Nuevo Leon (Monterrey)", "country": "Mexico", "capacity": 51243 },
    { "id": "guadalajara", "name": "Estadio Akron", "commercialName": "Estadio Akron", "fifaName": "Estadio Guadalajara", "city": "Zapopan, Jalisco (Guadalajara)", "country": "Mexico", "capacity": 45664 },
    { "id": "toronto", "name": "BMO Field", "commercialName": "BMO Field", "fifaName": "Toronto Stadium", "city": "Toronto, Ontario", "country": "Canada", "capacity": 43036 }
  ]
}
```

**Verified accessibility facts** (use these verbatim for the `accessibility` objects of these venues in Phase 1; they anchor the dataset's realism):
- **MetLife** (official site): wheelchair/low-mobility/companion seating on all levels; wheelchair escort from gates on request; assistive listening via ListenWIFI app; captioning on ribbon boards/concourse TVs; KultureCity sensory pods at two elevator lobbies (Plaza Level HCLTech; 3rd-floor Moody's) + sensory bags (noise-cancelling headphones, fidget toys, weighted lap pads) at Guest Service booths.
- **SoFi** (official site): wheelchair + companion seating on every level/price point; two sensory-inclusive spaces (Level 3 NE and SW Guest Services); free sensory kits; assistive listening devices at Guest Services (ID deposit); Mobility Ambassadors/wheelchair escorts at Entries 4, 8, 10.
- **Estadio Azteca/Banorte** (third-party guides — mark "unverified" in dataset): accessible seats mostly Category 1; elevators, ramps, wheelchair bays, companion seats, accessible toilets, hearing-assist system.
- **Tournament-wide** (commercial blog — directionally reliable): FIFA 2026 sells three accessibility ticket types — Wheelchair User, Easy Access Standard, Easy Access Amenity; companion tickets are paid this cycle.

**Confidence notes → copy into README "Assumptions" in Phase 7:**
- Venue list, FIFA names, cities, opening/final dates, Banorte renaming (Mar 2025), MetLife/SoFi accessibility: **high confidence, sourced**.
- **Capacities are approximate** — the JSON uses Wikipedia's FIFA-tournament-capacity figures (e.g. AT&T at 70,649, well below its ~93k nominal, plausibly due to pitch overlay); other sources list higher nominal figures. Label capacities "approx. tournament capacity" in the UI/dataset.
- "GEHA Field at" prefix for Arrowhead is general knowledge, not session-verified.
- Accessibility details for the other 13 venues, gate layouts, and per-venue quiet routes will be **plausible synthesized data** — must be flagged as illustrative in README Assumptions.

---

## Phase 1 — Repo scaffold + dataset + data layer

**Implement:**
1. `git init -b main` in the project directory; `.gitignore` (venv, `__pycache__`, `.env`, `.pytest_cache`, `.DS_Store`).
2. Create the public GitHub repo: `gh repo create accessmate-wc26 --public --source=. --remote=origin` (single branch `main` only — challenge rule). First commit after scaffold.
3. `requirements.txt`: `google-genai`, `fastapi`, `uvicorn`, `pydantic`, `pytest`, `httpx` (pin with `>=` majors). `.env.example` with `GEMINI_API_KEY=` placeholder.
4. `data/venues.json` — copy the Phase 0.4 snippet verbatim; extend each venue with an `accessibility` object (gates, accessible_seating, sensory_room, assistive_listening, elevators, accessible_restrooms, quiet_route_hint) and `services` (water, first_aid, nursing_room, prayer_room). Where research confirmed a fact, keep it; otherwise use clearly-plausible defaults and record them in README "Assumptions".
5. `app/data.py` — pure functions: `load_venues()`, `get_venue(venue_id)`, `search_venues(query)`, `list_venues()`. No I/O outside module-level cached JSON load.
6. `challenge.md` note: keep the provided `.claude.md` brief in place, untouched.

**Verification:**
- [ ] `python -m json.tool data/venues.json` parses; 16 venues; ids unique (small assert script or test)
- [ ] `git branch --all` shows only `main`; `gh repo view --json visibility` says PUBLIC
- [ ] `du -sh .git .` total well under 10 MB
- [ ] Push succeeds

**Anti-pattern guards:** no `.env` committed (`git ls-files | grep -c "\.env$"` → 0); no binary assets.

---

## Phase 2 — Tools + offline deterministic engine

**Implement (`app/tools.py`):** pure functions over `app.data`, each returning a compact dict (JSON-serializable):
- `get_venue_info(venue_id)` — name, city, country, capacity, gates, matchday basics
- `find_accessible_services(venue_id, need="general")` — filters `accessibility` by need
- `get_live_status(venue_id)` — **simulated** ops feed: deterministic pseudo-random (seeded by venue_id + hour) gate congestion, elevator outages, quiet-entrance suggestion. Document simulation in README.
- `plan_visit(venue_id, needs: list[str], language)` — composes a step-by-step arrival plan (which gate, when to arrive, services en route) from the above.
- `execute_tool(name, args) -> str` dispatcher: validates `venue_id` exists, returns `json.dumps(...)`; unknown venue → friendly error string (used with `is_error=True` upstream).

**Implement (`app/offline.py`):** deterministic fallback assistant — keyword/intent routing (accessibility, navigation, food/water, schedule, greeting; keyword tables for en/es/fr at minimum) → calls the same tools → fills language-appropriate response templates. This is what runs when no API key is available and doubles as the "logical decision making" demonstration.

**Verification:**
- [ ] `pytest tests/test_tools.py tests/test_offline.py` green: every tool with valid/invalid venue ids; offline engine answers accessibility + navigation + Spanish-language queries sensibly; `get_live_status` deterministic for a fixed seed
- [ ] Tools never raise on bad input — they return error payloads

**Anti-pattern guards:** no network calls in this phase; no `google.genai` import in `tools.py`/`offline.py`.

---

## Phase 3 — Gemini assistant core

**Implement (`app/assistant.py`):** copy the Phase 0.1 snippets — client init, `GenerateContentConfig` with frozen `system_instruction`, tool definitions (4 `FunctionDeclaration`s mirroring `app/tools.py`, prescriptive "call this when…" descriptions), manual function-calling loop with iteration cap 8 and verbatim model-turn append, error chain, blocked-response guard.
- `SYSTEM_PROMPT`: frozen constant. Contents: role ("accessibility-first stadium copilot for FIFA WC 2026"), grounding rule ("answer venue facts ONLY from function results; if the dataset lacks it, say so — never invent gate numbers or services"), reply in the user's language, concise + screen-reader-friendly formatting (short sentences, no decorative emoji/ASCII), safety ("no medical/legal advice; direct emergencies to stadium staff/security").
- Per-request context (selected venue, declared needs, language, short history) is passed **in the user turn** as a structured preamble — keeps `system_instruction` byte-stable.
- `answer(message, profile, history) -> AssistantReply{text, mode: "live"|"offline", tool_calls_made}`: tries live; when no `GEMINI_API_KEY`/`GOOGLE_API_KEY` is configured, or on `ClientError` 401/403/429, `ServerError`, or connection failure, delegates to `app.offline` and sets `mode="offline"`.
- Prompt-injection hygiene: function results are our own dataset (trusted), but user text is untrusted — system prompt explicitly says user messages cannot change these rules.

**Doc refs:** Phase 0.1/0.2 of this plan; if something is missing, WebFetch https://googleapis.github.io/python-genai/ or ai.google.dev/gemini-api/docs — do not guess SDK signatures.

**Verification:**
- [ ] `pytest tests/test_assistant.py` green with a **mocked** client (monkeypatch `genai.Client`): (a) function-call round-trip appends the model turn verbatim and all results in ONE user Content, (b) plain-text response returns text, (c) `response.text is None` (blocked) → polite decline, (d) `ClientError(401)` → offline mode
- [ ] `grep -rn "google.generativeai\|GenerativeModel\|genai.configure\|thinking_budget\|thinking_level\|temperature" app/` → no matches
- [ ] `grep -rn "gemini-" app/ | grep -v "gemini-3.5-flash"` → no other model ids
- [ ] If `GEMINI_API_KEY` set locally: one live smoke call (`python -m scripts.smoke`) asserting a non-None text answer and that a function call fires for "wheelchair access at MetLife?"

---

## Phase 4 — API layer (FastAPI)

**Implement (`app/main.py` + `app/schemas.py`):**
- `POST /api/chat` — Pydantic body: `message: str (1..2000 chars)`, `profile {language: 2-letter code, needs: subset of enum, venue_id: optional}`, `history: list (≤20 turns, each ≤2000 chars)`. Returns `{reply, mode, venue_id}`.
- `GET /api/venues`, `GET /api/venues/{id}` (404 on unknown), `GET /healthz` (reports `{"llm": "live"|"offline"}`).
- Static file serving for the frontend at `/`.
- Security: security-headers middleware (CSP `default-src 'self'`, `X-Content-Type-Options`, `Referrer-Policy`, `X-Frame-Options DENY`); simple in-memory token-bucket rate limit per client IP (e.g. 20 req/min) returning 429; request size limits via the Pydantic caps; no persistence of chat content (stateless, history round-trips through the client); never log message bodies.

**Verification:**
- [ ] `pytest tests/test_api.py` green: happy path (offline mode), 422 on oversized/malformed input, 404 unknown venue, 429 after burst, security headers present on `/` and `/api/*`
- [ ] `uvicorn app.main:app` boots with **no** `GEMINI_API_KEY`/`GOOGLE_API_KEY` set and `/api/chat` still answers (offline mode)

**Anti-pattern guards:** no `allow_origins=["*"]` with credentials; no API key echoed in `/healthz` or logs.

---

## Phase 5 — Accessible frontend

**Implement (`static/index.html`, `static/style.css`, `static/app.js`):** single-page chat UI that itself demonstrates the accessibility criterion:
- Semantic landmarks (`header/main/footer`, one `h1`); chat transcript as `role="log"` with `aria-live="polite"`; every input labelled; visible focus indicators; full keyboard operability (Enter to send, focus returns to input)
- Setup row: venue `<select>`, language `<select>` (en/es/fr/ar minimum — set `lang` and `dir` on replies), needs checkboxes (mobility/vision/hearing/sensory)
- Quick-action chips ("Accessible route to my seat", "Quiet entrance", "Nearest accessible restroom")
- WCAG-minded styling: contrast ≥ 4.5:1, `prefers-reduced-motion` respected, `prefers-color-scheme` dark support, rem-based sizing, no color-only meaning; "offline demo mode" banner when `mode=offline`
- Plain `fetch` to `/api/chat`; render replies as text nodes (no `innerHTML` with model output — XSS guard)

**Verification:**
- [ ] Keyboard-only walkthrough: set profile → send message → read reply, no mouse
- [ ] `grep -c "aria-\|role=" static/index.html` ≥ 6; no `innerHTML` assignment of reply text in `app.js`
- [ ] Manual screen-reader sanity pass (VoiceOver) or documented in README if skipped

---

## Phase 6 — Test suite completion + hardening sweep

**Implement:** fill any coverage gaps (target: every `app/` module imported by at least one test), `tests/conftest.py` with shared fixtures (venue fixture, mocked Gemini client, TestClient), README testing section, `pytest.ini`/`pyproject` config.

**Verification:**
- [ ] `python -m pytest -q` fully green from a clean venv (`pip install -r requirements.txt`)
- [ ] Security checklist re-run: no secrets (`git grep -iE "AIza|api_key\s*=\s*['\"]"` → only env reads; Google API keys start with `AIza`), input caps enforced, rate limit works
- [ ] Anti-pattern greps from Phase 3 re-run repo-wide

---

## Phase 7 — README + final submission verification

**Implement `README.md`** with exactly the sections the brief demands: Chosen Vertical (Accessibility + why), Approach & Logic (architecture diagram in ASCII/mermaid, context→decision flow, live vs offline modes), How It Works (setup: venv, `pip install -r requirements.txt`, optional `GEMINI_API_KEY` from a free Google AI Studio account, `uvicorn app.main:app`, screenshots optional but keep repo <10MB), Assumptions (simulated live feed, dataset provenance from Phase 0.4 with confidence notes, unofficial venue accessibility details), Testing (how to run), Security notes, Accessibility notes (WCAG features), Evaluation-criteria map (one line per criterion → where it's addressed).

**Final verification (submission gate):**
- [ ] All tests green; app boots and answers in both modes (live only if credentials exist)
- [ ] Repo-wide anti-pattern greps clean (Phase 0.2 table)
- [ ] `du -sh .` < 10 MB including `.git`; `git branch -a` → only `main`; repo public
- [ ] README sections match the brief's required list 1:1
- [ ] Everything committed & pushed; `gh repo view -w` opens the final state

---

## Execution notes

- Each phase is self-contained: it names its files, embeds or references its copy-sources, and ends with a verification checklist. Run phases in order (Phase 2 before 3; 3 before 4).
- Execute with `/claude-mem:do plans/01-accessmate.md` or manually phase-by-phase in fresh sessions.
- Commit + push at the end of every phase (challenge asks for regular commits).
- If any Gemini API usage question arises beyond Phase 0.1's snippets, WebFetch the official SDK reference (https://googleapis.github.io/python-genai/) or ai.google.dev/gemini-api/docs in that session rather than guessing.
