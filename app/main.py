"""FastAPI application layer for AccessMate.

Serves the accessible chat UI (static/) and a small JSON API. Security posture:
- Strict security headers on every response (CSP default-src 'self', nosniff,
  no-referrer, X-Frame-Options DENY) — set by middleware.
- Per-client-IP token-bucket rate limit on the chat endpoint (429 on burst).
- Input caps enforced by the Pydantic models in app.schemas (422 on violation).
- Stateless: chat content is never persisted and message bodies are never
  logged; history round-trips through the client.
- No CORS middleware / no wildcard origins: the UI is same-origin with the API.
- The Gemini API key is never returned or logged; /healthz reports only the
  live/offline mode.
"""

import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import assistant, data
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Health,
    VenueList,
    VenueSummary,
)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

#: Chat requests allowed per client IP per minute (free-tier-friendly ceiling).
RATE_LIMIT_PER_MIN = 20

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; "
        "object-src 'none'; img-src 'self' data:; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


class TokenBucketLimiter:
    """In-memory per-key token bucket. Thread-safe; monotonic-clock based."""

    def __init__(self, capacity: int, refill_seconds: float) -> None:
        self.capacity = float(capacity)
        self._refill_rate = capacity / refill_seconds  # tokens per second
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Consume one token for ``key``; False when the bucket is empty."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self._refill_rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True

    def reset(self) -> None:
        """Clear all buckets (used by tests)."""
        with self._lock:
            self._buckets.clear()


rate_limiter = TokenBucketLimiter(RATE_LIMIT_PER_MIN, 60.0)

app = FastAPI(
    title="AccessMate API",
    description="Accessibility-first stadium copilot for the FIFA World Cup 2026.",
    version="1.0.0",
)
app.state.rate_limiter = rate_limiter


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Attach strict security headers to every response."""
    response = await call_next(request)
    for header, value in _SECURITY_HEADERS.items():
        response.headers[header] = value
    return response


def enforce_rate_limit(request: Request) -> None:
    """Reject the request with 429 when the client's bucket is empty."""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment and try again.",
        )


@app.get("/healthz", response_model=Health)
def healthz() -> Health:
    """Liveness probe reporting whether the live LLM is available."""
    return Health(status="ok", llm="live" if assistant.api_key_configured() else "offline")


@app.get("/api/venues", response_model=VenueList)
def list_venues() -> VenueList:
    """All tournament venues (public summary fields only)."""
    return VenueList(
        venues=[
            VenueSummary(
                id=v["id"],
                name=v["name"],
                city=v["city"],
                country=v["country"],
                capacity=v["capacity"],
            )
            for v in data.list_venues()
        ]
    )


@app.get("/api/venues/{venue_id}")
def get_venue(venue_id: str) -> dict:
    """Full record for one venue; 404 when the id is unknown."""
    venue = data.get_venue(venue_id)
    if venue is None:
        raise HTTPException(status_code=404, detail=f"Unknown venue id {venue_id!r}.")
    return venue


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest, _rate: None = Depends(enforce_rate_limit)) -> ChatResponse:
    """Answer a chat message (live Gemini when keyed, deterministic offline otherwise).

    The message body is never logged or persisted; history is supplied by the
    client each turn.
    """
    reply = assistant.answer(
        body.message,
        profile=body.profile.model_dump(),
        history=[turn.model_dump() for turn in body.history],
    )
    return ChatResponse(reply=reply.text, mode=reply.mode, venue_id=body.profile.venue_id)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page accessible chat UI."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


# Static assets (style.css, app.js) referenced by index.html as /static/*.
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
