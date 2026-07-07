/*
  AccessMate front-end logic.

  Security posture:
    - ALL dynamic text (user input and assistant replies) is rendered with
      document.createElement + textContent / createTextNode. Dynamic content is
      never assigned to an element's markup property, so untrusted model/user
      text can never be parsed as markup (no XSS sink).
    - No inline event handlers; every listener is attached with
      addEventListener. This keeps a strict CSP (default-src 'self') working.

  Accessibility posture:
    - New messages are appended into a role="log" aria-live="polite" region,
      so screen readers announce each new reply.
    - Author is labelled in text ("You:" / "AccessMate:"), not colour alone.
    - Language / direction (LTR / RTL for Arabic) is reflected on the transcript.
*/

"use strict";

// ---- Constants ----
const HISTORY_LIMIT = 20;     // keep the last N turns (per API contract)
const MAX_MESSAGE_LEN = 2000; // matches the backend's 1..2000 bound

// Human-readable author labels; used as text prefixes, not colour cues.
const AUTHOR_LABEL = {
  user: "You",
  assistant: "AccessMate",
  status: "AccessMate",
};

// ---- State ----
/** @type {{role: "user"|"assistant", text: string}[]} */
let history = [];

// ---- Element references ----
const els = {
  venue: document.getElementById("venue-select"),
  language: document.getElementById("language-select"),
  transcript: document.getElementById("transcript"),
  form: document.getElementById("chat-form"),
  input: document.getElementById("message-input"),
  send: document.getElementById("send-button"),
  banner: document.getElementById("offline-banner"),
};

// =====================================================================
// Rendering helpers — every one builds DOM via createElement/textContent.
// =====================================================================

/**
 * Append a message bubble to the transcript.
 * @param {"user"|"assistant"|"status"} role
 * @param {string} text
 * @returns {HTMLElement} the created message element
 */
function appendMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg--" + role;

  // Errors ("status" bubbles) are announced immediately by assistive tech,
  // not queued behind the polite log.
  if (role === "status") {
    wrap.setAttribute("role", "alert");
  }

  const author = document.createElement("span");
  author.className = "msg__author";
  // textContent keeps this a plain string, never parsed as HTML.
  author.textContent = AUTHOR_LABEL[role] + ":";

  const body = document.createElement("span");
  body.className = "msg__text";
  body.textContent = text;

  wrap.appendChild(author);
  wrap.appendChild(body);

  // Arabic replies read right-to-left; everything else left-to-right.
  if (els.language.value === "ar") {
    wrap.setAttribute("dir", "rtl");
  } else {
    wrap.setAttribute("dir", "ltr");
  }

  els.transcript.appendChild(wrap);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return wrap;
}

/** Add a "typing" placeholder bubble; returns it so it can be updated later. */
function appendPending() {
  const wrap = appendMessage("assistant", "");
  wrap.classList.add("msg--pending");
  const body = wrap.querySelector(".msg__text");
  if (body) {
    body.textContent = ""; // CSS renders an ellipsis for the pending state
  }
  return wrap;
}

// =====================================================================
// History management
// =====================================================================

/**
 * Push a turn and cap the array at the last HISTORY_LIMIT entries.
 * @param {"user"|"assistant"} role
 * @param {string} text
 */
function pushHistory(role, text) {
  history.push({ role, text });
  if (history.length > HISTORY_LIMIT) {
    history = history.slice(history.length - HISTORY_LIMIT);
  }
}

// =====================================================================
// Offline-mode banner
// =====================================================================

/** Show/hide the non-alarming offline banner based on the reply mode. */
function setOfflineBanner(isOffline) {
  els.banner.hidden = !isOffline;
}

// =====================================================================
// Language / direction
// =====================================================================

/** Reflect the chosen language on the transcript and the composer input. */
function applyLanguage() {
  const lang = els.language.value;
  const dir = lang === "ar" ? "rtl" : "ltr";
  els.transcript.setAttribute("lang", lang);
  els.transcript.setAttribute("dir", dir);
  // The composer holds text typed in the chosen language, so screen readers
  // and the caret/text direction must follow it too (RTL for Arabic).
  els.input.setAttribute("lang", lang);
  els.input.setAttribute("dir", dir);
}

// =====================================================================
// Venue loading
// =====================================================================

/** Fetch the venue list and populate the select; fail gracefully. */
async function loadVenues() {
  try {
    const res = await fetch("/api/venues", { headers: { Accept: "application/json" } });
    if (!res.ok) {
      throw new Error("HTTP " + res.status);
    }
    const data = await res.json();
    const venues = Array.isArray(data.venues) ? data.venues : [];
    for (const v of venues) {
      const opt = document.createElement("option");
      opt.value = String(v.id);
      // Build a readable label from name + city/country (text only).
      const parts = [v.name];
      if (v.city) {
        parts.push(v.city);
      } else if (v.country) {
        parts.push(v.country);
      }
      opt.textContent = parts.join(" — ");
      els.venue.appendChild(opt);
    }
  } catch (err) {
    // Non-fatal: the app still works without a venue. Leave a hint in the list.
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Stadium list unavailable — you can still ask questions";
    opt.disabled = true;
    els.venue.appendChild(opt);
  }
}

// =====================================================================
// Health check (optional): pre-set the offline banner if the LLM is down.
// =====================================================================

async function checkHealth() {
  try {
    const res = await fetch("/api/healthz", { headers: { Accept: "application/json" } });
    if (!res.ok) {
      return;
    }
    const data = await res.json();
    if (data && data.llm === "offline") {
      setOfflineBanner(true);
    }
  } catch (err) {
    // Ignore — the banner will settle correctly on the first chat reply.
  }
}

// =====================================================================
// Sending a message
// =====================================================================

/** Build the profile object from the current form controls. */
function readProfile() {
  const needs = Array.from(
    document.querySelectorAll('input[name="needs"]:checked')
  ).map((el) => el.value);

  return {
    language: els.language.value,
    needs: needs,
    venue_id: els.venue.value ? els.venue.value : null,
  };
}

let inFlight = false;

/**
 * Send a message to the backend and render the reply as it streams in.
 *
 * The reply arrives as newline-delimited JSON frames from /api/chat/stream:
 *   {"type":"meta","mode":...}  then  {"type":"delta","text":...} pieces.
 * Deltas are typed into a bubble that is aria-hidden during streaming, so a
 * screen reader is NOT spammed with each fragment; on completion the streaming
 * bubble is swapped for a single, final bubble that is announced exactly once.
 * @param {string} rawText
 */
async function sendMessage(rawText) {
  const text = rawText.trim();
  if (!text || inFlight) {
    return;
  }
  if (text.length > MAX_MESSAGE_LEN) {
    appendMessage("status", "Message is too long. Please shorten it to 2000 characters or fewer.");
    return;
  }

  // Optimistically render the user's message and record it in history.
  appendMessage("user", text);
  pushHistory("user", text);

  // Lock the UI while the request is in flight. aria-busy tells assistive
  // tech the log is updating; queued changes are announced when it clears.
  inFlight = true;
  els.send.disabled = true;
  els.input.value = "";
  els.transcript.setAttribute("aria-busy", "true");

  // The streaming bubble is hidden from assistive tech: partial tokens must not
  // be announced one-by-one in the aria-live log. The completed reply is added
  // as a fresh (announced) bubble once the stream finishes.
  const pending = appendPending();
  pending.setAttribute("aria-hidden", "true");
  const pendingBody = pending.querySelector(".msg__text");

  const payload = {
    message: text,
    profile: readProfile(),
    history: history.slice(0, HISTORY_LIMIT),
  };

  let replyText = "";
  let started = false; // has the first delta arrived (bubble left pending state)?
  let errored = false;

  const handleFrame = (line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }
    let frame;
    try {
      frame = JSON.parse(trimmed);
    } catch (e) {
      return; // ignore a malformed frame rather than break the whole stream
    }
    if (frame.type === "meta") {
      setOfflineBanner(frame.mode === "offline");
    } else if (frame.type === "delta" && typeof frame.text === "string") {
      if (!started) {
        started = true;
        pending.classList.remove("msg--pending");
      }
      replyText += frame.text;
      if (pendingBody) {
        pendingBody.textContent = replyText;
      }
      els.transcript.scrollTop = els.transcript.scrollHeight;
    } else if (frame.type === "error") {
      errored = true;
    }
  };

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/x-ndjson",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok || !res.body) {
      throw new Error("HTTP " + res.status);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        handleFrame(buffer.slice(0, nl));
        buffer = buffer.slice(nl + 1);
      }
    }
    handleFrame(buffer); // final line, if not newline-terminated

    if (errored && !replyText) {
      throw new Error("stream error");
    }

    // Swap the (aria-hidden) streaming bubble for a final, announced bubble so
    // screen readers hear the complete reply exactly once.
    pending.remove();
    const finalText = replyText || "(No reply received.)";
    appendMessage("assistant", finalText);
    if (replyText) {
      pushHistory("assistant", replyText);
    }
  } catch (err) {
    // Polite, non-technical inline error — never a raw stack trace.
    pending.remove();
    appendMessage(
      "status",
      "Sorry, I could not reach the assistant just now. Please check your connection and try again."
    );
  } finally {
    inFlight = false;
    els.send.disabled = false;
    els.transcript.setAttribute("aria-busy", "false");
    els.input.focus(); // return focus to the input after sending
  }
}

// =====================================================================
// Event wiring — all via addEventListener (no inline handlers).
// =====================================================================

function wireEvents() {
  // Submit (covers Enter key and the Send button).
  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(els.input.value);
  });

  // Language change updates transcript lang/dir.
  els.language.addEventListener("change", applyLanguage);

  // Quick-action chips: fill and send their prompt.
  const chips = document.querySelectorAll(".chip");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt") || chip.textContent || "";
      sendMessage(prompt);
    });
  });
}

// =====================================================================
// Welcome message
// =====================================================================

// Static, first-load greeting. Rendered via the same textContent-only path as
// every other bubble (no markup), and deliberately NOT pushed into `history`,
// so the API contract (only real user/assistant turns) is unchanged.
const WELCOME_TEXT =
  "Hello! I'm AccessMate, your accessibility copilot for the FIFA World Cup 2026. " +
  "Pick a stadium and your access needs, or just ask me anything — wheelchair " +
  "routes, sensory rooms, assistive listening, and live gate status.";

function renderWelcome() {
  appendMessage("assistant", WELCOME_TEXT);
}

// =====================================================================
// Init
// =====================================================================

function init() {
  applyLanguage();
  renderWelcome();
  wireEvents();
  loadVenues();
  checkHealth();
}

// `defer` guarantees the DOM is parsed before this runs, but guard anyway.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
