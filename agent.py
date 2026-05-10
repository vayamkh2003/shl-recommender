"""
agent.py — Gemini-powered conversational SHL assessment recommender.

Uses the new `google-genai` SDK (google.generativeai is deprecated).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from google import genai
from google.genai import types

from catalog import Assessment, CatalogIndex

logger = logging.getLogger(__name__)

GEMINI_MODEL          = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
MAX_CATALOG_IN_PROMPT = int(os.getenv("MAX_CATALOG_IN_PROMPT", "60"))
REQUEST_TIMEOUT_MS    = int(os.getenv("GEMINI_TIMEOUT", "25000"))   # milliseconds
MAX_RETRIES           = int(os.getenv("GEMINI_RETRIES", "3"))


# ─────────────────────────────────────────────────────────────────────────────
# Lazy client — created once, not at import time
# ─────────────────────────────────────────────────────────────────────────────

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta",timeout=REQUEST_TIMEOUT_MS),
        )
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert SHL assessment consultant. You help hiring managers and
recruiters build the right assessment shortlist through natural dialogue.
You have deep knowledge of the SHL product catalog and always reason from it.

══════════════════════════════════════
HARD RULES — NEVER VIOLATE THESE
══════════════════════════════════════
1. CATALOG ONLY. Only recommend assessments that appear in the CATALOG section
   below. Copy every name and URL verbatim. Never invent assessments.

2. CLARIFY BEFORE RECOMMENDING. If the query is vague, ask the single most
   important missing piece — not all at once. Good things to clarify:
   - Role / job title
   - Seniority (graduate, entry, mid, senior IC, manager, director, C-suite)
   - Backend vs frontend vs full-stack (for technical roles)
   - Skills / competencies to measure
   - Selection vs development vs 360
   - Duration constraints
   - Language requirements
   - High-volume screening vs finalist-stage depth

3. OPQ32r DEFAULT. For any selection shortlist always include OPQ32r as the
   personality component unless the user explicitly removes it. Tell the user
   it is included and invite them to drop it if not needed.

4. STAY IN SCOPE. Politely refuse:
   - General hiring advice (salary, interviews, headcount)
   - Legal / compliance questions → "that is a legal question outside what I
     can advise on; your legal or compliance team is the right resource."
   - Anything not about SHL assessments.

5. REFINEMENT. When the user adds/removes/swaps items, update incrementally.
   Carry forward all items not mentioned in the update. Never restart.

6. NO SHORTER ALTERNATIVE IF NONE EXISTS. If the user asks for a shorter test
   and none exists in the catalog, say so clearly. Do not invent one.

7. CATALOG GAPS. If the catalog lacks a test for a skill, acknowledge the gap
   and suggest the closest alternatives with your reasoning.

8. COMPARISON. Answer comparison questions using only catalog data — purpose,
   scope, norm group, when to use each. Keep the existing shortlist active.

9. ACCENT / LANGUAGE. For SVAR tests, clarify which English accent variant is
   needed (US, UK, Australian, Indian) before recommending.

10. end_of_conversation = true ONLY when the user explicitly confirms they are
    done ("confirmed", "locking it in", "perfect", "that covers it",
    "that works, thanks"). Do not set it prematurely.

══════════════════════════════════════
RESPONSE FORMAT — STRICT JSON ONLY
══════════════════════════════════════
Return ONLY a JSON object. No markdown fences. No prose outside the object.

{{
  "reply": "<your conversational message>",
  "recommendations": <null or list>,
  "end_of_conversation": false
}}

RECOMMENDATIONS rules:
  null   → still clarifying, answering a side-question, or refusing
  []     → explicitly nothing to recommend this turn
  [1-10] → committed shortlist; repeat the FULL list every turn it is active

test_type codes (comma-separated for multi-category):
  K=Knowledge & Skills  A=Ability & Aptitude  P=Personality & Behavior
  B=Biodata/SJT  C=Competencies  D=Development/360  E=Exercises  S=Simulations

══════════════════════════════════════
CATALOG — YOUR ONLY SOURCE OF TRUTH
══════════════════════════════════════
{catalog}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Few-shot examples — behavior patterns only, NO hardcoded URLs
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT: list[dict] = [
    # Vague query → single clarifying question
    {
        "role": "user",
        "content": "We need a solution for senior leadership.",
    },
    {
        "role": "model",
        "content": '{"reply": "Happy to help. Is this for selecting external candidates against a leadership benchmark, or developmental feedback for leaders already in role?", "recommendations": null, "end_of_conversation": false}',
    },
    # Catalog gap → acknowledge + offer closest alternatives
    {
        "role": "user",
        "content": "I need a Rust programming test.",
    },
    {
        "role": "model",
        "content": '{"reply": "SHL\'s catalog does not include a Rust-specific test. The closest available options are Smart Interview Live Coding (your panel sets the Rust tasks), Linux Programming for systems depth, and Networking and Implementation for infrastructure breadth. Want me to build a shortlist from these?", "recommendations": null, "end_of_conversation": false}',
    },
    # Legal question → clean refusal
    {
        "role": "user",
        "content": "Are we legally required to test all staff who handle patient records?",
    },
    {
        "role": "model",
        "content": '{"reply": "That is a legal compliance question outside what I can advise on. Your legal or compliance team is the right resource for regulatory obligations. I can tell you what each assessment measures — shall we continue with the shortlist?", "recommendations": null, "end_of_conversation": false}',
    },
    # No shorter alternative → hold firm
    {
        "role": "user",
        "content": "Can you replace OPQ32r with something shorter? Candidates say it takes too long.",
    },
    {
        "role": "model",
        "content": '{"reply": "OPQ32r is the most comprehensive personality instrument in the catalog for this use case and there is no shorter alternative that provides equivalent coverage. I recommend keeping it.", "recommendations": null, "end_of_conversation": false}',
    },
    # Explicit confirmation → end_of_conversation true
    {
        "role": "user",
        "content": "Perfect, that is what we need.",
    },
    {
        "role": "model",
        "content": '{"reply": "Great. The shortlist is confirmed.", "recommendations": null, "end_of_conversation": true}',
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_catalog_block(results: list[Assessment]) -> str:
    rows = []
    for a in results:
        rows.append({
            "name":             a.name,
            "url":              a.url,
            "test_type":        a.test_type,
            "categories":       a.categories,
            "job_levels":       a.job_levels,
            "duration_minutes": a.duration_minutes,
            "remote":           a.remote_testing,
            "adaptive":         a.adaptive,
            "description":      a.description[:350],
        })
    return json.dumps(rows, indent=2)


def _extract_query_from_history(messages: list[dict]) -> str:
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_texts[-4:])


def _build_contents(messages: list[dict], system_prompt: str = "") -> list[types.Content]:
    """
    Build the `contents` list for the new SDK.

    Layout:  few-shot pairs  +  real history (all but last msg)
             The last user message is passed separately as the final item.

    Gemini requires strictly alternating user/model roles.
    Consecutive same-role turns are merged into one Content object.
    """
    raw: list[tuple[str, str]] = []  # (role, text)
    if system_prompt:
        raw.append(("user", f"[SYSTEM INSTRUCTIONS]\n{system_prompt}\n[END SYSTEM INSTRUCTIONS]\nAcknowledge these instructions."))
        raw.append(("model", '{"reply": "Understood. I will follow all instructions.", "recommendations": null, "end_of_conversation": false}'))
    # Few-shot examples
    for shot in FEW_SHOT:
        raw.append((shot["role"], shot["content"]))

    # Real conversation history (everything except the last user message)
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        raw.append((role, msg["content"]))

    # Merge consecutive same-role entries
    merged: list[tuple[str, list[str]]] = []
    for role, text in raw:
        if merged and merged[-1][0] == role:
            merged[-1][1].append(text)
        else:
            merged.append((role, [text]))

    # Must start with "user"
    while merged and merged[0][0] != "user":
        merged.pop(0)

    # Convert to types.Content
    contents = [
        types.Content(
            role=role,
            parts=[types.Part(text=t) for t in texts],
        )
        for role, texts in merged
    ]

    # Append the final user message
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=messages[-1]["content"])],
        )
    )

    return contents


def _call_gemini_with_retry(
    system_prompt: str,
    contents: list[types.Content],
) -> str:
    """
    Call the new Gemini SDK with retry on rate-limit / transient errors.
    Returns raw text. Raises RuntimeError on total failure.
    """
    client = _get_client()
    config = types.GenerateContentConfig(
        # system_instruction=system_prompt,
        temperature=0.15,
        max_output_tokens=2048,
    )

    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )

            # Safety / empty response guard
            if not response.candidates:
                raise RuntimeError("Gemini returned no candidates.")

            candidate = response.candidates[0]
            finish = str(getattr(candidate, "finish_reason", "")).upper()

            if finish in ("SAFETY", "RECITATION", "OTHER"):
                logger.warning("Gemini blocked: finish_reason=%s", finish)
                return json.dumps({
                    "reply": "I can't respond to that request. Could you rephrase?",
                    "recommendations": None,
                    "end_of_conversation": False,
                })

            return response.text

        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            
            if "404" in err_str or "not_found" in err_str:
                logger.error("Model not found: %s", exc)
                break  # don't retry, it won't help

            if any(k in err_str for k in ("429", "quota", "rate", "resource exhausted")):
                wait = 2 ** attempt
                logger.warning("Rate limit (attempt %d/%d). Retrying in %ds…",
                               attempt + 1, MAX_RETRIES, wait)
                time.sleep(wait)
                continue
            
            if any(k in err_str for k in ("timeout", "deadline", "timed out")):
                logger.warning("Timeout on attempt %d/%d.", attempt + 1, MAX_RETRIES)
                if attempt < MAX_RETRIES - 1:
                    continue
                break

            logger.error("Gemini error (attempt %d): %s", attempt + 1, exc)
            break

    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_exc}")


def _parse_agent_response(raw: str) -> dict:
    """Extract and validate JSON from model output."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in model output:\n{raw[:400]}")

    obj = json.loads(match.group())
    obj.setdefault("reply", "I am here to help you find the right assessments.")
    obj.setdefault("end_of_conversation", False)

    recs = obj.get("recommendations")
    if isinstance(recs, list):
        clean = []
        for r in recs:
            if isinstance(r, dict) and "name" in r and "url" in r:
                clean.append({
                    "name":      str(r["name"]).strip(),
                    "url":       str(r["url"]).strip(),
                    "test_type": str(r.get("test_type", "K")).strip(),
                })
        obj["recommendations"] = clean[:10]
    else:
        obj["recommendations"] = None

    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(messages: list[dict[str, Any]], index: CatalogIndex) -> dict:
    """
    messages: [{"role": "user"|"assistant", "content": str}, ...]
    Returns:  {"reply": str, "recommendations": None|list, "end_of_conversation": bool}
    """
    # 1. FAISS retrieval
    query = _extract_query_from_history(messages)
    retrieved = index.search(query, top_k=MAX_CATALOG_IN_PROMPT)
    catalog_block = _build_catalog_block(retrieved)

    # 2. System prompt with catalog injected
    system = SYSTEM_PROMPT.format(catalog=catalog_block)

    # 3. Build contents list (few-shots + history + last user msg)
    contents = _build_contents(messages, system)

    # 4. Call Gemini
    try:
        raw_text = _call_gemini_with_retry(system, contents)
        logger.debug("Gemini raw response:\n%s", raw_text)
    except RuntimeError as exc:
        logger.error("Gemini call failed: %s", exc)
        return {
            "reply": "I'm having trouble reaching the AI service. Please try again in a moment.",
            "recommendations": None,
            "end_of_conversation": False,
        }

    # 5. Parse JSON
    try:
        return _parse_agent_response(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("JSON parse error: %s\nRaw: %s", exc, raw_text)
        return {
            "reply": "I encountered a formatting issue. Could you rephrase your request?",
            "recommendations": None,
            "end_of_conversation": False,
        }