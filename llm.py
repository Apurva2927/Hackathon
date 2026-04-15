import os
import json
import time
import re
import threading
import logging
from dataclasses import dataclass
from typing import Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("lead_scorer.llm")


SYSTEM_PROMPT = """You are a senior B2B sales lead qualification expert.
Score inbound leads on a 1-10 integer scale using the BANT+F framework:

  - Budget        : Signals the prospect can pay (company size, funding, revenue, ARR).
  - Authority     : Title or role suggests decision-making power.
  - Need          : Stated pain, problem, or use case that maps to a typical solution.
  - Timing        : Urgency, evaluation window, deadline, or active project.
  - Fit           : Industry, geography, and company stage match a typical ICP.

SCALE ANCHORS (use the full range — most leads are 4-7):
  10  Perfect ICP, decision-maker, explicit budget, buying within 30 days.
   8-9 Strong ICP and clear need; 1 of {budget, timing, authority} missing.
   6-7 Good fit, real need, but evaluation is early or signals are mixed.
   4-5 Plausible fit but weak signals (vague description, junior contact, no timing).
   2-3 Poor ICP fit, tire-kicker, student, or competitor research.
   1   Spam, fake, irrelevant, or hostile.

RULES
  - Be skeptical: missing information is a negative signal, not neutral.
  - Penalize free email domains, generic descriptions, students, and job seekers.
  - Reward specifics: named tools being replaced, team size, deadlines, budget figures.
  - The reason MUST be one line, <=20 words, and reference the SPECIFIC signal that
    drove the score (not generic praise). Start with the strongest signal.

OUTPUT FORMAT — respond with EXACTLY two lines, nothing else:
Score: <integer 1-10>
Reason: <one short sentence citing the strongest signal>"""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    max_retries: int
    retry_backoff_seconds: float
    total_timeout_seconds: float
    cache_ttl_seconds: int
    cache_max_entries: int
    circuit_breaker_threshold: int
    circuit_breaker_cooldown_seconds: int


def _get_config() -> LLMConfig:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Set it in your environment or .env file.")
    if api_key.lower() in {"your_groq_api_key_here", "changeme", "replace_me"}:
        raise RuntimeError("GROQ_API_KEY is a placeholder value. Set a real Groq API key in .env.")

    return LLMConfig(
        api_key=api_key,
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "120")),
        max_retries=max(1, int(os.getenv("GROQ_MAX_RETRIES", "3"))),
        retry_backoff_seconds=max(0.1, float(os.getenv("GROQ_RETRY_BACKOFF_SECONDS", "0.6"))),
        total_timeout_seconds=max(1.0, float(os.getenv("GROQ_TOTAL_TIMEOUT_SECONDS", "12"))),
        cache_ttl_seconds=max(10, int(os.getenv("LLM_CACHE_TTL_SECONDS", "600"))),
        cache_max_entries=max(10, int(os.getenv("LLM_CACHE_MAX_ENTRIES", "1000"))),
        circuit_breaker_threshold=max(1, int(os.getenv("LLM_CB_FAILURE_THRESHOLD", "5"))),
        circuit_breaker_cooldown_seconds=max(5, int(os.getenv("LLM_CB_COOLDOWN_SECONDS", "60"))),
    )


_client = None
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()
_circuit_lock = threading.Lock()
_consecutive_failures = 0
_circuit_open_until = 0.0


def _get_client(config: LLMConfig) -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.api_key)
    return _client


def _cache_key(name: str, company: str, description: str) -> str:
    normalized_description = re.sub(r"\s+", " ", description.strip().lower())
    return f"{name.strip().lower()}|{company.strip().lower()}|{normalized_description}"


def _cache_get(key: str) -> dict[str, Any] | None:
    with _cache_lock:
        record = _cache.get(key)
        if not record:
            return None

        expires_at, payload = record
        if expires_at < time.time():
            _cache.pop(key, None)
            return None
        return dict(payload)


def _cache_set(key: str, payload: dict[str, Any], config: LLMConfig) -> None:
    with _cache_lock:
        if len(_cache) >= config.cache_max_entries:
            oldest_key = next(iter(_cache), None)
            if oldest_key:
                _cache.pop(oldest_key, None)
        _cache[key] = (time.time() + config.cache_ttl_seconds, dict(payload))


def _is_circuit_open() -> bool:
    with _circuit_lock:
        return _circuit_open_until > time.time()


def _record_failure(config: LLMConfig) -> None:
    global _consecutive_failures, _circuit_open_until
    with _circuit_lock:
        _consecutive_failures += 1
        if _consecutive_failures >= config.circuit_breaker_threshold:
            _circuit_open_until = time.time() + config.circuit_breaker_cooldown_seconds


def _record_success() -> None:
    global _consecutive_failures, _circuit_open_until
    with _circuit_lock:
        _consecutive_failures = 0
        _circuit_open_until = 0.0


def _estimate_confidence(score: int, reason: str, cached: bool) -> float:
    confidence = 0.55
    if score in (1, 10):
        confidence += 0.15
    elif score in (2, 3, 8, 9):
        confidence += 0.1
    confidence += min(len(reason), 120) / 600
    if cached:
        confidence -= 0.05
    return round(max(0.2, min(0.99, confidence)), 2)


def _neutral_result(reason: str, model: str, cached: bool = False) -> dict[str, Any]:
    return {
        "score": 5,
        "reason": reason,
        "model": model,
        "confidence": _estimate_confidence(5, reason, cached),
        "token_usage": None,
        "cached": cached,
    }


def score_lead_with_llm(name: str, company: str, description: str, request_id: str | None = None) -> dict[str, Any]:
    """
    Call Groq LLM to score a lead 1-10 with a one-line reason.
    Returns: {"score": int, "reason": str}
    """
    config = _get_config()
    if _is_circuit_open():
        logger.warning("circuit_open request_id=%s", request_id)
        return _neutral_result("LLM temporarily unavailable. Assigned neutral score.", config.model)

    key = _cache_key(name=name, company=company, description=description)
    cached = _cache_get(key)
    if cached:
        cached["cached"] = True
        return cached

    client = _get_client(config)

    user_prompt = (
        "Score this lead.\n\n"
        f"Contact: {name}\n"
        f"Company: {company}\n"
        f"Description: {description}"
    )

    started = time.time()
    last_error = None
    for attempt in range(1, config.max_retries + 1):
        if time.time() - started > config.total_timeout_seconds:
            break
        try:
            chat_completion = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            response_text = (chat_completion.choices[0].message.content or "").strip()
            parsed = _parse_score(response_text)
            token_usage = getattr(getattr(chat_completion, "usage", None), "total_tokens", None)
            result = {
                "score": parsed["score"],
                "reason": parsed["reason"],
                "model": config.model,
                "confidence": _estimate_confidence(parsed["score"], parsed["reason"], cached=False),
                "token_usage": token_usage,
                "cached": False,
            }
            _cache_set(key, result, config)
            _record_success()
            return result
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "llm_attempt_failed request_id=%s attempt=%s error=%s",
                request_id,
                attempt,
                last_error,
            )
            if attempt < config.max_retries:
                time.sleep(config.retry_backoff_seconds * attempt)

    _record_failure(config)
    if last_error:
        logger.error("llm_failed request_id=%s error=%s", request_id, last_error)
    return _neutral_result("LLM unavailable after retries. Assigned neutral score.", config.model)


def _parse_score(response_text: str) -> dict:
    """Parse the LLM response to extract score and reason."""
    score = 5
    reason = "Unable to parse LLM response."

    # Prefer JSON parsing first, but tolerate non-JSON fallback formats.
    try:
        payload = json.loads(response_text)
        if isinstance(payload, dict):
            raw_score = payload.get("score", score)
            raw_reason = payload.get("reason", reason)

            score = int(raw_score)
            score = max(1, min(10, score))
            reason = str(raw_reason).strip() or reason
            return {"score": score, "reason": reason[:240]}
    except Exception:
        pass

    score_match = re.search(r"Score\s*[:=-]?\s*(\d+)", response_text, re.IGNORECASE)
    reason_match = re.search(r"Reason\s*[:=-]?\s*(.+)", response_text, re.IGNORECASE)

    if score_match:
        score = int(score_match.group(1))
        score = max(1, min(10, score))
    else:
        first_num = re.search(r"\b(10|[1-9])\b", response_text)
        if first_num:
            score = int(first_num.group(1))

    if reason_match:
        reason = reason_match.group(1).strip()
    elif response_text:
        reason = response_text.strip().splitlines()[0]

    return {"score": score, "reason": reason[:240]}
