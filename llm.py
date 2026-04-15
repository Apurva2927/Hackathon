import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


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


def score_lead_with_llm(name: str, company: str, description: str) -> dict:
    """
    Call Groq LLM to score a lead 1-10 with a one-line reason.
    Returns: {"score": int, "reason": str}
    """
    user_prompt = (
        f"Score this lead.\n\n"
        f"Contact: {name}\n"
        f"Company: {company}\n"
        f"Description: {description}"
    )

    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=120,
    )

    response_text = chat_completion.choices[0].message.content
    return _parse_score(response_text)


def _parse_score(response_text: str) -> dict:
    """Parse the LLM response to extract score and reason."""
    score = 5  # default
    reason = "Unable to parse LLM response."

    score_match = re.search(r"Score:\s*(\d+)", response_text, re.IGNORECASE)
    reason_match = re.search(r"Reason:\s*(.+)", response_text, re.IGNORECASE)

    if score_match:
        score = int(score_match.group(1))
        score = max(1, min(10, score))  # clamp to 1-10
    if reason_match:
        reason = reason_match.group(1).strip()

    return {"score": score, "reason": reason}
