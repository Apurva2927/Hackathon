import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def score_lead_with_llm(name: str, company: str, description: str) -> dict:
    """
    Call Groq LLM to score a lead 1-10 with a one-line reason.
    Returns: {"score": int, "reason": str}
    """
    prompt = (
        f"You are a sales lead scoring expert. "
        f"Score this lead from 1 to 10 (10 = highest quality) and give exactly one short reason.\n\n"
        f"Lead Name: {name}\n"
        f"Company: {company}\n"
        f"Description: {description}\n\n"
        f"Respond ONLY in this exact format:\n"
        f"Score: <number>\nReason: <one line reason>"
    )

    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=100,
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
