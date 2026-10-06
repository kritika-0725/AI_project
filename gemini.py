from google import genai

from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please add it to your .env file."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


SYSTEM_PROMPT = """
You are NexaMind AI, an intelligent AI assistant.

Your job is to:
- Answer questions accurately.
- Explain concepts clearly.
- Be friendly and helpful.
- Use document context when it is provided.
- Never invent information from uploaded documents.
- If the answer is not present in the document,
  clearly tell the user.
"""


def generate_response(
    question: str,
    context: str = "",
    history: list | None = None,
) -> str:

    history = history or []

    conversation = ""

    for item in history[-10:]:
        conversation += (
            f"User: {item['user']}\n"
            f"NexaMind AI: {item['assistant']}\n\n"
        )

    prompt = f"""
{SYSTEM_PROMPT}

CONVERSATION HISTORY:
{conversation}

DOCUMENT CONTEXT:
{context if context else "No document has been provided."}

USER QUESTION:
{question}

Give the best possible answer.
"""

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        return response.text or (
            "I could not generate a response."
        )

    except Exception as error:

        return (
            "Sorry, I encountered an error while "
            f"communicating with Gemini.\n\nError: {error}"
        )
