from google import genai

from config.settings import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


client = genai.Client(
    api_key=GEMINI_API_KEY
)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(
                chunk.strip()
            )

        start += chunk_size - overlap

    return chunks


def create_embeddings(
    texts: list[str]
) -> list[list[float]]:

    if not texts:
        return []

    embeddings = []

    for text in texts:

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )

        embeddings.append(
            response.embeddings[0].values
        )

    return embeddings


def create_query_embedding(
    text: str
) -> list[float]:

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values
