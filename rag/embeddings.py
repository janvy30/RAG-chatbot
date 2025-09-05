from typing import List
from google import genai
from google.genai import types
import os

# Initialize client
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/embedding-001"


def embed_documents(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Embed a list of documents with batching (Gemini API allows max 100 per request)."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        res = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        if hasattr(res, "embeddings"):
            for item in res.embeddings:
                embeddings.append(item.values)
    return embeddings


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    res = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return res.embeddings[0].values
