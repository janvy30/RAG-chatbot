import io
import re
from typing import List
from pypdf import PdfReader

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200) -> List[str]:
    """
    Cut text into overlapping chunks.
    chunk_size = approx characters per chunk.
    overlap = how many characters to overlap between chunks.
    """
    text = normalize_ws(text)
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    if text_len <= chunk_size:
        return [text]
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunks.append(text[start:end])
        start += max(1, chunk_size - overlap)
    return chunks

def read_any_text(raw: bytes, filename: str, content_type: str | None) -> str:
    """
    Try to extract text from raw bytes based on filename/content_type.
    Supports PDF (via pypdf) and plain text. Returns empty string on failure.
    """
    name = (filename or "").lower()
    ctype = (content_type or "").lower()

    # PDF
    if name.endswith(".pdf") or "pdf" in ctype:
        try:
            buf = io.BytesIO(raw)
            reader = PdfReader(buf)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            return "\n".join(pages)
        except Exception:
            return ""

    # Attempt to decode as utf-8/plain text
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""
