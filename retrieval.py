"""
Document Retrieval + Parsing & Cleaning + Semantic Chunking
"""
import re
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PrivacyLensBot/1.0; +https://huggingface.co/spaces)"
}

MIN_HEADING_LEN = 3
MAX_HEADING_LEN = 120

CHUNK_TARGET_CHARS = 3000   # ~500-800 tokens
CHUNK_OVERLAP_CHARS = 400


def fetch_and_parse(url: str | None = None, text_override: str | None = None) -> list[dict]:
    """
    Returns a list of {"section": str, "content": str} preserving document structure.
    """
    if text_override is not None:
        raw_text = text_override
        return _sections_from_plaintext(raw_text)

    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")

    if "html" not in content_type and "<html" not in resp.text[:500].lower():
        # plaintext-ish response
        return _sections_from_plaintext(resp.text)

    return _sections_from_html(resp.text)


def _sections_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    # Strip clutter
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form", "iframe"]):
        tag.decompose()

    body = soup.body or soup

    sections: list[dict] = []
    current_heading = "Introduction"
    current_buf: list[str] = []

    def flush():
        text = " ".join(current_buf).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > 40:
            sections.append({"section": current_heading.strip(), "content": text})

    heading_tags = {"h1", "h2", "h3", "h4"}
    block_tags = {"p", "li", "td", "div"}

    for el in body.find_all(list(heading_tags | block_tags)):
        if el.name in heading_tags:
            heading_text = el.get_text(strip=True)
            if MIN_HEADING_LEN <= len(heading_text) <= MAX_HEADING_LEN:
                flush()
                current_heading = heading_text
                current_buf = []
        else:
            # skip if it's inside an already-captured heading area with no real text
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) > 15:
                # avoid re-adding nested duplicate text (div-in-div)
                if not el.find(list(heading_tags)):
                    current_buf.append(txt)

    flush()

    # Deduplicate near-identical sections (common with nested divs)
    seen = set()
    deduped = []
    for s in sections:
        key = s["content"][:200]
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    if not deduped:
        # fallback: just grab all text as one section
        text = re.sub(r"\s+", " ", body.get_text(" ", strip=True))
        if len(text) > 40:
            deduped = [{"section": "Full Document", "content": text}]

    return deduped


def _sections_from_plaintext(text: str) -> list[dict]:
    """
    Split pasted plaintext policy into sections using heading-like lines
    (short lines, Title Case / ALL CAPS, or markdown-style headers).
    """
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")

    sections: list[dict] = []
    current_heading = "Introduction"
    current_buf: list[str] = []

    def is_heading(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) > MAX_HEADING_LEN:
            return False
        if stripped.startswith("#"):
            return True
        if len(stripped) < 80 and stripped == stripped.title() and not stripped.endswith("."):
            return True
        if stripped.isupper() and 3 < len(stripped) < 80:
            return True
        if re.match(r"^\d+(\.\d+)*[\.\)]\s+[A-Z]", stripped) and len(stripped) < 100:
            return True
        return False

    def flush():
        text_block = " ".join(current_buf).strip()
        text_block = re.sub(r"\s+", " ", text_block)
        if len(text_block) > 40:
            sections.append({"section": current_heading.strip("# ").strip(), "content": text_block})

    for line in lines:
        if is_heading(line):
            flush()
            current_heading = line.strip()
            current_buf = []
        else:
            if line.strip():
                current_buf.append(line.strip())
    flush()

    if not sections:
        clean = re.sub(r"\s+", " ", text).strip()
        sections = [{"section": "Full Document", "content": clean}]

    return sections


def chunk_sections(sections: list[dict]) -> list[dict]:
    """
    Hierarchical chunking: chunk within each section by size with overlap.
    Returns list of {"section": str, "content": str, "chunk_id": str}
    """
    chunks = []
    chunk_idx = 0

    for sec in sections:
        content = sec["content"]
        section_name = sec["section"]

        if len(content) <= CHUNK_TARGET_CHARS:
            chunks.append({
                "section": section_name,
                "content": content,
                "chunk_id": f"chunk_{chunk_idx}",
            })
            chunk_idx += 1
            continue

        start = 0
        while start < len(content):
            end = min(start + CHUNK_TARGET_CHARS, len(content))
            # try to break on sentence boundary
            if end < len(content):
                boundary = content.rfind(". ", start, end)
                if boundary != -1 and boundary > start + 200:
                    end = boundary + 1

            piece = content[start:end].strip()
            if len(piece) > 40:
                chunks.append({
                    "section": section_name,
                    "content": piece,
                    "chunk_id": f"chunk_{chunk_idx}",
                })
                chunk_idx += 1

            if end >= len(content):
                break
            start = max(end - CHUNK_OVERLAP_CHARS, start + 1)

    return chunks
