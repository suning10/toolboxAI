"""
Small dependency-free text chunker - splits on paragraph boundaries
first (keeps SOP steps/sections intact where possible) and only falls
back to a hard character split for paragraphs longer than chunk_size.
Overlap carries trailing context from one chunk into the next so a
step split across a chunk boundary doesn't lose meaning.
"""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = current[-overlap:] + "\n\n" + para if current and overlap else para

        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - overlap :] if overlap else current[chunk_size:]

    if current:
        chunks.append(current)

    return chunks
