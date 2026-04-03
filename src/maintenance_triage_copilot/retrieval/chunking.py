"""Document chunking."""

from __future__ import annotations

from maintenance_triage_copilot.domain.models import CorpusDocument


def chunk_document(document: CorpusDocument, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Create overlapping text chunks with citation metadata."""
    text = document.body.strip()
    if not text:
        return []

    chunks: list[dict] = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        body = text[start:end].strip()
        if body:
            chunks.append(
                {
                    "chunk_id": f"{document.document_id}-chunk-{chunk_index}",
                    "document_id": document.document_id,
                    "title": document.title,
                    "source_type": document.source_type,
                    "equipment_family": document.equipment_family,
                    "text": body,
                    "tags": list(document.tags),
                }
            )
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
        chunk_index += 1
    return chunks
