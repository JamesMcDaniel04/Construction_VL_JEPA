from __future__ import annotations

from maintenance_triage_copilot.domain.models import CorpusDocument, CorpusSourceType
from maintenance_triage_copilot.retrieval.chunking import chunk_document


def test_chunk_document_preserves_document_identity() -> None:
    document = CorpusDocument(
        document_id="doc-1",
        source_type=CorpusSourceType.sop,
        title="Breaker SOP",
        body=(
            "Step 1 inspect breaker. Step 2 isolate incoming disconnect. "
            "Step 3 verify lamp state."
        )
        * 4,
        tags=["breaker"],
    )
    chunks = chunk_document(document, chunk_size=90, chunk_overlap=10)
    assert len(chunks) >= 2
    assert all(chunk["document_id"] == "doc-1" for chunk in chunks)
    assert chunks[0]["source_type"] == CorpusSourceType.sop
