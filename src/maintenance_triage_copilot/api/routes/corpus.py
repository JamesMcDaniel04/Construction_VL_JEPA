"""Corpus ingestion endpoints — documents, incidents, and file uploads."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from maintenance_triage_copilot.domain.models import (
    CorpusDocument,
    CorpusSourceType,
    IncidentRecord,
)

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.post("/documents")
async def add_document(document: CorpusDocument, request: Request) -> dict[str, object]:
    return cast(dict[str, object], request.app.state.service.add_document(document))


@router.post("/incidents")
async def add_incident(incident: IncidentRecord, request: Request) -> dict[str, object]:
    return cast(dict[str, object], request.app.state.service.add_incident(incident))


@router.post(
    "/upload",
    responses={
        400: {"description": "Empty file upload"},
        415: {"description": "Unsupported document type"},
    },
)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    document_id: Annotated[str, Form()],
    source_type: Annotated[CorpusSourceType, Form()] = CorpusSourceType.manual,
    equipment_family: Annotated[str, Form()] = "electrical_panel_family_a",
    title: Annotated[str | None, Form()] = None,
    tags: Annotated[str, Form()] = "",
) -> dict[str, object]:
    """Upload a PDF, TXT, or Markdown file for corpus ingestion.

    The file is parsed, chunked, and indexed for retrieval.
    """
    filename = file.filename or "document"
    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")

    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".rst"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported document type: {suffix}. Expected .pdf, .txt, or .md",
        )

    # Parse from bytes in memory to avoid sync tempfile in async context
    document = _parse_bytes(
        data,
        suffix=suffix,
        document_id=document_id,
        source_type=source_type,
        equipment_family=equipment_family,
        title=title or Path(filename).stem,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
    )

    result = request.app.state.service.add_document(document)
    return cast(dict[str, object], result)


def _parse_bytes(
    data: bytes,
    *,
    suffix: str,
    document_id: str,
    source_type: CorpusSourceType,
    equipment_family: str,
    title: str,
    tags: list[str],
) -> CorpusDocument:
    """Parse document bytes without touching the filesystem when possible."""
    if suffix in {".txt", ".md", ".rst"}:
        from maintenance_triage_copilot.retrieval.document_parser import _clean_markdown

        text = data.decode("utf-8", errors="replace")
        if suffix == ".md":
            text = _clean_markdown(text)
        return CorpusDocument(
            document_id=document_id,
            source_type=source_type,
            title=title,
            body=text,
            equipment_family=equipment_family,
            tags=tags,
        )

    # PDFs require a file path for the parsing libraries
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        from maintenance_triage_copilot.retrieval.document_parser import parse_pdf

        return parse_pdf(
            tmp.name,
            document_id=document_id,
            source_type=source_type,
            equipment_family=equipment_family,
            title=title,
            tags=tags,
        )
