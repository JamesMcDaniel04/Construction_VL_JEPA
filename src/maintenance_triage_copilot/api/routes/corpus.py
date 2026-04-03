"""Corpus ingestion endpoints."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request

from maintenance_triage_copilot.domain.models import CorpusDocument, IncidentRecord

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.post("/documents")
async def add_document(document: CorpusDocument, request: Request) -> dict[str, object]:
    return cast(dict[str, object], request.app.state.service.add_document(document))


@router.post("/incidents")
async def add_incident(incident: IncidentRecord, request: Request) -> dict[str, object]:
    return cast(dict[str, object], request.app.state.service.add_incident(incident))
