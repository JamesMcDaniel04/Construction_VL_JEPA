"""Offline triage benchmark runner."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, cast

from maintenance_triage_copilot.config import load_config
from maintenance_triage_copilot.domain.models import (
    CorpusDocument,
    IncidentRecord,
    ReferenceState,
    TriageRequest,
)
from maintenance_triage_copilot.encoding.text import MaintenanceTextEncoder
from maintenance_triage_copilot.models.adapter import VisualTextProjector
from maintenance_triage_copilot.models.assets import validate_model_assets
from maintenance_triage_copilot.models.backbones import IJEPAImageAdapter, VJEPAVideoAdapter
from maintenance_triage_copilot.models.policy import CalibratedTriagePolicy
from maintenance_triage_copilot.retrieval.index import VectorIndex
from maintenance_triage_copilot.services.triage import AppState, TriageService
from maintenance_triage_copilot.storage.memory import MemoryMetadataStore
from maintenance_triage_copilot.storage.object_store import MemoryObjectStore
from maintenance_triage_copilot.training.train_policy import train_policy_from_rows
from maintenance_triage_copilot.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run offline benchmark evaluation for triage")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = Path(args.bundle)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_profile = _validate_bundle(bundle)
    cfg = load_config(args.config)
    policy_checkpoint = _prepare_policy_checkpoint(bundle, output_dir)
    if policy_checkpoint is not None:
        cfg.policy.checkpoint_path = str(policy_checkpoint)

    service = _build_service(cfg)
    _index_bundle(service, bundle)
    summary, per_case = _run_holdout(service, bundle)
    summary["bundle_profile"] = bundle_profile
    summary["policy"] = {
        "checkpoint_path": cfg.policy.checkpoint_path,
        "metadata": service.state.triage_policy.metadata,
    }

    thresholds_path = bundle / "thresholds.json"
    if thresholds_path.exists():
        thresholds = json.loads(thresholds_path.read_text())
        summary["thresholds"] = thresholds
        summary["thresholds_met"] = {
            key: float(summary["metrics"].get(key, 0.0)) >= float(value)
            for key, value in thresholds.items()
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    with (output_dir / "per_case.jsonl").open("w") as handle:
        for row in per_case:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    _write_confusion_csv(
        output_dir / "issue_confusion.csv",
        per_case,
        "expected_issue_class",
        "pred_issue_class",
    )
    _write_confusion_csv(
        output_dir / "escalation_confusion.csv",
        per_case,
        "expected_escalation",
        "pred_escalation",
    )


def _prepare_policy_checkpoint(bundle: Path, output_dir: Path) -> Path | None:
    policy_train = bundle / "policy_train.jsonl"
    if not policy_train.exists():
        return None
    rows = _load_jsonl(policy_train)
    policy = train_policy_from_rows(rows, epochs=200, lr=5e-2)
    target = output_dir / "trained-policy.json"
    policy.save(target)
    return target


def _build_service(cfg) -> TriageService:
    assets = validate_model_assets(cfg)
    text_encoder = MaintenanceTextEncoder(cfg.text_encoder)
    image_backbone = IJEPAImageAdapter(cfg.image_backbone, runtime_spec=assets.image_backbone)
    video_backbone = VJEPAVideoAdapter(cfg.video_backbone, runtime_spec=assets.video_backbone)
    projector = VisualTextProjector(
        input_dim=image_backbone.embedding_dim,
        hidden_dim=cfg.adapter.hidden_dim,
        output_dim=cfg.adapter.output_dim,
    )
    import torch

    if cfg.adapter.checkpoint_path:
        projector.load_state_dict(torch.load(cfg.adapter.checkpoint_path, map_location="cpu"))
    else:
        with torch.no_grad():
            for parameter in projector.parameters():
                parameter.zero_()
    projector.eval()

    triage_policy = (
        CalibratedTriagePolicy.from_file(cfg.policy.checkpoint_path)
        if cfg.policy.checkpoint_path
        else CalibratedTriagePolicy.bootstrap()
    )
    state = AppState(
        config=cfg,
        text_encoder=text_encoder,
        image_backbone=image_backbone,
        video_backbone=video_backbone,
        projector=projector,
        triage_policy=triage_policy,
        vector_index=VectorIndex(),
        metadata_store=MemoryMetadataStore(),
        object_store=MemoryObjectStore(),
        asset_status=assets.status,
        auth_mode="evaluation",
        telemetry_mode="disabled",
        projector_checkpoint_path=cfg.adapter.checkpoint_path,
        projector_checkpoint_loaded=cfg.adapter.checkpoint_path is not None,
        policy_checkpoint_path=cfg.policy.checkpoint_path,
        policy_checkpoint_loaded=cfg.policy.checkpoint_path is not None,
    )
    return TriageService(state)


def _index_bundle(service: TriageService, bundle: Path) -> None:
    for row in _load_jsonl(bundle / "documents.jsonl"):
        service.add_document(CorpusDocument.model_validate(row))
    for row in _load_jsonl(bundle / "incidents.jsonl"):
        service.add_incident(IncidentRecord.model_validate(row))
    for row in _load_jsonl(bundle / "reference_states.jsonl"):
        resolved_row = _resolve_media_paths(row, bundle)
        service.add_reference_state(ReferenceState.model_validate(resolved_row))


def _run_holdout(
    service: TriageService,
    bundle: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _load_jsonl(bundle / "holdout_eval.jsonl")
    per_case: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    issue_top1 = 0
    issue_top3 = 0
    state_label_correct = 0
    state_match_correct = 0
    escalation_correct = 0
    escalation_true_positive = 0
    escalation_false_positive = 0
    escalation_false_negative = 0
    similar_incident_hits = 0
    citation_precision: list[float] = []
    citation_recall: list[float] = []
    step_precision: list[float] = []
    step_recall: list[float] = []

    for row in rows:
        request = TriageRequest.model_validate(_resolve_media_paths(row["request"], bundle))
        start = time.perf_counter()
        response = service.analyze(request)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency_ms)

        expected_issue = row.get("expected_issue_class")
        expected_state_label = row.get(
            "expected_matched_state_label",
            row.get("expected_state_label"),
        )
        request_expected_state_label = request.expected_state_label
        expected_match = row.get("expected_matches_expected")
        expected_escalation = row.get("expected_escalation")
        expected_incident_ids = set(row.get("expected_incident_ids", []))
        expected_citation_doc_ids = set(row.get("expected_citation_document_ids", []))
        expected_step_phrases = [
            str(item) for item in row.get("expected_step_phrases", []) if str(item).strip()
        ]

        predicted_issue = (
            response.issue_candidates[0].issue_class if response.issue_candidates else None
        )
        predicted_issues = [item.issue_class for item in response.issue_candidates[:3]]
        predicted_incidents = {item.incident_id for item in response.similar_incidents[:3]}
        predicted_steps = [item.step for item in response.next_steps[:3]]
        predicted_step_citation_doc_ids = {
            citation.document_id
            for next_step in response.next_steps[:3]
            for citation in next_step.citations
        }

        if expected_issue and predicted_issue == expected_issue:
            issue_top1 += 1
        if expected_issue and expected_issue in predicted_issues:
            issue_top3 += 1
        if (
            expected_state_label
            and response.state_assessment.matched_state_label == expected_state_label
        ):
            state_label_correct += 1
        if (
            expected_match is not None
            and response.state_assessment.matches_expected == expected_match
        ):
            state_match_correct += 1
        if expected_escalation and response.escalation_recommendation == expected_escalation:
            escalation_correct += 1
        predicted_escalated = response.escalation_recommendation != "proceed_with_guided_inspection"
        expected_escalated = expected_escalation not in {
            None,
            "proceed_with_guided_inspection",
        }
        if predicted_escalated and expected_escalated:
            escalation_true_positive += 1
        elif predicted_escalated and not expected_escalated:
            escalation_false_positive += 1
        elif not predicted_escalated and expected_escalated:
            escalation_false_negative += 1
        if expected_incident_ids and predicted_incidents.intersection(expected_incident_ids):
            similar_incident_hits += 1
        if expected_citation_doc_ids:
            correct = predicted_step_citation_doc_ids.intersection(expected_citation_doc_ids)
            citation_precision.append(len(correct) / max(len(predicted_step_citation_doc_ids), 1))
            citation_recall.append(len(correct) / len(expected_citation_doc_ids))
        if expected_step_phrases:
            step_hits = _count_matching_step_phrases(predicted_steps, expected_step_phrases)
            step_precision.append(step_hits / max(len(predicted_steps), 1))
            step_recall.append(step_hits / len(expected_step_phrases))

        per_case.append(
            {
                "case_id": row["case_id"],
                "expected_issue_class": expected_issue,
                "pred_issue_class": predicted_issue,
                "request_expected_state_label": request_expected_state_label,
                "expected_state_label": expected_state_label,
                "pred_state_label": response.state_assessment.matched_state_label,
                "expected_matches_expected": expected_match,
                "pred_matches_expected": response.state_assessment.matches_expected,
                "expected_escalation": expected_escalation,
                "pred_escalation": response.escalation_recommendation,
                "expected_step_phrases": expected_step_phrases,
                "pred_steps": predicted_steps,
                "latency_ms": round(latency_ms, 3),
            }
        )

    count = max(len(rows), 1)
    summary = {
        "cases": len(rows),
        "metrics": {
            "issue_top1_accuracy": issue_top1 / count,
            "issue_top3_accuracy": issue_top3 / count,
            "state_label_accuracy": state_label_correct / count,
            "expected_state_match_accuracy": state_match_correct / count,
            "escalation_accuracy": escalation_correct / count,
            "escalation_precision": _safe_ratio(
                escalation_true_positive,
                escalation_true_positive + escalation_false_positive,
            ),
            "escalation_recall": _safe_ratio(
                escalation_true_positive,
                escalation_true_positive + escalation_false_negative,
            ),
            "similar_incident_recall_at_3": similar_incident_hits / count,
            "next_step_citation_precision_at_3": _mean(citation_precision),
            "next_step_citation_recall_at_3": _mean(citation_recall),
            "next_step_precision_at_3": _mean(step_precision),
            "next_step_recall_at_3": _mean(step_recall),
            "latency_p50_ms": _percentile(latencies_ms, 50),
            "latency_p95_ms": _percentile(latencies_ms, 95),
        },
    }
    return summary, per_case


def _resolve_media_paths(payload: dict[str, Any], bundle: Path) -> dict[str, Any]:
    cloned = cast(dict[str, Any], json.loads(json.dumps(payload)))
    observation = cloned.get("observation")
    if isinstance(observation, dict) and observation.get("media_uri"):
        observation["media_uri"] = str(bundle / "media" / observation["media_uri"])
    if cloned.get("media_uri"):
        cloned["media_uri"] = str(bundle / "media" / cloned["media_uri"])
    return cloned


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _validate_bundle(bundle: Path) -> dict[str, Any]:
    rows = _load_jsonl(bundle / "holdout_eval.jsonl")
    unique_issues = {
        str(row.get("expected_issue_class"))
        for row in rows
        if row.get("expected_issue_class") is not None
    }
    unique_escalations = {
        str(row.get("expected_escalation"))
        for row in rows
        if row.get("expected_escalation") is not None
    }
    media_types = {
        str(cast(dict[str, Any], row["request"])["observation"]["media_type"]) for row in rows
    }
    if len(rows) < 4:
        raise ValueError("Benchmark bundles must include at least 4 holdout cases")
    if len(unique_issues) < 3:
        raise ValueError("Benchmark bundles must cover at least 3 distinct issue classes")
    if len(unique_escalations) < 2:
        raise ValueError("Benchmark bundles must cover at least 2 escalation outcomes")
    return {
        "holdout_cases": len(rows),
        "distinct_issue_classes": len(unique_issues),
        "distinct_escalations": len(unique_escalations),
        "media_types": sorted(media_types),
    }


def _count_matching_step_phrases(predicted_steps: list[str], expected_phrases: list[str]) -> int:
    normalized_steps = [_normalize_text(step) for step in predicted_steps]
    hits = 0
    for phrase in expected_phrases:
        normalized_phrase = _normalize_text(phrase)
        if any(normalized_phrase in step for step in normalized_steps):
            hits += 1
    return hits


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((percentile / 100) * (len(ordered) - 1)))))
    return round(float(ordered[index]), 3)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(statistics.mean(values)), 6)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator / denominator), 6)


def _write_confusion_csv(
    path: Path,
    rows: list[dict[str, Any]],
    expected_key: str,
    predicted_key: str,
) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        expected = str(row.get(expected_key) or "unknown")
        predicted = str(row.get(predicted_key) or "unknown")
        counts[(expected, predicted)] = counts.get((expected, predicted), 0) + 1

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected", "predicted", "count"])
        for (expected, predicted), count in sorted(counts.items()):
            writer.writerow([expected, predicted, count])
