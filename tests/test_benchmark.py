from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from maintenance_triage_copilot.evaluation import benchmark


def test_eval_triage_writes_reports_and_metrics(small_config, tmp_path, monkeypatch) -> None:
    bundle = Path(__file__).resolve().parent / "fixtures" / "benchmark_bundle"
    output_dir = tmp_path / "benchmark-output"
    config_path = tmp_path / "benchmark-config.yaml"

    config_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "mode": "development",
                    "model_dir": small_config.runtime.model_dir,
                    "allow_smoke_assets": True,
                },
                "database": {
                    "postgres_url": None,
                    "qdrant_url": None,
                    "collection_prefix": "benchmark-test",
                },
                "retrieval": {
                    "top_k_documents": 3,
                    "top_k_incidents": 3,
                    "top_k_states": 2,
                    "chunk_size": 120,
                    "chunk_overlap": 20,
                },
                "triage": {
                    "top_k_steps": 3,
                    "state_match_threshold": 0.8,
                    "escalation_threshold": 0.5,
                    "video_num_frames": 8,
                },
                "policy": {
                    "checkpoint_path": None,
                    "require_checkpoint": False,
                    "top_k_issues": 3,
                },
                "security": {"service_tokens": {}},
                "text_encoder": {
                    "backend": "mock",
                    "embedding_dim": 192,
                },
                "adapter": {
                    "hidden_dim": 192,
                    "output_dim": 192,
                    "checkpoint_path": None,
                },
                "image_backbone": {
                    "preset": "ijepa_vith14_224",
                    "require_checkpoint": False,
                    "use_timm": False,
                    "checkpoint_path": None,
                },
                "video_backbone": {
                    "preset": "vjepa_vith16_224_2x16x16",
                    "require_checkpoint": False,
                    "use_timm": False,
                    "checkpoint_path": None,
                },
            },
            sort_keys=True,
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mtc-eval-triage",
            "--bundle",
            str(bundle),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ],
    )
    benchmark.main()

    summary = json.loads((output_dir / "summary.json").read_text())
    per_case = (output_dir / "per_case.jsonl").read_text().strip().splitlines()
    assert summary["cases"] == 1
    assert summary["metrics"]["issue_top1_accuracy"] >= 1.0
    assert summary["metrics"]["state_label_accuracy"] >= 1.0
    assert "escalation_precision" in summary["metrics"]
    assert "escalation_recall" in summary["metrics"]
    assert summary["policy"]["metadata"]["trained_examples"] == "1"
    assert all(summary["thresholds_met"].values())
    assert len(per_case) == 1
    assert (output_dir / "issue_confusion.csv").exists()
    assert (output_dir / "escalation_confusion.csv").exists()
