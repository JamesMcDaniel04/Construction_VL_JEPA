from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from maintenance_triage_copilot.models.assets import checkpoint_sha256, validate_model_assets


def test_validate_model_assets_accepts_smoke_manifest(small_config) -> None:
    assets = validate_model_assets(small_config)
    assert assets.manifest_validated is True
    assert assets.manifest_version == "test-1"
    assert assets.image_backbone.preset == "ijepa_vith14_224"
    assert assets.video_backbone.preset == "vjepa_vith16_224_2x16x16"
    assert assets.image_backbone.smoke_stub is True
    assert assets.video_backbone.smoke_stub is True


def test_validate_model_assets_rejects_bad_image_checkpoint(small_config, tmp_path) -> None:
    model_dir = tmp_path / "models-copy"
    model_dir.mkdir(parents=True, exist_ok=True)

    source_dir = small_config.runtime.model_dir
    assert source_dir is not None

    shutil.copytree(Path(source_dir), model_dir, dirs_exist_ok=True)
    bad_checkpoint = model_dir / "bad_image_backbone.pt"
    torch.save({"encoder": {"not_an_encoder.weight": torch.ones(1)}}, bad_checkpoint)

    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"]["image_backbone"]["path"] = bad_checkpoint.name
    manifest["assets"]["image_backbone"]["sha256"] = checkpoint_sha256(bad_checkpoint)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    small_config.runtime.model_dir = str(model_dir)
    with pytest.raises(RuntimeError, match="missing expected encoder weights"):
        validate_model_assets(small_config)


def test_validate_model_assets_requires_manifest_in_production(small_config, tmp_path) -> None:
    small_config.runtime.mode = "production"
    small_config.runtime.model_dir = str(tmp_path / "empty-models")
    with pytest.raises(FileNotFoundError, match="manifest"):
        validate_model_assets(small_config)
