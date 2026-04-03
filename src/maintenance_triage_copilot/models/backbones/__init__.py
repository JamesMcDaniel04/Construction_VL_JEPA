"""JEPA backbone adapters."""

from maintenance_triage_copilot.models.backbones.image import IJEPAImageAdapter
from maintenance_triage_copilot.models.backbones.video import VJEPAVideoAdapter

__all__ = ["IJEPAImageAdapter", "VJEPAVideoAdapter"]
