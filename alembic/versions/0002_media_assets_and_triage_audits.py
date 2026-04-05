"""Add media assets and triage audits.

Revision ID: 0002_media_audits
Revises: 0001_initial
Create Date: 2026-04-04
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_media_audits"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "media_assets",
        sa.Column("asset_id", sa.String(), primary_key=True),
        sa.Column("asset_type", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_media_assets_asset_type", "media_assets", ["asset_type"])
    op.create_index("ix_media_assets_created_at", "media_assets", ["created_at"])

    op.create_table(
        "triage_audits",
        sa.Column("audit_id", sa.String(), primary_key=True),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("principal", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_triage_audits_request_id", "triage_audits", ["request_id"])
    op.create_index("ix_triage_audits_principal", "triage_audits", ["principal"])
    op.create_index("ix_triage_audits_created_at", "triage_audits", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_triage_audits_created_at", table_name="triage_audits")
    op.drop_index("ix_triage_audits_principal", table_name="triage_audits")
    op.drop_index("ix_triage_audits_request_id", table_name="triage_audits")
    op.drop_table("triage_audits")

    op.drop_index("ix_media_assets_created_at", table_name="media_assets")
    op.drop_index("ix_media_assets_asset_type", table_name="media_assets")
    op.drop_table("media_assets")
