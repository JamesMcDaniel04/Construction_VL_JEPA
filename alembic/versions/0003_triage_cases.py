"""Add persisted triage cases.

Revision ID: 0003_cases
Revises: 0002_media_audits
Create Date: 2026-04-05
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_cases"
down_revision = "0002_media_audits"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "triage_cases",
        sa.Column("case_id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("created_by_user_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_triage_cases_organization_id", "triage_cases", ["organization_id"])
    op.create_index("ix_triage_cases_created_by_user_id", "triage_cases", ["created_by_user_id"])
    op.create_index("ix_triage_cases_status", "triage_cases", ["status"])
    op.create_index("ix_triage_cases_created_at", "triage_cases", ["created_at"])
    op.create_index("ix_triage_cases_updated_at", "triage_cases", ["updated_at"])


def downgrade() -> None:
    op.drop_index("ix_triage_cases_updated_at", table_name="triage_cases")
    op.drop_index("ix_triage_cases_created_at", table_name="triage_cases")
    op.drop_index("ix_triage_cases_status", table_name="triage_cases")
    op.drop_index("ix_triage_cases_created_by_user_id", table_name="triage_cases")
    op.drop_index("ix_triage_cases_organization_id", table_name="triage_cases")
    op.drop_table("triage_cases")
