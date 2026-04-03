"""Initial maintenance triage schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-03
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "corpus_documents",
        sa.Column("document_id", sa.String(), primary_key=True),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_table(
        "incident_records",
        sa.Column("incident_id", sa.String(), primary_key=True),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_table(
        "reference_states",
        sa.Column("state_id", sa.String(), primary_key=True),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_table(
        "triage_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("response", sa.JSON(), nullable=False),
    )
    op.create_index("ix_triage_history_request_id", "triage_history", ["request_id"])


def downgrade() -> None:
    op.drop_index("ix_triage_history_request_id", table_name="triage_history")
    op.drop_table("triage_history")
    op.drop_table("reference_states")
    op.drop_table("incident_records")
    op.drop_table("corpus_documents")
