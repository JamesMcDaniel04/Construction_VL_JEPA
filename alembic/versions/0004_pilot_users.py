"""Add persisted pilot users.

Revision ID: 0004_pilot_users
Revises: 0003_cases
Create Date: 2026-04-05
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0004_pilot_users"
down_revision = "0003_cases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pilot_users",
        sa.Column("user_id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_pilot_users_organization_id", "pilot_users", ["organization_id"])
    op.create_index("ix_pilot_users_role", "pilot_users", ["role"])
    op.create_index("ix_pilot_users_created_at", "pilot_users", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_pilot_users_created_at", table_name="pilot_users")
    op.drop_index("ix_pilot_users_role", table_name="pilot_users")
    op.drop_index("ix_pilot_users_organization_id", table_name="pilot_users")
    op.drop_table("pilot_users")
