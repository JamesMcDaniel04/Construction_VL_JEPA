"""Add Supabase-backed pilot-user email and site/asset catalog tables.

Revision ID: 0005_supabase_catalog
Revises: 0004_pilot_users
Create Date: 2026-04-05
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0005_supabase_catalog"
down_revision = "0004_pilot_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "pilot_users",
        sa.Column("email", sa.String(), nullable=False, server_default=""),
    )
    op.create_index("ix_pilot_users_email", "pilot_users", ["email"])
    if op.get_bind().dialect.name != "sqlite":
        op.alter_column("pilot_users", "email", server_default=None)

    op.create_table(
        "sites",
        sa.Column("site_id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("active", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_sites_organization_id", "sites", ["organization_id"])
    op.create_index("ix_sites_active", "sites", ["active"])

    op.create_table(
        "assets",
        sa.Column("asset_id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("site_id", sa.String(), nullable=False),
        sa.Column("active", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_assets_organization_id", "assets", ["organization_id"])
    op.create_index("ix_assets_site_id", "assets", ["site_id"])
    op.create_index("ix_assets_active", "assets", ["active"])


def downgrade() -> None:
    op.drop_index("ix_assets_active", table_name="assets")
    op.drop_index("ix_assets_site_id", table_name="assets")
    op.drop_index("ix_assets_organization_id", table_name="assets")
    op.drop_table("assets")

    op.drop_index("ix_sites_active", table_name="sites")
    op.drop_index("ix_sites_organization_id", table_name="sites")
    op.drop_table("sites")

    op.drop_index("ix_pilot_users_email", table_name="pilot_users")
    op.drop_column("pilot_users", "email")
