"""create leads table with llm metadata

Revision ID: 0001_create_leads_table
Revises:
Create Date: 2026-04-15
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_create_leads_table"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "leads",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("company", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("score_reason", sa.Text(), nullable=True),
        sa.Column("llm_model", sa.String(length=120), nullable=True),
        sa.Column("llm_confidence", sa.Float(), nullable=True),
        sa.Column("llm_token_usage", sa.Integer(), nullable=True),
        sa.Column("llm_cached", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_leads_id"), "leads", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_leads_id"), table_name="leads")
    op.drop_table("leads")
