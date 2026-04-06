from __future__ import annotations

from sqlalchemy import create_engine, inspect

from maintenance_triage_copilot.storage.migrations import run_migrations


def test_run_migrations_creates_expected_tables(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'migrations.db'}"

    run_migrations(database_url)
    run_migrations(database_url)

    inspector = inspect(create_engine(database_url))
    assert set(inspector.get_table_names()) >= {
        "corpus_documents",
        "incident_records",
        "media_assets",
        "pilot_users",
        "reference_states",
        "sites",
        "assets",
        "triage_audits",
        "triage_cases",
        "triage_history",
    }
