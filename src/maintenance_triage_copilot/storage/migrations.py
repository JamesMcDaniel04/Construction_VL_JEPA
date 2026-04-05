"""Alembic migration helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from alembic.config import Config

from alembic import command


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def alembic_config(database_url: str) -> Config:
    root = _repo_root()
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def run_migrations(database_url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(database_url), revision)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Alembic migrations for maintenance triage")
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--revision", default="head")
    args = parser.parse_args()
    run_migrations(args.database_url, args.revision)


if __name__ == "__main__":
    main()
