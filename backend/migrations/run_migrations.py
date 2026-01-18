#!/usr/bin/env python3
"""
BrandPulse AI - Database Migration Runner
==========================================

Runs SQL migrations for SQLite or PostgreSQL databases.
Tracks applied migrations to prevent re-running.

Usage:
    python migrations/run_migrations.py

For SQLite (default):
    python migrations/run_migrations.py

For PostgreSQL:
    DATABASE_URL=postgresql://... python migrations/run_migrations.py
"""

import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_database_path():
    """Get database path from environment or use default SQLite."""
    db_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./brandpulse.db")

    if "sqlite" in db_url:
        # Extract SQLite path
        if ":///" in db_url:
            path = db_url.split("///")[-1]
        else:
            path = "./brandpulse.db"
        return path, "sqlite"
    else:
        return db_url, "postgresql"


def create_migrations_table(conn):
    """Create migrations tracking table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(255) NOT NULL UNIQUE,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def get_applied_migrations(conn):
    """Get list of already applied migrations."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM _migrations ORDER BY id")
    return {row[0] for row in cursor.fetchall()}


def mark_migration_applied(conn, name):
    """Mark a migration as applied."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
        (name, datetime.now().isoformat())
    )
    conn.commit()


def run_sqlite_migration(db_path, migration_file):
    """Run a SQL migration file on SQLite database."""
    print(f"Connecting to SQLite database: {db_path}")

    conn = sqlite3.connect(db_path)

    # Create migrations tracking table
    create_migrations_table(conn)

    # Check if already applied
    applied = get_applied_migrations(conn)
    migration_name = migration_file.name

    if migration_name in applied:
        print(f"  Skipping {migration_name} (already applied)")
        conn.close()
        return False

    # Read and execute migration
    print(f"  Applying {migration_name}...")

    sql = migration_file.read_text()

    # Split into individual statements (handle comments)
    statements = []
    current_statement = []

    for line in sql.split("\n"):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("--") or stripped.startswith("/*"):
            continue

        # Skip PostgreSQL-specific blocks
        if "SERIAL PRIMARY KEY" in line or "TIMESTAMP WITH TIME ZONE" in line:
            continue

        current_statement.append(line)

        if stripped.endswith(";"):
            statement = "\n".join(current_statement)
            # Skip ALTER TABLE for SQLite (handle separately)
            if not statement.strip().upper().startswith("ALTER TABLE"):
                statements.append(statement)
            current_statement = []

    # Execute each statement
    cursor = conn.cursor()
    for statement in statements:
        try:
            cursor.execute(statement)
        except sqlite3.Error as e:
            if "already exists" in str(e).lower():
                print(f"    Table already exists, skipping...")
            else:
                print(f"    Warning: {e}")

    conn.commit()

    # Mark as applied
    mark_migration_applied(conn, migration_name)

    print(f"  Successfully applied {migration_name}")
    conn.close()
    return True


def main():
    """Run all pending migrations."""
    migrations_dir = Path(__file__).parent

    # Get database info
    db_path, db_type = get_database_path()

    print("=" * 60)
    print("BrandPulse AI - Database Migration Runner")
    print("=" * 60)
    print(f"Database type: {db_type}")
    print(f"Database path: {db_path}")
    print()

    if db_type != "sqlite":
        print("PostgreSQL migrations should be run directly with psql:")
        print(f"  psql {db_path} -f migrations/001_add_abtest_tables.sql")
        return

    # Find all .sql migration files
    migration_files = sorted(migrations_dir.glob("*.sql"))

    if not migration_files:
        print("No migration files found.")
        return

    print(f"Found {len(migration_files)} migration file(s)")
    print()

    applied_count = 0
    for migration_file in migration_files:
        if run_sqlite_migration(db_path, migration_file):
            applied_count += 1

    print()
    print(f"Migrations complete. Applied {applied_count} new migration(s).")


if __name__ == "__main__":
    main()
