"""Database maintenance commands.

python -m src.database.maintenance init      # create the database and tables
python -m src.database.maintenance manifest  # write data/metadata/manifest_files.csv
python -m src.database.maintenance verify    # recompute SHA-256 and compare with the database
python -m src.database.maintenance counts    # count rows in each table
"""

from __future__ import annotations

import argparse
import sys

from src.database.db import connect, init_db
from src.database.manifest import export_manifest, verify_files
from src.database.repository import table_counts
from src.utils.config import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IERAS metadata database maintenance")
    parser.add_argument("command", choices=["init", "manifest", "verify", "counts"])
    parser.add_argument(
        "--root", default=None, help="project root (defaults to the repository directory)"
    )
    args = parser.parse_args(argv)

    settings = load_settings(root=args.root)
    paths = settings.paths
    paths.ensure_dirs()

    if args.command == "init":
        init_db(paths.db_path)
        print(f"Database ready: {paths.db_path}")
        return 0

    if args.command == "counts":
        init_db(paths.db_path)
        with connect(paths.db_path) as session:
            for table, count in table_counts(session).items():
                print(f"{table:24s} {count}")
        return 0

    if args.command == "manifest":
        init_db(paths.db_path)
        with connect(paths.db_path) as session:
            n = export_manifest(session, paths.manifest_path)
        print(f"Manifest: {paths.manifest_path} ({n} files)")
        return 0

    if args.command == "verify":
        init_db(paths.db_path)
        with connect(paths.db_path) as session:
            checks = verify_files(session, paths)
        bad = [c for c in checks if c.status != "ok"]
        for c in checks:
            print(f"{c.status:9s} {c.file_path}")
        print(f"Checked: {len(checks)}, with issues: {len(bad)}")
        return 1 if bad else 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
