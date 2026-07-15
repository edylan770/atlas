"""Make a copied Imagecb data directory portable inside a Docker image."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path, PurePosixPath


def _container_path(
    raw: str,
    source_data_dir: Path,
    target_data_dir: PurePosixPath,
    subdir: str,
) -> str:
    if raw.lower().startswith("s3://"):
        return raw
    normalized = raw.replace("\\", "/")
    marker = f"/data/{subdir}/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        return str(target_data_dir / subdir / relative)

    candidate = source_data_dir / subdir / Path(normalized).name
    if candidate.is_file():
        return str(target_data_dir / subdir / candidate.name)
    return raw


def prepare(source_data_dir: Path, target_data_dir: Path) -> int:
    source_data_dir = source_data_dir.resolve()
    database = source_data_dir / "imagecb.db"
    if not database.is_file():
        raise FileNotFoundError(f"Corpus database not found: {database}")

    container_data_dir = PurePosixPath(target_data_dir.as_posix())
    updated = 0
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT image_id, image_path, source_file FROM images"
        ).fetchall()
        for image_id, image_path, source_file in rows:
            portable_image_path = _container_path(
                image_path, source_data_dir, container_data_dir, "images"
            )
            portable_source_file = _container_path(
                source_file, source_data_dir, container_data_dir, "uploads"
            )
            if (portable_image_path, portable_source_file) != (
                image_path,
                source_file,
            ):
                connection.execute(
                    """
                    UPDATE images
                    SET image_path = ?, source_file = ?
                    WHERE image_id = ?
                    """,
                    (portable_image_path, portable_source_file, image_id),
                )
                updated += 1
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    for suffix in ("-shm", "-wal"):
        Path(f"{database}{suffix}").unlink(missing_ok=True)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_data_dir", type=Path)
    parser.add_argument("--target-data-dir", type=Path, required=True)
    args = parser.parse_args()
    updated = prepare(args.source_data_dir, args.target_data_dir)
    print(f"Prepared Docker corpus: rewrote {updated} image record(s)")


if __name__ == "__main__":
    main()
