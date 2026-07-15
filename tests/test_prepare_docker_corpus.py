import sqlite3

from scripts.prepare_docker_corpus import prepare


def test_prepare_rewrites_corpus_paths_for_container(tmp_path):
    data_dir = tmp_path / "data"
    images_dir = data_dir / "images"
    uploads_dir = data_dir / "uploads"
    images_dir.mkdir(parents=True)
    uploads_dir.mkdir()
    (images_dir / "cached.png").write_bytes(b"image")
    (uploads_dir / "source.jpg").write_bytes(b"source")

    database = data_dir / "imagecb.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE images (
                image_id TEXT PRIMARY KEY,
                image_path TEXT NOT NULL,
                source_file TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO images VALUES (?, ?, ?)",
            (
                "image-1",
                r"C:\project\data\images\cached.png",
                r"C:\project\data\uploads\source.jpg",
            ),
        )

    assert prepare(data_dir, tmp_path / "app" / "data") == 1

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT image_path, source_file FROM images"
        ).fetchone()

    assert row == (
        (tmp_path / "app" / "data" / "images" / "cached.png").as_posix(),
        (tmp_path / "app" / "data" / "uploads" / "source.jpg").as_posix(),
    )


def test_prepare_leaves_s3_references_unchanged(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = data_dir / "imagecb.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE images (image_id TEXT PRIMARY KEY, image_path TEXT, source_file TEXT)"
        )
        connection.execute(
            "INSERT INTO images VALUES (?, ?, ?)",
            (
                "image-s3",
                "s3://bucket/imagecb/images/image-s3.png",
                "s3://bucket/imagecb/uploads/aa/hash/source.pptx",
            ),
        )

    assert prepare(data_dir, tmp_path / "app" / "data") == 0
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT image_path, source_file FROM images"
        ).fetchone()
    assert row == (
        "s3://bucket/imagecb/images/image-s3.png",
        "s3://bucket/imagecb/uploads/aa/hash/source.pptx",
    )
