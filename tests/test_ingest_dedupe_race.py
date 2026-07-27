"""Regression: identical images ingested concurrently must not double-process.

Before the in-flight claim, N copies of the same image racing through the
worker pool all passed the duplicate check (the hash was only registered
after captioning/embedding), all paid model costs, and all but one died on
the content_hash UNIQUE constraint - counted as errors with orphaned blobs.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from imagecb.models.vlm import CaptionJSON
from imagecb.extractors.types import ExtractedImage, Provenance
from imagecb.ingest import _IngestWorkItem, _ingest_one_image


def _item(idx: int) -> _IngestWorkItem:
    return _IngestWorkItem(
        file_path=Path(f"/tmp/copy-{idx}.png"),
        extracted=ExtractedImage(
            image=MagicMock(),
            provenance=Provenance(
                source_file=f"/tmp/copy-{idx}.png",
                source_type="image",
            ),
        ),
    )


def test_identical_images_in_flight_processed_once():
    known: set[str] = set()
    in_flight: set[str] = set()
    known_lock = threading.Lock()
    barrier = threading.Barrier(4)
    model_calls = []

    def slow_caption_embed(*args, **kwargs):
        model_calls.append(1)
        time.sleep(0.05)  # widen the race window
        return CaptionJSON.empty(), np.zeros(4)

    def run(idx: int):
        barrier.wait()  # maximize overlap: all workers hash simultaneously
        return _ingest_one_image(
            _item(idx),
            known=known,
            known_lock=known_lock,
            in_flight=in_flight,
            force=False,
            skip_caption=False,
            skip_ocr=True,
            captioner=MagicMock(),
            embedder=MagicMock(),
            max_image_side=1024,
        )

    fake_session = MagicMock()
    fake_session.__enter__ = MagicMock(return_value=MagicMock())
    fake_session.__exit__ = MagicMock(return_value=False)

    with patch("imagecb.ingest._hash_image", return_value="same-hash"), patch(
        "imagecb.ingest.get_record_by_hash", return_value=None
    ), patch(
        "imagecb.ingest._cache_image", return_value="/tmp/cached.png"
    ), patch(
        "imagecb.ingest._cache_thumb", return_value="/tmp/thumb.jpg"
    ), patch(
        "imagecb.ingest._caption_and_embed", side_effect=slow_caption_embed
    ), patch(
        "imagecb.ingest.embed_caption_document", return_value=None
    ), patch(
        "imagecb.ingest.session_scope", return_value=fake_session
    ):
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(run, range(4)))

    added = [o for o in outcomes if o.added]
    skipped = [o for o in outcomes if o.skipped_duplicate]
    errors = [o for o in outcomes if o.error]
    assert len(added) == 1, f"expected exactly one processed, got {len(added)}"
    assert len(skipped) == 3
    assert not errors
    assert len(model_calls) == 1, "duplicates must not pay caption/embed costs"
    assert "same-hash" in known and not in_flight


def test_failed_claim_is_released_for_retry():
    known: set[str] = set()
    in_flight: set[str] = set()
    known_lock = threading.Lock()

    with patch("imagecb.ingest._hash_image", return_value="h2"), patch(
        "imagecb.ingest.get_record_by_hash", return_value=None
    ), patch(
        "imagecb.ingest._cache_image", side_effect=RuntimeError("disk full")
    ):
        outcome = _ingest_one_image(
            _item(0),
            known=known,
            known_lock=known_lock,
            in_flight=in_flight,
            force=False,
            skip_caption=True,
            skip_ocr=True,
            captioner=None,
            embedder=MagicMock(),
            max_image_side=1024,
        )

    assert outcome.error
    assert not in_flight, "failed claim must be released so a retry can process"
