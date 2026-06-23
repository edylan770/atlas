# Smoke-test corpus

A tiny set of synthetic images used to verify end-to-end search on the
deployed AWS environment before the real S3/AppFlow corpus pipeline exists.

On container startup, if `BOOTSTRAP_CORPUS_DIR` points here and the index is
empty, the app ingests these images in a background thread (see
`_lifespan` in `imagecb/api/server.py`). This requires Bedrock access on the
running task.

These are throwaway test assets, not production data. Keep this set small
(a few images). Real corpus content belongs in S3.
