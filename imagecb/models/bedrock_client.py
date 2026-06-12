"""Shared Bedrock client.

Cached boto3 client for ``bedrock-runtime`` (converse, invoke_model, rerank).
Auth is implicit: boto3 picks up ``AWS_BEARER_TOKEN_BEDROCK`` (loaded from
``.env`` via ``imagecb.config``) or any standard AWS credential source.

Concurrent calls are gated by a process-wide semaphore so ingest workers
cannot overwhelm Bedrock quotas regardless of ``INGEST_WORKERS``.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from imagecb.config import SETTINGS

_client: Optional[Any] = None
_semaphore: Optional[threading.Semaphore] = None


def _bedrock_config() -> Any:
    from botocore.config import Config

    return Config(
        connect_timeout=SETTINGS.bedrock_connect_timeout,
        read_timeout=SETTINGS.bedrock_read_timeout,
        retries={
            "max_attempts": SETTINGS.bedrock_max_retries,
            "mode": "adaptive",
        },
    )


def get_bedrock_runtime() -> Any:
    global _client
    if _client is None:
        import boto3

        _client = boto3.client(
            "bedrock-runtime",
            region_name=SETTINGS.aws_region,
            config=_bedrock_config(),
        )
    return _client


def _get_semaphore() -> threading.Semaphore:
    global _semaphore
    if _semaphore is None:
        n = max(1, SETTINGS.bedrock_max_concurrent)
        _semaphore = threading.Semaphore(n)
    return _semaphore


@contextmanager
def bedrock_call_gate() -> Iterator[None]:
    """Limit concurrent in-flight Bedrock API calls."""
    sem = _get_semaphore()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def bedrock_converse(**kwargs: Any) -> Any:
    with bedrock_call_gate():
        return get_bedrock_runtime().converse(**kwargs)


def bedrock_invoke_model(**kwargs: Any) -> Any:
    with bedrock_call_gate():
        return get_bedrock_runtime().invoke_model(**kwargs)


def bedrock_converse_stream(**kwargs: Any) -> Any:
    with bedrock_call_gate():
        return get_bedrock_runtime().converse_stream(**kwargs)
