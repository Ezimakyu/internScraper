"""Thin wrapper around the OpenAI client.

Centralizing client creation makes it easy to switch models or providers later
(e.g. an Anthropic backend) without touching the rest of the pipeline.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from openai import OpenAI


DEFAULT_MODEL = os.environ.get("INTERNSCRAPER_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=1)
def get_client(api_key: Optional[str] = None) -> OpenAI:
    """Return a cached OpenAI client.

    Reads ``OPENAI_API_KEY`` from the environment if ``api_key`` isn't given.
    Raising here is intentional: nothing downstream is useful without a key.
    """

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or place it in a .env file "
            "before running the scraper."
        )
    return OpenAI(api_key=key)


def model_name() -> str:
    return DEFAULT_MODEL
