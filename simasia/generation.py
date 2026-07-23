"""Pluggable text-generation backend for Simasia.

Used at training time to turn each on-brand chunk into an off-brand "opposite"
sample, so a brand can be calibrated from on-brand text alone.  A backend is
anything with a ``generate(prompt) -> str`` method; ``OpenAIGenerator`` is the
default.
"""

from __future__ import annotations

import os
from typing import Protocol


class GenerationModel(Protocol):
    """Minimal interface required from a generation backend."""

    def generate(self, prompt: str) -> str: ...


class OpenAIGenerator:
    """Generate text with an OpenAI chat model (default backend).

    The API key is read from ``GENERATION_KEY``, falling back to ``EMBEDDING_KEY``,
    unless passed explicitly.  ``model`` defaults to a small, cheap model since the
    opposite-example step is meant to be lightweight.  A pre-built ``client`` may
    be injected for testing.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.7,
        client: object | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        if client is None:
            api_key = (
                api_key
                or os.environ.get("GENERATION_KEY")
                or os.environ.get("EMBEDDING_KEY")
            )
            if not api_key:
                raise ValueError(
                    "No OpenAI API key found. Set GENERATION_KEY (or EMBEDDING_KEY) "
                    "or pass api_key=..."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "OpenAIGenerator requires the 'openai' package. "
                    "Install it with: pip install simasia[openai]"
                ) from exc
            client = OpenAI(api_key=api_key)
        self._client = client

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()
