"""Pluggable embedding backends for Simasia.

A backend is anything satisfying :class:`EmbeddingModel` — an object with an
``encode(list[str]) -> np.ndarray`` method.  ``OpenAIEmbedder`` is the default
used by :class:`~simasia.guard.SimasiaGuard`; ``LocalEmbedder`` keeps the
original offline sentence-transformer path available for consumers who install
the ``local`` extra.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    """Minimal interface required from an embedding backend."""

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray: ...


class OpenAIEmbedder:
    """Embed text with an OpenAI embedding model (default backend).

    The API key is read from the ``EMBEDDING_KEY`` environment variable unless
    passed explicitly.  ``dimensions`` is forwarded to the API to shorten the
    output vector when the model supports it (``text-embedding-3-*`` do).
    A pre-built ``client`` may be injected for testing.
    """

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        dimensions: int | None = None,
        client: object | None = None,
        batch_size: int = 1000,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        if client is None:
            api_key = api_key or os.environ.get("EMBEDDING_KEY")
            if not api_key:
                raise ValueError(
                    "No OpenAI API key found. Set the EMBEDDING_KEY environment "
                    "variable or pass api_key=..."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "OpenAIEmbedder requires the 'openai' package. "
                    "Install it with: pip install simasia[openai]"
                ) from exc
            client = OpenAI(api_key=api_key)
        self._client = client

    def encode(self, sentences: list[str], **_kwargs: object) -> np.ndarray:
        sentences = list(sentences)
        vectors: list[list[float]] = []
        # OpenAI caps inputs per request (2048), so send in batches.
        for start in range(0, len(sentences), self.batch_size):
            batch = sentences[start : start + self.batch_size]
            request: dict[str, object] = {"model": self.model, "input": batch}
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions
            response = self._client.embeddings.create(**request)
            vectors.extend(item.embedding for item in response.data)
        return np.asarray(vectors, dtype=np.float32)


class LocalEmbedder:
    """Embed text with a local, frozen sentence-transformer (offline backend).

    Defaults to ``BAAI/bge-small-en-v1.5`` loaded strictly from the local cache,
    so training and inference run on CPU without network access.  Pass a
    pre-loaded ``model`` to reuse a cached instance or to inject a fake.
    """

    DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        local_files_only: bool = True,
        model: object | None = None,
    ) -> None:
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "LocalEmbedder requires the 'sentence-transformers' package. "
                    "Install it with: pip install simasia[local]"
                ) from exc
            model = SentenceTransformer(model_name, local_files_only=local_files_only)
        self._model = model

    def encode(self, sentences: list[str], **_kwargs: object) -> np.ndarray:
        embeddings = self._model.encode(list(sentences), convert_to_numpy=True)
        return np.asarray(embeddings, dtype=np.float32)
