"""The embedding-plus-classification pipeline used by Simasia."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from .embeddings import EmbeddingModel, OpenAIEmbedder
from .generation import GenerationModel, OpenAIGenerator
from .sources import build_corpus, clean_corpus, fetch_url_text
from .storage import ArtifactStore, FileArtifactStore

ARTIFACT_FORMAT = "simasia-head/2"

# Tone-of-voice axes. Off-brand text is the on-brand text flipped on one axis, so
# the classifier learns a tight on-brand region hemmed in from every direction.
TONE_DIMENSIONS = {
    "directness": "direct and blunt vs. indirect and roundabout",
    "empathy": "warm and empathetic vs. cold and detached",
    "hedging": "confident and certain vs. hedged and tentative",
    "formality": "casual and approachable vs. stiff and corporate",
    "enthusiasm": "matter-of-fact vs. exaggerated and hyped",
    "technicality": "informative and specific vs. vague and empty",
}

DIMENSION_PROMPT = (
    "Rewrite the text below. Change ONLY its {dimension}: flip it to the opposite end "
    "of this axis ({axis}). Keep the topic, the length, and every other aspect of the "
    "tone unchanged. Return only the rewritten text, with no preamble.\n\nText: {chunk}"
)


class SimasiaGuard:
    """Classify short LLM responses as matching a brand's tone or not.

    The embedding backend is frozen and pluggable (OpenAI by default; see
    :mod:`simasia.embeddings`); only the small per-brand logistic-regression
    head is fitted.  The persisted artifact
    (``simasia_<brand_id>_head.joblib``) also stores the training chunks and
    their embeddings so :meth:`explain` can ground a verdict in the brand's own
    on-brand and off-brand samples.
    """

    def __init__(
        self,
        brand_id: str,
        artifact_dir: str | os.PathLike[str] = ".",
        embedding_model: EmbeddingModel | None = None,
        store: ArtifactStore | None = None,
        generator: GenerationModel | None = None,
    ) -> None:
        """Initialize a guard for ``brand_id``.

        ``embedding_model`` defaults to :class:`~simasia.embeddings.OpenAIEmbedder`
        (``text-embedding-3-small``, key from the ``EMBEDDING_KEY`` environment
        variable).  Pass any object implementing ``encode`` to choose a different
        model, supply your own key/dimensions, or run fully offline via
        :class:`~simasia.embeddings.LocalEmbedder`.

        ``store`` decides where the trained artifact lives.  It defaults to
        :class:`~simasia.storage.FileArtifactStore` writing under ``artifact_dir``;
        pass a custom :class:`~simasia.storage.ArtifactStore` to use a database or
        object store instead.
        """
        if not brand_id or not brand_id.strip():
            raise ValueError("brand_id must be a non-empty string")

        self.brand_id = brand_id
        self.store: ArtifactStore = store or FileArtifactStore(artifact_dir)
        self.embedding_model: EmbeddingModel = embedding_model or OpenAIEmbedder()
        # Created lazily only if opposite generation is actually needed.
        self.generator: GenerationModel | None = generator
        self.classifier: LogisticRegression | None = None
        self.on_chunks: list[str] | None = None
        self.on_embeddings: np.ndarray | None = None
        self.off_chunks: list[str] | None = None
        self.off_embeddings: np.ndarray | None = None

    def _chunk_text(self, raw_text: str, sentences_per_chunk: int = 2) -> list[str]:
        """Return overlapping sentence windows suitable for tone training.

        A chunk is retained only when the complete window has at least four words.
        This avoids discarding useful short sentences when they form a meaningful
        two-sentence sample with their neighbour.
        """
        if sentences_per_chunk < 1:
            raise ValueError("sentences_per_chunk must be at least 1")
        if not isinstance(raw_text, str):
            raise TypeError("raw_text must be a string")

        cleaned = re.sub(r"\s+", " ", raw_text).strip()
        if not cleaned:
            return []
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if sentence.strip()
        ]

        return [
            " ".join(sentences[index : index + sentences_per_chunk])
            for index in range(len(sentences) - sentences_per_chunk + 1)
            if len(" ".join(sentences[index : index + sentences_per_chunk]).split()) >= 4
        ]

    def train(
        self,
        on_brand: str | Path | list[str],
        off_brand: str | Path | list[str] | None = None,
        progress: Callable[[int, int], None] | None = None,
        max_chunks: int | None = None,
        dimensions: list[str] | None = None,
    ) -> float:
        """Train the brand model (the main training entry point).

        Each side accepts raw text (``str``), a file path (``pathlib.Path``, read
        as text), or a list of URLs (``list[str]``). If ``off_brand`` is omitted, an
        off-brand opposite is generated for each on-brand chunk on each tone
        dimension (see :meth:`calibrate_from_on_brand`). Returns training accuracy.

        ``progress`` is an optional callback ``progress(done, total)`` reporting how
        many opposite examples are generated so far (on-brand-only mode).
        ``max_chunks`` randomly samples that many on-brand chunks before generating
        opposites. ``dimensions`` selects which tone axes to flip (default: all of
        :data:`TONE_DIMENSIONS`).
        """
        if off_brand is None:
            return self.calibrate_from_on_brand(
                on_brand, progress=progress, max_chunks=max_chunks, dimensions=dimensions
            )
        return self.calibrate_weights(self._to_corpus(on_brand), self._to_corpus(off_brand))

    def calibrate_from_on_brand(
        self,
        on_brand: str | Path | list[str],
        progress: Callable[[int, int], None] | None = None,
        max_chunks: int | None = None,
        dimensions: list[str] | None = None,
    ) -> float:
        """Train from on-brand text alone, generating off-brand opposites.

        The on-brand corpus (raw text, a file path, or a list of URLs) is chunked.
        For each chunk, the generation backend produces one off-brand opposite per
        tone dimension (each flips a single axis). All chunks and opposites are then
        embedded, labelled, and fitted. ``progress(done, total)`` is called after
        each opposite. ``max_chunks`` caps how many on-brand chunks are used (random
        sample). ``dimensions`` selects which axes to flip.
        """
        on_chunks = self._chunk_text(self._to_corpus(on_brand))
        if not on_chunks:
            raise ValueError("Training input contains insufficient sentences to extract tone.")
        if max_chunks is not None and 0 < max_chunks < len(on_chunks):
            on_chunks = random.sample(on_chunks, max_chunks)
        off_chunks = self._generate_opposites(on_chunks, progress=progress, dimensions=dimensions)
        return self._fit(on_chunks, off_chunks)

    @staticmethod
    def _to_corpus(source: str | Path | list[str]) -> str:
        """Resolve a training source to clean raw text.

        ``Path`` is read as a text file, ``list``/``tuple`` is fetched as URLs, and
        ``str`` is used as-is (raw text — never treated as a path, to avoid guessing).
        Web boilerplate is stripped via :func:`clean_corpus` before training.
        """
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, (list, tuple)):
            text = build_corpus(list(source))
        elif isinstance(source, str):
            text = source
        else:
            raise TypeError(
                "Training source must be str (raw text), pathlib.Path (a file), or "
                "list[str] (URLs)."
            )
        return clean_corpus(text)

    def _generate_opposites(
        self,
        on_chunks: list[str],
        progress: Callable[[int, int], None] | None = None,
        dimensions: list[str] | None = None,
    ) -> list[str]:
        """Rewrite each on-brand chunk into one off-brand opposite per tone axis."""
        if self.generator is None:
            self.generator = OpenAIGenerator()
        dims = dimensions if dimensions is not None else list(TONE_DIMENSIONS)
        unknown = [name for name in dims if name not in TONE_DIMENSIONS]
        if unknown:
            raise ValueError(f"Unknown tone dimension(s): {unknown}. Choose from {list(TONE_DIMENSIONS)}.")

        total = len(on_chunks) * len(dims)
        opposites: list[str] = []
        done = 0
        for chunk in on_chunks:
            for name in dims:
                prompt = DIMENSION_PROMPT.format(
                    dimension=name, axis=TONE_DIMENSIONS[name], chunk=chunk
                )
                opposite = self.generator.generate(prompt).strip()
                if not opposite:
                    raise ValueError("Generation backend returned an empty opposite example.")
                opposites.append(opposite)
                done += 1
                if progress is not None:
                    progress(done, total)
        return opposites

    def calibrate_from_urls(
        self,
        on_brand_urls: list[str],
        off_brand_urls: list[str],
        fetcher: Callable[[str], str] = fetch_url_text,
    ) -> float:
        """Train from URLs: fetch and extract each page, then calibrate.

        Every on-brand URL's extracted text is concatenated into one corpus, and
        likewise for off-brand, before the usual chunking and fitting.  ``fetcher``
        is injectable for testing or for supplying pre-fetched content.
        """
        on_brand_raw = build_corpus(on_brand_urls, fetcher=fetcher)
        off_brand_raw = build_corpus(off_brand_urls, fetcher=fetcher)
        return self.calibrate_weights(on_brand_raw, off_brand_raw)

    def calibrate_weights(self, on_brand_raw: str, off_brand_raw: str) -> float:
        """Fit and persist the brand-specific logistic-regression head.

        On-brand and off-brand corpora are chunked independently and labelled
        1/0; the counts need not match.  The chunks and their embeddings are
        persisted alongside the head so verdicts can be explained against real
        samples.  Returns training accuracy.
        """
        on_chunks = self._chunk_text(on_brand_raw)
        off_chunks = self._chunk_text(off_brand_raw)
        if not on_chunks or not off_chunks:
            raise ValueError("Training inputs contain insufficient sentences to extract tone.")
        return self._fit(on_chunks, off_chunks)

    def train_from_labeled(
        self,
        examples: list[tuple[str, int]],
        include_existing: bool = False,
    ) -> float:
        """Train directly from labelled responses (e.g. production thumbs up/down).

        ``examples`` is a list of ``(text, label)`` pairs where label ``1`` is
        on-brand (thumbs up) and ``0`` is off-brand (thumbs down). Each response is
        a whole example — it is embedded as-is, not chunked, and no opposites are
        generated. This is the path for real user-labelled data.

        With ``include_existing=True``, the current brand's stored examples are
        reused (embeddings and all, no re-embedding) and the new ones are appended —
        so you can grow a bootstrap model with real feedback over time. Returns
        training accuracy.
        """
        if not examples:
            raise ValueError("examples must be a non-empty list of (text, label) pairs.")

        new_on: list[str] = []
        new_off: list[str] = []
        for text, label in examples:
            clean = self._validate_response(text)
            value = int(label)
            if value == 1:
                new_on.append(clean)
            elif value == 0:
                new_off.append(clean)
            else:
                raise ValueError(
                    "Each label must be 1 (on-brand / thumbs up) or 0 (off-brand / thumbs down)."
                )

        new_embeddings = self._encode(new_on + new_off)
        on_texts, on_embeddings = new_on, new_embeddings[: len(new_on)]
        off_texts, off_embeddings = new_off, new_embeddings[len(new_on) :]

        if include_existing:
            existing = self.store.load(self.brand_id)
            if isinstance(existing, dict) and existing.get("on_embeddings") is not None:
                on_texts = list(existing["on_chunks"]) + on_texts
                off_texts = list(existing["off_chunks"]) + off_texts
                on_embeddings = np.vstack([existing["on_embeddings"], on_embeddings])
                off_embeddings = np.vstack([existing["off_embeddings"], off_embeddings])

        if not on_texts or not off_texts:
            raise ValueError(
                "Training needs at least one on-brand (1) and one off-brand (0) example."
            )
        return self._fit_embedded(on_texts, on_embeddings, off_texts, off_embeddings)

    def _fit(self, on_chunks: list[str], off_chunks: list[str]) -> float:
        """Embed labelled chunks, fit the head, persist the artifact, return accuracy."""
        embeddings = self._encode(on_chunks + off_chunks)
        split = len(on_chunks)
        return self._fit_embedded(
            on_chunks, embeddings[:split], off_chunks, embeddings[split:]
        )

    def _fit_embedded(
        self,
        on_chunks: list[str],
        on_embeddings: np.ndarray,
        off_chunks: list[str],
        off_embeddings: np.ndarray,
    ) -> float:
        """Fit the head from pre-computed embeddings, persist, and return accuracy."""
        labels = np.concatenate(
            (np.ones(len(on_chunks), dtype=np.int64), np.zeros(len(off_chunks), dtype=np.int64))
        )
        embeddings = np.vstack([on_embeddings, off_embeddings]).astype(np.float32)

        self.classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.classifier.fit(embeddings, labels)

        self.on_chunks = list(on_chunks)
        self.off_chunks = list(off_chunks)
        self.on_embeddings = on_embeddings
        self.off_embeddings = off_embeddings

        self.store.save(
            self.brand_id,
            {
                "format": ARTIFACT_FORMAT,
                "classifier": self.classifier,
                "on_chunks": self.on_chunks,
                "off_chunks": self.off_chunks,
                "on_embeddings": self.on_embeddings,
                "off_embeddings": self.off_embeddings,
            },
        )
        return float(self.classifier.score(embeddings, labels))

    def evaluate_response(self, llm_generated_text: str) -> float:
        """Return the probability that a live response is on-brand."""
        text = self._validate_response(llm_generated_text)
        self._load_artifact()
        embedding = self._encode([text])
        return self._on_brand_probability(embedding)

    def pick_best(self, candidates: list[str]) -> dict[str, object]:
        """Score several candidate responses and return the most on-brand one.

        Best-of-N reranking: embed all candidates in one batch, score each, and
        return the highest. Because it only *ranks* candidates against each other,
        it is robust to a poorly-calibrated absolute threshold — the model just has
        to order them roughly right.

        Returns ``{"text", "score", "ranked"}`` where ``ranked`` is every candidate
        with its score, best first.
        """
        if not candidates:
            raise ValueError("candidates must be a non-empty list of strings")
        texts = [self._validate_response(candidate) for candidate in candidates]
        self._load_artifact()
        scores = self._on_brand_scores(self._encode(texts))
        order = np.argsort(scores)[::-1]
        ranked = [{"text": texts[i], "score": float(scores[i])} for i in order]
        return {"text": ranked[0]["text"], "score": ranked[0]["score"], "ranked": ranked}

    def explain(self, llm_generated_text: str) -> dict[str, object]:
        """Score a response and ground the verdict in the brand's own samples.

        Returns the on-brand probability plus the nearest on-brand and off-brand
        training chunks (by cosine similarity) — the concrete text the response
        reads most and least like. No generative model is involved.
        """
        text = self._validate_response(llm_generated_text)
        self._load_artifact()
        if self.on_embeddings is None or self.off_embeddings is None:
            raise ValueError(
                f"Artifact for brand '{self.brand_id}' has no stored exemplars. "
                "Retrain with calibrate_weights/calibrate_from_urls to enable explain()."
            )

        embedding = self._encode([text])
        score = self._on_brand_probability(embedding)
        query = embedding[0]
        return {
            "score": score,
            "verdict": "on-brand" if score >= 0.5 else "off-brand",
            "closest_on_brand": self._nearest(query, self.on_embeddings, self.on_chunks),
            "closest_off_brand": self._nearest(query, self.off_embeddings, self.off_chunks),
        }

    def refine(
        self,
        generate: Callable[[str | None], str],
        threshold: float = 0.7,
        max_attempts: int = 4,
    ) -> dict[str, object]:
        """Regenerate a response until it scores on-brand, using explain feedback.

        ``generate(feedback)`` is your LLM call: on the first attempt ``feedback``
        is ``None``; after a low score it receives a plain-text hint built from the
        nearest off-brand and on-brand samples. Stops as soon as a response reaches
        ``threshold``; otherwise returns the best attempt after ``max_attempts``.

        Returns ``{"text", "score", "passed", "attempts"}``.
        """
        if not callable(generate):
            raise TypeError("generate must be callable")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        best: dict[str, object] | None = None
        feedback: str | None = None
        for attempt in range(1, max_attempts + 1):
            text = generate(feedback)
            result = self.explain(text)
            score = float(result["score"])
            if best is None or score > float(best["score"]):
                best = {"text": text, "score": score}
            if score >= threshold:
                return {**best, "passed": True, "attempts": attempt}
            feedback = self._build_feedback(result)

        return {**best, "passed": False, "attempts": max_attempts}

    @staticmethod
    def _build_feedback(result: dict[str, object]) -> str:
        """Turn an explain() result into a hint for the next generation."""
        off = result["closest_off_brand"]["text"]
        on = result["closest_on_brand"]["text"]
        return (
            f"That response scored {float(result['score']):.2f} on-brand (too low). "
            f'It sounds too much like this off-brand sample: "{off}". '
            f'Rewrite it to sound more like this on-brand sample: "{on}".'
        )

    def _nearest(
        self, query: np.ndarray, matrix: np.ndarray, chunks: list[str]
    ) -> dict[str, object]:
        """Return the chunk in ``matrix`` most similar to ``query``."""
        similarities = self._cosine_similarity(query, matrix)
        best = int(np.argmax(similarities))
        return {"text": chunks[best], "similarity": float(similarities[best])}

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity between one vector and each row of ``matrix``."""
        query_norm = query / (np.linalg.norm(query) + 1e-12)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        return matrix_norm @ query_norm

    def _on_brand_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """On-brand probability for each row of an encoded batch."""
        assert self.classifier is not None  # set by _load_artifact
        probabilities = self.classifier.predict_proba(embeddings)
        on_brand_index = int(np.where(self.classifier.classes_ == 1)[0][0])
        return probabilities[:, on_brand_index]

    def _on_brand_probability(self, embedding: np.ndarray) -> float:
        """Probability of the on-brand class for a single encoded sample."""
        return float(self._on_brand_scores(embedding)[0])

    @staticmethod
    def _validate_response(llm_generated_text: str) -> str:
        if not isinstance(llm_generated_text, str) or not llm_generated_text.strip():
            raise ValueError("llm_generated_text must be a non-empty string")
        return llm_generated_text

    def _load_artifact(self) -> None:
        """Load the persisted head (and exemplars, if present) once per process.

        Supports both the current dict artifact and legacy files that stored a
        bare ``LogisticRegression``; legacy files load without exemplars.
        """
        if self.classifier is not None:
            return
        loaded = self.store.load(self.brand_id)
        if loaded is None:
            raise FileNotFoundError(
                f"No trained artifact for brand '{self.brand_id}'. Call "
                "calibrate_weights or calibrate_from_urls first."
            )
        if isinstance(loaded, dict):
            self.classifier = loaded["classifier"]
            self.on_chunks = loaded.get("on_chunks")
            self.off_chunks = loaded.get("off_chunks")
            self.on_embeddings = loaded.get("on_embeddings")
            self.off_embeddings = loaded.get("off_embeddings")
        else:  # legacy: bare classifier, no exemplars
            self.classifier = loaded

    def _encode(self, text_samples: list[str]) -> np.ndarray:
        """Embed text and validate the returned sample matrix.

        The embedding dimension is model-dependent and no longer fixed; sklearn
        enforces train/inference consistency, so we only check the matrix shape.
        """
        embeddings = np.asarray(
            self.embedding_model.encode(text_samples, convert_to_numpy=True), dtype=np.float32
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(text_samples):
            raise ValueError("Embedding model returned an invalid sample matrix.")
        return embeddings
