# Changelog

All notable changes to Simasia are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-07-23

First public release.

### Added
- Pluggable embedding backends (`EmbeddingModel`): `OpenAIEmbedder`
  (`text-embedding-3-small`, default) and `LocalEmbedder` (offline
  sentence-transformer). Key read from `EMBEDDING_KEY`.
- Unified `train(on_brand, off_brand=None)` entry point accepting raw text
  (`str`), a file path (`pathlib.Path`), or a list of URLs (`list[str]`), mixable
  per side.
- On-brand-only training: omit `off_brand` and a generation backend
  (`OpenAIGenerator`, key `GENERATION_KEY` → `EMBEDDING_KEY`) writes an off-brand
  opposite for each on-brand chunk.
- URL training via `trafilatura` (fetch + clean-text extraction), concatenated
  into one corpus per side.
- `explain()` returns the score plus the nearest on-brand and off-brand training
  samples (cosine similarity) — a grounded reason, no LLM.
- `refine(generate, ...)` steers an injected LLM callable, regenerating until a
  response clears the on-brand threshold, using `explain()` output as feedback.
- Pluggable `ArtifactStore` (default `FileArtifactStore`) with
  `serialize_artifact` / `deserialize_artifact` helpers for storing the model in
  a database or object store.
- Config-driven CLI: `simasia train` / `score` / `explain`, reading
  `simasia.toml` and loading keys from `.env` / `.env.local`.
- Packaging: `pyproject.toml` with extras (`openai`, `local`, `urls`, `test`),
  `simasia` console script, MIT license.

### Changed
- Embedding dimension is no longer hardcoded to 384; it is model-dependent and
  train/inference consistency is enforced by the classifier.
- The persisted artifact now stores the training chunks and their embeddings
  alongside the head (needed by `explain()`). Note: on-brand/off-brand source
  text is written into the artifact.

## [0.1.0] - internal

- Initial prototype: frozen local `BAAI/bge-small-en-v1.5` embeddings plus a
  per-brand `LogisticRegression` head, with paired on/off-brand chunk training.
  Never published to PyPI (used for TestPyPI only).

[0.2.0]: https://github.com/Armstrong2035/simasia/releases/tag/v0.2.0
