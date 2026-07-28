# Simasia: Technical Article Outline

Working outline for a technical article on Simasia — what it is, how it works
under the hood, and the decision-by-decision journey (including the AI-assisted
design process) that produced it. Bullets are prompts for writing, not final prose.

---

## 0. Framing

- **Working title options:** "Teaching a classifier your brand's voice",
  "A tone guardrail for LLM output in 300 lines", "Judge, then steer: keeping AI
  replies on-brand".
- **One-line hook:** Does an AI reply sound like *your* brand? Score it 0–1,
  explain why, and regenerate until it fits.
- **Audience:** engineers building on LLMs; ML-curious but not ML-expert.
- **TL;DR box:** frozen embedding model + a tiny per-brand logistic-regression
  head = cheap, fast, per-brand tone scoring. Then two things on top: a *reason*
  (nearest examples) and a *steering loop* (regenerate to pass).

## 1. The problem

- LLM output drifts off-brand: too formal, too casual, wrong voice.
- Existing options and why they fall short: prompt-only steering is unreliable;
  fine-tuning is heavy and per-brand-expensive; hand-written rules don't capture
  "tone".
- Requirements that shaped the design: cheap per brand, run anywhere (CPU),
  importable as a library, usable by non-ML people.

## 2. The core idea (and why)

- **Frozen embedding body + small trainable head.** Embeddings capture tone;
  a per-brand `LogisticRegression` is what actually learns "on vs off brand".
- Why this split: the expensive model never changes; each brand is just a few KB
  of weights. Train once, score forever.
- Trade-off named up front: embeddings are powerful **but opaque** — this comes
  back to bite the "explain" feature (Section 6).

## 3. Build it — a staged tutorial

> The spine of the article, written as a follow-along build. Each stage adds one
> capability, states the design choice *as a lesson* ("do X, because Y"), and ends
> with what the reader can now do. Every stage is a working increment.
>
> Format per stage: **Goal → Code you add → Design note (the "because") → You can
> now…**. Keep the reasoning as teaching callouts, not autobiography.

- **Stage 1 — The minimal core.**
  - Goal: score text as on/off-brand at all.
  - Build: frozen embedding model + a `LogisticRegression` head; `train(on, off)`
    on two text blobs; `evaluate_response(text)` → probability.
  - Design note: freeze the big model, train only a tiny head — cheap per brand,
    train once, score forever.
  - You can now: score a string 0–1.

- **Stage 2 — Make the embedding backend swappable.**
  - Build: an `EmbeddingModel` protocol; `OpenAIEmbedder` (default) and
    `LocalEmbedder`; remove the hardcoded 384-dim assumption.
  - Design note: program to an interface so users pick their model/keys and you
    can test with a fake. Dimension is model-dependent — let the classifier
    enforce consistency.
  - You can now: run on OpenAI or fully offline, same code.

- **Stage 3 — Train from URLs.**
  - Build: `sources.py` using `trafilatura` to fetch + extract clean article text;
    concatenate pages into one corpus per side, then chunk.
  - Design note: integrate, don't build — fetching is trivial, *clean extraction*
    is the hard part. A page is raw material, never a single chunk.
  - You can now: `train(["https://…", …])`.

- **Stage 4 — Loosen the training data.**
  - Build: drop any "equal on/off counts" requirement; just label and fit.
  - Design note: logistic regression needs *labeled* samples, not *paired* ones —
    and real web pages never produce equal counts.
  - You can now: feed uneven amounts of on/off text.

- **Stage 5 — Train from on-brand text alone.**
  - Build: a `GenerationModel` backend (`OpenAIGenerator`); for each on-brand
    chunk, generate an off-brand *opposite*, then embed both.
  - Design note: inject the generator like the embedder; one lightweight LLM call
    per chunk at train time removes the burden of authoring off-brand data.
  - You can now: `train(on_brand="…")` with no off-brand corpus.

- **Stage 6 — Explain the score (say *why*).**
  - Build: persist the training chunks + embeddings; at score time do a
    cosine-similarity search and return the nearest on-brand and off-brand samples.
  - Design note: the classifier's coefficients are over opaque embedding
    dimensions — meaningless to a human. Ground the reason in real examples
    instead. Trade-off to state plainly: you now store source text in the artifact.
  - You can now: `explain(text)` → score + closest samples (no LLM).

- **Stage 7 — Steer, don't just judge.**
  - Build: `refine(generate, threshold, max_attempts)` — call the user's LLM,
    score, and if low, feed the explain() reason back and regenerate.
  - Design note: the classifier is a *discriminator*, not a *writer*. Pair it with
    a generator you *inject*; the library orchestrates but never owns an LLM.
  - You can now: auto-regenerate replies until they pass.

- **Stage 8 — Abstract storage for production.**
  - Build: an `ArtifactStore` seam (`FileArtifactStore` default) plus
    `serialize_artifact` / `deserialize_artifact` for DB/S3 backends.
  - Design note: a local file is fine for dev but breaks on containers/serverless;
    abstract the seam early so prod storage is a drop-in.
  - You can now: keep the model in your own database.

- **Stage 9 — A config file + CLI for non-coders.**
  - Build: `simasia.toml` for settings, `.env.local` for keys (gitignored),
    `simasia train | score | explain`.
  - Design note: "edit a config, run one command." Keys go in env, never the
    committed config — a security rule the project learned the hard way.
  - You can now: use it with zero Python.

- **Stage 10 — One clean entry point.**
  - Build: `train()` accepts `str` (raw text), `Path` (file), or `list[str]`
    (URLs), mixable per side; `str` is never guessed as a path.
  - Design note: one obvious front door beats several near-duplicate methods.
  - You can now: point training at anything without picking a method.

## 4. How the system works (walkthrough)

- **Module map:** `guard.py` (orchestration), `embeddings.py`, `generation.py`,
  `sources.py` (URL → corpus), `storage.py`, `config.py`, `cli.py`.
- **Training flow:** source → clean text → sentence chunking (overlapping
  2-sentence windows, ≥4 words) → embed → fit `LogisticRegression` → persist
  {classifier, chunks, embeddings}.
- **On-brand-only flow:** chunk → per-chunk LLM "opposite" → embed both → fit.
- **Scoring flow:** load artifact once (cached in-process) → embed response →
  `predict_proba` → on-brand probability.
- **Explain flow:** embed once → score + cosine-similarity search over stored
  exemplars → nearest on/off samples.
- **Refine flow:** loop generate → score → (if low) feedback → regenerate.

## 5. Under the hood — the mechanics worth explaining

- **Why embeddings + logistic regression works for "tone":** semantic space,
  linear separability of a style axis.
- **The cost model:** training = N embedding calls (+ N generation calls if
  auto-opposite). Inference = **one embedding call per response**; the regression
  itself is a dot product (effectively free). Debunk "score is cached forever".
- **Cosine similarity for explain** and why normalization matters.
- **Discriminator vs generator** — the conceptual crux the whole feature set
  pivots on.
- **What's persisted and why it grew:** head-only → head + reference corpus (for
  explain), and the privacy implication.

## 6. Making it a real package

- Pluggable backends via `Protocol` (embedding, generation, storage) = the design
  pattern that keeps it library-friendly.
- Optional-dependency **extras** (`openai`, `local`, `urls`) so consumers install
  only what they use.
- Config + CLI + env-based secrets.

## 7. Shipping it — the boring lessons that bite

- **PyPI immutability:** a published version is frozen — you can't overwrite files
  or edit the description; changes require a new version (the 0.2.0 → 0.2.1 story).
- **TestPyPI vs PyPI** are separate accounts/tokens (the 403s).
- **`--index-url` replaces the whole index**, so deps must come via
  `--extra-index-url`.
- **Secret hygiene:** repeated token leaks → `.gitignore`, keys out of the repo,
  and the case for CI publishing with secrets instead of a local `.env`.

## 8. Optional sidebars: built with an AI pair

> In tutorial form this is *not* a standalone diary section — sprinkle these as
> short "Behind the build" callouts next to the relevant stage.

- Next to Stage 4/5: how a correction ("drop pairing", "generate opposites",
  "a page isn't a chunk") redirected the design.
- Next to Stage 6: the "we don't need the prose" call that kept explain() LLM-free.
- General callout: the assistant surfaced forks and the human chose; it also
  caught its own stale docs and an untested URL path (never actually run until
  verified live). Honest, not a sales pitch.
- If you'd rather keep the tutorial pure, cut these and spin the collaboration
  angle into a separate short piece.

## 9. Limitations & future work

- Opposite-generation quality depends on the prompt/model; long corpora = many
  train-time LLM calls (batching is a TODO).
- Exemplar scan is linear per explain call (fine at brand scale; ANN later).
- Threshold calibration is manual.
- Storing source text in the artifact — an opt-out / embeddings-only mode.
- No CHANGELOG-driven automated releases yet (GitHub Actions on tag).

## 10. Appendix

- Architecture diagram (source → chunk → embed → head → {score, explain, refine}).
- Minimal end-to-end code sample (train + refine).
- Links: repo, PyPI, CHANGELOG.

---

### Suggested tutorial arc
Problem → the cheap core (Stage 1) → build up capability stage by stage
(Stages 2–10) → package it → ship it (and the scars). Each stage leaves the reader
with something that runs. Teach the "judge vs writer" insight at Stage 7 — it's the
memorable turn where scoring becomes steering. Keep the AI-pair notes as optional
sidebars (Section 8), not the spine.
