# Simasia Tone Guardrail

**Does an AI reply sound like your brand? Simasia gives it a score from 0 to 1.**

What you can do:

- **Train** it once on your brand's writing.
- **Score** any AI response — how on-brand is it?
- **Explain** the score — see the closest on-brand and off-brand samples.
- **Refine** — keep regenerating a reply until it sounds right.

How it works, in one line: a shared model turns text into numbers, and a tiny
per-brand classifier learns your voice. Runs on CPU, no GPU needed.

One heads-up: the trained model is saved to a file that includes your training
text — keep it private if that text is sensitive.

Version history is in [CHANGELOG.md](CHANGELOG.md).

## Quick start

Two methods: **`train`** once per brand, then **`refine`** in your response pipeline.

```python
from simasia import SimasiaGuard

guard = SimasiaGuard(brand_id="fintech_core")   # key from EMBEDDING_KEY

# 1. Train once from ON-BRAND text only. For each on-brand chunk, a lightweight
#    LLM writes an off-brand opposite to train against (key: GENERATION_KEY, or
#    it falls back to EMBEDDING_KEY). Pass URLs instead of text if you prefer.
guard.train(on_brand="We build automated investment tools. They work fast.")

# Or supply both sides yourself and skip the LLM step entirely:
# guard.train(on_brand="...", off_brand="...")

# 2. Wire into the pipeline: keep regenerating until the reply is on-brand.
def generate(feedback):
    prompt = "Reply to the customer about their late transfer."
    if feedback:                 # a hint from the previous low-scoring attempt
        prompt += "\n\n" + feedback
    return my_llm(prompt)        # your LLM call

result = guard.refine(generate, threshold=0.7, max_attempts=4)
print(result)  # {"text": "...", "score": 0.82, "passed": True, "attempts": 2}
```

## Config-file usage (no code)

Prefer not to write Python? Configure once, then run one command.

1. `pip install "simasia[openai]"`
2. Copy `.env.example` → `.env.local`, add your key (`EMBEDDING_KEY=...`). It's gitignored — keys never go in the committed config.
3. Copy `simasia.example.toml` → `simasia.toml`, set your brand and training source.
4. Train, then score:

```bash
simasia train                       # reads simasia.toml + .env.local
simasia score "Hey! Quick heads up about your transfer."
simasia explain "Kindly be advised your request is under review."
```

`simasia train --config path/to/other.toml` points at a different config.

## Install

The package is parametric — install only the backend you want:

```bash
pip install .[openai]          # default backend: OpenAI text-embedding-3-small
pip install .[urls]            # train from URLs (fetch + clean-text extraction)
pip install .[local]           # offline backend: frozen sentence-transformer, CPU
pip install .[openai,urls,test]  # a typical dev setup
```

`requirements.txt` is a quick-start shortcut for the OpenAI + URLs combination.

## Choosing an embedding backend

The backend is any object with an `encode(list[str]) -> np.ndarray` method
(`EmbeddingModel`). The default is OpenAI, with the API key read from the
`EMBEDDING_KEY` environment variable:

```python
from simasia import SimasiaGuard, OpenAIEmbedder, LocalEmbedder

# Default: OpenAIEmbedder(), key from EMBEDDING_KEY
guard = SimasiaGuard(brand_id="fintech_core")

# Pick a model / shorten the vector / pass a key explicitly
guard = SimasiaGuard(
    brand_id="fintech_core",
    embedding_model=OpenAIEmbedder(model="text-embedding-3-small", dimensions=512),
)

# Fully offline (no network, CPU) — requires the `local` extra and a cached model
guard = SimasiaGuard(brand_id="fintech_core", embedding_model=LocalEmbedder())
```

## Training

`train(on_brand, off_brand=None)` is the one entry point. Each side accepts three
source types, and you can mix them:

```python
guard.train("We build automated investment tools. They work fast.")  # str  -> raw text
guard.train(Path("brand_voice.txt"))                                 # Path -> read file
guard.train(["https://brand.example/voice", "https://brand.example/blog"])  # list -> URLs
```

A `str` is always raw text (never guessed as a path); use `Path` for a file. URLs
are fetched and reduced to clean article text, then all on-brand pages are
concatenated into one corpus (same for off-brand) before chunking.

**On-brand only (opposites generated).** Omit `off_brand` and a lightweight LLM
writes an off-brand opposite for each on-brand chunk (key: `GENERATION_KEY`, or it
falls back to `EMBEDDING_KEY`):

```python
guard.train(on_brand="We build automated investment tools. They work fast.")
```

**Both sides supplied (no LLM).** Pass both to skip generation entirely:

```python
guard.train(
    on_brand="We build automated investment tools. They work fast.",
    off_brand="Our enterprise architecture processes asset allocations. "
              "Systems experience transactional delay cycles.",
)
```

The lower-level `calibrate_weights` (raw text) and `calibrate_from_urls` (URLs)
methods remain available if you want to call them directly.

## Scoring live responses

```python
score = guard.evaluate_response("Hey! Let's get your account squared away right now.")
print(score)  # probability in [0, 1] that the text is on-brand
```

## Explaining a verdict

`explain()` returns the score plus the nearest on-brand and off-brand training
chunks — the concrete text the response reads most and least like. No generative
model is involved; the "reason" is retrieved from the brand's own samples.

```python
result = guard.explain("Kindly be advised your request is under review.")
# {
#   "score": 0.21,
#   "verdict": "off-brand",
#   "closest_on_brand":  {"text": "We build automated tools...", "similarity": 0.44},
#   "closest_off_brand": {"text": "Systems experience delay cycles.", "similarity": 0.73},
# }
```

## Steering a response on-brand

`refine()` keeps asking your LLM for a response until it scores on-brand. You pass
a `generate(feedback)` function; the first call gets `feedback=None`, and after a
low score it gets a plain-text hint built from the nearest off/on-brand samples.

```python
def generate(feedback):
    prompt = "Reply to the customer."
    if feedback:
        prompt += "\n\n" + feedback   # explain()-based hint
    return my_llm(prompt)

result = guard.refine(generate, threshold=0.7, max_attempts=4)
# {"text": "...", "score": 0.82, "passed": True, "attempts": 2}
```

## Where the artifact is stored

By default the artifact is a file (`FileArtifactStore`). For production (many
servers, containers, serverless) put it in shared storage instead — your own
database or an S3/GCS bucket. Implement `ArtifactStore` (`save`, `load`,
`exists`) and pass it in; nothing in the guard changes.

Use `serialize_artifact` / `deserialize_artifact` so your store only handles
bytes. Example against any DB (shown with SQLite):

```python
import sqlite3
from simasia import SimasiaGuard, serialize_artifact, deserialize_artifact

class SQLiteStore:
    def __init__(self, conn):
        self.conn = conn
        conn.execute("CREATE TABLE IF NOT EXISTS simasia (brand TEXT PRIMARY KEY, blob BLOB)")

    def save(self, brand_id, artifact):
        self.conn.execute(
            "REPLACE INTO simasia (brand, blob) VALUES (?, ?)",
            (brand_id, serialize_artifact(artifact)),
        )
        self.conn.commit()

    def load(self, brand_id):
        row = self.conn.execute(
            "SELECT blob FROM simasia WHERE brand = ?", (brand_id,)
        ).fetchone()
        return deserialize_artifact(row[0]) if row else None

    def exists(self, brand_id):
        return self.conn.execute(
            "SELECT 1 FROM simasia WHERE brand = ?", (brand_id,)
        ).fetchone() is not None

guard = SimasiaGuard(brand_id="fintech_core", store=SQLiteStore(sqlite3.connect("brands.db")))
```
