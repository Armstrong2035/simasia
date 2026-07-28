# Build a Brand-Tone Guardrail for LLM Output

This tutorial shows you how to build Simasia. Simasia gives a text a score from 0
to 1. The score tells you if an AI reply sounds like your brand.

This document uses ASD-STE100 Simplified Technical English. The sentences are
short. The text uses the active voice. Each instruction is one step.

---

## What you build

Simasia does four tasks:

- **Train** — it learns your brand voice one time.
- **Score** — it rates any AI reply from 0 to 1.
- **Explain** — it shows the closest on-brand and off-brand examples.
- **Refine** — it makes the model write again until the reply is on-brand.

You build these tasks in stages. Each stage adds one capability. Each stage gives
you code that runs.

## The core idea

Simasia has two parts:

- A large embedding model. This model turns text into numbers. This model does not
  change.
- A small classifier. This classifier learns one brand. It is a logistic
  regression.

You freeze the large model. You train only the small classifier. This method is
cheap. You train one time for each brand. Then you score many replies.

**Note:** An embedding is a strong signal. But a person cannot read an embedding.
This fact is important in Stage 6.

---

## Stage 1 — Build the core

**Goal:** Give a score to a text.

Do these steps:

1. Load an embedding model.
2. Make a logistic-regression head.
3. Train the head on on-brand text and off-brand text.
4. Score a new text.

```python
from sklearn.linear_model import LogisticRegression

class SimasiaGuard:
    def train(self, on_brand: str, off_brand: str) -> float:
        on = self._chunk(on_brand)
        off = self._chunk(off_brand)
        samples = on + off
        labels = [1] * len(on) + [0] * len(off)
        vectors = self.embedder.encode(samples)
        self.head = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.head.fit(vectors, labels)
        return self.head.score(vectors, labels)

    def evaluate_response(self, text: str) -> float:
        vector = self.embedder.encode([text])
        return float(self.head.predict_proba(vector)[0, 1])
```

**Design note:** Freeze the large model. Train only the small head. This method
keeps the cost low. Each brand is only a few kilobytes of weights.

**Result:** You can score a string from 0 to 1.

## Stage 2 — Make the embedding backend replaceable

**Goal:** Let the user choose the embedding model.

Do these steps:

1. Write an interface with one method, `encode`.
2. Write an OpenAI backend. Make it the default.
3. Write a local backend for offline use.
4. Remove any fixed vector size from the code.

```python
from typing import Protocol
import numpy as np

class EmbeddingModel(Protocol):
    def encode(self, sentences: list[str]) -> np.ndarray: ...
```

**Design note:** Write to an interface. The user picks the model and the key. You
can also test with a fake backend. The vector size depends on the model, so the
classifier — not a fixed number — controls the size.

**Result:** The same code runs on OpenAI or fully offline.

## Stage 3 — Train from URLs

**Goal:** Train from web pages, not only from text.

Do these steps:

1. Add the `trafilatura` library.
2. Download each URL.
3. Extract the main article text from each page.
4. Join all pages into one text block for each side.
5. Chunk the text block as before.

```python
import trafilatura

def fetch_url_text(url: str) -> str:
    page = trafilatura.fetch_url(url)
    if page is None:
        raise ValueError(f"Could not download the URL: {url}")
    text = trafilatura.extract(page)
    if not text:
        raise ValueError(f"No text content at the URL: {url}")
    return text
```

**Design note:** Use a library. Do not write your own reader. A download is easy.
Clean text extraction is difficult, because a page also has menus and footers. A
page is raw material. A page is never one chunk.

**Result:** You can train from a list of URLs.

## Stage 4 — Loosen the training data

**Goal:** Accept different amounts of on-brand and off-brand text.

Do this step:

1. Remove the rule that the two sides must have an equal number of chunks.

**Design note:** A logistic regression needs labelled samples. It does not need
paired samples. Also, two web pages almost never give an equal number of chunks.
The rule was a limit with no benefit.

**Result:** You can give different amounts of on-brand and off-brand text.

## Stage 5 — Train from on-brand text only

**Goal:** Train a brand from on-brand text alone.

Do these steps:

1. Add a generation backend with one method, `generate`.
2. Chunk the on-brand text.
3. For each on-brand chunk, ask the model for an opposite chunk.
4. Use the on-brand chunks and the opposite chunks as the two sides.

```python
OPPOSITE_PROMPT = (
    "Rewrite the text below so it has the OPPOSITE tone and voice. "
    "Keep a similar topic and length. Return only the new text.\n\nText: {chunk}"
)

def generate_opposites(self, on_chunks: list[str]) -> list[str]:
    opposites = []
    for chunk in on_chunks:
        opposites.append(self.generator.generate(OPPOSITE_PROMPT.format(chunk=chunk)))
    return opposites
```

**Design note:** Inject the generator, like the embedder. The model makes one
lightweight call for each chunk at train time. The user does not write off-brand
data.

**Result:** You can call `train(on_brand="...")` with no off-brand text.

## Stage 6 — Explain the score

**Goal:** Show the user why a text got its score.

Do these steps:

1. Save the training chunks and their vectors, not only the head.
2. At score time, embed the new text one time.
3. Compare the new text to each saved chunk with cosine similarity.
4. Return the closest on-brand chunk and the closest off-brand chunk.

```python
def explain(self, text: str) -> dict:
    vector = self.embedder.encode([text])[0]
    return {
        "score": self._score(vector),
        "closest_on_brand": self._nearest(vector, self.on_vectors, self.on_chunks),
        "closest_off_brand": self._nearest(vector, self.off_vectors, self.off_chunks),
    }
```

**Design note:** The classifier has weights for each vector dimension. But a person
cannot read those dimensions. So do not use the weights as the reason. Instead,
show real examples from the brand.

**Trade-off:** You now store the source text in the model file. State this fact
clearly. Keep the file private if the text is sensitive.

**Result:** `explain(text)` returns the score and the closest examples. It uses no
language model.

## Stage 7 — Steer, do not only judge

**Goal:** Make the reply on-brand, not only measure it.

Do these steps:

1. Accept a `generate` function from the user.
2. Call the function to get a reply.
3. Score the reply.
4. If the score is too low, add a hint and call the function again.
5. Stop when the reply passes or the attempts run out.

```python
def refine(self, generate, threshold=0.7, max_attempts=4) -> dict:
    feedback = None
    best = None
    for attempt in range(1, max_attempts + 1):
        text = generate(feedback)
        result = self.explain(text)
        if best is None or result["score"] > best["score"]:
            best = {"text": text, "score": result["score"]}
        if result["score"] >= threshold:
            return {**best, "passed": True, "attempts": attempt}
        feedback = self._build_feedback(result)
    return {**best, "passed": False, "attempts": max_attempts}
```

**Design note:** The classifier is a judge. The classifier is not a writer. To
change a reply, you need a writer. So the user gives Simasia a language model.
Simasia controls the loop. Simasia does not own the model.

**Result:** Simasia writes a reply again until the reply passes.

## Stage 8 — Store the model for production

**Goal:** Keep the model in shared storage, not only on local disk.

Do these steps:

1. Write a storage interface with three methods: `save`, `load`, and `exists`.
2. Make a file backend the default.
3. Add helpers that turn the model into bytes and back.

```python
def serialize_artifact(artifact) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    return buffer.getvalue()
```

**Design note:** A local file is correct for one machine. But a container has no
permanent disk. A server farm has many machines. Abstract the storage seam early.
Then a database backend is a small change.

**Result:** You can keep the model in your own database.

## Stage 9 — Add a config file and a CLI

**Goal:** Let a person use Simasia with no Python code.

Do these steps:

1. Read the settings from a `simasia.toml` file.
2. Read the keys from a `.env.local` file.
3. Add the commands `train`, `score`, and `explain`.

```bash
simasia train                       # reads simasia.toml and .env.local
simasia score "Hey! Quick update on your transfer."
simasia explain "Kindly be advised of the delay."
```

**Design note:** The user edits a config file and runs one command. Put the keys
in the environment, never in the config file. This project learned this rule the
hard way, after real keys leaked.

**Result:** A person uses Simasia with no code.

## Stage 10 — One clean entry point

**Goal:** Give the user one method for all training inputs.

Do this step:

1. Make `train()` accept three input types:
   - a `str` for raw text,
   - a `Path` for a file,
   - a `list` for URLs.

**Design note:** One clear entry point is better than many similar methods. A
`str` is always raw text. The code never treats a `str` as a file path, because a
guess creates errors.

**Result:** You point training at any source with one method.

---

## How Simasia works internally

**The two flows.** The train flow turns text into chunks, then into vectors, then
into a fitted head. The score flow turns one reply into a vector, then into a
probability.

**The cost.** Training makes one embedding call for each chunk. It also makes one
generation call for each chunk in on-brand-only mode. A score makes one embedding
call for each reply. The classifier math is only a dot product. This math cost is
very low.

**A common error to correct.** The score is not stored forever. Each reply is new
text. So Simasia must embed and classify each reply.

**The key idea.** A classifier is a judge. A judge selects and rates. A judge does
not write. The score, the explanation, and the refine loop all follow from this
one idea.

## Make it a package

- Use a `Protocol` for each backend: embedding, generation, and storage. This
  pattern keeps the library open.
- Use optional dependencies (extras), so a user installs only the parts they need.
- Provide the config file, the CLI, and environment keys for people who do not
  write code.

## Ship it — the lessons that hurt

- **A published version is frozen.** You cannot overwrite the files of a version.
  You cannot change the description of a version. You must publish a new version.
- **The test index and the real index are separate.** They have separate accounts
  and separate keys. A wrong key gives a 403 error.
- **A private key must stay out of the repository.** Use a `.gitignore` file. Keep
  keys in the environment. Later, publish from CI with a stored secret.

## Limits and future work

- The opposite text depends on the prompt and the model. A long text makes many
  calls at train time. A batch mode is future work.
- The explain step compares the reply to every saved chunk. This method is correct
  at brand scale. A faster search is future work.
- The user sets the threshold by hand.
- The model file holds the source text. An embeddings-only mode is future work.

## Sidebars: built with an AI pair

> Add these as short notes next to the correct stage. Do not make one long section.

- Near Stage 4 and Stage 5: a human correction changed the design. Examples:
  "remove the pairing rule", "generate the opposite", and "a page is not a chunk".
- Near Stage 6: the human said "we do not need the prose". This choice kept
  `explain()` free of a language model.
- General note: the assistant offered the choices. The human made the decisions.
  The assistant also found its own old documents and one code path that no test
  had run. Keep this note honest, not a sales message.

## Minimal end-to-end example

```python
from simasia import SimasiaGuard

guard = SimasiaGuard(brand_id="fintech_core")   # key from EMBEDDING_KEY
guard.train(on_brand="We build automated investment tools. They work fast.")

def generate(feedback):
    prompt = "Reply to the customer about the late transfer."
    if feedback:
        prompt = prompt + "\n\n" + feedback
    return my_llm(prompt)

result = guard.refine(generate, threshold=0.7, max_attempts=4)
print(result)
```
