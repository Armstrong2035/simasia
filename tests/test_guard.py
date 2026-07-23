from __future__ import annotations

import os

import numpy as np
import pytest

from simasia import SimasiaGuard


class FakeEmbedder:
    def encode(self, sentences, **_kwargs):
        rows = []
        for text in sentences:
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = 1 if "friendly" in text.lower() else -1
            vector[1] = len(text)
            rows.append(vector)
        return np.asarray(rows)


def test_chunking_uses_overlapping_two_sentence_windows(tmp_path):
    guard = SimasiaGuard("demo", tmp_path, embedding_model=FakeEmbedder())
    assert guard._chunk_text("One sentence has enough words. Two has enough words. Three too.") == [
        "One sentence has enough words. Two has enough words.",
        "Two has enough words. Three too.",
    ]


def test_calibration_persists_and_loaded_head_evaluates(tmp_path):
    on_brand = "Friendly service solves your problem. Friendly humans reply promptly."
    off_brand = "Rigid processes delay your problem. Formal systems reply eventually."
    trained = SimasiaGuard("demo", tmp_path, embedding_model=FakeEmbedder())

    assert trained.calibrate_weights(on_brand, off_brand) >= 0.5
    assert (tmp_path / "simasia_demo_head.joblib").exists()

    loaded = SimasiaGuard("demo", tmp_path, embedding_model=FakeEmbedder())
    assert 0.0 <= loaded.evaluate_response("A friendly response is ready.") <= 1.0


def test_calibration_allows_unequal_chunk_counts(tmp_path):
    guard = SimasiaGuard("demo", tmp_path, embedding_model=FakeEmbedder())
    accuracy = guard.calibrate_weights(
        "Friendly first response is ready. Friendly second response follows. Friendly third response ends.",
        "Rigid response is delayed. Formal response follows.",
    )
    assert 0.0 <= accuracy <= 1.0


def test_calibrate_from_urls_builds_corpus_before_training(tmp_path):
    pages = {
        "https://brand.example/on": (
            "Friendly service solves your problem. Friendly humans reply promptly."
        ),
        "https://brand.example/off": (
            "Rigid processes delay your problem. Formal systems reply eventually."
        ),
    }
    guard = SimasiaGuard("demo", tmp_path, embedding_model=FakeEmbedder())

    accuracy = guard.calibrate_from_urls(
        ["https://brand.example/on"],
        ["https://brand.example/off"],
        fetcher=lambda url: pages[url],
    )

    assert accuracy >= 0.5
    assert (tmp_path / "simasia_demo_head.joblib").exists()


def test_empty_url_list_is_rejected(tmp_path):
    guard = SimasiaGuard("demo", tmp_path, embedding_model=FakeEmbedder())
    with pytest.raises(ValueError, match="At least one URL"):
        guard.calibrate_from_urls([], ["https://brand.example/off"], fetcher=lambda url: "x")


def test_explain_returns_score_and_nearest_exemplars(tmp_path):
    on_brand = "Friendly service solves your problem. Friendly humans reply promptly."
    off_brand = "Rigid processes delay your problem. Formal systems reply eventually."
    trained = SimasiaGuard("demo", tmp_path, embedding_model=FakeEmbedder())
    trained.calibrate_weights(on_brand, off_brand)

    # Reload from disk to prove exemplars were persisted, not just held in memory.
    loaded = SimasiaGuard("demo", tmp_path, embedding_model=FakeEmbedder())
    result = loaded.explain("A friendly response is ready right now for you.")

    assert 0.0 <= result["score"] <= 1.0
    assert result["verdict"] in {"on-brand", "off-brand"}
    assert result["closest_on_brand"]["text"] in on_brand
    assert result["closest_off_brand"]["text"] in off_brand
    assert -1.0 <= result["closest_on_brand"]["similarity"] <= 1.0


def test_explain_on_legacy_artifact_without_exemplars(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression

    # Simulate an old artifact: a bare classifier with no stored exemplars.
    clf = LogisticRegression().fit(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0, 1]))
    path = tmp_path / "simasia_legacy_head.joblib"
    joblib.dump(clf, path)

    guard = SimasiaGuard("legacy", tmp_path, embedding_model=FakeEmbedder())
    with pytest.raises(ValueError, match="no stored exemplars"):
        guard.explain("Anything at all here.")


def _trained_guard(tmp_path):
    on_brand = "Friendly service solves your problem. Friendly humans reply promptly."
    off_brand = "Rigid processes delay your problem. Formal systems reply eventually."
    guard = SimasiaGuard("demo", tmp_path, embedding_model=FakeEmbedder())
    guard.calibrate_weights(on_brand, off_brand)
    return guard


def test_refine_stops_once_threshold_is_met(tmp_path):
    guard = _trained_guard(tmp_path)
    # Second output is an exact on-brand sample, so it clears the threshold.
    outputs = [
        "Rigid processes delay your problem. Formal systems reply eventually.",
        "Friendly service solves your problem. Friendly humans reply promptly.",
    ]
    calls = {"n": 0}

    def generate(feedback):
        text = outputs[min(calls["n"], len(outputs) - 1)]
        calls["n"] += 1
        return text

    result = guard.refine(generate, threshold=0.6, max_attempts=4)

    assert result["passed"] is True
    assert result["attempts"] == 2  # first was off-brand, second passed
    assert "friendly" in result["text"].lower()


def test_refine_returns_best_effort_when_never_passing(tmp_path):
    guard = _trained_guard(tmp_path)

    def generate(feedback):
        return "Rigid formal notice is delayed."  # always off-brand

    result = guard.refine(generate, threshold=0.99, max_attempts=3)

    assert result["passed"] is False
    assert result["attempts"] == 3


def test_refine_feedback_is_none_first_then_populated(tmp_path):
    guard = _trained_guard(tmp_path)
    seen = []

    def generate(feedback):
        seen.append(feedback)
        return "Rigid formal notice is delayed."

    guard.refine(generate, threshold=0.99, max_attempts=2)

    assert seen[0] is None
    assert isinstance(seen[1], str) and "on-brand sample" in seen[1]


def test_config_resolve_side_text_file_and_urls(tmp_path):
    from simasia.config import resolve_side

    text_file = tmp_path / "brand.txt"
    text_file.write_text("Some on-brand copy.", encoding="utf-8")
    training = {
        "on_brand_text": "inline copy",
        "off_brand_file": str(text_file),
        "extra_urls": ["https://a", "https://b"],
    }

    assert resolve_side(training, "on_brand") == "inline copy"
    assert resolve_side(training, "off_brand") == "Some on-brand copy."
    assert resolve_side(training, "extra") == ["https://a", "https://b"]
    assert resolve_side(training, "missing") is None


def test_load_dotenv_sets_missing_keys_only(tmp_path, monkeypatch):
    from simasia.config import load_dotenv

    env_file = tmp_path / ".env.local"
    env_file.write_text('EMBEDDING_KEY="abc123"\n# comment\nGENERATION_KEY=xyz\n', encoding="utf-8")
    monkeypatch.delenv("EMBEDDING_KEY", raising=False)
    monkeypatch.setenv("GENERATION_KEY", "already-set")

    load_dotenv(env_file)

    assert os.environ["EMBEDDING_KEY"] == "abc123"       # filled from file
    assert os.environ["GENERATION_KEY"] == "already-set"  # not clobbered


def test_load_config_parses_toml(tmp_path):
    from simasia.config import load_config

    cfg_file = tmp_path / "simasia.toml"
    cfg_file.write_text(
        '[brand]\nid = "demo"\n[training]\non_brand_text = "hi there"\n', encoding="utf-8"
    )
    config = load_config(cfg_file)
    assert config["brand"]["id"] == "demo"
    assert config["training"]["on_brand_text"] == "hi there"


def test_cli_explain_prints_score_and_samples(monkeypatch, capsys):
    from simasia import cli

    class FakeGuard:
        def explain(self, text):
            return {
                "score": 0.21,
                "verdict": "off-brand",
                "closest_on_brand": {"text": "Warm and direct.", "similarity": 0.44},
                "closest_off_brand": {"text": "Rigid and formal.", "similarity": 0.73},
            }

    monkeypatch.setattr(cli, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(cli, "load_config", lambda *_: {"brand": {"id": "demo"}})
    monkeypatch.setattr(cli, "build_guard", lambda *_: FakeGuard())

    assert cli.main(["explain", "Kindly be advised."]) == 0
    out = capsys.readouterr().out
    assert "0.210 (off-brand)" in out
    assert "Rigid and formal." in out
    assert "Warm and direct." in out


class FakeGenerator:
    """Turns any on-brand chunk into an off-brand opposite (no 'friendly')."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return f"Rigid formal notice number {self.calls} is delayed and impersonal."


def test_train_from_on_brand_only_generates_opposites(tmp_path):
    on_brand = "Friendly service solves your problem. Friendly humans reply promptly."
    generator = FakeGenerator()
    guard = SimasiaGuard(
        "demo", tmp_path, embedding_model=FakeEmbedder(), generator=generator
    )

    accuracy = guard.train(on_brand)  # no off_brand -> opposites generated

    assert generator.calls >= 1  # the LLM was asked for opposites
    assert accuracy >= 0.5
    assert (tmp_path / "simasia_demo_head.joblib").exists()

    result = guard.explain("A friendly response is ready.")
    assert result["closest_off_brand"]["text"].startswith("Rigid formal notice")


def test_train_accepts_text_files_and_rejects_bad_types(tmp_path):
    from pathlib import Path

    on_brand = "Friendly service solves your problem. Friendly humans reply promptly."
    off_brand = "Rigid processes delay your problem. Formal systems reply eventually."

    # str = raw text
    text_guard = SimasiaGuard("t", tmp_path, embedding_model=FakeEmbedder())
    assert 0.0 <= text_guard.train(on_brand, off_brand) <= 1.0

    # Path = read the file as text
    on_file = tmp_path / "on.txt"
    off_file = tmp_path / "off.txt"
    on_file.write_text(on_brand, encoding="utf-8")
    off_file.write_text(off_brand, encoding="utf-8")
    file_guard = SimasiaGuard("f", tmp_path, embedding_model=FakeEmbedder())
    assert 0.0 <= file_guard.train(Path(on_file), Path(off_file)) <= 1.0

    # unsupported type is rejected
    with pytest.raises(TypeError, match="Training source must be"):
        text_guard.train(on_brand, 123)


def test_custom_store_is_used_instead_of_files(tmp_path):
    class MemoryStore:
        def __init__(self):
            self.data = {}

        def save(self, brand_id, artifact):
            self.data[brand_id] = artifact

        def load(self, brand_id):
            return self.data.get(brand_id)

        def exists(self, brand_id):
            return brand_id in self.data

    store = MemoryStore()
    on_brand = "Friendly service solves your problem. Friendly humans reply promptly."
    off_brand = "Rigid processes delay your problem. Formal systems reply eventually."

    guard = SimasiaGuard("demo", embedding_model=FakeEmbedder(), store=store)
    guard.calibrate_weights(on_brand, off_brand)

    assert "demo" in store.data  # written to the store, not the filesystem
    assert not (tmp_path / "simasia_demo_head.joblib").exists()

    reloaded = SimasiaGuard("demo", embedding_model=FakeEmbedder(), store=store)
    assert 0.0 <= reloaded.evaluate_response("A friendly reply.") <= 1.0


def test_db_style_store_via_serialize_helpers(tmp_path):
    import sqlite3

    from simasia import deserialize_artifact, serialize_artifact

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
            return (
                self.conn.execute("SELECT 1 FROM simasia WHERE brand = ?", (brand_id,)).fetchone()
                is not None
            )

    store = SQLiteStore(sqlite3.connect(":memory:"))
    on_brand = "Friendly service solves your problem. Friendly humans reply promptly."
    off_brand = "Rigid processes delay your problem. Formal systems reply eventually."

    guard = SimasiaGuard("demo", embedding_model=FakeEmbedder(), store=store)
    guard.train(on_brand, off_brand)

    reloaded = SimasiaGuard("demo", embedding_model=FakeEmbedder(), store=store)
    result = reloaded.explain("A friendly response is ready.")
    assert 0.0 <= result["score"] <= 1.0  # exemplars survived the DB round-trip
