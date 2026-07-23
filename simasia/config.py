"""Config-file driven setup: build a guard and run training from ``simasia.toml``.

Secrets stay in the environment (``EMBEDDING_KEY`` / ``GENERATION_KEY``, which a
``.env`` file can supply); the TOML file holds only non-secret settings, so it is
safe to commit.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # stdlib on 3.11+, backport on 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - version dependent
    import tomli as tomllib

from .embeddings import LocalEmbedder, OpenAIEmbedder
from .generation import OpenAIGenerator
from .guard import SimasiaGuard


def load_dotenv(path: str | os.PathLike[str]) -> None:
    """Load simple ``KEY=VALUE`` lines from a .env file into the environment.

    Existing environment variables win, so a real shell export is never clobbered.
    """
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | os.PathLike[str]) -> dict:
    """Parse a ``simasia.toml`` config file into a dict."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def build_guard(config: dict) -> SimasiaGuard:
    """Construct a :class:`SimasiaGuard` from a parsed config dict."""
    brand_id = config["brand"]["id"]

    embed_cfg = config.get("embedding", {})
    provider = embed_cfg.get("provider", "openai")
    if provider == "openai":
        embedder = OpenAIEmbedder(
            model=embed_cfg.get("model", OpenAIEmbedder.DEFAULT_MODEL),
            api_key=embed_cfg.get("api_key"),
            dimensions=embed_cfg.get("dimensions"),
        )
    elif provider == "local":
        embedder = LocalEmbedder(
            model_name=embed_cfg.get("model", LocalEmbedder.DEFAULT_MODEL_NAME)
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r} (use 'openai' or 'local').")

    # Only build a generator when configured; otherwise the guard makes a default
    # one lazily, and only if on-brand-only training actually needs it.
    gen_cfg = config.get("generation")
    generator = (
        OpenAIGenerator(
            model=gen_cfg.get("model", OpenAIGenerator.DEFAULT_MODEL),
            api_key=gen_cfg.get("api_key"),
        )
        if gen_cfg
        else None
    )

    storage_dir = config.get("storage", {}).get("dir", ".")
    return SimasiaGuard(
        brand_id,
        artifact_dir=storage_dir,
        embedding_model=embedder,
        generator=generator,
    )


def resolve_side(training: dict, side: str) -> str | list[str] | None:
    """Resolve one training side to raw text or a URL list.

    Accepts ``<side>_text`` (inline), ``<side>_file`` (path to a .txt), or
    ``<side>_urls`` (list). Returns ``None`` when the side is not configured.
    """
    if f"{side}_urls" in training:
        return list(training[f"{side}_urls"])
    if f"{side}_file" in training:
        return Path(training[f"{side}_file"]).read_text(encoding="utf-8")
    if f"{side}_text" in training:
        return training[f"{side}_text"]
    return None


def run_training(config: dict) -> float:
    """Build a guard from config and train it. Returns training accuracy."""
    guard = build_guard(config)
    training = config["training"]

    on_brand = resolve_side(training, "on_brand")
    if on_brand is None:
        raise ValueError(
            "config [training] needs on_brand_text, on_brand_file, or on_brand_urls."
        )
    off_brand = resolve_side(training, "off_brand")
    return guard.train(on_brand, off_brand)
