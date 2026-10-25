import os

import pytest

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_BACKEND", "hashing")

from retention_agent.config import Settings  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("ra")
    return Settings(llm_provider="mock", embedding_backend="hashing", data_dir=str(root / "data"), output_dir=str(root / "out"))


@pytest.fixture(scope="session")
def world(settings):
    """Generated data + indexed Chroma + pipeline, built once for the whole test session."""
    from retention_agent.data_gen import generate
    from retention_agent.graph import Pipeline
    from retention_agent.run_daily import prepare

    generate(settings)
    ds, store, llm = prepare(settings)
    return dict(s=settings, ds=ds, store=store, llm=llm, pipe=Pipeline(settings, ds, store, llm))
