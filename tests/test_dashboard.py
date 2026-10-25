from pathlib import Path


def test_dashboard_renders_without_exceptions(world, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from retention_agent.run_daily import run_daily

    s = world["s"]
    run_daily(s, as_of="2026-10-05", label="dash", limit=120, progress=False)
    monkeypatch.setenv("DATA_DIR", s.data_dir)
    monkeypatch.setenv("OUTPUT_DIR", s.output_dir)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 7 and len(at.metric) >= 6
