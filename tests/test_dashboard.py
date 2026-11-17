"""Every dashboard page must render without exceptions using the generated artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"
PAGES = sorted((DASH / "pages").glob("*.py"))
NEEDED = [ROOT / "artifacts" / "metrics" / "model_summary.json",
          ROOT / "data" / "processed" / "expression_matrix.parquet"]


@pytest.mark.data
@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_page_renders(page):
    if not all(p.exists() for p in NEEDED):
        pytest.skip("pipeline artifacts not generated")
    from streamlit.testing.v1 import AppTest
    if str(DASH) not in sys.path:
        sys.path.insert(0, str(DASH))
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert len(at.title) == 1


@pytest.mark.data
def test_missing_artifact_is_handled_gracefully(tmp_path, monkeypatch):
    """With artifacts absent, pages show a warning instead of crashing."""
    from streamlit.testing.v1 import AppTest
    if str(DASH) not in sys.path:
        sys.path.insert(0, str(DASH))
    from components import data
    monkeypatch.setattr(data, "METRICS", tmp_path)
    data.metric_json.clear()
    data.metric_csv.clear()
    at = AppTest.from_file(str(DASH / "pages" / "6_pathways.py"), default_timeout=60).run()
    assert not at.exception
    assert any("not found" in w.value for w in at.warning)
