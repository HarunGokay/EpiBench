"""Regression tests for Phase 1 packaging and independence defects (P1-P5)."""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib
import pytest

import epibench
from epibench.embedding import EmbeddingService
from epibench.runners.metering import SpendTracker
from epibench.runners.run_efficiency_benchmark import generate_synthetic_history
from epibench.utils import get_git_commit_sha


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_P1_single_package_name_no_alias_hack():
    """P1: Package is 'epibench' with no sys.modules['EpiBench'] alias hack."""
    assert epibench.__name__ == "epibench"
    assert "EpiBench" not in sys.modules
    init_text = (REPO_ROOT / "src" / "epibench" / "__init__.py").read_text(encoding="utf-8")
    assert "sys.modules" not in init_text


def test_P2_standalone_embedding_service_no_private_imports():
    """P2: VectorRAGAdapter and HybridRAGAdapter use epibench.embedding, never private modules."""
    for rel in (
        "src/epibench/adapters/vector_rag_adapter.py",
        "src/epibench/adapters/hybrid_rag_adapter.py",
    ):
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "from epibench.embedding import EmbeddingService" in source
        forbidden_prefix = "from " + "app."
        assert forbidden_prefix not in source

    sim_same = EmbeddingService.cosine_similarity([1.0, 0.0], [1.0, 0.0])
    sim_orth = EmbeddingService.cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert sim_same == pytest.approx(1.0)
    assert sim_orth == pytest.approx(0.0)


def test_P3_declared_dependencies_and_packaging():
    """P3: pyproject.toml declares httpx, drops unused rank-bm25, and uses src layout."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)

    deps = cfg["project"]["dependencies"]
    assert any(d.startswith("httpx") for d in deps)
    assert not any(d.startswith("rank-bm25") for d in deps)
    assert cfg["tool"]["setuptools"]["package-dir"] == {"": "src"}

    import httpx  # noqa: F401


def test_P4_efficiency_runner_session_date_key():
    """P4: Efficiency runner generates sessions with 'session_date', not 'date_str'."""
    sessions = generate_synthetic_history(10, seed=42)
    assert len(sessions) == 5
    for s in sessions:
        assert "session_date" in s
        assert s["session_date"]
        assert "date_str" not in s


def test_P5_metering_requires_cap_and_runtime_git_sha():
    """P5: SpendTracker requires explicit hard_cap_usd; git SHA is computed at runtime."""
    with pytest.raises(TypeError):
        SpendTracker()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        SpendTracker(hard_cap_usd=0.0)

    tracker = SpendTracker(hard_cap_usd=2.5)
    assert tracker.hard_cap_usd == 2.5

    sha = get_git_commit_sha(REPO_ROOT)
    assert isinstance(sha, str) and len(sha) > 0
    assert sha not in ("3fcd4ef", "c92afdac8fbd3b3c657722ca67ccdb394d053ea8")
