"""Tests for GBDTTool."""
import pytest
from agentic_rec.tools import GBDTTool


CORPUS = [
    {"id": "a", "title": "悬疑剧", "tags": ["悬疑", "犯罪"], "author": "A1",
     "ctr_prior": 0.20, "freshness": 0.9, "hot": 1200},
    {"id": "b", "title": "治愈片", "tags": ["治愈", "轻松"], "author": "A2",
     "ctr_prior": 0.05, "freshness": 0.2, "hot": 60},
    {"id": "c", "title": "科幻片", "tags": ["科幻"], "author": "A3",
     "ctr_prior": 0.10, "freshness": 0.5, "hot": 500},
]


def test_gbdt_deterministic():
    """Same inputs produce same scores."""
    tool = GBDTTool(CORPUS)
    r1 = tool(item_ids=["a", "b", "c"])
    r2 = tool(item_ids=["a", "b", "c"])
    for k in r1:
        assert r1[k] == pytest.approx(r2[k])


def test_gbdt_score_range():
    """Scores must be in [0, 1]."""
    tool = GBDTTool(CORPUS)
    result = tool(item_ids=["a", "b", "c"])
    for v in result.values():
        assert 0.0 <= v <= 1.0


def test_gbdt_better_features_score_higher():
    """Item with better ctr_prior/freshness/hot should score higher."""
    tool = GBDTTool(CORPUS)
    result = tool(item_ids=["a", "b"])
    assert result["a"] > result["b"], (
        f"High-CTR item a ({result['a']:.3f}) should outscore "
        f"low-CTR item b ({result['b']:.3f})"
    )


def test_gbdt_unknown_item():
    """Unknown item ID returns 0.0."""
    tool = GBDTTool(CORPUS)
    result = tool(item_ids=["nonexistent"])
    assert result["nonexistent"] == 0.0
