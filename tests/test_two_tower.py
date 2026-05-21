"""Tests for TwoTowerTool."""
import pytest
from agentic_rec.tools import TwoTowerTool


CORPUS = [
    {"id": "a", "title": "悬疑剧", "tags": ["悬疑", "犯罪"], "author": "A1", "ctr_prior": 0.15},
    {"id": "b", "title": "治愈片", "tags": ["治愈", "轻松"], "author": "A2", "ctr_prior": 0.10},
    {"id": "c", "title": "科幻片", "tags": ["科幻"], "author": "A3", "ctr_prior": 0.12},
]


def test_two_tower_deterministic():
    """Same inputs produce same scores every time."""
    tool = TwoTowerTool(CORPUS)
    user_profile = {"tags": {"悬疑": 0.9}}
    r1 = tool(item_ids=["a", "b", "c"], user_profile=user_profile)
    r2 = tool(item_ids=["a", "b", "c"], user_profile=user_profile)
    for k in r1:
        assert r1[k] == pytest.approx(r2[k])


def test_two_tower_score_range():
    """Scores should be in [0, 1] range."""
    tool = TwoTowerTool(CORPUS)
    result = tool(item_ids=["a", "b", "c"], user_profile={"tags": {"悬疑": 0.8}})
    for k, v in result.items():
        assert 0.0 <= v <= 1.0, f"{k} score {v} out of range"


def test_two_tower_relevant_higher():
    """Items matching user tags should score higher than unrelated items."""
    tool = TwoTowerTool(CORPUS)
    user = {"tags": {"悬疑": 0.9}}
    result = tool(item_ids=["a", "b"], user_profile=user)
    assert result["a"] > result["b"], f"悬疑-item should outscore 治愈-item for 悬疑 user"


def test_two_tower_empty_profile():
    """Empty user profile should still produce valid scores."""
    tool = TwoTowerTool(CORPUS)
    result = tool(item_ids=["a", "b"], user_profile={"tags": {}})
    assert len(result) == 2
    for v in result.values():
        assert 0.0 <= v <= 1.0
