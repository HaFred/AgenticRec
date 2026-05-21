"""Tests for LLM-driven RankAgent."""
import pytest
from agentic_rec.core import AgentMessage, Decision, Item, Memory, ToolRegistry
from agentic_rec.llm import MockLLM
from agentic_rec.agents import RankAgent
from agentic_rec.tools import FeatureTool, TwoTowerTool, GBDTTool


CORPUS = [
    {"id": f"mv_{i:04d}", "title": f"item_{i}", "tags": ["悬疑", "犯罪"],
     "author": f"A{i%5}", "ctr_prior": 0.10 + 0.01 * i, "freshness": 0.5, "hot": 500 + i * 10}
    for i in range(50)
]
# Add a few different-tag items
CORPUS += [
    {"id": "mv_heal_1", "title": "治愈花园", "tags": ["治愈", "轻松"],
     "author": "A9", "ctr_prior": 0.08, "freshness": 0.7, "hot": 200},
    {"id": "mv_heal_2", "title": "午后阳光", "tags": ["治愈", "轻松"],
     "author": "A8", "ctr_prior": 0.07, "freshness": 0.6, "hot": 150},
]


def make_items(ids, score=0.5):
    return [Item(id=i, score=score, features=c) for i, c in zip(ids, CORPUS)
            if c["id"] in ids]


def make_agent(llm=None):
    tools = ToolRegistry()
    tools.register(FeatureTool(CORPUS))
    tools.register(TwoTowerTool(CORPUS))
    tools.register(GBDTTool(CORPUS))
    memory = Memory()
    memory.update_profile("u1", tags={"悬疑": 0.9})
    return RankAgent(llm=llm or MockLLM(), tools=tools, memory=memory)


class TestRankAgentSkip:
    def test_skip_when_below_threshold(self):
        """Small candidate set should skip ranking entirely."""
        agent = make_agent()
        items = make_items(["mv_0000", "mv_0001", "mv_0002"])
        msg = AgentMessage("orch", "RankAgent", "request", content={"items": items})
        ctx = {"query": "test", "user_id": "u1", "scene": "feed_home"}
        d = agent.step(msg, ctx)
        assert d.action == "skip"
        assert len(d.payload) == len(items)

    def test_no_skip_when_above_threshold(self):
        """Large candidate set should not skip."""
        agent = make_agent()
        agent.SKIP_THRESHOLD = 5  # lower for test
        items = make_items([f"mv_{i:04d}" for i in range(10)])
        msg = AgentMessage("orch", "RankAgent", "request", content={"items": items})
        ctx = {"query": "test", "user_id": "u1", "scene": "feed_home"}
        d = agent.step(msg, ctx)
        assert d.action == "rank"


class TestRankAgentFusion:
    def test_fusion_sorts_by_score(self):
        """Fused scores should produce descending order."""
        agent = make_agent()
        agent.SKIP_THRESHOLD = 5
        items = make_items([f"mv_{i:04d}" for i in range(30)])
        msg = AgentMessage("orch", "RankAgent", "request", content={"items": items})
        ctx = {"query": "test", "user_id": "u1", "scene": "feed_home"}
        d = agent.step(msg, ctx)
        scores = [it.score for it in d.payload]
        assert scores == sorted(scores, reverse=True), "Scores must be descending"

    def test_truncates_to_threshold(self):
        """Output should be truncated to SKIP_THRESHOLD."""
        agent = make_agent()
        agent.SKIP_THRESHOLD = 5
        items = make_items([f"mv_{i:04d}" for i in range(30)])
        msg = AgentMessage("orch", "RankAgent", "request", content={"items": items})
        ctx = {"query": "test", "user_id": "u1", "scene": "feed_home"}
        d = agent.step(msg, ctx)
        assert len(d.payload) <= agent.SKIP_THRESHOLD


class TestRankAgentFallback:
    def test_fallback_on_llm_error(self):
        """If LLM returns unusable response, fall back to defaults."""
        agent = make_agent()

        class BrokenLLM:
            name = "broken"
            def chat(self, messages, **kwargs):
                return "nonsense blob {{{[/ not json"

        agent.llm = BrokenLLM()
        agent.SKIP_THRESHOLD = 5
        items = make_items([f"mv_{i:04d}" for i in range(30)])
        msg = AgentMessage("orch", "RankAgent", "request", content={"items": items})
        ctx = {"query": "test", "user_id": "u1", "scene": "feed_home"}
        d = agent.step(msg, ctx)
        assert d.action == "rank"
        assert len(d.payload) <= agent.SKIP_THRESHOLD


class TestRankAgentLLMConfig:
    def test_llm_with_valid_json_controls_fusion(self):
        """When LLM returns valid JSON config, those weights should be used."""
        agent = make_agent()
        agent.SKIP_THRESHOLD = 5

        class ConfigLLM:
            name = "config"
            def chat(self, messages, **kwargs):
                return '{"w_tower": 0.0, "w_gbdt": 1.0, "w_orig": 0.0, "reflect": false}'

        agent.llm = ConfigLLM()
        items = make_items([f"mv_{i:04d}" for i in range(10)])
        msg = AgentMessage("orch", "RankAgent", "request", content={"items": items})
        ctx = {"query": "test", "user_id": "u1", "scene": "feed_home"}
        d = agent.step(msg, ctx)
        assert d.action == "rank"
        # Should use GBDT-only weights, mentioned in thought
        assert "w_gbdt=1.00" in d.thought
