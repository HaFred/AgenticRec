"""High-level facade: AgenticPipeline.run(query, user_id, scene)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agents import (
    CriticAgent,
    ExplainAgent,
    OrchestratorAgent,
    RankAgent,
    RecallAgent,
    RerankAgent,
)
from .core import AgentMessage, Item, Memory, ToolRegistry, Trace
from .llm import BaseLLM, MockLLM
from .tools import BizRuleTool, FeatureTool, GBDTTool, HotTool, TagTool, TwoTowerTool, VectorTool


@dataclass
class PipelineResult:
    items: List[Item] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    total_ms: float = 0.0


class AgenticPipeline:
    """Wires up the council with sensible defaults so quickstart works in 3 lines."""

    def __init__(
        self,
        corpus: Optional[List[Dict[str, Any]]] = None,
        llm: Optional[BaseLLM] = None,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[Memory] = None,
        top_n: int = 10,
    ) -> None:
        self.llm = llm or MockLLM()
        self.memory = memory or Memory()
        self.tools = tools or ToolRegistry()
        self.top_n = top_n

        if corpus and not self.tools.names():
            self.tools.register(VectorTool(corpus))
            self.tools.register(TagTool(corpus))
            self.tools.register(HotTool(corpus))
            self.tools.register(TwoTowerTool(corpus))
            self.tools.register(GBDTTool(corpus))
            self.tools.register(FeatureTool(corpus))
            self.tools.register(BizRuleTool())

        self.recall = RecallAgent(llm=self.llm, tools=self.tools, memory=self.memory)
        self.rank = RankAgent(llm=self.llm, tools=self.tools, memory=self.memory)
        self.rerank = RerankAgent(llm=self.llm, tools=self.tools, memory=self.memory)
        self.explain = ExplainAgent(llm=self.llm, tools=self.tools, memory=self.memory)
        self.critic = CriticAgent(llm=self.llm, tools=self.tools, memory=self.memory)
        self.orch = OrchestratorAgent(
            llm=self.llm, tools=self.tools, memory=self.memory,
            recall=self.recall, rank=self.rank, rerank=self.rerank,
            explain=self.explain, critic=self.critic,
        )

    def run(self, query: str, user_id: str = "anon",
            scene: str = "feed_home") -> PipelineResult:
        trace = Trace()
        ctx = {"query": query, "user_id": user_id, "scene": scene, "trace": trace}
        d = self.orch.run(AgentMessage("user", self.orch.name, "request"), ctx)
        items = d.payload[: self.top_n]
        return PipelineResult(items=items, trace=trace.dump(),
                              total_ms=trace.total_ms())
