"""AgenticRec — From Pipeline to Council.

A lightweight agentic framework for search & recommendation.
"""
from .core import (
    AgentMessage,
    BaseAgent,
    Memory,
    Tool,
    ToolRegistry,
    Item,
    Decision,
    Trace,
)
from .llm import BaseLLM, MockLLM, DeepSeekLLM
from .agents import (
    RecallAgent,
    RankAgent,
    RerankAgent,
    ExplainAgent,
    CriticAgent,
    OrchestratorAgent,
)
from .pipeline import AgenticPipeline, PipelineResult
from .tools import (
    BizRuleTool,
    FeatureTool,
    GBDTTool,
    HotTool,
    TagTool,
    TwoTowerTool,
    VectorTool,
)

__all__ = [
    "AgentMessage",
    "BaseAgent",
    "Memory",
    "Tool",
    "ToolRegistry",
    "Item",
    "Decision",
    "Trace",
    "BaseLLM",
    "MockLLM",
    "DeepSeekLLM",
    "RecallAgent",
    "RankAgent",
    "RerankAgent",
    "ExplainAgent",
    "CriticAgent",
    "OrchestratorAgent",
    "AgenticPipeline",
    "PipelineResult",
    "VectorTool",
    "TagTool",
    "FeatureTool",
    "BizRuleTool",
    "GBDTTool",
    "HotTool",
    "TwoTowerTool",
]

__version__ = "0.1.0"
