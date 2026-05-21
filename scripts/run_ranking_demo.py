"""Standalone RankAgent demo: LLM-driven fusion in action.

Run:
    python scripts/run_ranking_demo.py
"""
import os

from agentic_rec import DeepSeekLLM, Memory, ToolRegistry
from agentic_rec.agents import RankAgent
from agentic_rec.core import AgentMessage
from agentic_rec.tools import TwoTowerTool, GBDTTool, FeatureTool


CORPUS = [
    {"id": "mv_8821", "title": "雾港谜局", "tags": ["悬疑", "都市"], "author": "A1",
     "ctr_prior": 0.18, "freshness": 0.9, "hot": 980},
    {"id": "mv_7102", "title": "山雨欲来", "tags": ["悬疑", "犯罪"], "author": "A2",
     "ctr_prior": 0.15, "freshness": 0.4, "hot": 420},
    {"id": "mv_6010", "title": "轻松小镇日记", "tags": ["治愈", "轻松"], "author": "A3",
     "ctr_prior": 0.10, "freshness": 0.7, "hot": 220},
    {"id": "mv_5511", "title": "暗夜推理者", "tags": ["悬疑", "推理"], "author": "A1",
     "ctr_prior": 0.12, "freshness": 0.2, "hot": 1200},
    {"id": "mv_4321", "title": "夜行列车", "tags": ["悬疑", "惊悚"], "author": "A4",
     "ctr_prior": 0.09, "freshness": 0.6, "hot": 90},
    {"id": "mv_3010", "title": "午后茶馆", "tags": ["治愈"], "author": "A5",
     "ctr_prior": 0.06, "freshness": 0.5, "hot": 60},
    {"id": "mv_2008", "title": "搞笑同事录", "tags": ["喜剧", "轻松"], "author": "A6",
     "ctr_prior": 0.20, "freshness": 0.8, "hot": 310},
    {"id": "mv_1207", "title": "迷雾追凶", "tags": ["悬疑", "犯罪"], "author": "A7",
     "ctr_prior": 0.13, "freshness": 0.3, "hot": 770},
    {"id": "mv_0904", "title": "巷口便利店", "tags": ["治愈", "轻松"], "author": "A8",
     "ctr_prior": 0.08, "freshness": 0.9, "hot": 130},
    {"id": "mv_0510", "title": "时间裂缝", "tags": ["科幻"], "author": "A9",
     "ctr_prior": 0.11, "freshness": 0.7, "hot": 540},
]


def main() -> None:
    # Setup
    tools = ToolRegistry()
    tools.register(TwoTowerTool(CORPUS))
    tools.register(GBDTTool(CORPUS))
    tools.register(FeatureTool(CORPUS))

    memory = Memory()
    memory.update_profile("u_42", tags={"悬疑": 0.9, "轻松": 0.4})

    llm = DeepSeekLLM() if os.environ.get("DEEPSEEK_API_KEY") else None
    if llm:
        print("Using DeepSeek LLM backbone")
    else:
        print("No DEEPSEEK_API_KEY set, using fallback weights (no LLM)")

    agent = RankAgent(llm=llm, tools=tools, memory=memory)
    agent.SKIP_THRESHOLD = 6  # lower for demo

    from agentic_rec.core import Item

    # Build items from corpus with initial vector scores
    items = [Item(id=c["id"], score=0.5, features=dict(c), source="recall")
             for c in CORPUS]

    msg = AgentMessage("demo", "RankAgent", "request", content={"items": items})
    ctx = {"query": "想看看悬疑剧", "user_id": "u_42", "scene": "feed_home"}
    d = agent.run(msg, ctx)

    print(f"\n=== RankAgent Decision ({d.elapsed_ms:.1f}ms) ===")
    print(f"Action: {d.action}")
    print(f"Thought: {d.thought}")
    print(f"\n=== Ranked Results (top {len(d.payload)}) ===")
    for i, it in enumerate(d.payload, 1):
        tags = it.features.get("tags", [])
        print(f"  {i}. {it.id} ({it.features.get('title','?')}) "
              f"score={it.score:.4f}  tags={tags}")


if __name__ == "__main__":
    main()
