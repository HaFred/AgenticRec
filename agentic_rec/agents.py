"""The five council members + the orchestrator.

Each agent is intentionally short — its job is to *route* tools, reflect on
the result, and produce a Decision. Replace MockLLM with a real backbone to
get real reasoning.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .core import AgentMessage, BaseAgent, Decision, Item


# ---------------------------------------------------------------------------
class RecallAgent(BaseAgent):
    name = "RecallAgent"

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        query = ctx["query"]
        user_id = ctx["user_id"]
        scene = ctx.get("scene", "feed_home")

        prof = self.memory.profile_of(user_id)
        user_tags = prof.get("tags", {})
        is_cold = not user_tags

        # ----- LLM-driven routing thought (mockable) -----
        thought = self.llm.chat([{"role": "user",
                                  "content": f"recall route for query='{query}' scene={scene}"}]) \
            if self.llm else "default"

        items: List[Item] = []
        plan: List[str] = []

        # Hot fallback first if cold start
        if is_cold and "hot" in self.tools.names():
            items += self.tools.get("hot")(top_k=20)
            plan.append("hot:20")

        if "vector" in self.tools.names():
            v = self.tools.get("vector")(query=query, top_k=40)
            items += v
            plan.append(f"vector:{len(v)}")

        if "tag" in self.tools.names():
            t = self.tools.get("tag")(query=query, user_tags=user_tags, top_k=40)
            items += t
            plan.append(f"tag:{len(t)}")

        # Reflection: if too narrow, widen with hot
        unique = {i.id: i for i in items}
        if len(unique) < 30 and "hot" in self.tools.names() and not is_cold:
            for it in self.tools.get("hot")(top_k=30):
                unique.setdefault(it.id, it)
            plan.append("hot+30(reflection)")

        merged = list(unique.values())
        return Decision(agent=self.name, thought=f"{thought} | plan={plan}",
                        action="recall", payload=merged)


# ---------------------------------------------------------------------------
class RankAgent(BaseAgent):
    """Coarse ranking with LLM-controlled fusion of two-tower + GBDT scores.

    The LLM receives candidate set statistics (count, score distribution,
    tag diversity) and returns a JSON fusion config. Tools run
    deterministically; the LLM controls *how* they're combined and
    whether to trigger reflection for pathological distributions.
    """

    name = "RankAgent"
    SKIP_THRESHOLD = 200

    # Default fusion weights used when LLM is unavailable or fails
    DEFAULT_WEIGHTS = {"w_tower": 0.4, "w_gbdt": 0.4, "w_orig": 0.2}

    def _candidate_stats(self, items: List[Item]) -> Dict[str, Any]:
        """Summarise candidate set for the LLM prompt."""

        scores = [it.score for it in items]
        tag_counter: Counter = Counter()
        for it in items:
            for t in it.features.get("tags", []):
                tag_counter[t] += 1

        top_tag, top_cnt = tag_counter.most_common(1)[0] if tag_counter else ("none", 0)

        return {
            "count": len(items),
            "score_min": round(min(scores), 4),
            "score_max": round(max(scores), 4),
            "score_avg": round(sum(scores) / len(scores), 4),
            "top_tag": top_tag,
            "tag_concentration": round(top_cnt / len(items), 3) if items else 0,
            "unique_tags": len(tag_counter),
        }

    def _build_prompt(self, stats: Dict[str, Any]) -> str:
        """Build the LLM prompt requesting a fusion config."""
        return (
            "You are a coarse ranking controller in a recommendation system.\n"
            f"Candidate set: {stats['count']} items\n"
            f"Score range: [{stats['score_min']}, {stats['score_max']}], "
            f"avg={stats['score_avg']}\n"
            f"Top tag: '{stats['top_tag']}' (concentration={stats['tag_concentration']})\n"
            f"Unique tags: {stats['unique_tags']}\n\n"
            "Decide the fusion weights and whether to reflect. Reply with ONLY valid JSON:\n"
            '{"skip": false, "w_tower": 0.4, "w_gbdt": 0.4, "w_orig": 0.2, "reflect": false}\n\n'
            "Rules:\n"
            "- skip=true if the candidate set is very small or already well-scored\n"
            "- w_tower + w_gbdt + w_orig should sum to 1.0\n"
            "- reflect=true if tag concentration > 0.7 (distribution abnormal, all same tag)\n"
            "- w_tower should be high when user-tag match matters; w_gbdt high for feature-driven ranking"
        )

    def _parse_llm_config(self, thought: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, falling back per-field."""
        import json as _json

        config = dict(self.DEFAULT_WEIGHTS, skip=False, reflect=False)

        try:
            # Extract JSON object from response (may have surrounding text)
            start = thought.find("{")
            end = thought.rfind("}")
            if start >= 0 and end > start:
                parsed = _json.loads(thought[start:end + 1])
                config.update(parsed)
        except (ValueError, _json.JSONDecodeError):
            pass  # fall back to defaults

        # Normalise weights to sum to 1.0
        w_sum = config.get("w_tower", 0) + config.get("w_gbdt", 0) + config.get("w_orig", 0)
        if w_sum > 0:
            config["w_tower"] = config.get("w_tower", 0.4) / w_sum
            config["w_gbdt"] = config.get("w_gbdt", 0.4) / w_sum
            config["w_orig"] = config.get("w_orig", 0.2) / w_sum

        return config

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        items: List[Item] = msg.content.get("items", [])

        # 1) Stats for LLM
        stats = self._candidate_stats(items)

        # 2) LLM decides fusion config
        if self.llm:
            prompt = self._build_prompt(stats)
            thought = self.llm.chat([{"role": "user", "content": prompt}])
        else:
            thought = "no LLM, using defaults"
        config = self._parse_llm_config(thought)

        # 3) Skip gate
        if config.get("skip") or len(items) <= self.SKIP_THRESHOLD:
            reason = (
                f"skip coarse ranking, candidates={len(items)} <= {self.SKIP_THRESHOLD}"
                if len(items) <= self.SKIP_THRESHOLD
                else "skip coarse ranking, LLM decided skip"
            )
            return Decision(
                agent=self.name,
                thought=reason,
                action="skip", payload=items,
            )

        # 4) Run TwoTower tool
        tower_scores: Dict[str, float] = {}
        tt = self.tools.get("two_tower") if "two_tower" in self.tools.names() else None
        if tt:
            prof = self.memory.profile_of(ctx["user_id"])
            ids = [it.id for it in items]
            tower_scores = tt(item_ids=ids, user_profile=prof)

        # 5) Run GBDT tool
        gbdt_scores: Dict[str, float] = {}
        gbt = self.tools.get("gbdt") if "gbdt" in self.tools.names() else None
        if gbt:
            ids = [it.id for it in items]
            gbdt_scores = gbt(item_ids=ids)

        # 6) Fuse scores
        w_tower = config.get("w_tower", self.DEFAULT_WEIGHTS["w_tower"])
        w_gbdt = config.get("w_gbdt", self.DEFAULT_WEIGHTS["w_gbdt"])
        w_orig = config.get("w_orig", self.DEFAULT_WEIGHTS["w_orig"])

        for it in items:
            s_tower = tower_scores.get(it.id, 0.0)
            s_gbdt = gbdt_scores.get(it.id, 0.0)
            it.score = w_tower * s_tower + w_gbdt * s_gbdt + w_orig * it.score

        # 7) Sort & truncate
        items.sort(key=lambda x: -x.score)

        # 8) Reflection: if tag concentration is high, boost minority-tag items
        if config.get("reflect") and stats["tag_concentration"] > 0.7:
            tag_counter: Counter = Counter()
            for it in items:
                for t in it.features.get("tags", []):
                    tag_counter[t] += 1
            dominant_tag = tag_counter.most_common(1)[0][0] if tag_counter else None
            if dominant_tag:
                boost = [it for it in items
                         if dominant_tag not in it.features.get("tags", [])]
                for it in boost:
                    it.score *= 1.15  # moderate diversity boost
                items.sort(key=lambda x: -x.score)

        kept = items[: self.SKIP_THRESHOLD]
        return Decision(
            agent=self.name,
            thought=f"LLM fusion: w_tower={w_tower:.2f} w_gbdt={w_gbdt:.2f} "
                    f"w_orig={w_orig:.2f} reflect={config.get('reflect')} | "
                    f"kept={len(kept)}",
            action="rank", payload=kept,
        )


# ---------------------------------------------------------------------------
class RerankAgent(BaseAgent):
    name = "RerankAgent"

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        items: List[Item] = msg.content.get("items", [])
        scene = ctx.get("scene", "feed_home")
        rule = self.tools.get("biz_rule") if "biz_rule" in self.tools.names() else None
        if rule:
            items = rule(items=items, scene=scene)
        thought = f"rerank by scene={scene}, dedup+freshness+ad"
        return Decision(agent=self.name, thought=thought, action="rerank", payload=items)


# ---------------------------------------------------------------------------
class ExplainAgent(BaseAgent):
    name = "ExplainAgent"

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        items: List[Item] = msg.content.get("items", [])
        prof = self.memory.profile_of(ctx["user_id"])
        user_tags = set(prof.get("tags", {}).keys())
        for it in items:
            if it.features.get("is_ad"):
                it.explain = "商业化插入位"
                continue
            tags = set(it.features.get("tags", []))
            hit = list(user_tags & tags)
            src = it.source
            if hit:
                it.explain = f"命中你常看的 {','.join(hit[:2])}（来源:{src}）"
            else:
                it.explain = f"来源:{src}, 评分{it.score:.2f}"
        return Decision(agent=self.name, thought="annotate explanations",
                        action="explain", payload=items)


# ---------------------------------------------------------------------------
class CriticAgent(BaseAgent):
    """Distribution / bias guard. Vetoes pathological outputs."""

    name = "CriticAgent"

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        items: List[Item] = msg.content.get("items", [])
        if not items:
            return Decision(agent=self.name, thought="empty result, veto",
                            action="veto", payload=True)
        # Check tag concentration
        cats = Counter()
        for it in items:
            for t in it.features.get("tags", []):
                cats[t] += 1
        if cats:
            top, top_cnt = cats.most_common(1)[0]
            if top_cnt > 0.7 * len(items):
                return Decision(
                    agent=self.name,
                    thought=f"tag '{top}' over-concentrated ({top_cnt}/{len(items)}), veto",
                    action="veto",
                    payload=True,
                )
        # Ad ratio sanity
        ads = sum(1 for it in items if it.features.get("is_ad"))
        if ads > max(1, len(items) // 5):
            return Decision(agent=self.name, thought="ad ratio too high",
                            action="veto", payload=True)
        return Decision(agent=self.name, thought="distribution ok", action="pass",
                        payload=False)


# ---------------------------------------------------------------------------
class OrchestratorAgent(BaseAgent):
    """Council chair: routes the message flow across agents."""

    name = "OrchestratorAgent"

    def __init__(self, llm=None, tools=None, memory=None,
                 recall=None, rank=None, rerank=None,
                 explain=None, critic=None, max_retry: int = 1) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory)
        self.recall = recall
        self.rank = rank
        self.rerank = rerank
        self.explain = explain
        self.critic = critic
        self.max_retry = max_retry

    def step(self, msg: AgentMessage, ctx: Dict[str, Any]) -> Decision:
        trace = ctx["trace"]

        for attempt in range(self.max_retry + 1):
            # 1) recall
            d_recall = self.recall.run(
                AgentMessage(self.name, self.recall.name, "request"), ctx)
            trace.add(d_recall)
            items = d_recall.payload

            # 2) rank
            d_rank = self.rank.run(
                AgentMessage(self.name, self.rank.name, "request",
                             content={"items": items}), ctx)
            trace.add(d_rank)
            items = d_rank.payload

            # 3) rerank
            d_re = self.rerank.run(
                AgentMessage(self.name, self.rerank.name, "request",
                             content={"items": items}), ctx)
            trace.add(d_re)
            items = d_re.payload

            # 4) critic
            d_crit = self.critic.run(
                AgentMessage(self.name, self.critic.name, "critique",
                             content={"items": items}), ctx)
            trace.add(d_crit)
            if d_crit.payload is True and attempt < self.max_retry:
                # vetoed → reflect: bias profile to broaden recall, retry
                prof = self.memory.profile_of(ctx["user_id"])
                prof["tags"] = {}  # widen
                continue
            break

        # 5) explain
        d_exp = self.explain.run(
            AgentMessage(self.name, self.explain.name, "request",
                         content={"items": items}), ctx)
        trace.add(d_exp)
        items = d_exp.payload

        return Decision(agent=self.name, thought="council finished",
                        action="orchestrate", payload=items)
