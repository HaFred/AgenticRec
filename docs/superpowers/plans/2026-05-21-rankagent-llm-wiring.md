# RankAgent LLM Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire RankAgent with DeepSeek LLM backbone to control ranking fusion, and add TwoTowerTool + GBDTTool deterministic rankers.

**Architecture:** LLM receives candidate stats (count, score range, tag diversity), returns JSON fusion config. TwoTowerTool and GBDTTool score all items deterministically. RankAgent fuses scores with LLM-chosen weights, truncates, and reflects on pathological distributions.

**Tech Stack:** Python 3.8+, zero external dependencies, `urllib` for LLM calls, `hashlib` for deterministic tool embeddings.

---

### Task 1: Add TwoTowerTool

**Files:**
- Modify: `agentic_rec/tools.py` (append after FeatureTool)
- Create: `tests/test_two_tower.py`

- [ ] **Step 1: Write the test file**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd d:/fredcode/hf_home/agenticrec_fork && python -m pytest tests/test_two_tower.py -v`
Expected: FAIL with ImportError (TwoTowerTool not defined)

- [ ] **Step 3: Add TwoTowerTool to tools.py**

Append after FeatureTool (line 113):

```python
# ---------------------------------------------------------------------------
class TwoTowerTool(Tool):
    """Deterministic two-tower model: user embedding dot item embedding.

    User tower: hashes user profile tags into a fixed 16-dim vector.
    Item tower: hashes item features (tags, author, category) into 16-dim.
    Score = cosine similarity of the two L2-normalized vectors.
    """

    name = "two_tower"
    description = "two-tower relevance scoring via deterministic embeddings"

    DIM = 16

    def __init__(self, corpus: List[Dict[str, Any]]) -> None:
        self._idx = {c["id"]: c for c in corpus}

    def _embed(self, tokens: List[str], seed: str = "") -> List[float]:
        """Hash a list of tokens into a DIM-dimensional vector."""
        vec = [0.0] * self.DIM
        if not tokens:
            return vec
        for i in range(self.DIM):
            val = 0.0
            for t in tokens:
                val += _h(f"{seed}|{t}|{i}", 1000) / 1000.0
            vec[i] = val / max(1, len(tokens))
        # L2-normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def __call__(
        self,
        item_ids: List[str],
        user_profile: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, float]:
        user_profile = user_profile or {}
        user_tags = list(user_profile.get("tags", {}).keys())

        user_vec = self._embed(user_tags, seed="user")

        scores: Dict[str, float] = {}
        for iid in item_ids:
            c = self._idx.get(iid)
            if c is None:
                scores[iid] = 0.0
                continue
            item_tokens = c.get("tags", []) + [c.get("author", ""), c.get("category", "")]
            item_vec = self._embed(item_tokens, seed="item")
            # Cosine similarity (vectors already L2-normalized)
            sim = sum(u * v for u, v in zip(user_vec, item_vec))
            scores[iid] = max(0.0, min(1.0, (sim + 1.0) / 2.0))
        return scores
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd d:/fredcode/hf_home/agenticrec_fork && python -m pytest tests/test_two_tower.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add agentic_rec/tools.py tests/test_two_tower.py
git commit -m "feat: add TwoTowerTool — deterministic user/item embedding scorer"
```

---

### Task 2: Add GBDTTool

**Files:**
- Modify: `agentic_rec/tools.py` (append after TwoTowerTool)
- Create: `tests/test_gbdt.py`

- [ ] **Step 1: Write the test file**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd d:/fredcode/hf_home/agenticrec_fork && python -m pytest tests/test_gbdt.py -v`
Expected: FAIL with ImportError (GBDTTool not defined)

- [ ] **Step 3: Add GBDTTool to tools.py**

Append after TwoTowerTool:

```python
# ---------------------------------------------------------------------------
class GBDTTool(Tool):
    """Deterministic lightweight GBDT ranker.

    Mimics a small gradient-boosted decision tree ensemble with fixed
    decision stumps over four features: ctr_prior, freshness, hot, tag_count.
    Score = sigmoid(sum of weighted stump decisions).
    """

    name = "gbdt"
    description = "lightweight GBDT ranking via deterministic feature stumps"

    # Fixed "stumps": (feature_index, threshold, weight)
    # feature order: ctr_prior, freshness, log_hot_norm, tag_count_norm
    STUMPS = [
        (0, 0.12, 0.8),   # ctr_prior > 12% → boost
        (0, 0.08, 0.4),   # ctr_prior > 8% → moderate boost
        (1, 0.5,  0.6),   # freshness > 0.5 → boost
        (1, 0.3,  0.3),   # freshness > 0.3 → moderate
        (2, 0.4,  0.5),   # log_hot > 0.4 → boost
        (3, 0.3,  0.4),   # tag_count_norm > 0.3 → boost
        (3, 0.1,  0.2),   # tag_count_norm > 0.1 → slight boost
    ]

    def __init__(self, corpus: List[Dict[str, Any]]) -> None:
        self._idx = {c["id"]: c for c in corpus}

    def _extract_features(self, item: Dict[str, Any], tag_count: int = 0) -> List[float]:
        """Extract normalized feature vector from item dict."""
        ctr = float(item.get("ctr_prior", 0))
        freshness = float(item.get("freshness", 0))
        hot = float(item.get("hot", 0))
        log_hot = math.log1p(hot) / 10.0  # normalize to ~[0, 1]
        tag_norm = min(1.0, tag_count / 5.0)
        return [ctr, freshness, log_hot, tag_norm]

    def __call__(self, item_ids: List[str], **_: Any) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for iid in item_ids:
            c = self._idx.get(iid)
            if c is None:
                scores[iid] = 0.0
                continue
            tag_count = len(c.get("tags", []))
            feats = self._extract_features(c, tag_count)

            # Sum weighted stump decisions
            total = 0.0
            for feat_idx, threshold, weight in self.STUMPS:
                if feats[feat_idx] > threshold:
                    total += weight

            # Sigmoid activation
            scores[iid] = 1.0 / (1.0 + math.exp(-total + 1.0))
        return scores
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd d:/fredcode/hf_home/agenticrec_fork && python -m pytest tests/test_gbdt.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add agentic_rec/tools.py tests/test_gbdt.py
git commit -m "feat: add GBDTTool — deterministic gradient-boosted decision tree ranker"
```

---

### Task 3: Rewrite RankAgent with LLM-driven fusion

**Files:**
- Modify: `agentic_rec/agents.py:63-85`
- Create: `tests/test_rank_agent.py`

- [ ] **Step 1: Write the test file**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd d:/fredcode/hf_home/agenticrec_fork && python -m pytest tests/test_rank_agent.py -v`
Expected: Some tests FAIL (old RankAgent doesn't use fusion)

- [ ] **Step 3: Rewrite RankAgent in agents.py**

Replace lines 63-85 with:

```python
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
        from collections import Counter

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
            if len(items) <= self.SKIP_THRESHOLD:
                return Decision(
                    agent=self.name,
                    thought=f"skip coarse ranking, candidates={len(items)} <= {self.SKIP_THRESHOLD}",
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
            from collections import Counter
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd d:/fredcode/hf_home/agenticrec_fork && python -m pytest tests/test_rank_agent.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add agentic_rec/agents.py tests/test_rank_agent.py
git commit -m "feat: wire RankAgent with LLM-driven fusion of TwoTower + GBDT scoring"
```

---

### Task 4: Register new tools in pipeline + add imports

**Files:**
- Modify: `agentic_rec/pipeline.py:17` (import line)
- Modify: `agentic_rec/pipeline.py:47-48` (tool registration)

- [ ] **Step 1: Update pipeline.py**

Change the import line (17):
```python
from .tools import BizRuleTool, FeatureTool, GBDTTool, HotTool, TagTool, TwoTowerTool, VectorTool
```

Add tool registration after HotTool (after line 47, before FeatureTool):
```python
            self.tools.register(TwoTowerTool(corpus))
            self.tools.register(GBDTTool(corpus))
```

- [ ] **Step 2: Run existing smoke test to verify nothing breaks**

Run: `cd d:/fredcode/hf_home/agenticrec_fork && python -m pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add agentic_rec/pipeline.py
git commit -m "feat: register TwoTowerTool and GBDTTool in pipeline defaults"
```

---

### Task 5: Create ranking demo launch script

**Files:**
- Create: `scripts/run_ranking_demo.py`

- [ ] **Step 1: Write the demo script**

```python
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
```

- [ ] **Step 2: Run the demo with MockLLM (no API key needed)**

Run: `cd d:/fredcode/hf_home/agenticrec_fork && python scripts/run_ranking_demo.py`
Expected: Prints ranked results with fallback weights, no errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_ranking_demo.py
git commit -m "feat: add standalone RankAgent demo script"
```

---

### Task 6: Export new tools from __init__.py

**Files:**
- Modify: `agentic_rec/__init__.py`

- [ ] **Step 1: Update __init__.py exports**

Read the current `__init__.py`, add `TwoTowerTool` and `GBDTTool` to the import line:
```python
from .tools import (
    BizRuleTool,
    FeatureTool,
    GBDTTool,
    HotTool,
    TagTool,
    TwoTowerTool,
    VectorTool,
)
```

And add them to `__all__`.

- [ ] **Step 2: Verify import works**

Run: `cd d:/fredcode/hf_home/agenticrec_fork && python -c "from agentic_rec import TwoTowerTool, GBDTTool; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agentic_rec/__init__.py
git commit -m "feat: export TwoTowerTool and GBDTTool from package"
```

---

### Task 7: Final integration verification

- [ ] **Step 1: Run full test suite**

Run: `cd d:/fredcode/hf_home/agenticrec_fork && python -m pytest tests/ -v`
Expected: ALL tests pass (smoke + bench + deepseek_api + two_tower + gbdt + rank_agent)

- [ ] **Step 2: Run the original quickstart to confirm no regression**

Run: `cd d:/fredcode/hf_home/agenticrec_fork && python examples/quickstart.py`
Expected: Works as before, RankAgent shows fusion thought in trace

- [ ] **Step 3: Commit any remaining changes**

```bash
git add -A
git commit -m "chore: final integration verification, all tests passing"
```
