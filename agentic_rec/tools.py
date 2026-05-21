"""Built-in tools: small, hash-based, fully deterministic.

Replace any of these with a real backend (Faiss/Milvus/feature service)
without changing agent code — the Tool interface stays identical.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional

from .core import Item, Tool


def _h(s: str, mod: int = 10_000) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


def _sim(a: str, b: str) -> float:
    """Crude string similarity: jaccard over char trigrams + hash perturbation."""
    def grams(x: str) -> set:
        x = "  " + x + "  "
        return {x[i:i + 3] for i in range(len(x) - 2)}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    j = len(ga & gb) / len(ga | gb)
    perturb = (_h(a + "|" + b, 100) / 100.0) * 0.1
    return min(1.0, j + perturb)


# ---------------------------------------------------------------------------
class VectorTool(Tool):
    """Approximate vector recall over an in-memory item corpus."""

    name = "vector"
    description = "dense retrieval by semantic similarity"

    def __init__(self, corpus: List[Dict[str, Any]]) -> None:
        # corpus item: {id, title, tags:[...]}
        self.corpus = corpus

    def __call__(self, query: str = "", top_k: int = 50, **_: Any) -> List[Item]:
        scored = []
        for c in self.corpus:
            s = _sim(query, c.get("title", ""))
            scored.append(Item(id=c["id"], score=s, features=dict(c), source="vector"))
        scored.sort(key=lambda x: -x.score)
        return scored[:top_k]


# ---------------------------------------------------------------------------
class TagTool(Tool):
    """Tag/keyword inverted-index style recall."""

    name = "tag"
    description = "sparse retrieval by tag overlap with query/user profile"

    def __init__(self, corpus: List[Dict[str, Any]]) -> None:
        self.corpus = corpus

    def __call__(
        self,
        query: str = "",
        user_tags: Optional[Dict[str, float]] = None,
        top_k: int = 50,
        **_: Any,
    ) -> List[Item]:
        user_tags = user_tags or {}
        out: List[Item] = []
        q_low = query.lower()
        for c in self.corpus:
            tags = c.get("tags", [])
            qhit = sum(1 for t in tags if t.lower() in q_low)
            uhit = sum(user_tags.get(t, 0.0) for t in tags)
            score = 0.6 * (qhit / max(1, len(tags))) + 0.4 * uhit
            if score > 0:
                out.append(Item(id=c["id"], score=score, features=dict(c), source="tag"))
        out.sort(key=lambda x: -x.score)
        return out[:top_k]


# ---------------------------------------------------------------------------
class HotTool(Tool):
    """Popularity / hot-list recall — the cold-start safety net."""

    name = "hot"
    description = "global popularity fallback"

    def __init__(self, corpus: List[Dict[str, Any]]) -> None:
        self.corpus = corpus

    def __call__(self, top_k: int = 30, **_: Any) -> List[Item]:
        ranked = sorted(self.corpus, key=lambda c: -c.get("hot", 0))
        return [
            Item(id=c["id"], score=math.log1p(c.get("hot", 0)) / 10,
                 features=dict(c), source="hot")
            for c in ranked[:top_k]
        ]


# ---------------------------------------------------------------------------
class FeatureTool(Tool):
    """Lightweight feature service: returns enrichments by item id."""

    name = "feature"
    description = "fetch dense features for ranking"

    def __init__(self, corpus: List[Dict[str, Any]]) -> None:
        self._idx = {c["id"]: c for c in corpus}

    def __call__(self, item_ids: List[str], **_: Any) -> Dict[str, Dict[str, Any]]:
        return {i: self._idx.get(i, {}) for i in item_ids}


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

    def _embed(self, tokens: List[str]) -> List[float]:
        """Hash a list of tokens into a DIM-dimensional vector."""
        vec = [0.0] * self.DIM
        if not tokens:
            return vec
        for i in range(self.DIM):
            val = 0.0
            for t in tokens:
                val += _h(f"{t}|{i}", 1000) / 1000.0
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

        user_vec = self._embed(user_tags)

        scores: Dict[str, float] = {}
        for iid in item_ids:
            c = self._idx.get(iid)
            if c is None:
                scores[iid] = 0.0
                continue
            item_tokens = c.get("tags", []) + [c.get("author", ""), c.get("category", "")]
            item_vec = self._embed(item_tokens)
            # Cosine similarity (vectors already L2-normalized)
            sim = sum(u * v for u, v in zip(user_vec, item_vec))
            scores[iid] = max(0.0, min(1.0, (sim + 1.0) / 2.0))
        return scores


# ---------------------------------------------------------------------------
class BizRuleTool(Tool):
    """Business rules executed during rerank (dedup, boost, insert ads, etc.)."""

    name = "biz_rule"
    description = "scene-aware reranking rules"

    def __call__(
        self,
        items: List[Item],
        scene: str = "feed_home",
        **_: Any,
    ) -> List[Item]:
        # 1) dedup by author/category if available
        seen_author: Dict[str, int] = {}
        deduped: List[Item] = []
        for it in items:
            a = str(it.features.get("author", ""))
            if seen_author.get(a, 0) >= 2:
                continue
            seen_author[a] = seen_author.get(a, 0) + 1
            deduped.append(it)

        # 2) freshness boost in feed scene
        if scene == "feed_home":
            for it in deduped:
                fresh = float(it.features.get("freshness", 0))
                it.score = it.score * (1 + 0.15 * fresh)

        # 3) insert 1 ad slot at position 4 (placeholder behaviour)
        if scene in {"feed_home", "search"} and len(deduped) > 4:
            ad = Item(id="ad_slot_1", score=0.0, features={"is_ad": True},
                      source="biz_rule", explain="商业化插入位")
            deduped.insert(3, ad)

        # 4) re-sort by score, keeping ad slot fixed
        ad_pos = next((i for i, x in enumerate(deduped) if x.features.get("is_ad")), -1)
        if ad_pos >= 0:
            ad = deduped.pop(ad_pos)
            deduped.sort(key=lambda x: -x.score)
            deduped.insert(min(3, len(deduped)), ad)
        else:
            deduped.sort(key=lambda x: -x.score)
        return deduped
