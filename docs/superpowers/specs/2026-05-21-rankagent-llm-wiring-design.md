# RankAgent LLM Wiring Design

## Summary

Wire the RankAgent with the real DeepSeek LLM backbone to control ranking fusion,
and add two deterministic ranking tools: TwoTowerTool and GBDTTool.

## Architecture

```
RankAgent.step()
  │
  ├─ 1. LLM analyzes candidate stats (count, score distribution, tag diversity)
  │     → returns JSON: {skip, w_tower, w_gbdt, w_orig, reflect}
  │
  ├─ 2. TwoTowerTool(user_profile, items)  ──→ tower scores
  │    GBDTTool(items)                      ──→ gbdt scores
  │
  ├─ 3. Fuse: final = w_tower*s_tower + w_gbdt*s_gbdt + w_orig*s_orig
  │    Sort, truncate to SKIP_THRESHOLD
  │
  └─ 4. Reflection gate: if LLM flagged distribution abnormal
        → de-concentrate by boosting minority-tag items
```

## New Tools

### TwoTowerTool
- User tower: hash user profile tags → 16-dim vector
- Item tower: hash item features (tags, author, category) → 16-dim vector
- Score: cosine similarity of L2-normalized vectors
- Deterministic, zero-dependency

### GBDTTool
- Mimics a lightweight GBDT ensemble with fixed decision stumps
- Input features: ctr_prior, freshness, hot (log-scaled), tag match count
- Fixed weighted stumps → sigmoid → score
- Deterministic, zero-dependency

## LLM Integration

- Prompt describes candidate stats, asks for JSON fusion config
- Fallback: if LLM call fails, use hardcoded defaults (w_tower=0.4, w_gbdt=0.4, w_orig=0.2)
- Malformed JSON → best-effort parse, fallback per-field

## Files Changed

| File | Change |
|------|--------|
| `agentic_rec/tools.py` | Add TwoTowerTool, GBDTTool |
| `agentic_rec/agents.py` | Rewrite RankAgent with LLM-driven step() |
| `agentic_rec/pipeline.py` | Register new tools |
| `scripts/run_ranking_demo.py` | New standalone ranking demo |
| `tests/test_two_tower.py` | New unit tests for TwoTowerTool |
| `tests/test_gbdt.py` | New unit tests for GBDTTool |
| `tests/test_rank_agent.py` | New unit tests for RankAgent |

## Testing

- TwoTowerTool: test determinism, score range [0,1], symmetry
- GBDTTool: test determinism, score range [0,1], feature sensitivity
- RankAgent: test skip threshold, fusion, reflection, LLM fallback
