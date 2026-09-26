# 全 PDF 抽取 v2 schema 关系（操作手册）

**目的**：把 Stage C 抽取从当前的 120 个雷达扩展到所有 ~331 个有 narrative 的雷达。

## 现状

- PDF 共 577 页，v1 抽取覆盖全部 577 页（产出 4335 条 v1-schema 三元组）
- v2 Stage C **目前只跑了 120 个雷达**（涉及 158 页 OCR），抽出 749 条 hasSubsystem / hasComponent / meetsStandard / usedIn
- **还有 ~211 个雷达**带 narrative 内容但没跑 v2 schema 抽取

## 一键全量重跑（3 个命令）

```bash
# 1. 抽取（OCR 增量 + LLM 抽取 + 已有缓存复用）
#    --all 处理所有 ~331 个 narrative-rich 雷达
#    OCR cache 在 data/v2/ocr_cache/，已有 158 页不会重 OCR
#    LLM cache 在 data/v2/stage_c_llm_cache.json，已抽过的 120 个雷达瞬间命中
python scripts/stage_c_extract.py --all

# 2. 整合到 v2 KG（幂等，重跑不会重复加边）
python scripts/stage_c_integrate.py

# 3. 清理（去掉 LLM 过度抽取的 usedIn + 孤立实体）
python scripts/stage_c_cleanup.py
```

## 时间和成本预估

| 阶段 | 工作量 | 时间 | LLM cost |
|---|---|---|---|
| OCR 增量 | ~250 新页（每页 ~6 秒）| **25-30 min** | $0 |
| LLM 抽取 | ~211 新雷达（每个 ~3 秒）| **10-12 min** | ~$0.30 |
| 整合 + 清理 | <1 分钟 | <1 min | $0 |
| **总计** | | **~40 min** | **~$0.30** |

## 预期产出

| 指标 | 当前 v2（120 雷达） | 全量 v2（331 雷达） |
|---|---|---|
| hasSubsystem | 446 | ~1100 (×2.5) |
| hasComponent | 194 | ~480 |
| meetsStandard | 89 | ~220 |
| usedIn | 13 | ~30-40 |
| **新增边总计** | **+742** | **~+1800** |
| Subsystem 实体 | 446 | ~1100 |
| Component 实体 | 71 | ~150 |
| Standard 实体 | 33 | ~50 |
| Conflict 实体 | 10 | ~20 |
| **v2 总边数** | 5129 | **~6200** |

## 跑完后验证

```bash
# 看 v2 KG 最终状态
python -c "
import json
with open('data/v2/radarkg_v2_edges.json', encoding='utf-8') as f:
    e = json.load(f)
with open('data/v2/radarkg_v2_entities.json', encoding='utf-8') as f:
    n = json.load(f)
print(f'Entities: {n[\"count\"]}, Edges: {e[\"count\"]}')
"

# 看新加的 stage_c 边的关系分布
python -c "
import json
from collections import Counter
with open('data/v2/radarkg_v2_edges.json', encoding='utf-8') as f:
    edges = json.load(f)['edges']
cc = Counter(e['relation'] for e in edges if e.get('source')=='stage_c_llm')
for r, c in cc.most_common():
    print(f'  {r:20s} {c}')
"
```

## 重要说明

1. **OCR/LLM 都有持久化缓存**：你可以中途 Ctrl-C，下次接着跑，不会重复 OCR/调 LLM
2. **integration 是幂等的**：你重新跑 `stage_c_integrate.py` 不会双重添加（已经被加过的 (head, rel, tail) 会跳过）
3. **如果中间出错**：检查 `data/v2/stage_c_raw_extractions.json` 是否生成。这是 LLM 抽取的原始结果，整合脚本读它。
4. **清理脚本独立可重跑**：发现遗漏可以重新调用

## 完成后下一步

跑完这 3 个命令后，整个 PDF 的 v2 schema 抽取就完成了，**接下来就可以做 Step 4（v2 上重跑实验）**。

Step 4 详细计划见 `RERUN_PLAN.md` Scenario B。
