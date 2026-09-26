# NL2SPARQL 对照实验设计（v1，供审阅）

## 目标
回应最强的新颖性攻击——"你的算子不就是退化版 NL2SPARQL?COUNT/交集/否定 SPARQL 早就能做"。
**把这个攻击点变成卖点**:证明在**多源噪声 + 中英混杂的工业 KG** 上,完整形式化查询(NL2SPARQL)恰恰**在 grounding（实体/字面量对齐）这一步系统性失败**,而我们的「别名解析 + 等价类回退 + 粗算子派发」正是让"集合运算式检索"在实践中跑得通的鲁棒性贡献。

**核心不是"NL2SPARQL 差、我们好",而是更诚实更强的三段论:**
1. 形式查询的"思想"没问题(SPARQL 原生支持聚合/交/否定);
2. 它在工业 KG 上**脆弱的具体环节是 grounding**(字面量 `USA` vs KG 里的 `美国`、`S band` vs `S`、`developedBy` 子图 vs `operatedBy` 子图不连通);
3. **我们的鲁棒层(别名+等价类)正是修复点**——甚至可以直接 bolt 到 NL2SPARQL 上救回来。

---

## 公平性保障（必须做到,否则审稿人反指我们造 strawman）
- 同一个强 LLM:**DeepSeek-Chat**(与全文一致)
- 给 NL2SPARQL **同样的 schema 信息**:`get_relation_specs()` 的关系列表(中文标签+类型签名),和我们 parser 拿到的完全一样
- **few-shot**:给 2–3 个 NL→SPARQL 示例(覆盖 count / filter-AND / ASK 否定),不让它吃 zero-shot 亏
- 允许 **1 次 repair 重试**:执行报错时把错误回喂让它修
- **同一套 scorer**(`scorer.score_question`)、**同样的 499 题**

---

## RDF 建模（用 rdflib 建可执行 SPARQL 端点）
把 `graphrag_index/merged_triples.json` 灌入 rdflib Graph:
- 每个不同表面串 → 一个资源 `radar:n/<hash>`,带 `rdfs:label "<原串>"`
- 每条三元组 (h,r,t) → `n(h) radar:<relation> n(t)`
- 关系用英文 schema id 作谓词(`radar:developedBy` 等);实体/取值**通过 rdfs:label 字面量匹配**

→ 这样 SPARQL 很自然:
```sparql
# 计数: 美国运营多少款雷达
SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE {
  ?s radar:operatedBy ?o . ?o rdfs:label "美国" . }
# 多约束 AND: 中国研制 + S 波段
SELECT ?l WHERE {
  ?s radar:countryOfOrigin ?c . ?c rdfs:label "中国" .
  ?s radar:hasFrequencyBand ?b . ?b rdfs:label "S" .
  ?s rdfs:label ?l . }
# 否定: AN/TPY-2 是否出口日本
ASK { ?s rdfs:label "AN/TPY-2" . ?s radar:exportedTo ?o . ?o rdfs:label "日本" . }
```
**脆弱点天然暴露在 `rdfs:label "…"` 这一步**:LLM 写 `"USA"`/`"S band"`/`"Raytheon公司"`,但 KG label 是 `"美国"`/`"S"`/`"Raytheon"` → 查询语法完全合法、能执行,但**返回空**。这正是我们要量的东西。

---

## 三个对照条件
1. **NL2SPARQL (raw)**:按 LLM 原样写的字面量执行 → 预期 grounding 处大量返回空
2. **NL2SPARQL + 我们的鲁棒层**:执行前把 SPARQL 里的 `rdfs:label` 字面量过一遍我们的别名解析(`_alias_resolve_tail`)+ 关系等价类回退(countryOfOrigin∪operatedBy),改写/扩展后再执行
   - 若这一条大幅救回 → **直接证明**:失败在 grounding,我们的层是修复点(可移植到 SPARQL)
3. **Strategy (ours)**:复用已有 88.6% 结果

---

## 关键产出:分阶段失败分解（这才是诊断,不是"谁赢"）
对每题记录 4 个阶段是否通过,算占比:

| 指标 | 含义 |
|---|---|
| **Syntactic valid %** | LLM 产出的 SPARQL 能被 rdflib 解析 |
| **Exec OK %** | 执行不报错 |
| **Non-empty %** | 返回非空结果 ← **grounding 死在这里** |
| **E2E accuracy %** | 经 scorer 判对(与 Strategy / baseline 同标尺) |

**预期叙事**:raw 的 Syntactic valid 很高(80%+),但 **Non-empty 断崖式下跌**(尤其 attr_filter / agg_count / negation,因为多约束/否定对字面量对齐最敏感);加我们的鲁棒层后 Non-empty 和 E2E 显著回升,但仍 < Strategy(因为完整解析对 schema 噪声更敏感)。

按题型(尤其 agg_count / attr_filter / negation —— "SPARQL 本应碾压"的三类)出 per-type 表,展示它反而被 grounding 卡住。

---

## 诚实预案（结果可能不如预期）
- 若 raw NL2SPARQL **意外地好**(比如 single_hop/count 上实体是 canonical 串)→ 那也是诚实发现,照报;说明我们的鲁棒性论点要收窄到"多约束/跨语言/否定"子集。**这个实验值得做,正因为两种结果都 informative。**
- 若 +鲁棒层后追平甚至超过 Strategy 在某些题型 → 说明"算子派发"相对"形式查询"的额外价值主要在鲁棒性而非表达力,论文据实写(更精确的定位,不丢人)。

---

## 实现 / 成本
- 依赖:`rdflib`(本地 minimind 环境装一下,纯 python)
- 流程:`build_rdf.py`(建图,几秒)→ `nl2sparql.py`(499 题生成+执行+两条件,~$0.05–0.1,~15 min)→ `eval_nl2sparql.py`(scorer + 分阶段统计出表)
- KG 已在本地;DeepSeek 本地可调；全程本地即可,不需要服务器

---

## 文件
```
experiments/nl2sparql/
  build_rdf.py          # triples -> rdflib Graph (+ label index)
  nl2sparql.py          # NL->SPARQL (few-shot) + execute (raw / +alias) + repair retry
  eval_nl2sparql.py     # 分阶段指标 + per-type + 4-way 对照表
results/nl2sparql_*.json / nl2sparql_summary.md
```

## 待你拍板
1. 用 **rdflib + 真 SPARQL**（推荐,最faithful）还是退而求其次用"NL→结构化程序在 KGIndex 上执行"（更省事但少了"真 SPARQL"的说服力）?
2. 要不要做**条件 2（+我们的鲁棒层）**?我强烈建议做——它把实验从"踩别人"升级为"诊断+证明我们的修复点",诚实且更强。
3. few-shot 示例给几个、覆盖哪些题型(默认:count / attr_filter / negation / single_hop 各 1)?
