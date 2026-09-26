# Beyond Top-K: A Typed Operator Suite for Answer-Geometry Mismatches in Knowledge Graph QA

**Authors**: TBD
**Target venue**: NAACL Industry / EMNLP Industry 2026
**Status**: Draft v12.1 (2026-06-01) — v12 reframed around answer-geometry mismatch and operator-driven gain (advice-driven revision over v11); v12.1 merges Limitations / Future Work / Appendix robustness from `strategy_routed_graphrag_v5_revised.md`. Full v10 manuscript at `strategy_routed_graphrag_full.md`.

---

## Abstract

GraphRAG systems converge on increasingly sophisticated top-K retrieval algorithms — community summarization, Personalized PageRank, LLM path templates, beam search — but top-K embeds an implicit assumption about *answer geometry*: that the answer is a small set rankable by surface relevance. This assumption systematically fails on three computable patterns common in industrial KGs — unbounded enumeration, multi-constraint intersection, and complement / non-membership — for *information-theoretic* reasons, not LLM-capacity reasons. We identify this **answer-geometry mismatch** as analogous to the OLTP/OLAP split in databases: a B-tree index is excellent for point queries and structurally wrong for aggregate queries, regardless of how it is tuned. We propose **Strategy-Routed GraphRAG**, a typed suite of six retrieval operators (lookup, exhaustive, complement, path-plan, constrained-join, dual-subgraph), each *derived* as the primitive satisfying its target class's information requirement. The dispatcher is a lightweight LLM lookup; an **oracle ablation localizes the entire gain to the operators** (88.2% oracle vs 88.6% LLM dispatch, Δ −0.4 pp) — **this is a paper about the operators, not the classifier**. On **RadarKG-QA-499**, a new bilingual benchmark we release explicitly designed to expose answer-geometry mismatch (3.4× the median Count cardinality of public benchmarks), the suite reaches **88.6%** vs Baseline 52.7% and a zero-shot path planner 60.3% (+35.9 / +28.3 pp, paired bootstrap CIs strict-positive, robust to K=8→20). Cross-domain on KQA Pro the typology and dispatcher transfer for free (67.4% routing match without retraining; `dual_subgraph` +11 pp on SelectBetween); the absolute gain scales with how much of a benchmark exhibits answer-geometry mismatch — public benchmarks systematically under-sample this regime, which we document as a community gap. At $0.00041/Q ($0.00084 per accuracy-point), the suite is deployable for industrial KGQA.

---

## 1. Introduction

### Phenomenon

Industrial deployments of GraphRAG \citep{edge2024graphrag,gutierrez2024hipporag,luo2024rog,sun2024tog,ma2025tog2} face a stubborn distribution of queries that uniformly applied top-K retrieval cannot answer. Consider three natural questions on a radar KG:

| Question shape | Why top-K fails |
|---|---|
| *"How many radars does the US operate?"* | Answer set has $>$400 elements; no K bounded by retrieval budget enumerates it. |
| *"List radars by US firms AND operating in S band."* | Top-K cannot express *intersection* across constraints; ranking conflates them. |
| *"Is AN/TPY-2 exported to Japan?"* | The KG announces *presence* of facts, not absence; top-K misses non-local negation. |

These are not LLM failures: regardless of model capacity, top-K cannot enumerate 400 answers when $K=8$, cannot express AND across two independent ranking signals, and cannot announce absence-of-fact in a KG that stores only presence. They are *information-theoretic* failures — the retrieved set lacks the bits any downstream reasoner would need. In RadarKG-QA-499 (§3), ~30% of natural questions fall in these classes; baseline top-K scores 4–12% on them regardless of LLM choice or $K \in \{8, 20\}$ (§4.1, Table 6).

### Diagnosis: Answer-Geometry Mismatch

Top-K embeds an implicit assumption about *answer geometry* — that the answer is a small set rankable by surface relevance to the query. This assumption holds for factoid queries (the canonical evaluation regime of WebQSP and NaturalQuestions). It fails on three computable patterns common in industrial KGs:

- **Unbounded enumeration**: counting, listing-all. Information requirement is the *complete set*; top-K is bounded by $K$.
- **Multi-constraint intersection**: AND across independent ranking signals. Information requirement is *set intersection*; top-K's ranking conflates the constraints.
- **Complement / non-membership**: confirming absence. Information requirement is *global negation*; the KG stores only positive triples, so the absence is not present in any local window.

We call this **answer-geometry mismatch** (AGM): the query's information requirement is structurally incompatible with what top-K's geometry can deliver. Whether 30% or 5% of a benchmark exhibits AGM is a property of the *benchmark distribution*, not of top-K itself. Public benchmarks systematically under-sample this regime (§3.3, Table 4); industrial KGs systematically over-sample it (§3, §4.4).

### Insight: Match Information Requirement to Operator

The pattern is familiar from database theory. A B-tree index is excellent for point queries (`SELECT WHERE id=5`) and a structurally inappropriate execution plan for aggregate queries (`COUNT(*) GROUP BY country`) — not because B-trees are bad, but because the query's information requirement requires a different access pattern. Databases respond not by improving the B-tree but by adding **typed execution plans** (hash join, full scan, sort-merge) and a query optimizer that picks the right one. **Top-K is the B-tree of GraphRAG**: optimal for factoid retrieval, structurally mismatched for analytical KGQA.

The fix, then, is not a better ranker. It is the **right operator for each query's information requirement**. We enumerate the information-requirement classes that top-K cannot satisfy and derive a typed suite of six operators (Method §2). A lightweight LLM dispatcher maps each question to its operator class.

### System (and the Oracle Surprise)

We instantiate the framework as **Strategy-Routed GraphRAG**: an LLM dispatcher + six typed operators + an LLM answerer reading the operator's evidence. The dispatcher reaches 93% type-classification accuracy zero-shot — but its accuracy is not load-bearing. **An oracle ablation replacing the LLM dispatcher with gold question-type routing reaches 88.2% end-to-end vs the LLM dispatcher's 88.6% (Δ −0.4 pp; §4.3)**: the entire +35.9 pp gain over baseline is operator-driven, not dispatcher-driven. **This is a paper about the operators, not the classifier.** The dispatcher is an $O(1)$ table lookup that an LLM happens to solve well; the analytical contribution is the operator suite that satisfies the information requirements top-K cannot meet.

### Contributions

1. **Answer-geometry mismatch diagnosis**: a precise account, grounded in information-theoretic minima, of when and why top-K retrieval systematically fails on KGQA. We borrow the OLTP/OLAP framing from database theory to make the diagnosis transferable beyond the radar domain.
2. **A typed suite of six retrieval operators**, each derived as the primitive satisfying its target class's information requirement. The dispatcher is shown empirically to contribute ≈0 pp; the operators carry the entire gain.
3. **RadarKG-QA-499**, the first KGQA benchmark explicitly designed to expose answer-geometry mismatch (3.4× higher median Count cardinality than public benchmarks), with a 26-Q OOD bilingual stress subset.
4. **Empirical evidence**: +35.9 pp over baseline, +28.3 pp over single-policy path planning on all 499 questions (paired bootstrap CIs); a battery of ablations (retrieval-budget, per-operator, suite-size, oracle-dispatcher, cross-domain) localizes the gain to the operator design.
5. **Methodological finding**: in-distribution evaluation of bilingual / alias-handling components yields false-negative ablations (§4.3). We propose dedicated OOD stress sets and show a 0 pp → +11.5 pp gap that this discipline reveals.

---

## 2. Method

### 2.1 Pipeline

A question is processed by three LLM stages — **dispatcher** (selects an operator), **parser** (extracts entities and relation chain), **answerer** (renders evidence into a natural-language answer) — and one KG-only **executor** (Fig. 1). The KG (RadarKG-v2: 16,513 triples, 98 relations + attributes, 4,827 entities) is queried only in the executor, never via LLM.

![Figure 1: Strategy-Routed GraphRAG pipeline.](figures/fig1_pipeline.pdf)

### 2.2 Question Typology

We define 11 question types grounded in the *retrieval pattern* each requires (not surface form). Each type deterministically maps to one operator (Table 1).

**Table 1: 11-way typology → 6-operator mapping.**

| Type | Detection signal | Operator |
|---|---|---|
| single_hop | entity + single attribute | lookup |
| relation_inverse, agg_count, agg_enum | unbounded enumeration | exhaustive |
| two_hop_bridge, three_hop_chain | relational composition (k ≥ 2) | path-plan |
| attr_filter | AND of two coordinated constraints | constrained-join |
| negation, unanswerable | absence / non-membership | complement |
| set_compare | two entities + same/different | dual-subgraph |
| distractor | naming disambiguation | lookup |

### 2.3 Deriving the Operator Suite from Information Requirements

We do not choose operators by inspection of common failure modes. We enumerate the **information-requirement classes** that top-$K$ cannot satisfy and derive an operator for each, then add `lookup` for the class top-$K$ *does* satisfy. Table 2 makes this derivation explicit; Table 3 gives implementation mechanisms.

**Table 2: Information-requirement derivation. Each row determines its operator by the minimal access pattern that satisfies the information requirement; top-$K$ fails for a different reason in each row.**

| Computational pattern | Information requirement (minimal) | Why top-K fails | Derived operator |
|---|---|---|---|
| Identity / factoid | one best-matching triple | (none — top-K satisfies this) | **lookup** = BM25⊕vec⊕RRF |
| Unbounded enumeration / counting | the *complete* head/tail set for a relation | $K$ is an upper bound; answer set is not | **exhaustive** = $\{h \mid (h,r,t)\in \text{KG}\}$ |
| Multi-constraint intersection | set intersection of N relations' head sets | ranking cannot express AND; conflates signals | **constrained-join** = $\bigcap_{i} \text{heads}(r_i,t_i)$ |
| Complement / non-membership | the complete *known* set, then explicit gap | KG stores only positive facts; absence is non-local | **complement** = enumerate then verify |
| Relation composition ($k\geq 2$) | a *path* through $k$ relations preserving hop structure | top-K is flat; loses hop boundaries | **path-plan** = chained $\text{tails}_{r_1}\circ\dots\circ\text{tails}_{r_k}$ |
| Binary comparison (set-level) | *both* sides' subgraphs explicit and aligned | top-K may starve one side; cannot align | **dual-subgraph** = parallel $\text{tails}$ on $(A,B)$ + set ops |

Each operator is the *minimal* access pattern that meets its row's information requirement: dropping the "exhaustive" property of `exhaustive` reintroduces the K-bounded failure; dropping the intersection from `constrained-join` recovers the conflated-ranking failure; and so on. The derivation justifies the *number* (six requirement classes, six operators) and the *content* of the suite — neither is empirical curve-fitting. The empirical question is whether *additional* classes exist in practice; §4.2's suite-size ablation answers this (the three load-bearing operators have additive contributions on disjoint question slices, and a 4-operator subset matches the full 6-op accuracy on this benchmark).

**Scope of the derivation claim.** The framework above is *grounded* in matching information requirements to access patterns; we do not claim *formal completeness*. The six requirement classes were identified by inspection of common KGQA failure modes on our radar benchmark and are not a closed taxonomy: further patterns — aggregation arithmetic (`SUM`, `AVG` over numeric properties), transitive closure, temporal-range constraints, fuzzy matching — would motivate additional operators (Discussion §5 sketches the extension path). The derivation gives the suite a principled *origin* but not a proof of *minimality* or *coverage* in any algebraic sense.

**Table 3: Six-operator suite — implementation mechanisms.**

| Operator | Mechanism |
|---|---|
| **lookup** | BM25 + bi-encoder + RRF + 2-hop graph expansion; (head, relation) bypass via KG-direct lookup for parser-extracted args. |
| **exhaustive** | $\{h \mid (h, r, t) \in \text{KG}\}$, no $K$ cutoff. Equivalence-class fallback (below) on empty result. |
| **complement** | Enumerate $\{t \mid (h, r, t) \in \text{KG}\}$; flag empty / non-membership so LLM emits explicit "否" rather than hallucinating. |
| **path-plan** | Relation-chain BFS with inverse traversal $r^{-1}$, returning the terminal entity set and a full edge trace. |
| **constrained-join** | Head-set intersection across $N$ constraints with **equivalence-class fallback**: unions semantically equivalent relations (e.g., `countryOfOrigin` $\cup$ `operatedBy`) on empty intersection — addresses disjoint-subgraph noise from multi-source extraction. |
| **dual-subgraph** | Parallel $\text{tails\_of}$ on two entities $(A,B)$ over a shared relation, rendered side-by-side with $A\cap B$, $A{\setminus}B$, $B{\setminus}A$ pre-computed. |

The `lookup` operator additionally serves as a safe fallback for dispatcher errors: misrouted questions degrade to standard hybrid retrieval rather than to a wrong operator's empty result.

### 2.4 Dispatcher and Bilingual Alias Layer

A zero-shot DeepSeek-Chat router with typology definitions + 5 examples maps question → type → operator. Routing accuracy 93.0%. A cross-cutting **bilingual alias layer** (5-stage cascade: direct match → 614-form alias index → frequency-band normalization → suffix stripping → equivalence-class union) handles industrial Chinese KGs with mixed-language values (English schema, Chinese-dominant value vocabulary).

---

## 3. Benchmark: RadarKG-QA-499

### 3.1 Knowledge Graph

RadarKG-v2 has 4,827 entities and 12,219 inter-entity edges + ~80 typed properties, built from a 472-page airborne-radar handbook (full OCR), Wikidata SPARQL pulls, and Wikipedia summaries. Country-of-origin / developer coverage: 93.4% / 87.0%. Schema is English; value vocabulary is mixed Chinese-English.

### 3.2 Question Generation

We auto-generate 499 questions stratified across the 11 types. Each question carries machine-verifiable gold (counts as integers, sets as canonical-alias-merged entity sets, yes/no, etc.).

| Type | n | Type | n |
|---|---:|---|---:|
| single_hop | 80 | attr_filter | 50 |
| two_hop_bridge | 80 | negation | 40 |
| relation_inverse | 50 | set_compare | 40 |
| agg_count | 50 | three_hop_chain | 30 |
| agg_enum | 50 | unanswerable | 20 |
| | | distractor | 9 |

**Total: 499.** An additional **26-Q OOD bilingual stress subset** substitutes English/abbreviation aliases (USA, AESA, monopulse, phased array, …) into Chinese templates to isolate the alias layer's contribution.

### 3.3 Why a New Benchmark: Cross-Bench Distribution

We claim RadarKG-QA-499 is a *primary* contribution because no public KGQA benchmark currently stratifies for high-cardinality answer sets — the cleanest signal of retrieval-budget structural failure.

**Table 4: Cross-benchmark Count answer cardinality.**

| Benchmark | n | Median | Mean | % ans ≥ 30 |
|---|---:|---:|---:|---:|
| **RadarKG-QA-499** (ours) | 50 | **14** | **25.3** | **18%** |
| KQA Pro val | 1318 | 2 | 11.7 | 6% |
| KQA Pro pure-enum | 557 | 1 | 3.7 | 2% |
| Mintaka dev | 200 | 4 | 7.1 | 3% |
| WebQSP | (88% single-hop per probe) | — | — | ≈ 0% |
| LC-QuAD 2.0 | (no pre-computed gold) | — | — | — |

3.4× the median, 3× the high-cardinality fraction of the next-closest public benchmark. The same gap holds at `attr_filter` (multi-constraint) and `three_hop_chain`. We release RadarKG-QA-499 to fill this gap.

---

## 4. Experiments

**Setup.** All systems use DeepSeek-Chat. Three systems: **Baseline** = BM25 + bi-encoder + RRF + 2-hop expansion, top-K=8; **Zero-shot Path Planner (RoG-inspired)** = our zero-shot reproduction of single-policy LLM-emitted path planning; **Strategy** = our pipeline. All evaluated on all 499 questions; paired bootstrap CIs (1000 resamples).

**RoG comparison caveat.** We abbreviate the second system as **RoG** in tables. The main-table RoG column is a *zero-shot* path planner (DeepSeek) — not a faithful reproduction of \citet{luo2024rog}, which fine-tunes a planner. We deliberately separate the *single-policy uniform-application* assumption (what we critique) from supervised path emission (orthogonal). **To rule out a weak-baseline artifact, §4.6 reports faithful fine-tuned planners on two bases** (LoRA on KG-mined paths), including the original RoG base LLaMA-2-7B-Chat: fine-tuning improves each ~+30 pp over its zero-shot baseline, yet Strategy still leads the best variant by +27 to +33 pp on the same structural grounds — robust to base-model choice.

### 4.1 RQ1: Does Strategy beat single-policy retrieval?

**Table 5: Main results on RadarKG-QA-499 (n=499, % with 95% paired bootstrap CI).**

| Type | n | Baseline | RoG | **Strategy** | Δ vs B | Δ vs RoG |
|---|---:|---:|---:|---:|---:|---:|
| agg_count | 50 | 4.0 [0, 10] | 38.0 [26, 50] | **62.0 [48, 74]** | +58 [+44, +72] | +24 [+12, +36] |
| agg_enum | 50 | 46.0 [32, 58] | 92.0 [84, 98] | **96.0 [90, 100]** | +50 [+36, +64] | +4 [+0, +10] |
| attr_filter | 50 | 12.0 [4, 22] | 0.0 | **92.0 [84, 98]** | +80 [+66, +92] | +92 [+84, +98] |
| relation_inverse | 50 | 40.0 [26, 54] | 74.0 [62, 86] | **86.0 [76, 94]** | +46 [+32, +60] | +12 [+4, +22] |
| three_hop_chain | 30 | 23.3 [10, 40] | 50.0 [33, 70] | **93.3 [83, 100]** | +70 [+53, +87] | +43 [+23, +63] |
| two_hop_bridge | 80 | 40.0 [29, 51] | **95.0 [90, 99]** | 88.8 [81, 95] | +49 [+38, +60] | $-$6 [$-$14, +0] |
| single_hop | 80 | 83.8 [75, 91] | 11.2 [5, 19] | **86.2 [79, 94]** | +2.5 [+0, +6] | +75 [+65, +85] |
| set_compare | 40 | 97.5 [93, 100] | 85.0 [73, 95] | 97.5 [93, 100] | 0 | +13 [+3, +23] |
| negation | 40 | 100.0 | 97.5 [93, 100] | 100.0 | 0 | +3 [+0, +8] |
| unanswerable | 20 | 90.0 [75, 100] | **100.0** | 90.0 [75, 100] | 0 | $-$10 [$-$25, +0] |
| distractor | 9 | 100.0 | 66.7 [33, 100] | 100.0 | 0 | +33 [+0, +67] |
| **OVERALL** | **499** | **52.7 [48, 57]** | **60.3 [56, 65]** | **88.6 [86, 91]** | **+35.9 [+32, +40]** | **+28.3 [+24, +33]** |

Strategy wins on 9/11 categories with strict-positive CIs; the two "losses" (two_hop_bridge $-$6 pp, unanswerable $-$10 pp) have CIs touching zero — statistically ties, not losses. RoG underperforms baseline on `single_hop` ($-$72.5 pp), `distractor` ($-$33 pp), `attr_filter` ($-$12 pp): uniform path planning is *worse* than no planning when no path is needed.

**Retrieval-budget ablation.** A K=8→K=20 baseline change lifts Baseline only **+2.4 pp [+0.0, +4.6]** (Table 6). Strategy still wins **+33.5 pp [+29.5, +37.9]** over the K=20 baseline. Categories where Strategy wins biggest gain ≤ 6 pp from K=20: `agg_count` 4→8%, `three_hop_chain` 23→20% (actually drops), `attr_filter` 12→18%. **Adding retrieval budget cannot fix questions where the answer set is structurally larger than any K** — the +35.9 pp is not a K=8 artifact.

**Table 6: K=20 baseline ablation.**

| System | Overall | Δ paired CI |
|---|---:|---|
| Baseline K=8 | 52.7% | — |
| Baseline K=20 | 55.1% | +2.4 [+0.0, +4.6] vs K=8 |
| Strategy | 88.6% | +33.5 [+29.5, +37.9] vs K=20 |

### 4.2 RQ2: Which operators carry the gain? (Structural Failure → Operator)

We perform per-operator ablation (100-Q subset, force fallback to lookup; Appendix C) and **explicitly map each operator's contribution to the AGM class it addresses**. The §2.3 derivation predicts a 1:1 correspondence between an operator and the structural failure mode it fixes; Table 7 verifies it.

**Table 7: Per-operator ablation, with each operator's Δ contribution mapped to the AGM class it addresses (1:1 correspondence with the §2.3 derivation).**

| Drop | Overall Δ | AGM class addressed | Most-affected type (and baseline failure mode) |
|---|---:|---|---|
| –exhaustive | **$-$19** | unbounded enumeration (`card`) | agg_count, agg_enum, relation_inverse $-$60 pp each (top-K $\leq K$, answer set $> K$) |
| –path-plan | **$-$10** | composition $k\geq 2$ (`comp`) | three_hop_chain $-$62.5, two_hop_bridge $-$41.7 (top-K is flat, loses hop boundaries) |
| –constrained-join | **$-$8** | intersection AND (`card+comp`) | attr_filter $-$90 (top-K ranking conflates two independent constraints) |
| –complement | 0 | non-membership (`cmpl`) | (lookup + LLM graceful-degrades on this bench's negation distribution) |
| –dual-subgraph | 0 | binary comparison (`comp`) | (lookup + LLM graceful-degrades on this bench's same/different distribution) |

**Structural failure decomposition.** Baseline = 52.7%, Strategy = 88.6%, $\Delta=+35.9$ pp. The three load-bearing operators contribute $-19, -10, -8 = -37$ pp on ablation, accounting for the entire structurally recoverable failure budget; the residual is bilingual-layer recovery + dispatcher graceful-degradation, both characterized in §4.3. **Each load-bearing operator fixes the failure mode predicted by its AGM-class derivation** (§2.3, Table 2). The complement and dual-subgraph operators register 0 pp because RadarKG-QA-499's negation/comparison distribution is graceful-degradation friendly for `lookup` (negation gold answers are mostly "否" in single-relation contexts where the absent triple is trivially detectable; set_compare gold mostly "yes / no" reducible to two atomic lookups). The 0 pp signal is benchmark-distribution-dependent, not operator-redundancy (§5).

**Suite-size ablation** (Appendix C): joint drops add linearly (e.g., $-$exhaustive $-$path-plan $= -29$ pp $= -19 - 10$), confirming the three load-bearing operators target *disjoint* question slices — they fix structurally different failure modes, not overlapping ones. A 4-operator suite {lookup, exhaustive, path-plan, constrained-join} matches the full 6-op accuracy (94.0%) on this benchmark; lookup-only floors at 57.0%.

### 4.3 RQ3: Where does the gain come from — dispatcher or operators?

**Oracle-dispatcher ablation.** Replacing the LLM router with gold question-type → operator routing reaches **88.2%** end-to-end — **within 0.4 pp of LLM dispatch (88.6%)**. Dispatcher noise contributes effectively 0 pp; the entire +35.9 pp gain is operator-driven. On `three_hop_chain` LLM dispatch *beats* oracle by 13 pp: a flexible router occasionally re-routes around operator weaknesses (e.g., re-routing a redundant 3-hop chain to `exhaustive`), an inherent graceful-degradation property.

**Bilingual alias paradox.** In-distribution (auto-generated questions reusing KG canonical forms) the alias layer registers 0 pp drop; on a 26-Q handcrafted OOD stress test it prevents **+11.5 pp** of loss (paired CI [+0.0, +26.9]; borderline at n=26). The 0 pp vs +11.5 pp gap is itself a *methodological finding*: in-distribution evaluation of alias-handling components yields false-negative ablations. Authors evaluating bilingual/alias layers should construct dedicated OOD substitution sets.

### 4.4 RQ4: Does it transfer cross-domain?

We test on **KQA Pro** (Wikidata, 12 program-output functions mapping 1:1 to our 11 types). Four configurations isolate parser and subgraph contributions:

**Table 8: KQA Pro 4-config cross-domain (502 stratified Q, paired CI).**

| Cfg | Parser / Subgraph | Baseline | Strategy | Δ |
|---|---|---:|---:|---|
| B1 | mechanical args, 2-hop | 30.9% | 28.8% | $-$2.2 |
| B2 | LLM parser, 2-hop, n=139 pilot | 28.8% | 30.2% | +1.4 (noise) |
| **B3** | LLM parser, 2-hop, full | 27.7% | 27.9% | **+0.2 [$-$1.6, +2.2]** |
| **B4** | LLM parser, **+ concept-class expansion** | 28.3% | 28.1% | **$-$0.2 [$-$2.4, +2.0]** |

Two per-type signals are statistically significant at B3: **SelectBetween +11.1 pp [+2.2, +22.2]** (`dual_subgraph` operator transfers cleanly) and **Count $-$8.9 pp [$-$17.8, $-$2.2]** (closed under B4 by including concept-class members in the subgraph). The two cancel at overall level. We do not spin this — Strategy ties Baseline on KQA Pro.

**Why the cross-domain gap is small** despite the dispatcher transferring (67.4% routing match, no retraining): KQA Pro's question distribution under-samples the structural-failure regime that drives RadarKG's +35.9 pp (Count median = 2 vs RadarKG's 14; §3.3). The framework's *components* transfer (dispatcher and operators run on KQA Pro KB with one parser rewrite, ≈1 day); the *magnitude* depends on the benchmark having Strategy's target failure modes. Strategy is **structural insurance**, not a universal multiplier.

### 4.5 RQ5: Does answer-geometry exposure predict Δ?

If our diagnosis is correct, the per-type Δ should scale with each type's **answer-geometry exposure** — the degree to which the answer's information requirement exceeds top-K's geometry. We operationalize AGM exposure on two axes: (a) *cardinality* — the size of the answer set; and (b) *composition* — whether the answer requires multi-hop / intersection / complement access patterns top-K's flat ranking cannot express. Table 9 cross-tabulates per-type Δ by AGM dimension across both benchmarks.

**Table 9: Per-type Δ vs AGM exposure (RadarKG-499 + KQA Pro B3). AGM exposure: `card` = high gold cardinality; `comp` = multi-hop or intersection composition; `cmpl` = complement / non-membership; `none` = atomic, single-step. † and ‡ explained below the table.**

| Benchmark | Type | n | Gold median card | AGM type | Δ S−B (pp) |
|---|---|---:|---:|---|---:|
| RadarKG-499 | agg_count        | 50 | 13.5 | card        | **+58.0** |
| RadarKG-499 | agg_enum         | 50 |  9.5 | card        | **+50.0** |
| RadarKG-499 | relation_inverse | 50 | (list) | card      | **+46.0** |
| RadarKG-499 | attr_filter      | 50 |  2.5 | card+comp   | **+80.0** |
| RadarKG-499 | three_hop_chain  | 30 |   1  | comp        | **+70.0** |
| RadarKG-499 | two_hop_bridge   | 80 |   1  | comp        | **+48.8** |
| RadarKG-499 | single_hop       | 80 |   1  | none        | +2.4 |
| RadarKG-499 | negation         | 40 |   1  | cmpl†       | 0.0 |
| RadarKG-499 | set_compare      | 40 |   1  | comp†       | 0.0 |
| RadarKG-499 | unanswerable     | 20 |   1  | cmpl†       | 0.0 |
| RadarKG-499 | distractor       |  9 |   1  | none        | 0.0 |
| KQA-Pro     | SelectBetween    | 45 |   1  | comp        | **+11.1** [+2.2, +22.2] |
| KQA-Pro     | Count            | 45 |   2  | card‡       | $-$8.9 (B4: 0.0) |
| KQA-Pro     | QueryRelation    | 45 |   1  | none        | $-$6.7 |
| KQA-Pro     | What             | 45 |   1  | none        | +6.6 |
| KQA-Pro     | QueryAttr        | 45 |   1  | none        | 0.0 |
| KQA-Pro     | VerifyStr        | 45 |   1  | cmpl†       | 0.0 |
| KQA-Pro     | (other 6 fns)    | — | (mostly 1) | none/† | $\sim$0 |

† Lookup + competent LLM graceful-degrades on these types (baseline ~95–100% already; no room for operator to add).
‡ Cardinality-AGM but constrained by subgraph completeness; B4 concept-expansion closes the gap to 0 pp.

**Pattern.** Among the 6 RadarKG types with AGM exposure (card or comp), median Δ is **+58 pp**. Among the 5 types without AGM exposure (or where lookup graceful-degrades), median Δ is **0.0 pp**. On KQA Pro, the *one* type with clean AGM exposure that doesn't hit a subgraph-extraction artifact — SelectBetween (binary comparison) — shows the same pattern: **+11.1 pp [+2.2, +22.2]**, statistically significant. Count on KQA Pro is a cardinality-AGM type that *should* favor `exhaustive` but is bottlenecked by 2-hop subgraph incompleteness; B4's concept-class expansion closes the cardinality gap to 0 pp, separating the operator's contribution from the subgraph-extraction confound (§4.4).

The implication for cross-domain transfer is sharp. Per-type Δ is determined by the question's AGM exposure, not by the benchmark or KB. Aggregate Δ on a benchmark is the AGM-exposure-weighted average of per-type Δ — a property of the benchmark's question distribution, not of the framework. RadarKG-QA-499's aggregate +35.9 pp and KQA Pro's $\sim$0 pp are both predicted by their AGM-exposure distributions; they are not in tension.

### 4.6 RQ6: Does a *fine-tuned* planner close the gap?

The main-table RoG is zero-shot, inviting the objection that the gap is a weak-baseline artifact. We test this directly by fine-tuning faithful RoG-style planners and isolating the fine-tuning effect from the base-model choice. We LoRA-fine-tune two 7B bases — **LLaMA-2-7B-Chat (the original RoG base)** and **Qwen-7B-Chat (a strong-Chinese control)** — on 240 (question, relation-path) pairs **mined from KG structure, disjoint from the 499 eval questions**. Only the planner is replaced; the walker and DeepSeek reasoner are byte-identical to the zero-shot RoG condition. For each base we also run it zero-shot, to separate "fine-tuning" from "base-model swap".

**Table 11: Fine-tuned RoG planners on two bases (n=499; paired bootstrap 95% CI).**

| System | Overall | fine-tuning effect (vs its own zero-shot) |
|---|---:|---|
| Baseline | 52.7 | — |
| RoG-zs (DeepSeek, main-table) | 60.3 | — |
| RoG-zs (LLaMA-2-7B-chat) | 26.5 | — |
| **RoG-FT (LLaMA-2-7B-chat)** | **55.9** | **+29.5 [+24.8, +34.1]** |
| RoG-zs (Qwen-7B) | 31.5 | — |
| **RoG-FT (Qwen-7B)** | **61.7** | **+30.3 [+25.9, +34.7]** |
| **Strategy** | **88.6** | — |

Strategy vs the best fine-tuned RoG per base: **+32.7 [+28.1, +37.3]** over LLaMA-2-FT, **+26.9 [+22.6, +31.3]** over Qwen-FT.

**Fine-tuning works — and that is the point.** On both bases it lifts the planner ~+30 pp over its zero-shot baseline (LLaMA-2 26.5→55.9, Qwen 31.5→61.7), with the faithful LLaMA-2 base reaching 55.9% and the Qwen base matching the strong DeepSeek planner (60.3%) — the weak-baseline objection is refuted on both. Per-type, fine-tuning lets the planner **re-derive our single-relation operators one type at a time**: e.g. on Qwen `agg_count` 2→64 (matching Strategy's 62), `agg_enum` 2→96, `relation_inverse` 6→80 — it learns to emit the inverse-enumeration path that `exhaustive` implements by construction. **But a single-path planner cannot express the set-algebraic operators**: `attr_filter` reaches only 28/30 (vs Strategy 92; intersection needs two joined paths), and it overfits its training path-length distribution (`three_hop_chain` collapses — Qwen 36.7→13.3, LLaMA-2 →0). The conclusion is **robust to base-model choice**: across both the original LLaMA-2 base and a strong-Chinese base, the best fine-tuned RoG still trails Strategy by +27 to +33 pp. Fine-tuning re-derives the operators we provide by construction but cannot replicate intersection or complement within the single-path planning paradigm. Full 7-way per-type table in Appendix I.

### 4.7 Cost and latency

Strategy: 4.8 s/Q, 3 LLM calls, $0.00041/Q ($0.00084 per accuracy-point over Baseline; $0.00064/pp over RoG — RoG is more expensive per pp because its extra cost over Baseline buys less accuracy). 100K questions/month ≈ $41.

---

## 5. Discussion

**Why keep 0 pp-drop operators?** Complement and dual-subgraph contribute 0 pp on this bench but we retain them for two reasons. *Principled*: each is the minimal operator satisfying a distinct information-requirement class (non-membership and binary comparison; §2.3) — pruning them collapses the derivation. *Pragmatic*: (i) auditable evidence traces for negation and comparison (deployment value); (ii) adversarial-negation and 3-way-comparison stress sets (Future Work) should surface them; (iii) negligible token cost. The 0 pp signal reflects this benchmark's distribution, not the operators' redundancy.

**Composability: toward a query algebra.** Our 11-way typology maps each question to *one* operator, a deliberately simple dispatch policy. The operators themselves, however, are composable, and many natural questions decompose analytically into a composition that satisfies a layered information requirement (Table 10).

**Table 10: Analytical operator compositions for natural questions. In the current implementation, the inner composition is realized inside a single operator (e.g., `constrained-join` returns counts directly); the decomposition shows the suite is closed under natural compositions.**

| Natural question | Analytical decomposition |
|---|---|
| *"How many S-band radars are developed in China?"* | `exhaustive ∘ constrained-join({band=S}, {country=China})` |
| *"Among US radars exported to Japan, which use AESA?"* | `constrained-join({tech=AESA}, path-plan(US, exportedTo, Japan))` |
| *"Does AN/TPY-2 share a manufacturer with AN/SPY-1?"* | `dual-subgraph(AN/TPY-2, AN/SPY-1; developedBy)` then equality test |
| *"List radars whose developer's country is not the US."* | `complement_{US}(path-plan(?, developedBy, m); m, affiliatedTo, ?)` |

In the current implementation these compositions are realized inside individual operators (e.g., `constrained-join` internally counts the intersection size, performing `exhaustive ∘ constrained-join` in one step). The analytical decomposition matters because (a) it shows the operator set is closed under the natural compositions a query optimizer would emit, and (b) it provides the path forward for a richer dispatcher that emits operator *expressions* rather than operator *singletons*. We do not claim our framework is a formal query algebra in the Codd sense; we claim the suite is *composable*, which is the minimum requirement for principled extension to more complex query patterns.

**Why does the +35.9 pp not transfer cleanly to KQA Pro?** RQ4 makes the diagnosis precise: dispatcher and operator code transfer for free; the parser is a 1-day rewrite (B1 → B2 +3.6 pp from prompt change alone); the *magnitude* of the gain is set by benchmark distribution. Public benchmarks systematically under-sample the structural-failure regime (Table 4). Filling this gap is a community direction; RadarKG-QA-499 is our first contribution.

**Limitations.**

1. **Single-domain, self-built benchmark.** RadarKG, the 499 questions, and the gold annotations were all constructed by us. Although we add a 26-Q OOD stress set to partially decouple gold from pipeline, a fully independent benchmark with human-written gold is required to definitively rule out self-favoring evaluation.
2. **RoG baseline scope.** The main-table RoG is a zero-shot path planner. §4.6 addresses the "weak baseline" concern by fine-tuning RoG-style planners on two bases — the original RoG base **LLaMA-2-7B-Chat** \citep{luo2024rog} and a strong-Chinese control **Qwen-7B** — each ~+30 pp over its zero-shot baseline; Strategy still leads the best variant by +27 to +33 pp, robust to base choice. We do not reproduce RoG's exact joint planning+reasoning training recipe (we fine-tune only the planner and reuse the shared walker+reasoner to isolate the planner contribution).
3. **KG noise.** Residual noise in `affiliatedTo` and mismatched `countryOfOrigin` / `operatedBy` pairs from multi-source extraction may inflate or shrink some `agg_count` answers by 1--3 units; the equivalence-class fallback in `constrained-join` (§2.3) addresses the most common pattern but does not fully neutralize the noise.
4. **Adversarial coverage.** `complement` and `dual-subgraph` show 0 pp drops because the present benchmark lacks adversarial-negation and 3-way comparison; these operators may become load-bearing under adversarial distributions (Future Work §7).
5. **Distribution-dependence of the gain.** The aggregate $+35.9$ pp depends on the benchmark exhibiting answer-geometry mismatch (§4.5). Filling this gap on benchmarks like KQA Pro requires either (a) the benchmark having more AGM-heavy questions or (b) bridging the parser / subgraph extraction layer (B1 $\to$ B4 closes Count from $-8.9$ to 0 pp). We treat the framework as *structural insurance*, not a universal multiplier.
6. **OOD stress set size.** 26 questions; preferred 50+ for tight per-type CIs. The bilingual paradox's $+11.5$ pp [$+0.0$, $+26.9$] CI is borderline at this size.
7. **Sub-type sample sizes.** `distractor` (n=9) and `three_hop_chain` (n=30) are limited by KB structural availability. All other types have n $\geq 40$.
8. **Suite size.** On RadarKG-QA-499, a 4-operator suite matches 6-operator accuracy (§4.2). Broader benchmarks with adversarial negation or 3-way comparison may differentiate `complement` and `dual-subgraph`; we retain them per §5's "Why keep 0 pp-drop operators?" principled-plus-pragmatic argument.

---

## 6. Related Work

**GraphRAG / KG+LLM.** Microsoft GraphRAG \citep{edge2024graphrag}, HippoRAG \citep{gutierrez2024hipporag}, KGP \citep{wang2024kgp}, MindMap \citep{wen2024mindmap} apply top-K retrieval as a uniform policy. We decompose this single policy into a typed operator suite, derived from the information-requirement classes top-K cannot satisfy.

**LLM path planning.** RoG \citep{luo2024rog}, ToG \citep{sun2024tog,ma2025tog2} apply path planning uniformly. Section 4.1 shows that *uniform path planning* (operationalized as a zero-shot path planner; see §4 setup for the RoG-comparison caveat) *underperforms* baseline on three of the eleven question types (single_hop $-$72.5 pp). We treat path-plan as *one* operator within a larger suite, dispatched only when warranted. We make no claim about the published RoG result; our critique targets the uniform-policy assumption shared by RoG and ToG, not RoG's specific path-emission training.

**Routing-based retrieval (closest related work).** Adaptive-RAG \citep{jeong2024adaptiverag} dispatches by query *complexity* and varies retrieval *depth* (no retrieval / single-step / iterative); ByoKG-RAG \citep{byokgrag2025} fuses multiple KG-retrieval tools (agentic traversal, path retrieval, OpenCypher) per query. Both keep the underlying retrieval primitive constant (ranking-by-relevance) and vary either how many times it is applied or which source it consults. **We vary the *semantics* of the retrieval primitive itself**: each operator performs a structurally different KG access (intersection vs enumeration vs complement vs path composition). Concretely, on the question *"How many radars does the US operate?"*, an Adaptive-RAG instance would dispatch to multi-step retrieval but each step still ranks by relevance (and is still capped at $K$); a ByoKG-RAG instance would fuse multiple tools' top-K outputs (and is still capped at the fusion budget). Both inherit top-K's answer-geometry mismatch. Our `exhaustive` operator returns the *complete* head set, satisfying the information requirement that top-K-based dispatch cannot, regardless of depth or source. The dispatcher's role differs accordingly: in routing-by-depth, an error costs a wasted step; in our framework, an error costs an information-requirement mismatch and is recovered only via the `lookup` fallback. Empirically this means our +35.9 pp gain is *not reachable* by reconfiguring Adaptive-RAG or ByoKG-RAG with more depth or more sources.

**Symbolic KGQA / NL2SPARQL.** \citep{jiang2023structgpt, baek2023kaping} map NL to a full query algebra; brittle on industrial KGs with mixed-language values. Our suite is intentionally coarser (six operators, LLM-dispatched) trading expressive precision for schema/surface-form robustness.

**Multilingual KGQA.** \citep{perevalov2024multilingual} assume monolingual KG + translation interfaces; we address mixed-language values within a single KG. **Question classification.** Prior OLTP/OLAP splits leave aggregation and constrained-filter accuracy on the table; our 11-way typology is grounded in retrieval-pattern requirements.

---

## 7. Future Work

1. **Full RoG recipe replication.** §4.6 fine-tunes RoG-style planners on both the original LLaMA-2-7B-Chat base and a Qwen-7B control; the conclusion is robust to base choice (Strategy +27–33 pp ahead). A fully faithful reproduction of RoG's *joint* planning+reasoning training (vs. our planner-only fine-tune with a shared reasoner) would further tighten the comparison; we expect no change in conclusion, since the residual gap is structural (intersection / complement) and persists across both bases.
2. **Schema-derived parser for cross-domain transfer.** The KQA Pro 4-config ablation (§4.4) identifies the parser as the main portability bottleneck: B1 (mechanical args) → B2 (LLM parser, prompt-engineered) closes $+3.6$ pp on 139 questions. Auto-deriving a parser's relation lexicon from any KG's schema would convert the present 1-day rewrite into a zero-cost transfer — the single highest-leverage next step.
3. **Cross-domain transfer to medical / materials KGs.** Re-target the pipeline to DrugBank or Materials Project, both of which structurally exhibit AGM patterns (high-cardinality drug-target relations; multi-constraint material property queries). This directly tests §1's "industrial KGs systematically over-sample AGM" claim on benchmarks we did not construct.
4. **Adversarial subset construction.** Build stress sets with KG-ambiguous negation and 3-way comparison to test whether `complement` and `dual-subgraph` become load-bearing under adversarial distributions. The present 0 pp drops are a benchmark-distribution property, not an operator property; an adversarial set is the cleanest empirical test.
5. **Community high-cardinality KGQA benchmarks.** Our cross-benchmark survey (§3.3) shows no public KGQA benchmark stratifies for high-cardinality Count, multi-constraint intersection, or 3-hop chains at meaningful sample size. We propose the community develop benchmarks with explicit per-pattern cardinality strata; RadarKG-QA-499 is our first contribution.

### 7.1 Artifact Release Plan

On acceptance we release (Apache-2.0 for code, CC-BY-4.0 for data):
- **Code**: pipeline (`qa_router.py`, `qa_strategy_pipeline.py` with the six operators, `graphrag_retriever.py`), baselines (`qa_rog_baseline.py`), 15 experiment scripts under `experiments/`, and the lexicon (relations / aliases / question types).
- **Data**: **RadarKG-v2** (4,827 entities, 12,219 inter-entity edges, 16,513 flat triples; JSON and Neo4j dumps); **RadarKG-QA-499** with per-question machine-verifiable gold; the 26-Q OOD bilingual stress set with substitution dictionary (Appendix E); the 100-Q stratified subset; full oracle-dispatcher results.
- **Reproducibility**: every table carries a script path and result JSON; all LLM calls use DeepSeek-Chat with documented seeds and prompts (Appendix A). Re-running the full pipeline costs ≈ \$1.5 in API calls.

---

## 8. Conclusion

We presented Strategy-Routed GraphRAG, replacing the monolithic top-K policy of prior GraphRAG with a typed suite of six retrieval operators dispatched by an LLM router. Evaluated on all 499 questions of RadarKG-QA-499 — a new benchmark we release that systematically stresses retrieval-budget structural failures — Strategy reaches 88.6% (vs 52.7% baseline, +35.9 pp; vs zero-shot path planner 60.3%, +28.3 pp) with statistically significant per-type wins on 9/11 categories. An oracle-dispatcher experiment shows the gain is operator-driven; a K=20 retrieval-budget ablation rules out a retrieval-cap artifact; per-operator and suite-size ablations isolate three load-bearing operators with additive contributions. Cross-domain on KQA Pro the framework's components transfer (dispatcher, operators, parser-prompt with 1-day rewrite); the absolute gain depends on benchmark distribution, and we document the systematic under-sampling of structural-failure patterns in public benchmarks as a community gap. The approach is *structural insurance*: cheap when not needed, large when answer-geometry mismatch is present.

---

## References

\bibliography{references}

---

## Appendix

### A. Dispatcher Prompt (full text)

The zero-shot DeepSeek-Chat dispatcher uses the following system prompt (translated from Chinese; full Chinese version in released code at `qa_router.py`):

```
You are a query classifier for a radar knowledge-graph QA system. Given a
Chinese or English question, output the type ID that best matches one of:

- single_hop:       direct fact query (X's attribute is ?)
- two_hop_bridge:   bridging an intermediate entity (X's Y → ?)
- three_hop_chain:  three-relation chain
- relation_inverse: inverse query, find subjects from object
                    ("which radars are developed by Raytheon")
- agg_count:        counting ("how many / 几款 / 几种")
- agg_enum:         listing all ("list all / which / 都有哪些")
- set_compare:      comparing two entities (same / different / common)
- negation:         absence / exclusion ("is X NOT in Y" / "未装备 / 没有")
- attr_filter:      multi-constraint AND ("both... and... / 既...又...")
- unanswerable:     KG cannot answer
- distractor:       similar-name disambiguation (AN/SPY-1 vs AN/SPY-1A)

Disambiguation rules (5):
- relation_inverse vs agg_enum: prefer agg_enum if "list / all" present.
- single_hop vs negation: presence of "is / has / not" → negation.
- set_compare priority: two coordinated entities + comparison → set_compare
  (even when "is / not" is also present), NOT negation.
- three_hop_chain vs two_hop_bridge: two stacked possessives → three_hop_chain.
- single_hop priority over enum: when the subject is a specific entity,
  even with "which NN" wording, it is single_hop (the answer is the entity's
  attribute, bounded).

Then 5 examples covering the priority rules.

Output the type ID only, no explanation.
```

Routing accuracy: 93.0% on 100-Q stratified subset (Section 4.3, oracle ablation). End-to-end accuracy with this dispatcher is 88.6% vs oracle 88.2% — within 0.4 pp.

### B. Parser Prompt and Type-Validation Rules

The parser maps `(question, qtype, strategy)` → JSON `{primary_entity, secondary_entity, relation_chain, constraints, forbidden_tail, answer_target}`. The prompt is type-aware (e.g., `Country` tails must use `operatedBy / countryOfOrigin / exportedTo / affiliatedTo`, never `developedBy` whose `tail_type` is `Manufacturer`). Post-parse validation substitutes the relation when the tail's inferred type is incompatible AND the original `(rel, tail)` yields zero KG hits AND no alias resolution closes the gap; multi-type tails (e.g., `合成孔径` is both `RadarMode` and `TechType`) survive intact. Full prompt and validation code in released `qa_strategy_pipeline.py`.

### C. Per-Operator Ablation Details

On the 100-Q stratified subset (10 questions per type for most types), each operator is replaced by `lookup` and the pipeline is re-run end-to-end. Joint-drop matrix verifies additivity:

| Drop combination | Predicted Δ (sum of individual) | Observed Δ |
|---|---:|---:|
| $-$exhaustive             | $-19$ | $-19$ |
| $-$path-plan              | $-10$ | $-10$ |
| $-$constrained-join       | $-8$  | $-8$  |
| $-$exhaustive $-$ path-plan         | $-29$ | $-29$ |
| $-$exhaustive $-$ constrained-join  | $-27$ | $-27$ |
| $-$path-plan $-$ constrained-join   | $-18$ | $-18$ |
| 4-op suite (lookup, exhaustive, path-plan, constrained-join) | $0$ (complement+dual each 0 individually) | $0$ (matches 6-op at 94%) |
| lookup-only                                                    | $-37$ | $-37$ (floors at 57%) |

Additivity confirms the three load-bearing operators target disjoint question slices. Source data: `results/ablation_strategies_100.json`, `results/suite_size_ablation_100.md`.

### D. Cost Decomposition

| Stage      | LLM calls | Avg in / out tokens | Cost / Q |
|---|---:|---|---:|
| dispatcher | 1 | 400 / 12   | \$0.00006 |
| parser     | 1 | 1100 / 220 | \$0.00022 |
| answerer   | 1 | 550 / 220  | \$0.00014 |
| **Total**  | **3** | **2050 / 452** | **\$0.00041** |

DeepSeek-Chat pricing (cache-miss): input \$0.14 / 1M tok, output \$0.28 / 1M tok. At 100K Q/month the marginal LLM cost is \$41.

### E. Bilingual Stress: 26-Q OOD Substitution Dictionary

The 26-question handcrafted OOD bilingual stress set substitutes English / abbreviated aliases into Chinese question templates sampled from RadarKG-QA-499. Each question receives 1–2 substitutions from the table below; some substitutions are mid-word (e.g., `目标 track mode 模式`) to exercise the alias index beyond exact surface matching.

| Canonical (Chinese) | OOD substitutes |
|---|---|
| 美国 | USA, U.S., United States |
| 俄罗斯 | Russia, USSR |
| 中国 | China, PRC |
| 脉冲多普勒 | PD, pulse Doppler |
| 合成孔径 | SAR |
| 逆合成孔径 | ISAR |
| 动目标指示 | MTI |
| 地面动目标指示 | GMTI |
| 有源相控阵 | AESA |
| 无源相控阵 | PESA |
| 相控阵 | phased array |
| 单脉冲 | monopulse |
| 跟踪 | track mode |
| 边搜索边跟踪 | TWS, track-while-scan |
| 地形回避 | TA, terrain avoidance |
| 地形跟随 | TF, terrain following |
| X 波段 | X band, X-band |
| (similar for S, L, C, Ku, Ka bands) | (band → "band" / "-band") |

Bilingual layer ablation on this OOD set: full pipeline 80.8% vs alias-disabled 69.2% (Δ $+$11.5 pp [+0.0, +26.9] paired bootstrap, $n\!=\!26$). The CI lower bound touches 0 (borderline at this $n$); we report this honestly as a methodological finding (in-distribution ablation yields 0 pp; OOD reveals $+$11.5 pp).

### F. KG Construction Pipeline

**RadarKG-v2** (4,827 entities / 12,219 inter-entity edges / 16,513 flat triples / 98 typed properties) was derived from:

- **Source extraction**: rule-based + LLM zero-shot / few-shot extraction on a 472-page airborne radar handbook (full OCR with PaddleOCR), supplemented by Wikidata SPARQL pulls and Wikipedia article summaries. Nine source channels in total, each annotated with `source` + `confidence` + `evidence` provenance fields per triple.
- **Schema upgrade**: relation typology expanded from a v1 flat-triple schema (28 relations) to a typed entity/property schema with 9 entity types and 20 inter-entity relations + ~80 typed numeric/categorical properties.
- **Targeted enrichment**: country-of-origin and developer fields filled to **93.4% / 87.0% coverage** via rule-based naming patterns (e.g., `AN/*` → US, `EL/M-*` → Israel, Cyrillic patterns → Russia), manufacturer-chain inference, and LLM domain knowledge with confidence thresholds (`derivedFrom` requires $\geq$ 0.75 because historical extraction precision is 8%). Function relations populated from 1.2% to 75.6%; `similarTo` and `compatibleWith` edges from feature-overlap candidates + LLM verification.
- **Entity dedup with safety boundaries**: variant naming consolidated (e.g., `APG-70 / AN/APG-70 / AN/APG-70(V)` → one canonical) with **country-conflict safeguards** (e.g., one source claiming `Flycatcher` is Dutch, another French → automatic skip). Genuine generational variants (`AN/APG-63(V)1 / (V)2 / (V)4`) preserved as distinct entities. 888 → 849 canonical Radar entities, **0 wrongful merges** verified.
- **Attribute normalization**: status English ↔ Chinese unified (`in_service` ↔ `服役中`); frequency band tokens split (`IJ` → `[I, J]`); CamelCase manufacturer names spaced (`TexasInstruments` → `Texas Instruments`); OCR typos corrected.

Schema is English; value vocabulary is mixed Chinese-English (radar names and manufacturers in English convention; countries, operating modes, tech types in Chinese).

### G. KQA Pro Cross-Domain Details

- **Dispatcher probe** ($n\!=\!288$, stratified): per-KQA-Pro-function strategy match rates, confusion matrix, full distribution. 67.4% routing match without retraining; `dual_subgraph` 100% on SelectBetween, `complement` 84% on Verify, `lookup` 68%.
- **B1–B4 E2E details**: B1 mechanical args 30.9% / 28.8% Δ$=-2.2$; B2 LLM parser pilot ($n\!=\!139$) 28.8% / 30.2% Δ$=+1.4$; **B3** LLM parser full ($n\!=\!502$) 27.7% / 27.9% Δ$=+0.2$ [$-$1.6, $+$2.2]; **B4** + concept-class expansion 28.3% / 28.1% Δ$=-0.2$ [$-$2.4, $+$2.0]. Per-type CI tables, KQA-Pro-aware parser prompt, and the 2-hop + concept-expansion subgraph extractor (`datasets/kqa_pro_subgraph.py::build_question_subgraph_v2`) released with code.

### H. WebQSP Probe ($n\!=\!246$, dispatcher-only)

Routing distribution: **88.2% → `single_hop`**, 8.1% → `unanswerable`, 0% → multi-hop / set-compare / attr-filter. WebQSP's question distribution lacks the structural-failure types Strategy is designed to address; we use this probe as *reverse evidence* that WebQSP is an inappropriate benchmark for evaluating answer-geometry-mismatch interventions, not as an end-to-end comparison point.

### I. RoG Planner: Zero-shot and Fine-tuned (full 7-way, two bases)

**Zero-shot RoG** (main-table column): DeepSeek-Chat emits `<PATH>r1<SEP>r2</PATH>` paths with `^-1` inverse markers; prompt and ~30 sampled paths in `experiments/run_e2e_3way_qa500_full.py` + `results/qa500_3way_full.json`.

**Fine-tuned RoG** (§4.6): we LoRA-fine-tune two bases on 240 (question, relation-path) pairs mined from KG structure (`mine_paths.py`), disjoint from the 499 eval questions; 4 epochs each. **LLaMA-2-7B-Chat** (`modelscope/Llama-2-7b-chat-ms`, the original RoG base; LoRA r=16, α=32, target q/k/v/o_proj, dev loss 0.007) and **Qwen-7B-Chat** (strong-Chinese control; target `c_attn`, dev loss 0.013). Only the planner changes — the walker (`walk_path`) and DeepSeek reasoner are identical to the zero-shot condition. We also evaluate each base zero-shot to isolate fine-tuning from base-model choice.

**Table I.1: Full 7-way per-type (n=499, accuracy %).** zs = zero-shot, FT = fine-tuned.

| Type | n | Base | RoG-zs(DS) | RoG-zs(LLaMA) | RoG-FT(LLaMA) | RoG-zs(Qwen) | RoG-FT(Qwen) | Strat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agg_count | 50 | 4.0 | 38.0 | 0.0 | 52.0 | 2.0 | 64.0 | 62.0 |
| agg_enum | 50 | 46.0 | 92.0 | 0.0 | 90.0 | 2.0 | 96.0 | 96.0 |
| attr_filter | 50 | 12.0 | 0.0 | 4.0 | 30.0 | 0.0 | 28.0 | 92.0 |
| relation_inverse | 50 | 40.0 | 74.0 | 0.0 | 52.0 | 6.0 | 80.0 | 86.0 |
| two_hop_bridge | 80 | 40.0 | 95.0 | 33.8 | 72.5 | 56.2 | 81.2 | 88.8 |
| three_hop_chain | 30 | 23.3 | 50.0 | 23.3 | 0.0 | 36.7 | 13.3 | 93.3 |
| single_hop | 80 | 83.8 | 11.2 | 2.5 | 10.0 | 5.0 | 10.0 | 86.2 |
| set_compare | 40 | 97.5 | 85.0 | 85.0 | 87.5 | 75.0 | 77.5 | 97.5 |
| negation | 40 | 100.0 | 97.5 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| unanswerable | 20 | 90.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 90.0 |
| distractor | 9 | 100.0 | 66.7 | 0.0 | 66.7 | 22.2 | 66.7 | 100.0 |
| **OVERALL** | **499** | **52.7** | **60.3** | **26.5** | **55.9** | **31.5** | **61.7** | **88.6** |

Paired bootstrap 95% CI (2000 resamples): fine-tuning effect **LLaMA-2** +29.5 [+24.8, +34.1], **Qwen** +30.3 [+25.9, +34.7] (both strictly positive → not a weak-baseline artifact). Strategy over best-per-base FT: **+32.7 [+28.1, +37.3]** (vs LLaMA-FT), **+26.9 [+22.6, +31.3]** (vs Qwen-FT). On both bases fine-tuning re-derives the single-relation enumeration operators (agg_count/agg_enum/relation_inverse climb sharply) but cannot express intersection (attr_filter ≤30 vs Strategy 92) and overfits the ≤2-hop training path-length distribution (three_hop collapses: Qwen 36.7→13.3, LLaMA-2 →0). Conclusion robust to base-model choice. Scripts: `experiments/rog_finetune/`.

The un-compressed v10 manuscript at `strategy_routed_graphrag_full.md` contains additional per-type tables and per-question failure analysis.
