# Strategy-Routed GraphRAG — Paper Outline

**Target venue**: NAACL Industry Track 2026 / EMNLP Industry Track 2026
**Status**: Draft v3 (2026-05-18) — operator-algebra reframing

---

## Core argument (one sentence)

GraphRAG's convergence on "better top-K retrieval" (community detection, beam-search planning, RL traversal) is a monolithic single-policy approach that is **structurally inadequate** for ~30% of natural KGQA questions (counting, multi-constraint, negation); the right fix is to **decompose retrieval into a closed Retrieval Operator Algebra indexed by question typology and dispatched by an LLM**, not to improve a single retrieval algorithm.

---

## Framing: Retrieval Operator Algebra

This draft commits to the operator-algebra framing throughout:

- **Algebra**: `A = {lookup, exhaustive, complement, path_plan, constrained_join, dual_subgraph}` — six primitive operators forming a closed cover of computable patterns in natural KGQA.
- **Typology**: 11 question types grounded in retrieval-pattern requirements (not surface form).
- **Dispatcher**: zero-shot LLM (DeepSeek-Chat) mapping type → operator.
- **Cross-cutting layer**: bilingual alias resolution (operator-orthogonal).
- **Closure claim**: every typology pattern maps to exactly one operator; 0 pp ablations are evidence of algebraic closure on the present distribution, not redundancy.

System brand name "Strategy-Routed GraphRAG" preserved; "strategy" and "operator" are interchangeable in the text, with "operator" used in formal/method contexts and "strategy" used in narrative/system-level contexts.

---

## Section structure

### 1. Introduction
- Three example questions where top-K cannot win regardless of K (count enumeration, set intersection, absence-of-fact)
- Argue this is structural, not an edge case (~30% of natural usage)
- Pose the fix as **the right operator for the question's computational pattern**, enumerated as six bullets matching the six operators
- **Contributions**:
  1. **Retrieval Operator Algebra** — 6 primitive operators, closed cover
  2. **11-way question typology** grounded in retrieval-pattern requirements + zero-shot LLM dispatcher (93% routing accuracy)
  3. **Bilingual alias resolution layer** for industrial Chinese KGs
  4. Open benchmark **RadarKG-QA-499** + 26-Q OOD bilingual stress
  5. +34 pp over baseline / +23 pp over RoG-style at $0.00088 marginal $/pp

### 2. Related Work
- Lead with **three-communities table**: GraphRAG / LLM Path Planning / Symbolic KGQA — each has one piece, missing the others
- §2.1 GraphRAG systems (Microsoft GraphRAG, HippoRAG, KGP) — uniform top-K policy
- §2.2 LLM path planning (RoG, ToG/ToG-2, RDPG, RAR) — uniform single policy of *path emission*
- §2.3 Symbolic KGQA / NL2SPARQL — precise algebra but brittle and schema-locked
- §2.4 Multilingual KGQA (mKGQAgent) — assumes monolingual KG; we handle mixed-language values
- §2.5 Question classification (OLTP/OLAP) — coarse; we go fine-grained pattern-grounded

### 3. Method: A Retrieval Operator Algebra for KGQA
- §3.1 Pipeline: 3 LLM stages (dispatcher / parser / answerer) + 1 KG stage (operator executor with 6 branches)
- §3.2 **Question typology** — 11 types with computable-pattern column added to the table; pattern-grounded, not surface-grounded
- §3.3 **The Six-Operator Algebra**:
  - Formal definition: `O: (Question, KG) → Evidence`
  - Operator table with **computable pattern** + **mechanism** + **patterns covered** columns
  - **Closure claim** stated explicitly
- §3.4 **Type-to-Operator Dispatch** — zero-shot LLM, decoupled from typology and operators (each layer independently auditable / upgradable)
- §3.5 **Bilingual Alias Resolution (cross-cutting)** — operator-orthogonal: relation lexicon (34) + entity aliases (614 surface forms) + 5-stage tail resolution + type validation + equivalence-class fallback

### 4. Dataset: RadarKG-QA-499
- §4.1 KG: 16,513 triples / 98 relations and attributes / 4,817 nodes / 849 radar entities
  - Multi-stage pipeline: rule + LLM extraction → Wikidata/Wikipedia enrichment → entity dedup → attribute normalization
  - Mixed-language values (radar names in English, countries/modes in Chinese)
- §4.2 499 questions stratified across 11 types
- §4.3 26-question handcrafted OOD bilingual stress (USA, AESA, MTI, monopulse, phased array, track mode, X band, etc.)

### 5. Experiments (all on DeepSeek-Chat for fairness)
- §5.1 **Main results** (Table 1, 100-Q stratified):
  - Baseline 57.0% / RoG 68.0% / **Strategy 91.0%** (+34.0 / +23.0 pp)
  - Widest gaps on attr_filter (B 20% → S 100%) and three_hop_chain (B 25% → S 88%)
- §5.2 **Per-Operator Ablation (Algebraic Closure Analysis)** (Table 2):
  - Load-bearing: exhaustive (-19), path_plan (-10), constrained_join (-8)
  - 0 pp on bench: complement, dual_subgraph — framed as **closure**, not redundancy
- §5.3 **Bilingual layer** — two-bench finding:
  - In-distribution: 0 pp drop (bench artifact, reinforced by v2 entity dedup)
  - 26-Q handcrafted OOD: **-11.5 pp drop** (real signal)
  - Three questions only succeed with layer ON: phased array→相控阵, track mode→跟踪, monopulse→单脉冲
- §5.4 **Dispatcher robustness**: 93% strategy-matching / 91% end-to-end / 85.7% graceful degradation / 1 catastrophic misroute. Frame lookup as **identity element** in the algebra (safe fallback under dispatch error).
- §5.5 **Cost & latency**: Strategy 4.8s avg, $0.00041/Q, $0.00088 marginal cost per accuracy point

### 6. Discussion
- §6.1 **Algebraic Closure, Not Empirical Redundancy** — 0 pp ablations are coverage-relation properties, not pruning candidates; missing operators leave silent failure modes; auditable traces matter beyond closed-bench accuracy
- §6.2 **Why Single-Operator Policies Underperform** — RoG-style path-emission over-explores for single_hop, mis-aggregates for counting, structurally fails for comparison; the lesson generalizes to any single-policy retrieval, however sophisticated
- §6.3 **The Bilingual-Bench Paradox** — methodological contribution: alias-handling ablations are false-negative on auto-gen benches; require handcrafted OOD substitution to measure
- §6.4 **Generalizing the Algebra** — patterns (identity, enumeration, composition, intersection, complement, comparison) are domain-general; principled extensions (aggregation arithmetic, transitive closure, temporal range, probabilistic membership) for other industrial domains. Analogy to Codd's relational algebra (closed core, principled extensions).
- §6.5 Limitations: bench distribution, OOD set size (26), residual KG noise on agg_count, no demonstrated domain transfer

### 7. Conclusion
- Operator algebra is cheap, interpretable, extensible, deployable
- Argument generalizes beyond radar KG via the pattern-grounded typology

---

## Figures
- **Fig 1** Pipeline architecture (3 LLM + 1 KG stage with 6 operator branches)
- **Fig 2** Per-type accuracy bar chart (Baseline / RoG / Strategy on 11 types)
- **Fig 3** Per-operator ablation drop chart (-19/-10/-8/0/0)
- (Optional **Fig 4**) Operator × pattern coverage matrix — visual closure argument

## Tables
- **Tab 1** Main results 3-way × 11 types
- **Tab 2** Per-operator ablation × 11 types (renamed from per-strategy)
- **Tab 3** Bilingual layer OOD ablation × 5 types
- **Tab 4** Cost & latency by system

## Appendices
- A: Question-type detection prompt
- B: Per-stage LLM cost breakdown
- C: Example full-pipeline traces (one per operator)
- D: OOD bilingual substitution dictionary

---

## Story-arc rationale

1. **Hook**: top-K is structurally broken on counting/intersection/negation — give 3 concrete examples a reader can verify mentally.
2. **Reframe**: the field's "better retrieval algorithm" framing is solving the wrong problem; the right framing is **decompose retrieval into a closed algebra over question typology**.
3. **Method**: operator algebra (6 primitives, closure claim) + typology (11 types, pattern-grounded) + LLM dispatcher (decoupled from both).
4. **Quantify**: +34 pp over baseline at $0.00088/pp marginal cost. Per-operator ablation pinpoints exhaustive / path_plan / constrained_join as load-bearing; complement / dual_subgraph 0 pp is **closure evidence**, not redundancy.
5. **Method note**: bilingual layer in-dist vs OOD divergence (0 pp vs -11.5 pp) is itself a publishable methodological finding for the broader community evaluating alias-handling components.
6. **Generalize**: pattern set is domain-general; algebra is closed but extensible — Codd analogy.

---

## Naming and terminology (consistency checklist)

- **Algebra / operator** in formal/method/closure contexts
- **Strategy** in system-name / narrative contexts (preserves "Strategy-Routed GraphRAG" brand)
- **Dispatcher** (PL-flavored), not "router" or "classifier" — when referring to the LLM stage. (Some legacy "router" references retained for non-technical descriptions where appropriate.)
- **Typology**, not "classification" or "taxonomy"
- **Closed cover** / **closure** for the 6-operator coverage claim
- **Pattern-grounded**, not "surface-grounded" — when distinguishing our typology from prior work

---

## Open items before submission

- [ ] Generate Fig 1 (pipeline diagram with 6-operator algebra branches) — draw.io
- [ ] (Optional) Fig 4 operator × pattern coverage matrix to visualize closure claim
- [ ] Regenerate Fig 2 and Fig 3 against v2 numbers in `scripts/make_paper_figures.py`
- [ ] Re-capture Appendix C example traces against current v2 pipeline (one per operator)
- [ ] Expand bibliography for LaTeX conversion (include Codd 1970 for algebra analogy)
- [ ] Optional follow-up: larger OOD bilingual set (50+ Q) to tighten per-type CIs
- [ ] Optional follow-up: cross-domain transfer test (e.g., medical equipment KG) to validate generalization claim in §6.4
- [ ] Optional follow-up: adversarial subset (KG-ambiguous negation, 3-way comparison) to surface load-bearing contribution of complement / dual_subgraph beyond closure argument
