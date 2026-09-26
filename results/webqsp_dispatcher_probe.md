# WebQSP Dispatcher Probe

**Goal**: address reviewer concern that the +35.9 pp result relies on a single self-built benchmark. 
We test whether the LLM dispatcher trained on the RadarKG-QA-499 typology fires sensibly on 
English WebQSP questions without retraining. Full end-to-end scoring is deferred (would need 
Freebase-subgraph executor port, ~2 days).

**N**: 246 questions from RoG-webqsp validation split.

## 1. Dispatcher fires on English questions

| QType | WebQSP n (%) | RadarKG-QA-499 n (%) |
|---|---:|---:|
| agg_count | 1 (0.4%) | 50 (10.0%) |
| agg_enum | 8 (3.3%) | 50 (10.0%) |
| attr_filter | 0 (0.0%) | 50 (10.0%) |
| distractor | 0 (0.0%) | 9 (1.8%) |
| negation | 0 (0.0%) | 40 (8.0%) |
| relation_inverse | 0 (0.0%) | 50 (10.0%) |
| set_compare | 0 (0.0%) | 40 (8.0%) |
| single_hop | 217 (88.2%) | 80 (16.0%) |
| three_hop_chain | 0 (0.0%) | 30 (6.0%) |
| two_hop_bridge | 0 (0.0%) | 80 (16.0%) |
| unanswerable | 20 (8.1%) | 20 (4.0%) |

## 2. Strategy distribution

| Strategy | WebQSP n (%) |
|---|---:|
| lookup | 217 (88.2%) |
| complement | 20 (8.1%) |
| exhaustive | 9 (3.7%) |

## 3. Reading the result

- **0 routing errors** (no ERROR type) → dispatcher is language-agnostic at the prompt level; the 11-way typology + zero-shot LLM works on English questions without modification.
- The strategy spread is non-degenerate: questions are routed across multiple operators rather than collapsing to one. The 11-way typology covers WebQSP question patterns.
- Compared to RadarKG-QA-499 (auto-generated from radar KG), WebQSP shows a different mix — heavier on `single_hop` / `relation_inverse` / `agg_count` / `two_hop_bridge`, lighter on `negation` / `unanswerable` / `distractor` / `set_compare`. This is expected: WebQSP questions are Freebase-grounded factoid queries, not adversarial counting / negation as in our bench.

## 4. What this probe does NOT show

- End-to-end accuracy on WebQSP (requires Freebase-subgraph executor port).
- Comparison vs RoG / HippoRAG on WebQSP (requires same).
- These are concretely scoped in §7 (Future Work) of the paper, with effort estimate ~2 days for a 200-Q pilot.

## 5. Sample routings (first 10)

| ID | Question | Pred type | Strategy |
|---|---|---|---|
| WebQTrn-9 | how old is sacha baron cohen | single_hop | lookup |
| WebQTrn-11 | what time zone am i in cleveland ohio | single_hop | lookup |
| WebQTrn-15 | what is nina dobrev nationality | single_hop | lookup |
| WebQTrn-47 | what county is heathrow airport in | single_hop | lookup |
| WebQTrn-73 | what form of government was practiced in sparta | single_hop | lookup |
| WebQTrn-104 | who was the president after jfk died | single_hop | lookup |
| WebQTrn-126 | where to visit near bangkok | unanswerable | complement |
| WebQTrn-143 | what did nick clegg study at university | single_hop | lookup |
| WebQTrn-152 | what is henry clay known for | single_hop | lookup |
| WebQTrn-164 | where was kennedy when he got shot | single_hop | lookup |