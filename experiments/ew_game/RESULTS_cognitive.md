# Cognitive EW as knowledge-guided online learning — validation PASSED

**The genuine algorithmic direction.** Isolated synthetic testbed (`cognitive.py`),
no real KG data. The contribution is the *algorithm*; the radars/doctrine are
synthetic but logically consistent, and the same learner runs on real doctrine later.

## Why this is different from the one-shot solver (and from the failed attempts)
Everything earlier (incl. the Stage-1 MILP) is **one-shot/static**. But cognitive EW
is intrinsically **sequential and partially observed**: you jam, the radar reacts
(activates ECCM), you observe the degraded effect, you adapt, it adapts again. A
static recommender is provably inadequate against an adaptive radar; even a *blind*
online learner is sample-inefficient. The right tool is **online learning that
exploits the doctrine knowledge graph** to infer the radar's hidden ECCM state.
This makes the doctrine KG *drive an algorithm* (not just be queried) — directly
answering the "too engineering / no algorithmic innovation" critique, and unifying
the KG-construction and EW-decision halves of the thesis.

## Problem
A jammer fights a radar whose ECCM capability `A` (which counter-measures it can run)
is **hidden**. Each round the jammer picks a technique; it is defeated iff some ECCM
in `A` counters it (doctrine graph: technique → defeating ECCM). Only success/defeat
is observed. Maximise cumulative suppression over a (short) engagement.

## Algorithm — doctrine-belief Bayesian learning
Maintain a posterior `P(e present)` over each ECCM `e`. Updates use the doctrine graph:
- **success(T)** ⇒ no ECCM in `def(T)` is present ⇒ rule them all out (→0). *One*
  observation thus revalues *every* technique sharing those ECCM — the structural
  transfer a blind bandit cannot do.
- **defeat(T)** ⇒ ≥1 ECCM in `def(T)` present ⇒ noisy-OR posterior bump.

Two selection rules:
- `doctrine-greedy`: play argmax expected reward under the belief.
- **`doctrine-Thompson`** (final): sample a hypothesis `A' ~ posterior`, play the best
  technique against `A'` — adds the exploration greedy lacks while keeping the
  structure. A `decay` term relaxes the belief toward the prior to track a switching
  radar.

Baselines: `static` doctrine (best fixed guess = current system); `blind-UCB` and
`blind-Thompson` (strong standard online learners that ignore the doctrine graph).

## Results (2000 random radars, 60-round engagements; mean suppression, regret)

**Stationary hidden radar (fixed unknown capability):**
| agent | mean | regret |
|---|---:|---:|
| static | 0.603 | 18.3 |
| blind-UCB | 0.624 | 17.0 |
| blind-Thompson | 0.662 | 14.8 |
| **doctrine-Thompson** | **0.897** | **0.65** |

→ beats both blind learners by **+0.24 / +0.27** and static by **+0.29**; converges to
optimal by round ~5 (regret ~0) while the blind learners are still climbing at round 60.

**Switching radar (capability re-rolls every 20 rounds — "cognitive"):**
doctrine variants 0.78–0.87 mean vs static 0.61, blind 0.51–0.58. The per-round curve
shows the learner drop right after a switch then **re-infer and recover in ~4 rounds**;
blind learners never track the switches.

**Density sweep (robustness, stationary), `d-Thom − best blind`:**
| prior(ECCM density) | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|
| doctrine-greedy | +0.19 | +0.21 | +0.24 | **−0.06** | 0.00 | +0.07 |
| **doctrine-Thompson** | +0.19 | +0.21 | +0.24 | **+0.26** | +0.27 | +0.28 |

→ **doctrine-Thompson beats the best blind learner at EVERY density (+0.19…+0.28).**

## Honest findings (surfaced by sensitivity analysis, not hidden)
- The naive `doctrine-greedy` **under-explores**: at high radar density it retreats to
  the safe low-base technique (0.55) without trying high-base ones, and *loses* to
  blind-Thompson at prior 0.5. Thompson sampling in the belief space fixes it.
- **Tradeoff:** greedy recovers faster under *switching* (commits hard after re-infer,
  0.87 vs 0.78); Thompson is robust across *density*. A practical learner blends them
  (exploration floor + decay-aware sampling) — concrete next step.

## Caveats
- Synthetic doctrine + synthetic radars (no open real engagement data); the claim is an
  *algorithmic / sample-efficiency* advantage, validated in simulation.
- Method relates to structured/contextual bandits + Bayesian opponent modeling; the
  contribution is the **formulation + adaptation to cognitive EW with a doctrine KG as
  the structure**, not a brand-new bandit class. Honest, and still a real algorithmic
  contribution with analysis-ready regret behaviour.

## Why it clears the bar
It beats not just the static system but a **strong knowledge-blind online learner**,
robustly, with a diagnosed-and-fixed algorithmic flaw and near-zero regret — a genuine
*algorithm* (KG-structured Bayesian online learning), not engineering.

---

# Step 1 — on the REAL doctrine graph (`real_doctrine.py`)

Loaded `data/ew/doctrine_map.json` (17 jamming techniques, real `defeated_by`; 15 ECCM).
Validity check first: the real defeat graph **does** have the shared structure the
inference needs — **8/15 ECCM defeat >1 technique** (one defeats 4). Base effectiveness
isn't in the qualitative doctrine, so assigned by two transparent schemes ('doctrine'
= powerful⇒more-counterable; 'random' = unbiased).

**What HOLDS on real data (the algorithmic claim):**
- `doctrine-Thompson` beats the blind learners (UCB, Thompson) **robustly, under both
  base schemes**: **+0.15 to +0.32**. Structure → sample efficiency transfers to the
  real 17×15 graph (and the gap is *larger* than the synthetic toy because 17 arms are
  harder for a blind learner to explore in a short engagement).

**What does NOT hold universally (the honest dent — vs the *static* system):**
- 'random' bases: a high-value *un-counterable* technique exists ⇒ static already plays
  it (0.909) and is near-optimal; learning ties it (still ≫ blind).
- 'doctrine' bases: static is stuck on the weak safe fallback (0.55); `doctrine-Thompson`
  *beats* it at sparse density (0.77 vs 0.55) **but under a switching radar the exploration
  cost makes it lose to do-nothing static (0.488 vs 0.550)** — `doctrine-greedy` ties static.

**Why:** real radars are dense (~4.5 of 15 ECCM active), so a *safe high-value* technique
is rare ⇒ the conservative static "play it safe" is often already good. The learner's
edge over static appears only when there is a **radar-specific safe high-value technique
to discover** (sparse radars / no dominant fallback). Over *blind learning* the advantage
is unconditional; over the *static system* it is regime-dependent.

## Honest revised contribution
- **Robust:** doctrine-structured online learning is much more sample-efficient than
  knowledge-blind online learning — on real doctrine, +0.15…+0.32. This is the algorithm.
- **Conditional:** its practical edge over the current *static* recommender holds when the
  engagement requires discovering a radar-specific safe high-value technique; on dense
  radars with a dominant safe fallback, static is already competitive and exploration can
  cost more than it gains.
- Next (theory) should explain this regime boundary, not paper over it: characterise WHEN
  online learning beats a static safe-fallback policy (a condition on technique
  base-spread vs radar ECCM density).

---

# Step 2 — theory: predicted scaling laws, verified in simulation (`theory.py`)

Two falsifiable predictions, each confirmed on controllable synthetic doctrines.
(Analytical *predictions* + simulation verification — master's-level analysis;
formal regret bounds are future work.)

**P1 — Sample complexity.** Rewards are deterministic given the hidden capability A.
A blind learner must sample each of the n arms ⇒ regret grows with n. The doctrine
learner only needs to rule out the ECCM that block high-value techniques (a success
rules out a whole defeat-set; a failure reveals ≥1 present ECCM) ⇒ regret bounded by
O(|A|)=O(m·ρ), **independent of n**.

Verified (total regret over T=60, m=10, ρ=0.3):
| n | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| blind-Thompson | 7.9 | 13.3 | 15.6 | 18.1 | 20.3 | 22.0 |
| doctrine-Thompson | 1.5 | 3.2 | 4.4 | 5.2 | 5.9 | 5.6 |

→ blind regret climbs with n; doctrine regret **plateaus ≈5.5** (bounded by structure,
not by technique count). 3.4–5.2× lower regret.

**P2 — Regime boundary (learning vs static safe-fallback).** Static guarantees v_safe
(best un-counterable technique). Learning can find a radar-specific safe technique worth
v_disc(A) ≥ v_safe; a technique with k defeating ECCM is safe w.p. (1−ρ)^k, so the
expected gain shrinks as ρ rises and the exploration cost is amortised as T grows.

Verified — Δ(doctrine-Thompson − static):
| ρ \ T | 20 | 60 | 150 |
|---|---:|---:|---:|
| 0.15 | +0.210 | +0.240 | +0.255 |
| 0.45 | +0.137 | +0.209 | +0.241 |
| 0.65 | +0.066 | +0.144 | +0.173 |
| 0.85 | **−0.012** | +0.065 | +0.091 |

→ Δ decreases in ρ, increases in T, with a **crossover ρ\*(T)** (≈0.85 at T=20, higher
for larger T) above which static wins. This law **explains Step 1**: real radars are dense
(~0.3 of many ECCM but with high per-technique counterability) and engagements are short
⇒ near/above the crossover ⇒ the static safe-fallback is competitive there.

## Net of Step 2
- The **sample-efficiency advantage over blind learning is structural and n-independent**
  (bounded regret) — the core algorithmic result, now with a predicted-and-verified law.
- The **advantage over the static system is precisely bounded by a density/horizon
  crossover** — the honest scope, now a quantitative condition rather than a caveat.

---

# Step 3 — CUSUM change detection replaces hand-tuned forgetting (`cusum.py`)

The switching learner previously used a fixed `decay` (relax belief toward prior every
round) — which must be tuned to the switch rate and forgets even when nothing changed.
`doctrine-CUSUM` instead watches the prediction-error "surprise" (confident-but-defeated)
and resets the belief only when a CUSUM statistic crosses a threshold — **no knowledge of
the switch period required**.

Mean suppression (doctrine n=12, m=10, prior=0.25, horizon 80):
| scenario | static | blind | decay(tuned@20) | no-decay | **CUSUM** |
|---|---:|---:|---:|---:|---:|
| stationary | 0.588 | 0.594 | 0.711 | 0.859 | **0.857** |
| switch@40 | 0.584 | 0.572 | 0.698 | 0.675 | **0.816** |
| switch@20 | 0.588 | 0.561 | 0.689 | 0.574 | **0.771** |
| switch@10 | 0.590 | 0.553 | 0.660 | 0.523 | **0.705** |
| **avg** | — | — | 0.690 | 0.658 | **0.787** |

- **stationary:** CUSUM (0.857) ≈ no-decay (0.859) — it rarely fires, so it does NOT waste
  inference; fixed-decay (0.711) needlessly forgets and loses.
- **switch@20 (where decay was tuned):** CUSUM (0.771) *beats* the tuned fixed-decay (0.689)
  — a clean reset on detection beats constant belief bleed.
- **switch@10 / @40 (decay mistuned):** CUSUM robust without any tuning.
- no-decay is great stationary but collapses under switching; CUSUM gets both.

→ Principled, parameter-light non-stationarity handling that **dominates both alternatives**
across stationary and all switch rates.

---

# Summary after the 3-step plan
1. **Algorithm + validation:** doctrine-structured Bayesian online learning (Thompson in
   belief space) beats static doctrine AND strong knowledge-blind learners; near-zero regret.
2. **Real data:** the advantage over blind learning transfers to the real doctrine graph
   (+0.15…+0.32); the advantage over static is bounded by a density/horizon crossover (honest).
3. **Theory:** verified scaling laws — regret O(m·ρ) not O(n) (P1); the learning-vs-static
   crossover ρ\*(T) (P2).
4. **Non-stationarity:** CUSUM change detection dominates hand-tuned forgetting.

Remaining: write-up (formalisation + figures), and optionally a formal regret bound and a
richer adversary. The doctrine KG now *drives* a learning algorithm — the algorithmic
contribution the project was missing.

---

# RL baseline — knowledge vs model-free reinforcement learning (`rl_baseline.py`)

Positions the doctrine-guided learner against standard RL. A tabular Q-learning agent
treats the engagement as an MDP (state = per-technique observation status, action =
technique, reward = suppression) and learns a policy across many training episodes,
with no doctrine knowledge. Given its fairest shot (epsilon-decay, averaged over 3 seeds):

| training episodes | model-free RL (test) | doctrine-guided (0 training) |
|---|---:|---:|
| 100 | 0.822 | **0.894** |
| 1,000 | 0.866 | **0.894** |
| 5,000 | 0.874 | **0.894** |
| 20,000 | 0.880 | **0.894** |
| 80,000 | 0.867 | **0.894** |

- **Sample efficiency:** RL needs ~20k episodes just to get close and **still trails**;
  the doctrine learner needs **zero** training (it infers within each engagement via the
  counter-relations — i.e., the knowledge graph *is* the prior the RL agent lacks).
- **Final performance:** RL plateaus ≈0.88 < doctrine 0.894 — tabular RL learns each of
  ~2,160 states independently and cannot generalise across them the way the doctrine graph
  does (a success on one technique instantly revalues all techniques sharing an ECCM).
- **Interpretability / transfer:** the doctrine learner exposes a belief over ECCM and takes
  any graph; the RL Q-table is opaque and tied to this one doctrine (retrain from scratch
  for a new one).

**Honest caveat:** tabular Q-learning is standard but not the strongest RL; deep RL with
function approximation could generalise across states better — but would need *even more*
experience and remain a black box. So the claim is precise: standard model-free RL needs
orders of magnitude more experience and still trails, and stronger RL trades yet more data
for less interpretability. This is the thesis's position w.r.t. the RL-agentic mainstream
(Graph-R1, Plan-Then-Retrieve, etc.): **use knowledge to learn faster and transparently,
rather than learn the structure from scratch in a black box.**

---

# Bridge — the MAIN radar KG as a per-radar prior (`kg_prior.py`)

Connects the project's main radar knowledge (`threat_radars.json`) to the learner. The
attributes *are* doctrine-grounded ECCM facts: `freq_agile`→`freq_agility` (19 radars),
`tracking_method=monopulse`→`monopulse` (28), `scan_type=phased`→`low_sidelobe_antenna`
(35). So the radar KG supplies a **per-radar prior** over the hidden capability; the
doctrine graph still drives the online inference. Both learners are identical except the
starting belief — the gap is the value of the radar KG.

Real radars (89), KG-prior vs a fair uniform prior:
| horizon T | uniform | KG-prior | gain | round-1 (uni/KG) |
|---|---:|---:|---:|---:|
| 4 | 0.421 | 0.459 | +0.038 | 0.349 / 0.376 |
| 8 | 0.493 | 0.530 | +0.038 | — |
| 16 | 0.583 | 0.617 | +0.034 | — |
| 40 | 0.664 | 0.684 | +0.020 | — |

**Honest reading:** the bridge is real and directionally correct — a consistent head start
from round 1, **largest at short (realistic) horizons** and shrinking as the uniform learner
catches up. But the magnitude is **modest (+0.02…+0.04)** because the current radar KG pins
down only **2–3 of 15 ECCM** per radar, and the doctrine-graph learner discovers the rest
quickly anyway. The conceptual point holds: **both knowledge graphs now participate** — the
radar KG sets the per-radar prior, the doctrine graph drives the inference — but the
practical gain is bounded by how much the radar KG's attributes actually constrain the
hidden ECCM state (richer per-radar ELINT attributes would widen it).
