# EW countermeasure allocation as a Stackelberg game — validation

**Status: first validation PASSED.** Isolated synthetic testbed (`experiments/ew_game/`,
no real KG data). The contribution under test is the *algorithm*; parameters are
synthetic-but-physically-consistent and randomly generated. Data-agnostic: real EOB
parameters plug into the same model later.

## Problem
A modern (cognitive) radar can **reactively activate ECCM** to defeat the jamming
technique it is being hit with. The current advisor / greedy set-cover commits to
techniques that are optimal against the radar's *known/default* config; an adaptive
adversary then activates ECCM and defeats them. We formulate countermeasure
allocation as a Stackelberg game (defender commits, adversary best-responds) and
compute the **robust** allocation (max worst-case priority-weighted suppression).

Baselines are NOT strawmen: `static-opt` is the *globally optimal* allocation
against the known config (as smart as the current advisor); it just doesn't
anticipate reactive ECCM.

## Result (`sim.py`, 300 random scenarios, 4 threats / 2 jammers)

| strategy | REALIZED (adversary adapts) | nominal (adversary static) |
|---|---:|---:|
| greedy | 0.372 | 0.514 |
| static-opt | 0.381 | 0.527 |
| **robust (game)** | **0.486** | 0.486 |

- robust beats static-opt by **+0.105**, greedy by **+0.114** under adaptation (~28% rel).
- **robust ≥ static-opt in 300/300 scenarios; strictly > in 196.**
- cost of robustness: ~−0.04 vs static-opt when the adversary does NOT adapt.
- robust is indifferent to the adversary's move (realized = nominal) — the minimax property.

## Sensitivity (`sweep.py`) — confirms it is not an artifact

| config | greedy | static | robust | rob−stat |
|---|---:|---:|---:|---:|
| T3 J2 pen0.1 | 0.337 | 0.346 | 0.445 | +0.100 |
| T4 J2 pen0.1 | 0.338 | 0.349 | 0.450 | +0.102 |
| T5 J2 pen0.1 | 0.304 | 0.315 | 0.416 | +0.101 |
| T4 J2 pen0.3 | 0.368 | 0.380 | 0.451 | +0.071 |
| T4 J2 pen0.5 | 0.399 | 0.411 | 0.451 | +0.040 |
| T4 J2 pen0.8 | 0.452 | 0.465 | 0.475 | +0.010 |

Stable across sizes; the advantage **scales with the strength of the ECCM↔jamming
interaction** (large when ECCM strongly defeats jamming, →0 when it barely does).
This graceful degradation is what a real effect looks like.

## Honest caveats / what is NOT yet done
- **The solver is brute-force enumeration** — it does not scale (T=6/J=3 already
  blows up). The actual algorithmic contribution to build is a **scalable Stackelberg
  solver** (MILP / double-oracle / column generation) with a quality/▲bound.
- Result magnitude depends on the doctrine model (technique table, J/S curve, defeat
  penalty); the sensitivity sweep shows the *qualitative* result is stable.
- Adversary model is per-threat independent ECCM activation; a richer game would add
  a global ECCM budget / joint response.

## Next (the real build)
1. Scalable solver for the robust allocation (the algorithmic core).
2. Richer adversary (ECCM budget) → fuller game; bound the price of robustness.
3. Data-agnostic adapter: map a real EOB (threats + available jammers) into the model.

---

# Stage 1 — exact scalable solver (`solver.py`)

The robust allocation (separable adversary) as a MILP (HiGHS via scipy), technique
pre-collapsed, with load-indicator linearization for the power-dilution coupling.

- **Correctness: EXACT.** 200 random instances vs brute force → 0 mismatches,
  max |brute−MILP| = 1.8e-15 (machine epsilon).
- **Scaling:** solves instances brute force cannot touch:
  | M,K | brute-force search space | MILP value | MILP time |
  |---|---:|---:|---:|
  | 12,4 | ~3e9 | 10.2 | 0.1s |
  | 16,5 | ~1.6e19 | 35.5 | 4.1s |
  | 20,5 | ~2.8e24 | 46.3 | 2.1s |

  → exact optimum at 20 threats where enumeration is 10^24 assignments.

# Stage 2 — REMOVED (randomization gave no improvement)

A double-oracle solver for a budgeted-adversary game was built and verified exact,
but the budgeted game turned out to have a **pure** optimum (= the Stage-1 MILP plan)
because the adversary's damage is additive — so mixed strategies / randomization
bought **nothing**. It was not a contribution and has been removed (2026-06-23).
Mixing could only matter under a *non-separable* adversary; not pursued.

## Status
The only verified-useful artifact here is the **Stage-1 exact MILP** (robust one-shot
allocation), which beats greedy/static under an adaptive adversary. Honestly, that is
a *formulation + standard-solver* contribution (moderate), not a novel algorithm — it
now serves as the **one-shot/static baseline** for a stronger, genuinely algorithmic
direction: cognitive EW as **online/sequential learning** against an adaptive radar
(see the project discussion / next experiment).
