"""
EW countermeasure allocation as a Stackelberg game — synthetic validation
=========================================================================
ISOLATED from the real KG. All radar/jammer parameters are SYNTHETIC but
physically self-consistent. The contribution under test is the ALGORITHM; the
synthetic world is a controlled testbed. Same algorithm runs on real EOB data
later (data-agnostic).

Core question (validate before building): does a robust / game-theoretic jammer
allocation beat greedy and static-optimal allocation WHEN the adversary can
reactively activate ECCM — without rigging the generator?

Setup (one engagement):
  threats : band, ERP(dBW), priority(1-5),
            default_eccm  = ECCM currently active (KNOWN to the defender),
            eccm_capability ⊇ default = ECCM the radar CAN activate reactively.
  jammers : band coverage, power(dBW), technique repertoire.
  doctrine: each technique has base effectiveness + which ECCM DEFEAT it.

Effectiveness(jammer→threat via technique T, under adversary ECCM set a):
  0 if band/technique infeasible; else sigmoid(J/S) * base_eff(T), with J/S using
  the jammer's power DILUTED across the threats it is assigned to (burnthrough),
  and a 0.1x penalty if any ECCM in `a` defeats T.

Three defender strategies — all evaluated under the adversary's BEST RESPONSE:
  greedy     : per-threat best technique vs DEFAULT eccm (like the current advisor)
  static-opt : globally optimal assignment vs DEFAULT eccm
  robust     : assignment maximizing WORST-CASE value over adversary capability
The baselines are NOT strawmen: they already avoid default-defeated techniques;
the game's edge is anticipating reactive ECCM activation.

Run:  python experiments/ew_game/sim.py
"""
import random, math, itertools
from statistics import mean

BANDS = ["L", "S", "C", "X", "Ku"]
ECCM = ["freq_agility", "monopulse", "prf_jitter"]

# technique id -> (base effectiveness, ECCM that defeat it). doctrine-inspired:
# high-power techniques are strong but defeatable; robust techniques are weaker.
TECH = {
    "spot_noise":   (0.95, {"freq_agility"}),                 # strong, agility kills it
    "barrage_noise":(0.55, set()),                            # weak but un-defeatable noise
    "inverse_gain": (0.90, {"monopulse"}),                    # strong angle dec., monopulse kills
    "cross_eye":    (0.60, set()),                            # beats monopulse, modest
    "rgpo":         (0.85, {"prf_jitter"}),                   # strong range dec., prf jitter kills
    "barrage+rgpo": (0.50, set()),                            # conservative combo, robust
}
ALL_TECH = list(TECH)


def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))


def gen_scenario(rng, n_threat, n_jammer):
    threats = []
    for _ in range(n_threat):
        cap = set(rng.sample(ECCM, rng.randint(0, 3)))     # what it CAN activate
        default = set(c for c in cap if rng.random() < 0.5)  # what's already on (known)
        threats.append({
            "band": rng.choice(BANDS),
            "erp": rng.uniform(60, 90),                      # dBW
            "prio": rng.randint(1, 5),
            "default_eccm": default,
            "capability": cap,
        })
    jammers = []
    for _ in range(n_jammer):
        nb = rng.randint(2, 4)
        jammers.append({
            "bands": set(rng.sample(BANDS, nb)),
            "power": rng.uniform(70, 95),                    # dBW
            "tech": set(rng.sample(ALL_TECH, rng.randint(3, len(ALL_TECH)))),
        })
    return threats, jammers


def eff(jammer, threat, tech, adv_eccm, load):
    """Suppression in [0,1]. load = #threats this jammer is assigned to (dilution)."""
    if threat["band"] not in jammer["bands"]:
        return 0.0
    if tech not in jammer["tech"]:
        return 0.0
    base, defeated_by = TECH[tech]
    js = (jammer["power"] - 10 * math.log10(load)) - threat["erp"]   # dB, power split
    val = sigmoid(0.25 * js) * base
    if defeated_by & adv_eccm:
        val *= 0.1
    return val


# an assignment = tuple over threats of (jammer_idx, tech) or None
def options(threats, jammers):
    opt = []
    for t in threats:
        choices = [None]
        for ji, j in enumerate(jammers):
            for tech in j["tech"]:
                if t["band"] in j["bands"]:
                    choices.append((ji, tech))
        opt.append(choices)
    return opt


def jammer_load(assign):
    load = {}
    for a in assign:
        if a is not None:
            load[a[0]] = load.get(a[0], 0) + 1
    return load


def value(assign, threats, jammers, adv_mode):
    """Weighted suppression. adv_mode: 'default' | 'best' (adversary best-responds)."""
    load = jammer_load(assign)
    total = 0.0
    for t, a in zip(threats, assign):
        if a is None:
            continue
        ji, tech = a
        if adv_mode == "default":
            adv = t["default_eccm"]
        else:  # adversary activates whatever defeats the assigned technique, if it can
            adv = {e for e in t["capability"] if e in TECH[tech][1]} or t["default_eccm"]
        total += t["prio"] * eff(jammers[ji], t, tech, adv, load[ji])
    return total


def best_assignment(threats, jammers, objective):
    """Enumerate joint assignments; return the one maximizing `objective(assign)`."""
    opt = options(threats, jammers)
    best, best_v = None, -1.0
    for assign in itertools.product(*opt):
        v = objective(assign)
        if v > best_v:
            best, best_v = assign, v
    return best


def greedy_assignment(threats, jammers):
    """Per-threat (priority order) best technique vs DEFAULT eccm, with running load."""
    order = sorted(range(len(threats)), key=lambda i: -threats[i]["prio"])
    assign = [None] * len(threats)
    load = {}
    for i in order:
        t = threats[i]
        best, best_v = None, 0.0
        for ji, j in enumerate(jammers):
            for tech in j["tech"]:
                nl = load.get(ji, 0) + 1
                v = eff(j, t, tech, t["default_eccm"], nl)
                if v > best_v:
                    best, best_v = (ji, tech), v
        assign[i] = best
        if best is not None:
            load[best[0]] = load.get(best[0], 0) + 1
    return assign


def run(n_scen=300, n_threat=4, n_jammer=2, seed=0):
    rng = random.Random(seed)
    realized = {"greedy": [], "static-opt": [], "robust": []}
    nominal = {"greedy": [], "static-opt": [], "robust": []}   # if adversary does NOT adapt
    for _ in range(n_scen):
        threats, jammers = gen_scenario(rng, n_threat, n_jammer)
        wmax = sum(t["prio"] for t in threats) or 1

        a_greedy = greedy_assignment(threats, jammers)
        a_static = best_assignment(threats, jammers,
                                   lambda a: value(a, threats, jammers, "default"))
        a_robust = best_assignment(threats, jammers,
                                   lambda a: value(a, threats, jammers, "best"))

        for name, a in [("greedy", a_greedy), ("static-opt", a_static), ("robust", a_robust)]:
            realized[name].append(value(a, threats, jammers, "best") / wmax)
            nominal[name].append(value(a, threats, jammers, "default") / wmax)

    print(f"Scenarios: {n_scen}  ({n_threat} threats, {n_jammer} jammers)\n")
    print("Weighted suppression (fraction of max), mean over scenarios:")
    print(f"  {'strategy':<12}{'REALIZED (adv adapts)':>24}{'nominal (adv static)':>22}")
    for name in ("greedy", "static-opt", "robust"):
        print(f"  {name:<12}{mean(realized[name]):>24.3f}{mean(nominal[name]):>22.3f}")
    # head-to-head: how often robust >= static-opt on the realized metric
    wins = sum(r >= s - 1e-9 for r, s in zip(realized["robust"], realized["static-opt"]))
    strict = sum(r > s + 1e-9 for r, s in zip(realized["robust"], realized["static-opt"]))
    print(f"\n  robust >= static-opt (realized): {wins}/{n_scen};  strictly > : {strict}")
    g = mean(realized["robust"]) - mean(realized["greedy"])
    s = mean(realized["robust"]) - mean(realized["static-opt"])
    print(f"  robust - greedy   (realized): {g:+.3f}")
    print(f"  robust - static-opt(realized): {s:+.3f}")


if __name__ == "__main__":
    run()
