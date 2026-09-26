"""Sensitivity sweep for the EW game validation (tractable configs only).
Shows the robust>baseline result is not an artifact of one configuration.
Run: python experiments/ew_game/sweep.py"""
import importlib.util, random, math
from statistics import mean

spec = importlib.util.spec_from_file_location("sim", "experiments/ew_game/sim.py")
sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)


def sweep(n_threat, n_jammer, penalty, n_scen=150, seed=1):
    def eff2(j, t, tech, adv, load):
        if t["band"] not in j["bands"] or tech not in j["tech"]:
            return 0.0
        base, defeated = sim.TECH[tech]
        js = (j["power"] - 10 * math.log10(load)) - t["erp"]
        val = sim.sigmoid(0.25 * js) * base
        if defeated & adv:
            val *= penalty
        return val
    orig = sim.eff; sim.eff = eff2
    rng = random.Random(seed); R = {"greedy": [], "static-opt": [], "robust": []}
    for _ in range(n_scen):
        th, ja = sim.gen_scenario(rng, n_threat, n_jammer)
        wmax = sum(t["prio"] for t in th) or 1
        ag = sim.greedy_assignment(th, ja)
        ast = sim.best_assignment(th, ja, lambda a: sim.value(a, th, ja, "default"))
        ar = sim.best_assignment(th, ja, lambda a: sim.value(a, th, ja, "best"))
        for nm, a in [("greedy", ag), ("static-opt", ast), ("robust", ar)]:
            R[nm].append(sim.value(a, th, ja, "best") / wmax)
    sim.eff = orig
    return {k: mean(v) for k, v in R.items()}


print(f"{'config':<28}{'greedy':>8}{'static':>8}{'robust':>8}{'rob-stat':>10}", flush=True)
for nt, nj, pen in [(3, 2, 0.1), (4, 2, 0.1), (5, 2, 0.1), (4, 2, 0.3), (4, 2, 0.5), (4, 2, 0.8)]:
    r = sweep(nt, nj, pen)
    print(f"  T={nt} J={nj} pen={pen:<13}{r['greedy']:>8.3f}{r['static-opt']:>8.3f}"
          f"{r['robust']:>8.3f}{r['robust']-r['static-opt']:>+10.3f}", flush=True)
