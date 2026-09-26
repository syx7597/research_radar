"""
Step 2 — theory: analytical predictions + simulation verification
=================================================================
Two falsifiable predictions about the doctrine-guided learner, verified on
controllable synthetic doctrines (n techniques, m ECCM, density rho, horizon T).

P1  SAMPLE COMPLEXITY.
    Rewards are deterministic given the hidden capability A. A blind learner must
    sample each of the n arms to know it -> convergence Theta(n). The doctrine-guided
    learner needs only to rule out the ECCM that block high-value techniques; each
    failure reveals >=1 present ECCM and each success rules out a whole defeat-set,
    so it converges in O(|A|) = O(m*rho) trials, INDEPENDENT of n.
    => Prediction: as n grows (m fixed), blind convergence grows ~linearly, guided
       stays flat (set by m*rho).

P2  REGIME BOUNDARY (learning vs static safe-fallback).
    Static guarantees v_safe (best un-counterable technique). Learning can discover a
    radar-specific safe technique worth v_disc(A) >= v_safe. A technique with k
    defeating ECCM is safe w.p. (1-rho)^k, so the expected gain E[v_disc]-v_safe
    SHRINKS as rho rises and the exploration cost is amortised as T rises.
    => Prediction: Delta(learn - static) decreases in rho, increases in T; there is a
       crossover density rho* above which static wins.

Run:  python experiments/ew_game/theory.py
"""
import importlib.util, random
from statistics import mean

_spec = importlib.util.spec_from_file_location("cog", __file__.replace("theory.py", "cognitive.py"))
cog = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cog)


def gen_doctrine(n, m, spread=0.40, n_safe=2, seed=0):
    rng = random.Random(seed)
    eccm = [f"E{i}" for i in range(m)]
    doc = {}
    for i in range(n_safe):                              # un-counterable low-base fallbacks
        doc[f"S{i}"] = (0.55, set())
    for i in range(n - n_safe):                          # counterable: more counterable = higher base
        k = rng.randint(1, 3)
        df = set(rng.sample(eccm, k))
        base = min(0.97, max(0.5, 0.60 + spread * (k / 3) + rng.uniform(-0.04, 0.04)))
        doc[f"T{i}"] = (round(base, 3), df)
    return doc, eccm


def avg_curve(agent_name, doc, eccm, prior, T, N, seed):
    cog.set_doctrine(doc, eccm, prior)
    rng = random.Random(seed)
    curve = [0.0] * T; opt = 0.0
    for _ in range(N):
        rs = random.Random(rng.random()); A = cog.gen_radar(rs)
        opt += cog.best_reward(A)
        ag = cog.make_agents(rng)[agent_name]
        for r in range(T):
            Tk = ag.select(); d = cog.defeated(Tk, A); ag.update(Tk, d)
            curve[r] += cog.reward(Tk, A)
    return [c / N for c in curve], opt / N


def conv_round(curve, opt, frac=0.90):
    thr = frac * opt
    for r, v in enumerate(curve):
        if v >= thr:
            return r + 1
    return len(curve)


def mean_reward(agent_name, doc, eccm, prior, T, N, seed):
    c, _ = avg_curve(agent_name, doc, eccm, prior, T, N, seed)
    return mean(c)


def total_regret(agent_name, doc, eccm, prior, T, N, seed):
    cog.set_doctrine(doc, eccm, prior)
    rng = random.Random(seed); reg = 0.0
    for _ in range(N):
        rs = random.Random(rng.random()); A = cog.gen_radar(rs)
        opt = cog.best_reward(A)
        ag = cog.make_agents(rng)[agent_name]
        for r in range(T):
            Tk = ag.select(); d = cog.defeated(Tk, A); ag.update(Tk, d)
            reg += opt - cog.reward(Tk, A)
    return reg / N


def part1():
    print("=" * 64)
    print("P1  SAMPLE COMPLEXITY — total regret (horizon T=60) vs #techniques n  (m=10, rho=0.3)")
    print("    prediction: blind regret ~ linear in n (must explore each arm);")
    print("                doctrine-guided regret ~ flat in n (bounded by m*rho)")
    print("=" * 64)
    m, prior, T, N = 10, 0.3, 60, 2000
    print(f"  {'n':>4}{'blind-Thompson':>17}{'doctrine-Thompson':>20}{'ratio b/g':>11}")
    for n in [4, 8, 12, 16, 24, 32]:
        doc, eccm = gen_doctrine(n, m, seed=n)
        rb = total_regret("blind-Thompson", doc, eccm, prior, T, N, seed=1)
        rg = total_regret("doctrine-Thompson", doc, eccm, prior, T, N, seed=1)
        print(f"  {n:>4}{rb:>17.2f}{rg:>20.2f}{rb/max(rg,1e-9):>11.1f}")
    print("    blind regret climbs with n; guided stays ~flat -> the structural advantage")


def part2():
    print("\n" + "=" * 64)
    print("P2  REGIME BOUNDARY — Delta(doctrine-Thompson - static) vs density rho and horizon T")
    print("    prediction: Delta decreases in rho, increases in T; crossover rho* exists")
    print("=" * 64)
    doc, eccm = gen_doctrine(n=12, m=10, seed=5)
    Ts = [20, 60, 150]
    print(f"  {'rho':>6}" + "".join(f"{'T='+str(t):>12}" for t in Ts))
    for rho in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
        row = []
        for T in Ts:
            g = mean_reward("doctrine-Thompson", doc, eccm, rho, T, 1500, seed=2)
            s = mean_reward("static", doc, eccm, rho, T, 1500, seed=2)
            row.append(g - s)
        print(f"  {rho:>6.2f}" + "".join(f"{d:>+12.3f}" for d in row))
    print("    (Delta>0: learning beats static; the sign flip across rho is the boundary)")


if __name__ == "__main__":
    part1()
    part2()
