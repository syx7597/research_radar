"""
Step 3 — CUSUM change detection vs hand-tuned forgetting (switching radar)
=========================================================================
The switching-radar learner previously used a fixed `decay` (relax belief toward the
prior every round). That is crude: it must be TUNED to the switch rate, and it forgets
even when nothing changed (hurting the stationary case). Here a CUSUM detector resets
the belief only when a switch is actually detected.

Claim under test: doctrine-CUSUM matches/﻿beats the fixed-decay learner that was TUNED
to one period (=20), across DIFFERENT (unknown) periods, AND does not hurt the
stationary case. Baselines: static, blind-Thompson.

Run:  python experiments/ew_game/cusum.py
"""
import importlib.util, random
from statistics import mean

_spec = importlib.util.spec_from_file_location("cog", __file__.replace("cusum.py", "cognitive.py"))
cog = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cog)


def gen_doctrine(n, m, spread=0.40, n_safe=2, seed=0):
    rng = random.Random(seed); eccm = [f"E{i}" for i in range(m)]; doc = {}
    for i in range(n_safe):
        doc[f"S{i}"] = (0.55, set())
    for i in range(n - n_safe):
        k = rng.randint(1, 3); df = set(rng.sample(eccm, k))
        base = min(0.97, max(0.5, 0.60 + spread * (k / 3) + rng.uniform(-0.04, 0.04)))
        doc[f"T{i}"] = (round(base, 3), df)
    return doc, eccm


def run_period(doc, eccm, prior, period, T, N, seed):
    cog.set_doctrine(doc, eccm, prior)
    rng = random.Random(seed)
    def build():
        return {
            "static": cog.StaticDoctrine(),
            "blind-Thompson": cog.BlindThompson(rng, forget=0.10),
            "dThom-decay(tuned@20)": cog.DoctrineThompson(rng, decay=0.12),
            "dThom-nodecay": cog.DoctrineThompson(rng, decay=0.0),
            "dCUSUM": cog.DoctrineCUSUM(rng),
        }
    names = list(build().keys()); cum = {a: 0.0 for a in names}
    for _ in range(N):
        nb = T // period + 2
        blocks = [{e for e in eccm if rng.random() < prior} for _ in range(nb)]
        As = [blocks[r // period] for r in range(T)]
        for a, ag in build().items():
            for r in range(T):
                A = As[r]; Tk = ag.select(); d = cog.defeated(Tk, A); ag.update(Tk, d)
                cum[a] += cog.reward(Tk, A)
    return {a: cum[a] / (N * T) for a in names}


def main():
    doc, eccm = gen_doctrine(n=12, m=10, seed=5)
    prior, T, N = 0.25, 80, 2000
    BIG = 10 ** 9
    periods = [("stationary", BIG), ("switch@40", 40), ("switch@20", 20), ("switch@10", 10)]
    names = ["static", "blind-Thompson", "dThom-decay(tuned@20)", "dThom-nodecay", "dCUSUM"]
    print(f"doctrine n=12 m=10  prior={prior}  horizon={T}  (mean suppression)\n")
    print(f"  {'scenario':<14}" + "".join(f"{a.split('(')[0]:>16}" for a in names))
    rows = {}
    for label, p in periods:
        r = run_period(doc, eccm, prior, p, T, N, seed=3)
        rows[label] = r
        print(f"  {label:<14}" + "".join(f"{r[a]:>16.3f}" for a in names))
    print("\n  Reading:")
    print("  - stationary: dCUSUM should ~= dThom-nodecay (rarely fires) and >> fixed-decay")
    print("                (fixed decay needlessly forgets -> worse when nothing switches)")
    print("  - switch@20 : fixed-decay was TUNED here; dCUSUM should match it")
    print("  - switch@10/40: fixed-decay is mistuned; dCUSUM should be robust (no tuning)")
    # headline robustness: average rank/score of CUSUM vs tuned-decay across periods
    dc = mean(rows[l]["dCUSUM"] for l, _ in periods)
    dd = mean(rows[l]["dThom-decay(tuned@20)"] for l, _ in periods)
    nd = mean(rows[l]["dThom-nodecay"] for l, _ in periods)
    print(f"\n  avg over all scenarios:  dCUSUM={dc:.3f}  fixed-decay(tuned@20)={dd:.3f}  "
          f"no-decay={nd:.3f}")


if __name__ == "__main__":
    main()
