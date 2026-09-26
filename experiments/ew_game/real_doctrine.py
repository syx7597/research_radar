"""
Step 1: run the cognitive-EW learner on the REAL doctrine graph
===============================================================
Loads data/ew/doctrine_map.json (17 jamming techniques x 15 ECCM, real `defeated_by`
relations) and runs the SAME agents from cognitive.py on it — the data-agnostic claim.

The real defeat graph genuinely has shared structure (8/15 ECCM defeat >1 technique),
which is what the doctrine-guided inference exploits. Base effectiveness is NOT in the
(qualitative) doctrine, so we assign it by two transparent schemes and report both:
  - 'doctrine': monotone in counterability (more-counterable = higher power = higher
    base; un-counterable fallbacks lower) — the real EW tradeoff.
  - 'random'  : seeded-random base in [0.5,0.95] — unbiased; if the advantage holds
    here it is NOT an artifact of the base choice.

Run:  python experiments/ew_game/real_doctrine.py
"""
import importlib.util, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("cog", __file__.replace("real_doctrine.py", "cognitive.py"))
cog = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cog)


def load_real_doctrine(base_scheme="doctrine", seed=0):
    dm = json.load(open(ROOT / "data" / "ew" / "doctrine_map.json", encoding="utf-8"))
    J = dm["jamming_techniques"]
    rng = random.Random(seed)
    eccm = sorted({e for j in J for e in j.get("defeated_by", [])})
    doctrine = {}
    for j in J:
        df = set(j.get("defeated_by", []))
        if base_scheme == "random":
            base = round(rng.uniform(0.5, 0.95), 3)
        else:  # 'doctrine': powerful techniques are more counterable
            base = round(min(0.95, 0.55 + 0.08 * len(df)), 3)
        doctrine[j["id"]] = (base, df)
    return doctrine, eccm


def stationary_mean(N=2000, T=60, seed=0):
    rng = random.Random(seed)
    names = list(cog.make_agents(rng).keys())
    cum = {a: 0.0 for a in names}
    for _ in range(N):
        rs = random.Random(rng.random()); A = cog.gen_radar(rs)
        ags = cog.make_agents(rng)
        for a, ag in ags.items():
            for r in range(T):
                Tk = ag.select(); d = cog.defeated(Tk, A); ag.update(Tk, d)
                cum[a] += cog.reward(Tk, A)
    return {a: cum[a] / (N * T) for a in names}


def main():
    print("REAL doctrine_map.json — 17 techniques, defeat graph is real; base assigned.\n")
    for scheme in ["doctrine", "random"]:
        doctrine, eccm = load_real_doctrine(scheme)
        print(f"================  base scheme = {scheme}  ================")
        print(f"  techniques={len(doctrine)}  ECCM={len(eccm)}")
        for prior in [0.2, 0.3, 0.4, 0.5]:
            cog.set_doctrine(doctrine, eccm, prior=prior)
            r = stationary_mean()
            bm = max(r["blind-UCB"], r["blind-Thompson"])
            print(f"  prior={prior:.2f}  static={r['static']:.3f}  "
                  f"blindUCB={r['blind-UCB']:.3f}  blindThom={r['blind-Thompson']:.3f}  "
                  f"dGreedy={r['doctrine-greedy']:.3f}  dThom={r['doctrine-Thompson']:.3f}  "
                  f"| dThom-blind={r['doctrine-Thompson']-bm:+.3f}")
        print()
    # switching on real doctrine (doctrine base scheme, prior 0.3)
    doctrine, eccm = load_real_doctrine("doctrine")
    cog.set_doctrine(doctrine, eccm, prior=0.3)
    print("================  SWITCHING radar on real doctrine (prior=0.3)  ================")
    cog.run_switching(2000, 60)


if __name__ == "__main__":
    main()
