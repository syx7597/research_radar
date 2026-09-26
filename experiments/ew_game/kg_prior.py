"""
The bridge: main radar KG attributes -> a per-radar ECCM prior (real data)
=========================================================================
Connects the project's MAIN radar knowledge (threat_radars.json: tracking_method,
freq_agile, scan_type) to the cognitive learner. Those attributes *mean* the radar
carries specific ECCM (doctrine-grounded): freq_agile -> freq_agility; monopulse ->
monopulse ECCM; phased array -> low-sidelobe antenna. So the radar KG tells us part
of the hidden capability A, as a per-radar PRIOR — the rest the learner discovers
online (over the doctrine graph, as before).

Test: does the KG-informed prior beat the uniform "don't know" prior? Both learners
are identical except the starting belief; the gap = the value of the radar KG.

Run:  python experiments/ew_game/kg_prior.py
"""
import importlib.util, json, random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
def _load(name):
    s = importlib.util.spec_from_file_location(name, __file__.replace("kg_prior.py", name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cog = _load("cognitive"); real = _load("real_doctrine")

# doctrine-grounded map: radar KG attribute -> ECCM it implies (and how certain)
def kg_known_eccm(radar):
    certain = set()
    if radar.get("freq_agile"):
        certain.add("freq_agility")
    if radar.get("tracking_method") == "monopulse":
        certain.add("monopulse")
    phased = (radar.get("scan_type") == "phased")
    return certain, phased


def main():
    doctrine, eccm = real.load_real_doctrine("doctrine")
    cog.set_doctrine(doctrine, eccm, prior=0.3)
    radars = json.load(open(ROOT / "data" / "ew" / "threat_radars.json", encoding="utf-8"))["radars"]
    rho_hidden = 0.25           # density of the UNKNOWN (non-attribute) ECCM
    rng = random.Random(0)

    # --- empirical marginal P(e present), to give the UNIFORM prior a fair scalar ---
    present = 0; total = 0
    for radar in radars:
        certain, phased = kg_known_eccm(radar)
        for e in eccm:
            p = 1.0 if e in certain else (0.8 if (phased and e == "low_sidelobe_antenna") else rho_hidden)
            present += p; total += 1
    uni = present / total
    print(f"real radars={len(radars)}  ECCM={len(eccm)}  fair uniform prior={uni:.3f}\n")

    def run(prior_mode, T, N_per_radar=30, seed=1):
        r = random.Random(seed)
        curve = [0.0] * T; tot = 0.0; cnt = 0
        for radar in radars:
            certain, phased = kg_known_eccm(radar)
            for _ in range(N_per_radar):
                # true capability A: attribute-ECCM present + random hidden ECCM
                A = set(certain)
                if phased and r.random() < 0.8:
                    A.add("low_sidelobe_antenna")
                for e in eccm:
                    if e not in certain and not (phased and e == "low_sidelobe_antenna"):
                        if r.random() < rho_hidden:
                            A.add(e)
                opt = cog.best_reward(A)
                if prior_mode == "kg":
                    ip = {e: (0.97 if e in certain else
                              (0.8 if (phased and e == "low_sidelobe_antenna") else rho_hidden))
                          for e in eccm}
                else:
                    ip = {e: uni for e in eccm}
                ag = cog.DoctrineThompson(r, init_prior=ip)
                for t in range(T):
                    Tk = ag.select(); d = cog.defeated(Tk, A); ag.update(Tk, d)
                    rew = cog.reward(Tk, A); curve[t] += rew; tot += rew
                cnt += 1
        return [c / cnt for c in curve], tot / (cnt * T)

    print(f"  {'horizon T':<10}{'uniform-prior':>16}{'KG-prior':>12}{'  KG gain':>10}{'  round-1 (uni/KG)':>22}")
    for T in [4, 8, 16, 40]:
        cu, mu = run("uniform", T)
        ck, mk = run("kg", T)
        print(f"  T={T:<8}{mu:>16.3f}{mk:>12.3f}{mk-mu:>+10.3f}"
              f"{cu[0]:>13.3f} /{ck[0]:>7.3f}")
    print("\n  KG-prior knows the attribute-implied ECCM from round 1 -> a head start that")
    print("  matters most at SHORT horizons (real engagements); uniform must discover it.")


if __name__ == "__main__":
    main()
