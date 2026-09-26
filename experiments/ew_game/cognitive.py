"""
Cognitive EW as online learning — minimal validation
=====================================================
ISOLATED synthetic testbed. A single jammer fights a radar whose ECCM capability
A (which counter-measures it can run) is HIDDEN. Each round the jammer picks a
technique; it is defeated iff some ECCM in A counters it. The jammer only observes
success/defeat. Goal: maximise cumulative suppression over T rounds.

Question under test (the real bar): does a DOCTRINE-GUIDED learner beat BOTH
  (1) static doctrine   — the current system: best fixed guess, no learning, and
  (2) a knowledge-BLIND bandit (UCB1) — a strong standard online learner?

Why doctrine should help: the doctrine graph (technique -> ECCM that defeat it)
lets one observation inform MANY techniques. If technique T succeeds, every ECCM
in def(T) is ruled out, which instantly revalues every other technique sharing
those ECCM. A blind bandit must learn each technique independently. So the guided
learner should infer the hidden A in ~|ECCM| observations instead of ~|techniques|.

Run:  python experiments/ew_game/cognitive.py
"""
import random, math
from statistics import mean

ECCM = ["E1", "E2", "E3", "E4", "E5"]
# technique -> (base effectiveness, set of ECCM that defeat it)
DOCTRINE = {
    "T1": (0.95, {"E1"}),
    "T2": (0.90, {"E2"}),
    "T3": (0.88, {"E3"}),
    "T4": (0.85, {"E1", "E4"}),
    "T5": (0.80, {"E2", "E5"}),
    "T6": (0.70, {"E3", "E4"}),
    "T7": (0.55, set()),          # always-safe fallback, low base
    "T8": (0.60, {"E5"}),
}
TECHS = list(DOCTRINE)
DEFEAT = {T: DOCTRINE[T][1] for T in TECHS}
BASE = {T: DOCTRINE[T][0] for T in TECHS}
RHO = 0.1
PRIOR = 0.4                       # prior prob each ECCM is present


def set_doctrine(doctrine, eccm, prior=0.4):
    """Rebind the module globals to an arbitrary doctrine (e.g. the real
    doctrine_map.json). The agents read these globals at runtime, so injecting a
    new doctrine lets the SAME learner run on real data — the data-agnostic claim."""
    global DOCTRINE, TECHS, DEFEAT, BASE, ECCM, PRIOR
    DOCTRINE = doctrine
    TECHS = list(doctrine)
    DEFEAT = {T: set(doctrine[T][1]) for T in TECHS}
    BASE = {T: doctrine[T][0] for T in TECHS}
    ECCM = list(eccm)
    PRIOR = prior


def defeated(T, A): return bool(DEFEAT[T] & A)
def reward(T, A): return BASE[T] * (RHO if defeated(T, A) else 1.0)
def best_reward(A): return max(reward(T, A) for T in TECHS)   # hindsight optimal


def gen_radar(rng):
    return {e for e in ECCM if rng.random() < PRIOR}


# ── agents ──────────────────────────────────────────────────────────────────
class StaticDoctrine:
    """Best fixed technique under the PRIOR over A (no learning) = current system."""
    def __init__(self):
        # expected reward of T under prior: defeated w.p. 1-prod(1-PRIOR) over def(T)
        def exp_r(T):
            p_safe = math.prod(1 - PRIOR for _ in DEFEAT[T])
            return BASE[T] * (p_safe + RHO * (1 - p_safe))
        self.fixed = max(TECHS, key=exp_r)
    def select(self): return self.fixed
    def update(self, T, was_defeated): pass


class BlindUCB:
    """UCB1 over techniques — strong knowledge-blind online learner."""
    def __init__(self):
        self.n = {T: 0 for T in TECHS}; self.s = {T: 0.0 for T in TECHS}; self.t = 0
    def select(self):
        self.t += 1
        for T in TECHS:
            if self.n[T] == 0:
                return T
        return max(TECHS, key=lambda T: self.s[T] / self.n[T]
                   + math.sqrt(2 * math.log(self.t) / self.n[T]))
    def update(self, T, was_defeated):
        r = BASE[T] * (RHO if was_defeated else 1.0)
        self.n[T] += 1; self.s[T] += r


class BlindThompson:
    """Gaussian Thompson sampling over techniques — a strong sample-efficient
    knowledge-blind learner (fairer than UCB at short horizons)."""
    def __init__(self, rng, forget=0.0):
        self.rng = rng; self.forget = forget
        self.n = {T: 0 for T in TECHS}; self.mu = {T: 0.5 for T in TECHS}
    def select(self):
        best, bv = None, -1e9
        for T in TECHS:
            var = 1.0 / (self.n[T] + 1)           # shrinking posterior variance
            theta = self.rng.gauss(self.mu[T], math.sqrt(var))
            if theta > bv:
                best, bv = T, theta
        return best
    def update(self, T, was_defeated):
        r = BASE[T] * (RHO if was_defeated else 1.0)
        self.n[T] += 1
        lr = self.forget if self.forget > 0 else 1.0 / self.n[T]   # recency vs running mean
        self.mu[T] += lr * (r - self.mu[T])


class DoctrineGuided:
    """Bayesian belief over hidden A via the doctrine graph; belief-greedy choice.
    success(T) -> every ECCM in def(T) ruled out;
    defeat(T)  -> at least one ECCM in def(T) present (noisy-OR posterior bump).
    `decay` relaxes beliefs toward the prior each round to track a SWITCHING radar."""
    def __init__(self, rng=None, decay=0.0):
        self.p = {e: PRIOR for e in ECCM}; self.decay = decay
    def _exp_reward(self, T):
        p_safe = math.prod(1 - self.p[e] for e in DEFEAT[T]) if DEFEAT[T] else 1.0
        return BASE[T] * (p_safe + RHO * (1 - p_safe))
    def select(self):
        return max(TECHS, key=self._exp_reward)
    def update(self, T, was_defeated):
        S = DEFEAT[T]
        if S:
            if not was_defeated:
                for e in S:
                    self.p[e] = 0.05 * PRIOR if self.decay else 0.0   # ruled out
            else:
                p_or = 1 - math.prod(1 - self.p[e] for e in S)
                if p_or > 1e-9:
                    for e in S:
                        self.p[e] = min(1.0, self.p[e] / p_or)
        if self.decay:                        # forget toward prior (non-stationary)
            for e in ECCM:
                self.p[e] = (1 - self.decay) * self.p[e] + self.decay * PRIOR


class DoctrineThompson:
    """Doctrine-belief Thompson sampling: same KG-driven posterior over the hidden
    capability A, but SELECT by sampling a hypothesis radar A' ~ posterior and
    playing the best technique against it. This adds the exploration the greedy
    variant lacks (it will try high-base techniques while A is still uncertain),
    while still exploiting doctrine structure. `decay` tracks a switching radar."""
    def __init__(self, rng, decay=0.0, init_prior=None):
        self.rng = rng; self.decay = decay
        self.p = dict(init_prior) if init_prior else {e: PRIOR for e in ECCM}
        self.p0 = dict(self.p)                              # remember start (for decay reset)
    def select(self):
        A = {e for e in ECCM if self.rng.random() < self.p[e]}     # sampled hypothesis
        return max(TECHS, key=lambda T: reward(T, A))
    def update(self, T, was_defeated):
        S = DEFEAT[T]
        if S:
            if not was_defeated:
                for e in S:
                    self.p[e] = 0.05 * PRIOR if self.decay else 0.0
            else:
                p_or = 1 - math.prod(1 - self.p[e] for e in S)
                if p_or > 1e-9:
                    for e in S:
                        self.p[e] = min(1.0, self.p[e] / p_or)
        if self.decay:
            for e in ECCM:
                self.p[e] = (1 - self.decay) * self.p[e] + self.decay * PRIOR


class DoctrineCUSUM:
    """doctrine-Thompson + CUSUM change detection. No constant forgetting; instead
    watch the prediction-error 'surprise' (confident-but-defeated). When the CUSUM
    statistic crosses a threshold, declare a radar switch and RESET the belief to the
    prior, then re-infer. Needs NO knowledge of the switch period (unlike fixed decay)."""
    def __init__(self, rng, k=0.30, h=0.90):
        self.rng = rng; self.p = {e: PRIOR for e in ECCM}
        self.S = 0.0; self.k = k; self.h = h; self.resets = 0
    def _safe_prob(self, T):
        return math.prod(1 - self.p[e] for e in DEFEAT[T]) if DEFEAT[T] else 1.0
    def select(self):
        A = {e for e in ECCM if self.rng.random() < self.p[e]}
        return max(TECHS, key=lambda T: reward(T, A))
    def update(self, T, was_defeated):
        pred_safe = self._safe_prob(T)
        actual = 0.0 if was_defeated else 1.0
        surprise = abs(pred_safe - actual)                 # prediction error
        self.S = max(0.0, self.S + surprise - self.k)
        if self.S > self.h:                                # change detected -> reset
            self.p = {e: PRIOR for e in ECCM}; self.S = 0.0; self.resets += 1
        S = DEFEAT[T]                                       # belief update (post-reset if any)
        if S:
            if not was_defeated:
                for e in S:
                    self.p[e] = 0.0
            else:
                p_or = 1 - math.prod(1 - self.p[e] for e in S)
                if p_or > 1e-9:
                    for e in S:
                        self.p[e] = min(1.0, self.p[e] / p_or)


def make_agents(rng, decay=0.0, forget=0.0):
    return {
        "static": StaticDoctrine(),
        "blind-UCB": BlindUCB(),
        "blind-Thompson": BlindThompson(rng, forget),
        "doctrine-greedy": DoctrineGuided(decay=decay),
        "doctrine-Thompson": DoctrineThompson(rng, decay=decay),
    }


def run_experiment(title, schedule_fn, n_radars, T_rounds, seed, decay, forget):
    """schedule_fn(rng_engagement, T) -> list of the hidden ECCM set A per round."""
    rng = random.Random(seed)
    names = list(make_agents(rng).keys())
    cum = {a: 0.0 for a in names}; curve = {a: [0.0] * T_rounds for a in names}
    regret = {a: 0.0 for a in names}
    for _ in range(n_radars):
        rstate = random.Random(rng.random())
        As = schedule_fn(rstate, T_rounds)
        opt = [best_reward(A) for A in As]
        agents = make_agents(rng, decay=decay, forget=forget)
        for a, agent in agents.items():
            for r in range(T_rounds):
                A = As[r]
                T = agent.select(); d = defeated(T, A); rew = reward(T, A)
                agent.update(T, d)
                cum[a] += rew; curve[a][r] += rew; regret[a] += (opt[r] - rew)
    N = n_radars
    print(f"\n=== {title} ===")
    print(f"  {'agent':<17}{'r=1':>7}{'r=2':>7}{'r=3':>7}{'r=5':>7}{'r=10':>7}{'r=30':>7}"
          f"{'r=60':>7}{'  mean':>8}{'regret':>8}")
    for a in names:
        c = curve[a]; cols = [c[0], c[1], c[2], c[4], c[9], c[29], c[59]]
        print(f"  {a:<17}" + "".join(f"{v/N:>7.3f}" for v in cols)
              + f"{cum[a]/(N*T_rounds):>8.3f}{regret[a]/N:>8.2f}")
    g = cum["doctrine-Thompson"] / (N * T_rounds)
    bu = cum["blind-UCB"] / (N * T_rounds); bt = cum["blind-Thompson"] / (N * T_rounds)
    s = cum["static"] / (N * T_rounds)
    print(f"  doctrine-Thompson vs:  static={g-s:+.3f}  blind-UCB={g-bu:+.3f}  "
          f"blind-Thompson={g-bt:+.3f}  (must beat BOTH blind learners)")


def main():
    N, T = 2000, 60
    # (1) stationary hidden radar: capability FIXED per engagement, unknown
    run_experiment("STATIONARY hidden radar (fixed capability, unknown)",
                   lambda rs, T: [gen_radar(rs)] * T, N, T, seed=0, decay=0.0, forget=0.0)
    # (2) SWITCHING radar (cognitive): capability re-rolls every 20 rounds
    run_switching(N, T)


def run_switching(N, T, period=20):
    """Radar capability switches every `period` rounds within an engagement."""
    rng = random.Random(1)
    names = list(make_agents(rng).keys())
    cum = {a: 0.0 for a in names}; curve = {a: [0.0] * T for a in names}; regret = {a: 0.0 for a in names}
    for _ in range(N):
        # schedule of capabilities, one per block
        blocks = [{e for e in ECCM if rng.random() < PRIOR} for _ in range((T // period) + 1)]
        As = [blocks[r // period] for r in range(T)]
        opt = [best_reward(A) for A in As]
        agents = make_agents(rng, decay=0.12, forget=0.12)
        for a, agent in agents.items():
            for r in range(T):
                A = As[r]; Tk = agent.select(); d = defeated(Tk, A); rew = reward(Tk, A)
                agent.update(Tk, d); cum[a] += rew; curve[a][r] += rew; regret[a] += opt[r] - rew
    print(f"\n=== SWITCHING radar (capability re-rolls every {period} rounds — 'cognitive') ===")
    print(f"  {'agent':<17}{'r=1':>7}{'r=5':>7}{'r=19':>7}{'r=21':>7}{'r=25':>7}{'r=41':>7}"
          f"{'r=45':>7}{'  mean':>8}{'regret':>8}")
    for a in names:
        c = curve[a]; cols = [c[0], c[4], c[18], c[20], c[24], c[40], c[44]]
        print(f"  {a:<17}" + "".join(f"{v/N:>7.3f}" for v in cols)
              + f"{cum[a]/(N*T):>8.3f}{regret[a]/N:>8.2f}")
    g = cum["doctrine-Thompson"]/(N*T); bu = cum["blind-UCB"]/(N*T)
    bt = cum["blind-Thompson"]/(N*T); s = cum["static"]/(N*T)
    print(f"  (rounds 21,41 are right AFTER a switch — watch who recovers)")
    print(f"  doctrine-Thompson vs:  static={g-s:+.3f}  blind-UCB={g-bu:+.3f}  blind-Thompson={g-bt:+.3f}")


if __name__ == "__main__":
    main()
