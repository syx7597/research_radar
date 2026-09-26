"""
RL baseline: model-free Q-learning vs the doctrine-guided learner
=================================================================
Positions the doctrine-guided learner against standard reinforcement learning.
The RL agent (tabular Q-learning) treats the engagement as an MDP: state = the
per-technique observation status so far (untried / succeeded / defeated), action =
which technique to jam with, reward = suppression. It LEARNS a policy across many
training episodes (random radars), with NO doctrine knowledge.

Claim under test: the doctrine graph makes learning vastly more sample-efficient.
The doctrine-guided learner needs ZERO training episodes (it infers within each
engagement via the counter-relations); the RL agent must see many thousands of
episodes to approach the same level — and its Q-table is opaque and tied to one
doctrine, whereas the doctrine learner is interpretable and takes any graph.

Run:  python experiments/ew_game/rl_baseline.py
"""
import importlib.util, random
from collections import defaultdict
from statistics import mean

_spec = importlib.util.spec_from_file_location("cog", __file__.replace("rl_baseline.py", "cognitive.py"))
cog = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cog)


class QLearner:
    """Tabular Q-learning over the per-technique observation status (model-free RL).
    Given its fairest shot: epsilon decays as training proceeds (more exploration
    early, near-greedy late)."""
    def __init__(self, rng, alpha=0.4, gamma=0.9):
        self.rng = rng; self.alpha = alpha; self.gamma = gamma
        self.Q = defaultdict(lambda: {T: 0.0 for T in cog.TECHS}); self.trained = 0

    @staticmethod
    def _state(status):
        return tuple(status[T] for T in cog.TECHS)        # 0 untried / 1 defeated / 2 success

    def train_episode(self, A, T_rounds):
        eps = max(0.03, 0.30 * 5000 / (5000 + self.trained))    # decay 0.30 -> 0.03
        status = {T: 0 for T in cog.TECHS}
        for _ in range(T_rounds):
            s = self._state(status)
            if self.rng.random() < eps:
                a = self.rng.choice(cog.TECHS)
            else:
                a = max(cog.TECHS, key=lambda T: self.Q[s][T])
            d = cog.defeated(a, A); r = cog.reward(a, A)
            status[a] = 1 if d else 2
            s2 = self._state(status)
            self.Q[s][a] += self.alpha * (r + self.gamma * max(self.Q[s2].values()) - self.Q[s][a])
        self.trained += 1

    def eval_episode(self, A, T_rounds):
        status = {T: 0 for T in cog.TECHS}; tot = 0.0
        for _ in range(T_rounds):
            s = self._state(status)
            a = max(cog.TECHS, key=lambda T: self.Q[s][T])   # greedy
            d = cog.defeated(a, A); tot += cog.reward(a, A)
            status[a] = 1 if d else 2
        return tot / T_rounds


def doctrine_mean(n_eval, T, rng, seed):
    r = random.Random(seed); tot = 0.0
    for _ in range(n_eval):
        A = cog.gen_radar(random.Random(r.random()))
        ag = cog.DoctrineThompson(r)
        for _ in range(T):
            Tk = ag.select(); d = cog.defeated(Tk, A); ag.update(Tk, d); tot += cog.reward(Tk, A)
    return tot / (n_eval * T)


def main():
    T, n_eval = 60, 2000
    rng = random.Random(0)
    # doctrine-guided: NO training
    dg = doctrine_mean(n_eval, T, rng, seed=1)

    print(f"Synthetic doctrine: {len(cog.TECHS)} techniques, {len(cog.ECCM)} ECCM, "
          f"engagement T={T}\n")
    print("Model-free RL (tabular Q-learning, eps-decay, avg of 3 seeds) — test suppression:")
    print(f"  {'train episodes':>16}{'RL test':>12}{'doctrine-guided (no training)':>32}")
    eval_rng = random.Random(123)
    eval_radars = [cog.gen_radar(random.Random(eval_rng.random())) for _ in range(n_eval)]
    targets = [100, 1000, 5000, 20000, 80000]
    SEEDS = [7, 17, 27]
    qls = {sd: QLearner(random.Random(sd)) for sd in SEEDS}
    trn = {sd: random.Random(1000 + sd) for sd in SEEDS}
    nstates = 0
    for target in targets:
        vals = []
        for sd in SEEDS:
            ql = qls[sd]
            while ql.trained < target:
                A = cog.gen_radar(random.Random(trn[sd].random()))
                ql.train_episode(A, T)
            vals.append(mean(ql.eval_episode(A, T) for A in eval_radars))
            nstates = max(nstates, len(ql.Q))
        print(f"  {target:>16}{mean(vals):>12.3f}{dg:>32.3f}")
    print(f"\n  states visited by RL Q-table: ~{nstates}")
    print("  Reading: RL must see many thousands of episodes to approach what the")
    print("  doctrine-guided learner achieves with ZERO training (it infers within each")
    print("  engagement via the counter-relations). The Q-table is also opaque and tied")
    print("  to this one doctrine; the doctrine learner is interpretable and graph-agnostic.")


if __name__ == "__main__":
    main()
