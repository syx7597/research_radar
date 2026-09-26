"""
Scalable solvers for the EW countermeasure Stackelberg game
===========================================================
Stage 1 (this file): EXACT robust allocation for the separable adversary, as a
mixed-integer program (MILP, HiGHS via scipy), replacing the brute-force
enumeration in sim.py. Verified to match brute force exactly on small instances,
then run where brute force is infeasible.

Model (identical to sim.py, shared by import — single source of truth):
  robust reward of assigning threat t -> (jammer j, technique T) at jammer-load L:
      r_{tjT}(L) = sigmoid(0.25*(p_j - 10*log10(L) - erp_t)) * base(T)
                   * (rho  if  def(T) ∩ capability_t != ∅  else 1)   [feasible only]
  The defender maximizes  sum_t w_t * r_{t,j(t),T(t)}(load_{j(t)}).
  Only coupling between threats = the load L_j (power dilution / burnthrough).

MILP (technique pre-collapsed: r_{tj}(L) = max_T r_{tjT}(L)):
  binaries  x[t,j,L] = threat t on jammer j while j carries load exactly L
            z[j,L]   = jammer j carries load exactly L
  max  sum  r_{tj}(L) * x[t,j,L]
  s.t. sum_{j,L} x[t,j,L] <= 1                      (each threat assigned once)
       x[t,j,L] <= z[j,L]                           (only the chosen load level)
       sum_{t,L} x[t,j,L] = sum_L L * z[j,L]        (declared load = actual count)
       sum_L z[j,L] <= 1                            (one load level per jammer)

Run:  python experiments/ew_game/solver.py
"""
import importlib.util, math, time, random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

_spec = importlib.util.spec_from_file_location("sim", __file__.replace("solver.py", "sim.py"))
sim = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sim)
RHO = 0.1  # ECCM defeat penalty (matches sim.TECH semantics)


def reward_q(threat, jammer, tech, load, qt):
    """Expected reward of (threat,jammer,tech) at given load when the adversary
    defeats this threat with probability qt (if the technique is defeatable).
    qt=1 -> worst case (Stage-1 robust); qt=0 -> nominal. Reuses sim.eff."""
    if threat["band"] not in jammer["bands"] or tech not in jammer["tech"]:
        return 0.0
    base_val = sim.eff(jammer, threat, tech, set(), load)            # undefeated value
    defeatable = bool(set(threat["capability"]) & sim.TECH[tech][1])
    pen_val = base_val * RHO if defeatable else base_val
    return (1 - qt) * base_val + qt * pen_val


def best_tech(threat, jammer, load, qt=1.0):
    """Best feasible technique and its (qt-)expected reward for (threat,jammer) at load.
    The argmax technique itself depends on qt: under threat (high qt) a robust
    low-base technique can beat a high-base defeatable one."""
    best_T, best_r = None, 0.0
    for T in jammer["tech"]:
        r = reward_q(threat, jammer, T, load, qt)
        if r > best_r:
            best_T, best_r = T, r
    return best_T, best_r


def solve_separable_milp(threats, jammers, q=None, time_limit=30.0):
    M, K = len(threats), len(jammers)
    loads = range(1, M + 1)
    qv = [1.0] * M if q is None else list(q)        # per-threat defeat probability

    # ---- variables ----
    xcol = {}          # (t,j,L) -> col index ; with precomputed reward + argmax tech
    xrew, xtech = [], []
    for t in range(M):
        for j in range(K):
            if threats[t]["band"] not in jammers[j]["bands"]:
                continue
            for L in loads:
                T, r = best_tech(threats[t], jammers[j], L, qv[t])
                if T is None or r <= 0:
                    continue
                # objective is priority-weighted suppression (matches sim.value)
                xcol[(t, j, L)] = len(xrew); xrew.append(threats[t]["prio"] * r)
                xtech.append((t, j, L, T))
    nx = len(xrew)
    zcol = {}
    for j in range(K):
        for L in loads:
            zcol[(j, L)] = nx + len(zcol)
    nz = len(zcol)
    n = nx + nz
    if nx == 0:
        return 0.0, [None] * M

    c = np.zeros(n)
    for (t, j, L), col in xcol.items():
        c[col] = -xrew[col]                       # milp minimizes -> negate to maximize

    rows_lb, rows_ub, A_rows = [], [], []
    def add(coeffs, lb, ub):
        r = np.zeros(n)
        for col, v in coeffs:
            r[col] += v
        A_rows.append(r); rows_lb.append(lb); rows_ub.append(ub)

    # each threat assigned at most once
    for t in range(M):
        coeffs = [(xcol[(t, j, L)], 1) for j in range(K) for L in loads if (t, j, L) in xcol]
        if coeffs:
            add(coeffs, -np.inf, 1)
    # x[t,j,L] <= z[j,L]
    for (t, j, L), col in xcol.items():
        add([(col, 1), (zcol[(j, L)], -1)], -np.inf, 0)
    # actual count on j == sum_L L*z[j,L]
    for j in range(K):
        coeffs = [(xcol[(t, j, L)], 1) for t in range(M) for L in loads if (t, j, L) in xcol]
        coeffs += [(zcol[(j, L)], -L) for L in loads]
        add(coeffs, 0, 0)
    # at most one load level per jammer
    for j in range(K):
        add([(zcol[(j, L)], 1) for L in loads], -np.inf, 1)

    A = sparse.csr_matrix(np.array(A_rows))
    cons = LinearConstraint(A, np.array(rows_lb), np.array(rows_ub))
    res = milp(c, constraints=cons, integrality=np.ones(n),
               bounds=Bounds(0, 1), options={"time_limit": time_limit})
    if not res.success or res.x is None:
        return None, None
    assign = [None] * M
    for (t, j, L, T) in xtech:
        if res.x[xcol[(t, j, L)]] > 0.5:
            assign[t] = (j, T)
    return -res.fun, assign


# ───────────────────────── verification + scaling ──────────────────────────
def main():
    rng = random.Random(7)
    print("=" * 64)
    print("STAGE 1 — exact MILP vs brute force (correctness)")
    print("=" * 64)
    mism = 0; checked = 0
    worst = 0.0
    for _ in range(200):
        M = rng.randint(2, 5); K = rng.randint(1, 3)
        th, ja = sim.gen_scenario(rng, M, K)
        bf_assign = sim.best_assignment(th, ja, lambda a: sim.value(a, th, ja, "best"))
        bf_val = sim.value(bf_assign, th, ja, "best")
        ms_val, ms_assign = solve_separable_milp(th, ja)
        checked += 1
        d = abs(bf_val - ms_val)
        worst = max(worst, d)
        # MILP assignment, re-scored through sim.value, must equal its own objective
        resc = sim.value(ms_assign, th, ja, "best")
        if d > 1e-6 or abs(resc - ms_val) > 1e-6:
            mism += 1
            if mism <= 5:
                print(f"  MISMATCH M={M} K={K}: bf={bf_val:.6f} milp={ms_val:.6f} rescored={resc:.6f}")
    print(f"  checked {checked} random instances; mismatches: {mism}; "
          f"max |bf-milp| = {worst:.2e}")
    print("  => MILP is EXACT" if mism == 0 else "  => DISCREPANCY (investigate)")

    print("\n" + "=" * 64)
    print("STAGE 1 — scaling where brute force is infeasible")
    print("=" * 64)
    print(f"  {'M':>3}{'K':>3}{'brute-force assigns':>22}{'MILP value':>13}{'MILP time(s)':>14}")
    for M, K in [(6, 3), (8, 3), (10, 4), (12, 4), (16, 5), (20, 5)]:
        th, ja = sim.gen_scenario(rng, M, K)
        # size of brute-force search space (options per threat ^ M) — just to report
        opt = sim.options(th, ja)
        space = 1
        for o in opt:
            space *= max(1, len(o))
        t0 = time.time()
        val, _ = solve_separable_milp(th, ja, time_limit=60)
        dt = time.time() - t0
        sp = f"{space:.1e}" if space < 1e18 else f"{space:.1e}"
        print(f"  {M:>3}{K:>3}{sp:>22}{val:>13.3f}{dt:>14.3f}")


if __name__ == "__main__":
    main()
