"""
Is the Phase 2 gap inherited from the end-of-Phase-1 Q-table? (CLAUDE.md §8)

Each agent runs Phase 1 as usual. Phase 2 is then run by a fresh recipient
(baseline or mood) that takes the DONOR's end-of-Phase-1 Q table, model,
visited list and RNG state, with M=0:
    donors:     baseline α=0.10, baseline α=0.15, mood
    recipients: baseline α in {0.10, 0.12, 0.15, 0.19}, mood
(learning-rate-matched arms added 2026-09-27). The recipient == donor cells
reproduce the ordinary runs exactly (M is ~0 at the end of Phase 1), which
checks the transplant.

Also runs control_arms.YokedMood (M replayed from seed (i+7)%30's per-step mood trace)
through both phases, to read its end-of-Phase-1 Q(start,right) and Phase 2
flips.

Config: eta=0.1, lam=0.7 (half-life 1.94 real steps), planning_steps=0,
eps=0.1, lower_len=4, q_init=0, mood real/real, 400 ep/phase, 30 paired seeds.

    python3 transplant.py      -> logs/transplant/*.csv
"""
import copy
import os

import numpy as np

from agent import DynaQ
from control_arms import YokedMood
from env import Maze, Config, make_maze_spec
from mood import MoodyDynaQ, half_life_steps
from train import run_phase, write_log

ETA, LAM, PLAN, EPS, LOWER, N_EP, SEEDS = 0.1, 0.7, 0, 0.1, 4, 400, range(30)
SPEC = make_maze_spec(2, LOWER)
CFG = {p: Config.matched(lower_len=LOWER, phase=p) for p in (1, 2)}
LOGDIR = os.path.join("logs", "transplant")
ALPHAS = (0.10, 0.12, 0.15, 0.19)
RECIPIENTS = [f"a{a:.2f}" for a in ALPHAS] + ["mood"]
DONORS = ("a0.10", "a0.15", "mood")


def make(kind, seed, trace=None):
    common = dict(epsilon=EPS, planning_steps=PLAN, seed=seed)
    if kind.startswith("a0"):
        return DynaQ(SPEC["valid_actions"], alpha=float(kind[1:]), **common)
    if kind == "yoked":
        return YokedMood(SPEC["valid_actions"], eta=ETA, lam=LAM, trace=trace, **common)
    return MoodyDynaQ(SPEC["valid_actions"], eta=ETA, lam=LAM, **common)


def sustained(g, target):
    run = 0
    for i, x in enumerate(g):
        run = run + 1 if x == target else 0
        if run >= 20:
            return i - 19
    return N_EP


def stats(log):
    """P1 sustained, P2 first 'upper', P2 sustained, P2 flips after first."""
    g1 = [r["greedy_route"] for r in log if r["phase"] == "phase1"]
    g = [r["greedy_route"] for r in log if r["phase"] == "phase2"]
    first = g.index("upper") if "upper" in g else N_EP
    flips = sum(g[i] != g[i - 1] for i in range(first + 1, len(g)))
    return sustained(g1, "lower"), first, sustained(g, "upper"), flips


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():6.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):5.2f}"


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    res, p1_right, donors = {}, {}, {}
    for seed in SEEDS:
        for kind in RECIPIENTS:
            agent, log = make(kind, seed), []
            run_phase(Maze(CFG[1]), agent, N_EP, "phase1", log, 0)
            donors[kind, seed] = (agent, log)
            p1_right.setdefault(kind, []).append(agent.q("start", "right"))
    for seed in SEEDS:
        for donor in DONORS:
            d, dlog = donors[donor, seed]
            for rec in RECIPIENTS:
                a = make(rec, seed)
                a.Q, a.model = copy.deepcopy(d.Q), copy.deepcopy(d.model)
                a.visited = list(d.visited)
                a.rng.setstate(d.rng.getstate())
                if rec == "mood":
                    a.mood.M = 0.0
                log = copy.deepcopy(dlog)
                run_phase(Maze(CFG[2]), a, N_EP, "phase2", log, N_EP)
                write_log(log, os.path.join(LOGDIR, f"{donor}Q_{rec}_seed{seed:02d}.csv"))
                res[donor, rec, seed] = stats(log)

    # Yoked: needs each mood seed's full two-phase per-step trace.
    full = {}
    for seed in SEEDS:
        a, log = make("mood", seed), []
        run_phase(Maze(CFG[1]), a, N_EP, "phase1", log, 0)
        run_phase(Maze(CFG[2]), a, N_EP, "phase2", log, N_EP)
        full[seed] = a.mood_trace
    seeds, yk = list(SEEDS), []
    for i, seed in enumerate(seeds):
        a, log = make("yoked", seed, trace=full[seeds[(i + 7) % len(seeds)]]), []
        run_phase(Maze(CFG[1]), a, N_EP, "phase1", log, 0)
        p1_right.setdefault("yoked", []).append(a.q("start", "right"))
        run_phase(Maze(CFG[2]), a, N_EP, "phase2", log, N_EP)
        write_log(log, os.path.join(LOGDIR, f"yoked_seed{seed:02d}.csv"))
        yk.append(stats(log))

    lab = lambda k: k if k == "mood" else f"baseline α={k[1:]}"
    print(f"eta={ETA} lam={LAM} (half-life {half_life_steps(LAM):.2f} real steps) "
          f"planning_steps={PLAN} eps={EPS} lower_len={LOWER} q_init=0 mood real/real "
          f"{N_EP} ep/phase, n={len(SEEDS)} paired seeds; Phase 2 starts with M=0\n")
    print("Ordinary runs (own table):")
    print("| agent | end-P1 Q(start,right) | P1 sustained | P2 first | P2 sustained | P2 flips |")
    print("|---|---|---|---|---|---|")
    for k in RECIPIENTS:
        # own-table run: the donor==recipient cell if k is a donor, else continue it
        rows = [res[k, k, s] if k in DONORS else own_run(k, s, donors) for s in SEEDS]
        p1, f, su, fl = np.array(rows, float).T
        print(f"| {lab(k)} | {ms(p1_right[k])} | {ms(p1)} | {ms(f)} | {ms(su)} | {ms(fl)} |")
    p1, f, su, fl = np.array(yk, float).T
    print(f"| yoked | {ms(p1_right['yoked'])} | {ms(p1)} | {ms(f)} | {ms(su)} | {ms(fl)} |")

    for donor in DONORS:
        print(f"\nPhase 2 from {lab(donor)}'s end-of-Phase-1 table:")
        print("| recipient | P2 first | P2 sustained | P2 flips | paired first − mood's |")
        print("|---|---|---|---|---|")
        mf = np.array([res[donor, "mood", s][1] for s in SEEDS], float)
        for rec in RECIPIENTS:
            _, f, su, fl = np.array([res[donor, rec, s] for s in SEEDS], float).T
            print(f"| {lab(rec)} | {ms(f)} | {ms(su)} | {ms(fl)} | {ms(f - mf)} |")


def own_run(kind, seed, donors):
    d, dlog = donors[kind, seed]
    a = copy.deepcopy(d)
    log = copy.deepcopy(dlog)
    run_phase(Maze(CFG[2]), a, N_EP, "phase2", log, N_EP)
    return stats(log)


if __name__ == "__main__":
    main()
