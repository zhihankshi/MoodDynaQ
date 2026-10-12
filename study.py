"""
Redesigned study (2026-10-03): gamma=0.9, epsilon decaying 1.0 -> 0.1 during
Phase 1, Phase 2 first-passage metrics. See CLAUDE.md and PREDICTIONS.md.

Arms: baseline alpha in {0.10, 0.12, 0.15, 0.19}; mood eta=0.1, lam=0.7
(half-life 1.94 real steps), mood_updates_from=mood_biases="real".
planning_steps=0, lower_len=4, q_init=0, eps floor 0.1.

    python3 study.py pilot      -> design numbers on held-out seeds 100-129
    python3 study.py            -> the study, 30 paired seeds 0-29
    python3 study.py fig        -> figures/fig_study.png
    python3 study.py reversed       -> the study with the phases swapped (2026-10-04):
    python3 study.py fig reversed      Phase 1 = shield broken, Phase 2 = shield works
                                      (Phase 2 = N2_REVERSED = 1000 episodes)
    python3 study.py reversed reset     -> exploratory (2026-10-05): Phase 2 reuses
    python3 study.py fig reversed reset    the Phase 1 eps schedule; Phase 2 = N2_RESET
"""
import sys

import numpy as np

from env import Maze, Config, make_maze_spec
from agent import DynaQ
from mood import MoodyDynaQ, half_life_steps
from optimal import optimal_q
from train import (run_phase, linear_decay, episodes_to_greedy_convergence,
                   flip_backs, start_q_error)

LOWER, PLAN, EPS_START, EPS_FLOOR = 4, 0, 1.0, 0.1
ETA, LAM = 0.1, 0.7
N_DECAY, N1, N2 = 200, 500, 100          # set from the pilot; see CLAUDE.md
N2_REVERSED = 1000   # reversed order: Phase 2 is silent, switches at 99-550 (PREDICTIONS.md)
N2_RESET = 500       # with the eps reset, Phase 2 is as long as Phase 1 (exploratory)
CHECKPOINTS = (5, 10, 25, 50)
ARMS = [("base α=0.10", "baseline", 0.10), ("base α=0.12", "baseline", 0.12),
        ("base α=0.15", "baseline", 0.15), ("base α=0.19", "baseline", 0.19),
        ("mood η=0.10", "mood", 0.10)]
Q1 = optimal_q(Config.matched(lower_len=LOWER, phase=1))[0]
Q2 = optimal_q(Config.matched(lower_len=LOWER, phase=2))[0]
# by env config phase: 1 = shield works (lower optimal), 2 = shield broken (upper optimal)
QSTAR, OPT = {1: Q1, 2: Q2}, {1: "lower", 2: "upper"}
REVERSED = (2, 1)


def run(kind, rate, seed, n1, n2, n_decay, order=(1, 2), reset=False):
    """`order` gives the env config phase used for study Phase 1 and Phase 2.
    `reset` reruns the Phase 1 eps schedule in Phase 2 (exploratory, 2026-10-05)."""
    va = make_maze_spec(2, LOWER)["valid_actions"]
    common = dict(epsilon=EPS_START, planning_steps=PLAN, seed=seed)
    agent = (DynaQ(va, alpha=rate, **common) if kind == "baseline"
             else MoodyDynaQ(va, eta=rate, lam=LAM, **common))
    log = []
    run_phase(Maze(Config.matched(lower_len=LOWER, phase=order[0])), agent, n1,
              "phase1", log, 0, epsilon_schedule=linear_decay(EPS_START, EPS_FLOOR, n_decay))
    if n2:
        run_phase(Maze(Config.matched(lower_len=LOWER, phase=order[1])), agent, n2,
                  "phase2", log, n1,
                  epsilon_schedule=linear_decay(EPS_START, EPS_FLOOR, n_decay) if reset else None)
    return log


def params(seeds, n1, n2, n_decay):
    return (f"gamma=0.9  eta/alpha as labelled  lam={LAM} (half-life "
            f"{half_life_steps(LAM):.2f} real steps)  planning_steps={PLAN}  "
            f"lower_len={LOWER}  q_init=0  eps {EPS_START}->{EPS_FLOOR} linear over "
            f"{n_decay} P1 ep  P1={n1} P2={n2} ep  seeds {seeds.start}-{seeds.stop-1}")


def pilot(seeds=range(100, 130)):
    """Phase 1 length (Q(start,right) within 0.1 of Q*) and Phase 2 length."""
    max_n1, grid = 800, (100, 200, 300, 400, 500, 600, 800)
    for n_decay in (100, 200):
        print(f"\n## Phase 1, eps decay over {n_decay} episodes  "
              f"[{params(seeds, max_n1, 0, n_decay)}]")
        print("Seeds (of 30) with |Q(start,right) − Q*| < 0.1 at end of Phase 1 episode N, "
              "and the worst seed's error")
        print(f"  {'arm':12s} " + " ".join(f"{'N=' + str(n):>12s}" for n in grid))
        for label, kind, rate in ARMS:
            logs = [run(kind, rate, s, max_n1, 0, n_decay) for s in seeds]
            cells = []
            for n in grid:
                e = [abs(l[n - 1]["q_start_right"] - Q1[("start", "right")]) for l in logs]
                cells.append(f"{sum(x < 0.1 for x in e):2d} ({max(e):5.2f})")
            print(f"  {label:12s} " + " ".join(f"{c:>12s}" for c in cells))

    print(f"\n## At the switch, chosen design  [{params(seeds, N1, 0, N_DECAY)}]")
    print(f"  Q*: right {Q1[('start','right')]:.3f}  down {Q1[('start','down')]:.3f}")
    for label, kind, rate in ARMS:
        ends = [run(kind, rate, s, N1, 0, N_DECAY)[-1] for s in seeds]
        r = np.array([e["q_start_right"] for e in ends])
        d = np.array([e["q_start_down"] for e in ends])
        m = np.array([e["mood_M"] for e in ends])
        er = np.abs(r - Q1[("start", "right")])
        print(f"  {label:12s} Q(start,right) {r.mean():8.3f} ± {r.std(ddof=1)/np.sqrt(len(r)):.3f} "
              f"(max err {er.max():.3f}, {int((er < 0.1).sum())}/30 < 0.1)   "
              f"Q(start,down) {d.mean():8.3f} (max err {np.abs(d - Q1[('start','down')]).max():.3f})"
              + ("" if kind == "baseline" else f"   M {m.mean():+.4f} (max |M| {np.abs(m).max():.4f})"))

    long2 = 300
    print(f"\n## Phase 2 length check: Phase 2 run for {long2} episodes  "
          f"[{params(seeds, N1, long2, N_DECAY)}]")
    print("  censoring only (arm comparisons withheld until the study is approved)")
    for label, kind, rate in ARMS:
        firsts, lasts = [], []
        for s in seeds:
            log = run(kind, rate, s, N1, long2, N_DECAY)
            f = episodes_to_greedy_convergence(log, "phase2", "upper", sustain=1)
            firsts.append(long2 if f is None else f - N1)
            g = [r["greedy_route"] for r in log if r["phase"] == "phase2"]
            backs = [i + 1 for i in range(len(g) - 1) if g[i] == "upper" and g[i + 1] == "lower"]
            lasts.append(backs[-1] if backs else 0)
        print(f"  {label:12s} max first passage {max(firsts):3d}   last flip-back at "
              f"Phase 2 episode {max(lasts):3d}   seeds with first passage >= 100: "
              f"{sum(x >= 100 for x in firsts)}   flip-backs after ep 100: "
              f"{sum(x >= 100 for x in lasts)} seeds")


def p1_settled(log, target):
    """Phase 1 episodes until the greedy route is `target` for good (last wrong + 1)."""
    p1 = [r for r in log if r["phase"] == "phase1"]
    return max([r["episode"] + 1 for r in p1 if r["greedy_route"] != target], default=0)


def study(seeds=range(30), order=(1, 2), reset=False):
    t1, t2, qs1, qs2 = OPT[order[0]], OPT[order[1]], QSTAR[order[0]], QSTAR[order[1]]
    n2 = N2_RESET if reset else N2_REVERSED if order == REVERSED else N2
    print(f"[{params(seeds, N1, n2, N_DECAY)}  phase order (env config) {order}: "
          f"P1 optimal {t1}, P2 optimal {t2}"
          + (f"  EPS RESET: P2 eps {EPS_START}->{EPS_FLOOR} over {N_DECAY} ep" if reset else "") + "]")
    res = {}
    for label, kind, rate in ARMS:
        rows = []
        for s in seeds:
            log = run(kind, rate, s, N1, n2, N_DECAY, order, reset)
            f = episodes_to_greedy_convergence(log, "phase2", t2, sustain=1)
            end1 = log[N1 - 1]
            row = {"first": n2 if f is None else f - N1, "censored": f is None,
                   "backs": 0 if f is None else flip_backs(log, "phase2", t2, f),
                   "p1": p1_settled(log, t1),
                   "sw_right": abs(end1["q_start_right"] - qs1[("start", "right")]),
                   "sw_down": abs(end1["q_start_down"] - qs1[("start", "down")])}
            for k in CHECKPOINTS:
                e = start_q_error(log, "phase2", qs2, k)
                row[("right", k)], row[("down", k)] = e["right"], e["down"]
            rows.append(row)
        res[label] = rows

    def ms(v):
        v = np.asarray(v, float)
        return f"{v.mean():6.2f} ± {v.std(ddof=1)/np.sqrt(len(v)):4.2f}"
    print(f"\nPhase 1: episodes until greedy '{t1}' for good (last wrong + 1); "
          "|Q − Q*| at start at the switch")
    for label in res:
        print(f"  {label:12s} {ms([r['p1'] for r in res[label]])}   "
              f"right {ms([r['sw_right'] for r in res[label]])}   "
              f"down {ms([r['sw_down'] for r in res[label]])}")
    print(f"\nPrimary: Phase 2 first greedy '{t2}' (episodes into Phase 2; censored "
          f"at {n2}) and flip-backs after it")
    for label in res:
        print(f"  {label:12s} first {ms([r['first'] for r in res[label]])}   "
              f"median {np.median([r['first'] for r in res[label]]):5.1f}   "
              f"censored {sum(r['censored'] for r in res[label]):2d}/{len(seeds)}   "
              f"flip-backs {ms([r['backs'] for r in res[label]])}")
    print("\nSecondary: |Q − Q*| at start, k episodes after the switch "
          f"(the abandoned route's error reflects exploration, not learning)")
    for a in ("right", "down"):
        print(f"  Q(start,{a})")
        for label in res:
            print(f"    {label:12s} " + "  ".join(
                f"k={k}: {ms([r[(a, k)] for r in res[label]])}" for k in CHECKPOINTS))
    mood = np.array([r["first"] for r in res["mood η=0.10"]], float)
    for label in res:
        if label.startswith("base"):
            d = np.array([r["first"] for r in res[label]], float) - mood
            print(f"  paired first passage, {label} − mood: {ms(d)}")
    return res


FIG_ARMS = [("baseline α=0.10", "baseline", 0.10, "#2b6cb0"),
            ("mood η=0.10", "mood", 0.10, "#c53030")]     # CLAUDE.md §12 colours


def figure(seeds=range(30), w=3, p1_shown=40, zoom=(-5, 30), order=(1, 2), n2=N2, w2=None,
           reset=False):
    """
    fig_study: baseline (alpha=0.10) vs mood (eta=0.10). Curves are the fraction
    of seeds whose greedy route at `start` is optimal, smoothed within each
    phase (w episodes in panel A, w2 in panel B; w2 defaults to w), ± SEM.
      A  Phase 1 acquisition, first p1_shown episodes (both stay at 1.0 to the switch)
      B  Phase 2 adaptation around the switch
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plots import smooth_by_phase, mean_sem

    w2 = w if w2 is None else w2
    curve = {}      # (label, window) -> (mean, sem)
    for label, kind, rate, _ in FIG_ARMS:
        oks = []
        for s in seeds:
            log = run(kind, rate, s, N1, n2, N_DECAY, order, reset)
            ok = np.array([r["greedy_route"] == OPT[order[0] if r["phase"] == "phase1" else order[1]]
                           for r in log], float)
            oks.append(ok)
        for win in {w, w2}:
            curve[(label, win)] = mean_sem([smooth_by_phase(ok, N1, win) for ok in oks])
    n = len(seeds)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.25))

    def draw(ax, sl, x, w):
        ax.spines[["top", "right"]].set_visible(False)
        for label, kind, _, col in FIG_ARMS:
            m, se = curve[(label, w)][0][sl], curve[(label, w)][1][sl]
            ax.fill_between(x, m - se, m + se, color=col, alpha=0.18, lw=0)
            ax.plot(x, m, color=col, lw=2.2, label=f"{label}  (n={n})")
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel(f"P(greedy route optimal)\nfraction of seeds, smoothed {w} ep")

    draw(axA, slice(0, p1_shown + 1), np.arange(p1_shown + 1), w)
    axA.set_xlim(0, p1_shown)
    axA.set_xlabel("Phase 1 episode")
    axA.set_title(f"A   Phase 1: learn the {OPT[order[0]]} route", loc="left", fontsize=11)
    axA.text(0.98, 0.42, f"both stay at 1.0 until\nthe switch (ep {N1})",
             transform=axA.transAxes, ha="right", fontsize=8, color="#4a5568")
    axA.text(0.98, 0.04, f"ε 1.0 → {EPS_FLOOR} over the first {N_DECAY} ep",
             transform=axA.transAxes, ha="right", fontsize=7.5, color="#718096")
    axA.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize=8.5, frameon=False)

    lo, hi = zoom
    draw(axB, slice(N1 + lo, N1 + hi + 1), np.arange(lo, hi + 1), w2)
    axB.axvline(-0.5, ls="--", color="#4a5568", lw=1)
    if hi - lo < 100:      # no room for the label on a long Phase 2 axis
        axB.text(-1, 0.5, "Phase 1", ha="right", fontsize=9, color="#4a5568", rotation=90, va="center")
    if n2 != N2 and not reset:
        axB.text(0.04, 0.9, f"Phase 2 = {n2} ep in this order\n(at {N2} every seed is censored)",
                 transform=axB.transAxes, fontsize=8, color="#4a5568", va="top")
    axB.text(hi - 0.5, 0.12, f"Phase 2: shield {'stops working' if order[1] == 2 else 'starts working'},"
             f"\n{OPT[order[1]]} route now optimal",
             fontsize=8.5, color="#4a5568", va="center", ha="right")
    axB.set_xlim(lo, hi)
    axB.set_xlabel("episodes after the switch")
    axB.set_title(f"B   Phase 2: switch to the {OPT[order[1]]} route", loc="left", fontsize=11)

    fig.text(0.5, -0.08, "γ=0.9   mood: η=0.1, λ=0.7 (half-life "
             f"{half_life_steps(LAM):.2f} real steps), mood_updates_from=mood_biases='real'   "
             f"planning_steps={PLAN}   upper_len=2, lower_len={LOWER}, T=10   q_init=0\n"
             f"ε 1.0→{EPS_FLOOR} over P1 ep 0–{N_DECAY - 1}, then {EPS_FLOOR}"
             + (f", RESET at the switch: again 1.0→{EPS_FLOOR} over P2 ep 0–{N_DECAY - 1}" if reset else "")
             + "   "
             f"P1={N1} / P2={n2} ep   seeds {seeds.start}–{seeds.stop - 1}, paired"
             + ("   PHASES REVERSED: P1 shield broken, P2 shield works" if order == REVERSED else ""),
             ha="center", fontsize=7.5, color="#555555")
    return fig


if __name__ == "__main__":
    args = sys.argv[1:]
    order = REVERSED if "reversed" in args else (1, 2)
    reset = "reset" in args
    if args == ["pilot"]:
        pilot()
    elif "fig" in args:
        import os
        os.makedirs("figures", exist_ok=True)
        path = "figures/fig_study%s%s.png" % ("_reversed" if order == REVERSED else "",
                                              "_reset" if reset else "")
        # reversed: Phase 2 is silent and takes ~100-550 episodes (N2_REVERSED);
        # the axis stops at 700, after the last seed has switched. Each seed
        # switches once, so unsmoothed the curve is 30 steps spread over ~450
        # episodes; w2=25 is to this axis what w=5 is to panel A's.
        kw = (dict(n2=N2_REVERSED, p1_shown=150, zoom=(-20, 700), w=5, w2=25)
              if order == REVERSED else {})
        if reset:
            # every seed switches within ~70 ep, so the no-reset zoom/smoothing doesn't fit
            kw.update(n2=N2_RESET, reset=True, zoom=(-5, 80), w2=3)
        figure(order=order, **kw).savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")
    else:
        study(order=order, reset=reset)
