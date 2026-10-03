"""
Q-table snapshots at fixed episodes, and steps per episode.

WHAT FIG 6 IS FOR
A gridworld value map has one number per cell. This maze is a corridor pair,
and every state except `start` has exactly one legal action, so one number per
state IS that state's Q-value -- nothing is hidden by a max. `start` is the
exception and the whole point of the maze: it has two actions, so its cell is
split, upper half = Q(start,right), lower half = Q(start,down). The greedy
policy is whichever half is less negative, and that is the only decision the
agent ever makes.

Each row of the figure ends with the exact Q* for that phase, from optimal.py,
drawn the same way. The panels are meant to be read against that column: how
far the table has got, and in which direction it is wrong.

STEPS PER EPISODE (fig 7) is degenerate here and worth saying rather than
plotting twice: there is no way back and the only choice is at `start`, so an
episode is `upper_len` (2) or `lower_len` (4) steps, and
    mean steps = upper_len + dL * P(lower executed).
It is the executed route rescaled, and it carries the epsilon floor, which is
why convergence is read off the greedy argmax instead (CLAUDE.md section 9).

    python3 q_snapshots.py   -> figures/fig6_q_snapshots.png
                                figures/fig7_steps_per_episode.png
                                + the steps table printed to stdout
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from env import Maze, Config, make_maze_spec
from agent import DynaQ
from mood import MoodyDynaQ, half_life_steps
from optimal import optimal_q
from plots import (BASE_C, MOOD_C, TRAP_C, OUT, conv_in_phase,
                   run_one, mean_sem, smooth_by_phase, param_note)
from train import run_phase

SNAP_EPISODES = (1, 10, 50, 100)
CMAP = plt.cm.viridis
CELL_W, CELL_H = 0.66, 0.46
START_W, START_H = 0.80, 0.58   # start holds two values, so it is bigger


# ---------------------------------------------------------------- running ---

def make_agent(kind, seed, eta, lam, planning_steps, epsilon, spec):
    if kind == "baseline":
        return DynaQ(spec["valid_actions"], alpha=eta, epsilon=epsilon,
                     planning_steps=planning_steps, seed=seed)
    return MoodyDynaQ(spec["valid_actions"], eta=eta, lam=lam, epsilon=epsilon,
                      planning_steps=planning_steps, seed=seed)


def run_with_snapshots(kind, seed, snap_episodes, eta, lam, planning_steps,
                       lower_len, n_ep, epsilon=0.1):
    """
    One agent through both phases, copying the Q table after the requested
    episode of each phase (1-indexed, counted within the phase).

    Episodes are run one at a time through `train.run_phase`, so the loop, the
    logging and the ordering are the ones every other script uses; only the
    snapshotting is new. The copy is read-only: the greedy rollout below uses
    the frozen Q, never `agent.greedy_action`, which would consume the agent's
    rng and change the run.
    """
    spec = make_maze_spec(2, lower_len)
    agent = make_agent(kind, seed, eta, lam, planning_steps, epsilon, spec)
    log, snaps, offset = [], {}, 0

    for phase in (1, 2):
        cfg = Config.matched(lower_len=lower_len, phase=phase)
        env = Maze(cfg)
        for i in range(1, n_ep + 1):
            offset = run_phase(env, agent, 1, f"phase{phase}", log, offset)
            if i in snap_episodes:
                snaps[(phase, i)] = {
                    "Q": {sa: agent.q(*sa) for sa in spec["transitions"]},
                    "seen": set(agent.Q),
                    "cfg": cfg,
                }
    return snaps, log


def greedy_rollout(Q, cfg, max_steps=20):
    """Walk from `start` under a frozen Q with exploration off."""
    env = Maze(cfg)
    state, path = env.reset(), []
    for _ in range(max_steps):
        actions = env.valid_actions(state)
        best = max(Q[(state, a)] for a in actions)
        path.append((state, [a for a in actions if Q[(state, a)] == best][0]))
        state, _, done = env.step(path[-1][1])
        if done:
            break
    return path


# --------------------------------------------------------------- one panel ---

def layout(spec):
    upper = ["start"] + [f"u_{i}" for i in range(1, spec["upper_len"])] + ["u_goal"]
    lower = (["start", "shield"]
             + [f"d_{i}" for i in range(1, spec["lower_len"] - 1)] + ["d_goal"])
    pos = {s: (i, 1.0) for i, s in enumerate(upper)}
    pos.update({s: (i - 1, 0.0) for i, s in enumerate(lower) if i > 0})
    return pos


def draw_panel(ax, Q, seen, cfg, spec, norm, note, show_names=False):
    """
    One value map: one cell per state, `start` split into its two actions.

    Deliberately unlabelled -- no per-edge numbers, no state names except in
    the key panel, no parameter text. The structure is in fig0_maze; this
    figure carries values and the greedy route and nothing else.
    """
    pos = layout(spec)
    trans, valid = spec["transitions"], spec["valid_actions"]
    terminals = spec["terminals"]
    trap_at = {ns for (ns, is_trap, _) in trans.values() if is_trap}
    path = set(greedy_rollout(Q, cfg))

    def size(state):
        return (START_W, START_H) if state == "start" else (CELL_W, CELL_H)

    # Route and arrows run centre to centre and sit BEHIND the cells, so the
    # segment crossing a cell is hidden and every line stays straight; which
    # half of `start` was chosen is shown by that half's border instead.
    for (s, a) in path:
        x1, y1 = pos[s]
        x2, y2 = pos[trans[(s, a)][0]]
        ax.plot([x1, x2], [y1, y2], color="#38bdf8", lw=6, alpha=0.85,
                solid_capstyle="round", zorder=1)

    for (s, a), (ns, _, _) in trans.items():
        (x1, y1), (x2, y2) = pos[s], pos[ns]
        dx, dy = x2 - x1, y2 - y1
        n = (dx ** 2 + dy ** 2) ** 0.5
        pads = [(size(st)[0] / 2 if dy == 0 else size(st)[1] / 2) + 0.03
                for st in (s, ns)]
        ax.annotate("", xy=(x2 - dx / n * pads[1], y2 - dy / n * pads[1]),
                    xytext=(x1 + dx / n * pads[0], y1 + dy / n * pads[0]),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#a0aec0"),
                    zorder=2)

    def cell(x, y, w, h, value, seen_flag, edge, lw=1.0, fontsize=10):
        fc = CMAP(norm(value)) if seen_flag else "#ffffff"
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec=edge,
                               lw=lw, zorder=3))
        if seen_flag:
            ax.text(x, y, f"{value:.1f}", ha="center", va="center", zorder=4,
                    fontsize=fontsize, fontweight="bold",
                    color="#ffffff" if norm(value) < 0.55 else "#1a202c")

    for s, (x, y) in pos.items():
        if s in terminals:
            ax.add_patch(Rectangle((x - CELL_W / 2, y - CELL_H / 2), CELL_W,
                                   CELL_H, fc="#ffffff", ec="#cbd5e0", lw=1.0,
                                   zorder=3))
            ax.text(x, y, "goal", ha="center", va="center", fontsize=8.5,
                    color="#718096", zorder=4)
        elif s == "start":
            # the only two-action state: upper half = right, lower half = down
            for a, off in (("right", START_H / 4), ("down", -START_H / 4)):
                chosen = (s, a) in path
                cell(x, y + off, START_W, START_H / 2, Q[(s, a)],
                     (s, a) in seen, "#0284c7" if chosen else "#1a202c",
                     lw=2.6 if chosen else 1.2, fontsize=9.5)
        else:
            a = valid[s][0]
            cell(x, y, CELL_W, CELL_H, Q[(s, a)], (s, a) in seen,
                 TRAP_C if s in trap_at else "#cbd5e0",
                 lw=1.8 if s in trap_at else 1.0)

        if show_names:
            ax.text(x, y - size(s)[1] / 2 - 0.08, s, ha="center", va="top",
                    fontsize=7, color="#718096", zorder=4)

    ax.text(0.5, -0.06, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#1a202c")
    ax.set_xlim(-0.55, spec["lower_len"] - 1 + 0.55)
    ax.set_ylim(-0.50, 1.50)
    ax.axis("off")


# ---------------------------------------------------------------- figure 6 ---

def fig_q_snapshots(eta=0.1, lam=0.7, planning_steps=0, lower_len=4,
                    n_ep=150, seed=0, epsilon=0.1,
                    snap_episodes=SNAP_EPISODES):
    """
    Fig 6: the Q table at fixed episodes, baseline vs mood, both phases,
    with exact Q* as the last column.

    ONE seed, paired between the two agents: a Q table averaged over seeds is
    not a table any agent ever held.
    """
    spec = make_maze_spec(2, lower_len)
    runs, logs = {}, {}
    for k in ("baseline", "mood"):
        runs[k], logs[k] = run_with_snapshots(k, seed, snap_episodes, eta, lam,
                                              planning_steps, lower_len, n_ep,
                                              epsilon)
    qstar = {ph: optimal_q(Config.matched(lower_len=lower_len, phase=ph))[0]
             for ph in (1, 2)}

    # one colour scale for every panel, anchored on Q*, or the panels cannot
    # be compared to each other or to the target
    lo = min([min(s["Q"].values()) for r in runs.values() for s in r.values()]
             + [min(q.values()) for q in qstar.values()])
    norm = Normalize(vmin=np.floor(lo), vmax=0.0)

    rows = [(kind, ph) for ph in (1, 2) for kind in ("baseline", "mood")]
    ncol = len(snap_episodes) + 1
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(2.9 * ncol, 1.95 * len(rows)))

    for r, (kind, ph) in enumerate(rows):
        target = "lower" if ph == 1 else "upper"
        cfg = Config.matched(lower_len=lower_len, phase=ph)
        panels = [(runs[kind][(ph, ep)]["Q"], runs[kind][(ph, ep)]["seen"])
                  for ep in snap_episodes]
        panels.append((qstar[ph], set(qstar[ph])))
        for c, (Q, seen) in enumerate(panels):
            route = "lower" if greedy_rollout(Q, cfg)[0][1] == "down" else "upper"
            mark = "✓" if route == target else "✗"
            note = f"greedy: {route} {mark}"
            draw_panel(axes[r, c], Q, seen, cfg, spec, norm, note,
                       show_names=(r == 0 and c == 0))
            if r == 0:
                head = (f"episode {snap_episodes[c]}" if c < len(snap_episodes)
                        else "Q* (target)")
                axes[r, c].set_title(head, fontsize=11, pad=8)
        # conv_in_phase counts episodes before the stable run (0-indexed);
        # panels are 1-indexed, so the stable run starts at episode conv + 1
        conv = conv_in_phase(logs[kind], f"phase{ph}", target, n_ep)
        stable = (f"stable from ep {conv + 1}" if conv < n_ep
                  else f"never stable by ep {n_ep}")
        axes[r, 0].text(-0.10, 0.5, f"{kind}\nPhase {ph}",
                        transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=11, fontweight="bold",
                        color=BASE_C if kind == "baseline" else MOOD_C)
        axes[r, 0].text(-0.10, 0.20, stable, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=8.5, color="#718096")
        axes[r, -1].patch.set_alpha(0.0)

    fig.subplots_adjust(left=0.09, right=0.90, top=0.90, bottom=0.10,
                        hspace=0.45, wspace=0.06)
    cax = fig.add_axes([0.925, 0.30, 0.012, 0.40])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("Q value", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("Q values over learning — one cell per state, start split "
                 "into its two actions", fontsize=13, y=0.965)
    fig.text(0.5, 0.055,
             f"seed {seed}   eta={eta}   lam={lam} (half-life "
             f"{half_life_steps(lam):.1f} real steps)   planning_steps={planning_steps}   "
             f"eps={epsilon}   mood_biases=real   geometry 2/{lower_len}, T=10   "
             "|   blue = greedy route and the action it takes at start   orange = trap step   "
             "blank = never visited   episodes counted within phase\n"
             "A tick is the greedy argmax IN THAT EPISODE. It flips for a long time "
             "before it settles, because the two values at start stay within ~1 of each "
             "other; 'stable from' is the first of 20 consecutive correct episodes.",
             fontsize=8, color="#718096", ha="center", va="top")
    return fig


# ---------------------------------------------------------------- figure 7 ---

def steps_curves(eta=0.1, lam=0.7, planning_steps=0, lower_len=4, n_ep=150,
                 seeds=range(30), epsilon=0.1):
    """Steps per episode for every seed, seed-paired between the two agents."""
    curves = {k: [] for k in ("baseline", "mood")}
    for seed in seeds:
        for kind in curves:
            log, _ = run_one(kind, seed, eta, lam, planning_steps, lower_len,
                             n_ep, epsilon)
            curves[kind].append([r["steps"] for r in log])
    return {k: np.array(v, float) for k, v in curves.items()}


def steps_table(n_ep=150, snap_episodes=SNAP_EPISODES, **kw):
    """Mean steps at the snapshot episodes -- the numbers behind fig 6's titles."""
    curves = steps_curves(n_ep=n_ep, **kw)
    n = len(next(iter(curves.values())))
    print(f"\nSteps in the episode itself (mean over {n} seeds; 2 = upper, 4 = lower)")
    print("  phase   agent     " + "".join(f"{'ep ' + str(e):>8}" for e in snap_episodes))
    for phase, off in (("phase1", 0), ("phase2", n_ep)):
        for kind, arr in curves.items():
            cells = "".join(f"{arr[:, off + e - 1].mean():>8.2f}" for e in snap_episodes)
            print(f"  {phase}  {kind:<10}{cells}")
    return curves


def fig_steps_per_episode(eta=0.1, lam=0.7, planning_steps=0, lower_len=4,
                          n_ep=150, seeds=range(30), epsilon=0.1):
    """Fig 7: steps per episode, seed-paired baseline vs mood, both phases."""
    curves = steps_curves(eta, lam, planning_steps, lower_len, n_ep, seeds,
                          epsilon)

    x = np.arange(2 * n_ep)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for kind, color in (("baseline", BASE_C), ("mood", MOOD_C)):
        arr = np.array([smooth_by_phase(c, n_ep, w=5) for c in curves[kind]])
        m, sem = mean_sem(arr)
        for row in arr:
            ax.plot(x, row, color=color, lw=0.4, alpha=0.10)
        ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.25, lw=0)
        ax.plot(x, m, color=color, lw=2.2,
                label=f"{kind} (n={len(curves[kind])})")

    ax.axvline(n_ep, color="#000000", ls="--", lw=1.2)
    ax.text(n_ep, 1.07, "  shield goes inert", transform=ax.get_xaxis_transform(),
            fontsize=8, va="bottom", ha="left")
    ax.text(n_ep * 0.5, 1.01, "Phase 1 — lower optimal (4 steps)",
            transform=ax.get_xaxis_transform(), fontsize=8, va="bottom", ha="center")
    ax.text(n_ep * 1.5, 1.01, "Phase 2 — upper optimal (2 steps)",
            transform=ax.get_xaxis_transform(), fontsize=8, va="bottom", ha="center")

    ax.set_xlabel("episode")
    ax.set_ylabel("steps per episode\n(5-episode mean, smoothed within phase)")
    ax.set_ylim(1.8, 4.2)
    ax.set_yticks([2, 3, 4])
    right = ax.twinx()
    right.set_ylim(1.8, 4.2)
    right.set_yticks([2, 4])
    right.set_yticklabels(["all upper", "all lower"], fontsize=8)
    for a in (ax, right):
        a.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.spines["top"].set_visible(False)

    ax.legend(loc="center left", frameon=False, fontsize=9)
    ax.set_title("Steps per episode — a rescaling of the executed route, "
                 "since the only choice is at start", pad=30)
    param_note(ax, loc="lower left", eta=eta, lam=f"{lam} (t½={half_life_steps(lam):.1f} steps)",
               planning_steps=planning_steps, eps=epsilon,
               geometry=f"2/{lower_len}", mood_biases="real")
    fig.tight_layout()
    return fig



# ---------------------------------------------------------------- figure 8 ---

def flip_episodes(log, phase, target, n_ep, start_ep=10):
    """
    Episodes (1-indexed, within phase) where the greedy argmax at `start`
    changes, from `start_ep` up to sustained convergence. The last one is the
    convergence episode itself: conv_in_phase counts episodes BEFORE the
    stable run (0-indexed), so the stable run begins at episode conv + 1.
    """
    rows = [r for r in log if r["phase"] == phase]
    conv = conv_in_phase(log, phase, target, n_ep) + 1
    changes = [e for e in range(start_ep + 1, conv + 1)
               if rows[e - 1]["greedy_route"] != rows[e - 2]["greedy_route"]]
    return changes, conv, rows


def flip_caption(rows, ep):
    """
    Two lines, in time order: what the agent did DURING episode `ep`, then
    what the Q table drawn above it (its state AFTER the episode) now picks.
    'on-policy' = it followed the greedy choice it had going in.
    """
    r, prev = rows[ep - 1], rows[ep - 2]
    how = "on-policy" if r["route"] == prev["greedy_route"] else "random, ε"
    return (f"ep {ep}: ran {r['route']} ({how})\n"
            f"after it, greedy = {r['greedy_route']}")


def fig_q_first_flip(eta=0.1, lam=0.7, planning_steps=0, lower_len=4,
                     n_ep=150, seed=0, epsilon=0.1, start_ep=10):
    """
    Fig 8a: Phase 1 only. Every episode from `start_ep` up to and including
    each agent's first flip of the greedy argmax away from lower -- the
    run-up to one flip, episode by episode.
    """
    spec = make_maze_spec(2, lower_len)
    cfg = Config.matched(lower_len=lower_len, phase=1)
    data = {}
    for kind in ("baseline", "mood"):
        snaps, log = run_with_snapshots(kind, seed, range(1, n_ep + 1), eta,
                                        lam, planning_steps, lower_len, n_ep,
                                        epsilon)
        changes, _, rows = flip_episodes(log, "phase1", "lower", n_ep,
                                         start_ep)
        data[kind] = (snaps, rows, list(range(start_ep, changes[0] + 1)))

    lo = min(min(snaps[(1, e)]["Q"].values())
             for snaps, _, eps in data.values() for e in eps)
    norm = Normalize(vmin=np.floor(lo), vmax=0.0)

    ncol = max(len(eps) for *_, eps in data.values())
    fig, axes = plt.subplots(2, ncol, figsize=(2.9 * ncol, 2.1 * 2),
                             squeeze=False)
    for r, kind in enumerate(("baseline", "mood")):
        snaps, rows, eps = data[kind]
        for c in range(ncol):
            if c >= len(eps):
                axes[r, c].axis("off")
                continue
            s = snaps[(1, eps[c])]
            draw_panel(axes[r, c], s["Q"], s["seen"], cfg, spec, norm,
                       flip_caption(rows, eps[c]))
            if c == len(eps) - 1:
                axes[r, c].set_title("first flip", fontsize=9,
                                     color="#c05621", pad=2)
        axes[r, 0].text(-0.10, 0.5, kind, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=11,
                        fontweight="bold",
                        color=BASE_C if kind == "baseline" else MOOD_C)

    fig.subplots_adjust(left=0.10, right=0.88, top=0.84, bottom=0.20,
                        hspace=0.85, wspace=0.06)
    cax = fig.add_axes([0.91, 0.30, 0.015, 0.40])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("Q value", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"Phase 1: episode {start_ep} to the first flip of the "
                 "greedy route", fontsize=12, y=0.98)
    fig.text(0.5, 0.06,
             f"seed {seed}   eta={eta}   lam={lam} (half-life "
             f"{half_life_steps(lam):.1f} real steps)   planning_steps={planning_steps}   "
             f"eps={epsilon}   geometry 2/{lower_len}, T=10   q_init=0\n"
             "Each panel is the Q table at the END of that episode; blue = the "
             "greedy route it now picks.",
             fontsize=8, color="#718096", ha="center", va="top")
    return fig


def fig_q_flips(eta=0.1, lam=0.7, planning_steps=0, lower_len=4, n_ep=150,
                seed=0, epsilon=0.1, start_ep=10):
    """
    Fig 8: Phase 1 only. The Q table at episode `start_ep` and at every
    episode where the greedy argmax at `start` changes, until it settles.

    Every-episode snapshots would be ~50 panels per agent; between flips the
    argmax does not change, so only the flip episodes are drawn. The argmax is
    binary, so flips alternate: each column is one pair, top = the episode it
    flips to upper (wrong), bottom = the episode it flips back to lower. The
    route the agent actually RAN in that episode is written under the panel,
    because that run is what moved the value that flipped.
    """
    spec = make_maze_spec(2, lower_len)
    cfg = Config.matched(lower_len=lower_len, phase=1)
    qstar = optimal_q(cfg)[0]
    data = {}
    for kind in ("baseline", "mood"):
        snaps, log = run_with_snapshots(kind, seed, range(1, n_ep + 1), eta,
                                        lam, planning_steps, lower_len, n_ep,
                                        epsilon)
        changes, conv, rows = flip_episodes(log, "phase1", "lower", n_ep,
                                            start_ep)
        # pairs assume the agent is on lower at start_ep, as in the question
        assert rows[start_ep - 1]["greedy_route"] == "lower", (kind, seed)
        data[kind] = (snaps, changes, conv, rows)

    lo = min([min(s["Q"].values()) for sn, *_ in data.values()
              for (ph, _), s in sn.items() if ph == 1] + list(qstar.values()))
    norm = Normalize(vmin=np.floor(lo), vmax=0.0)

    ncol = 1 + max(len(d[1]) // 2 for d in data.values())
    fig, axes = plt.subplots(4, ncol, figsize=(2.5 * ncol, 1.95 * 4))

    for b, kind in enumerate(("baseline", "mood")):
        snaps, changes, conv, rows = data[kind]
        top, bot = axes[2 * b], axes[2 * b + 1]
        s = snaps[(1, start_ep)]
        draw_panel(top[0], s["Q"], s["seen"], cfg, spec, norm,
                   flip_caption(rows, start_ep))
        draw_panel(bot[0], qstar, set(qstar), cfg, spec, norm, "Q* (target)")
        bot[0].set_title("for reference", fontsize=9, color="#718096", pad=2)
        for c in range(1, ncol):
            pair = changes[2 * (c - 1): 2 * c]
            for ax, ep in zip((top[c], bot[c]), pair + [None] * (2 - len(pair))):
                if ep is None:
                    ax.axis("off")
                    continue
                s = snaps[(1, ep)]
                note = flip_caption(rows, ep)
                draw_panel(ax, s["Q"], s["seen"], cfg, spec, norm, note)
                if ep == conv:
                    ax.set_title("stable from here", fontsize=9,
                                 color="#2f855a", pad=2)
        color = BASE_C if kind == "baseline" else MOOD_C
        top[0].text(-0.10, 0.5, f"{kind}\nflips to\nupper (wrong)", transform=top[0].transAxes,
                    ha="right", va="center", fontsize=11, fontweight="bold",
                    color=color)
        bot[0].text(-0.10, 0.5, f"{kind}\nflips back to\nlower (right)", transform=bot[0].transAxes,
                    ha="right", va="center", fontsize=11, fontweight="bold",
                    color=color)
        bot[0].text(-0.10, -0.05, f"{len(changes)} flips after ep {start_ep}",
                    transform=bot[0].transAxes, ha="right", va="center",
                    fontsize=8.5, color="#718096")

    axes[0, 0].set_title(f"episode {start_ep}", fontsize=11, pad=8)
    for c in range(1, ncol):
        axes[0, c].set_title(f"flips {2 * c - 1} & {2 * c}", fontsize=11, pad=8)

    fig.subplots_adjust(left=0.09, right=0.92, top=0.90, bottom=0.14,
                        hspace=0.85, wspace=0.06)
    # divider between the two agents, just above the mood block's top titles
    y = axes[2, 0].get_position().y1 + 0.03
    fig.add_artist(plt.Line2D([0.02, 0.92], [y, y], color="#cbd5e0", lw=1))
    cax = fig.add_axes([0.94, 0.30, 0.010, 0.40])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("Q value", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"Phase 1: every flip of the greedy argmax after episode "
                 f"{start_ep}, until it stays on lower", fontsize=13, y=0.965)
    fig.text(0.5, 0.045,
             f"seed {seed}   eta={eta}   lam={lam} (half-life "
             f"{half_life_steps(lam):.1f} real steps)   planning_steps={planning_steps}   "
             f"eps={epsilon}   mood_biases=real   geometry 2/{lower_len}, T=10   q_init=0   "
             "|   episodes counted within phase\n"
             "Each panel is the Q table at the END of that episode; blue = the greedy "
             "route it now picks. The caption's first line is the route run DURING the "
             "episode (on-policy, or a random ε step) — that run is what moved the values.\n"
             "Episodes between flips are omitted — the argmax does not change there.",
             fontsize=8, color="#718096", ha="center", va="top")
    return fig


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig6_q_snapshots", fig_q_snapshots),
                     ("fig7_steps_per_episode", fig_steps_per_episode),
                     ("fig8_q_flips", fig_q_flips),
                     ("fig8a_q_first_flip", fig_q_first_flip)]:
        fig = fn()
        path = os.path.join(OUT, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
    # a single episode's mean is noisy across 30 seeds; the smoothed curve in
    # fig 7 is the reliable read. These are the snapshot episodes themselves.
    steps_table()
