"""
Reference figures for the baseline vs mood comparison.

This file sets the plotting standard for the project. New figures should
follow the same conventions (see PLOTTING RULES in CLAUDE.md):
  - baseline blue, mood red, everywhere
  - mean line with a shaded SEM band; faint per-seed lines underneath
  - the phase switch always marked with a dashed vertical line
  - the parameters that produced the figure written on the figure itself
  - every axis labelled with units

    python3 plots.py                  -> figures/*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env import Maze, Config, make_maze_spec
from agent import DynaQ
from mood import MoodyDynaQ, flip_threshold
from train import run_phase, episodes_to_greedy_convergence

BASE_C, MOOD_C = "#2b6cb0", "#c53030"
OUT = "figures"


def run_one(kind, seed, eta, lam, planning_steps, lower_len, n_ep,
            epsilon=0.1, mood_updates_from="real", mood_biases="real"):
    """
    One agent through both phases. Returns (log, agent).

    `kind` is "baseline" or "mood". The mood flags are ignored for the
    baseline, which has no mood term at all.
    """
    spec = make_maze_spec(2, lower_len)
    if kind == "baseline":
        agent = DynaQ(spec["valid_actions"], alpha=eta, epsilon=epsilon,
                      planning_steps=planning_steps, seed=seed)
    else:
        agent = MoodyDynaQ(spec["valid_actions"], eta=eta, lam=lam,
                           epsilon=epsilon, planning_steps=planning_steps,
                           seed=seed, mood_updates_from=mood_updates_from,
                           mood_biases=mood_biases)
    log = []
    run_phase(Maze(Config.matched(lower_len=lower_len, phase=1)),
              agent, n_ep, "phase1", log, 0)
    run_phase(Maze(Config.matched(lower_len=lower_len, phase=2)),
              agent, n_ep, "phase2", log, n_ep)
    return log, agent


def run_pair(seed, eta, lam, planning_steps, lower_len, n_ep, epsilon=0.1,
             mood_updates_from="real", mood_biases="real"):
    """Seed-paired baseline and mood runs through both phases."""
    logs, agents = {}, {}
    for kind in ("baseline", "mood"):
        logs[kind], agents[kind] = run_one(
            kind, seed, eta, lam, planning_steps, lower_len, n_ep, epsilon,
            mood_updates_from, mood_biases)
    return logs, agents["mood"]


def conv_in_phase(log, phase, target, n_ep):
    """
    Episodes to greedy convergence, counted FROM THE START OF THE PHASE.

    episodes_to_greedy_convergence returns an absolute episode index, so phase2
    needs the offset removed. `None` (never converged) is censored at n_ep --
    keep this convention identical across every script (CLAUDE.md §9).
    """
    c = episodes_to_greedy_convergence(log, phase, target)
    if c is None:
        return n_ep
    return c - (n_ep if phase == "phase2" else 0)


def correct_greedy(log, n_ep):
    """1 where the greedy policy picks the currently-optimal route, else 0."""
    out = []
    for r in log:
        target = "lower" if r["episode"] < n_ep else "upper"
        out.append(1.0 if r["greedy_route"] == target else 0.0)
    return np.array(out)


def smooth(x, w=5):
    if w <= 1:
        return x
    k = np.ones(w) / w
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def smooth_by_phase(x, n_ep, w=5):
    """
    Smooth each phase separately.

    A window straddling the switch would average pre-switch and post-switch
    episodes together and blur the very transition the figure is about.
    """
    return np.concatenate([smooth(x[:n_ep], w), smooth(x[n_ep:], w)])


def mean_sem(arr):
    arr = np.asarray(arr)
    m = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else 0 * m
    return m, sem


def param_note(ax, loc="lower right", **kw):
    """Write the parameters that produced the figure onto the figure itself.

    `loc` because the default corner is not always free -- check the rendered
    png for collisions with the data or another label before settling on one.
    """
    x, ha = (0.99, "right") if loc.endswith("right") else (0.01, "left")
    y, va = (0.02, "bottom") if loc.startswith("lower") else (0.98, "top")
    text = "   ".join(f"{k}={v}" for k, v in kw.items())
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=7.5, color="#555555")


def fig_learning_curves(eta=0.1, lam=0.7, planning_steps=0, lower_len=4,
                        n_ep=150, seeds=range(30)):
    """Fig 1: is the greedy policy correct, over episodes, both phases."""
    curves = {"baseline": [], "mood": []}
    for seed in seeds:
        logs, _ = run_pair(seed, eta, lam, planning_steps, lower_len, n_ep)
        for name in curves:
            curves[name].append(smooth_by_phase(correct_greedy(logs[name], n_ep), n_ep))

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(2 * n_ep)
    for name, color in [("baseline", BASE_C), ("mood", MOOD_C)]:
        for c in curves[name]:
            ax.plot(x, c, color=color, alpha=0.08, lw=0.8)
        m, sem = mean_sem(curves[name])
        ax.plot(x, m, color=color, lw=2, label=f"{name} (n={len(seeds)})")
        ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.2, lw=0)

    ax.axvline(n_ep, color="black", ls="--", lw=1)
    ax.text(n_ep * 0.5, 1.06, "Phase 1: shield protects", ha="center", fontsize=9)
    ax.text(n_ep * 1.5, 1.06, "Phase 2: shield inert", ha="center", fontsize=9)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("P(greedy policy is optimal)")
    ax.set_title("Greedy-policy correctness, baseline vs mood", pad=18)
    ax.legend(loc="center right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    param_note(ax, eta=eta, lam=lam, planning=planning_steps,
               lower_len=lower_len, smooth=5)
    fig.tight_layout()
    return fig


def per_episode_mood(mood_trace, log):
    """
    Mean M within each episode. Indexing by EPISODE (not raw step) keeps the
    phase switch at the same x-position for every seed -- seeds take routes of
    different lengths, so the switch falls at a different raw step for each.
    """
    out, i = [], 0
    for r in log:
        n = r["steps"]
        chunk = mood_trace[i:i + n]
        out.append(np.mean(chunk) if chunk else np.nan)
        i += n
    return np.array(out)


def fig_mood_trace(eta=0.1, lam=0.7, planning_steps=0, lower_len=4,
                   n_ep=150, seeds=range(30)):
    """Fig 2: per-episode mean M, aligned on the phase switch."""
    traces = []
    for seed in seeds:
        logs, mood_agent = run_pair(seed, eta, lam, planning_steps, lower_len, n_ep)
        traces.append(per_episode_mood(mood_agent.mood_trace, logs["mood"]))
    arr = np.array(traces)
    thr = flip_threshold(eta, margin=5.0, dL=lower_len - 2)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(arr.shape[1])
    for t in arr:
        ax.plot(x, t, color=MOOD_C, alpha=0.08, lw=0.8)
    m, sem = mean_sem(arr)
    ax.plot(x, m, color=MOOD_C, lw=2, label=f"mood M, per-episode mean (n={len(seeds)})")
    ax.fill_between(x, m - sem, m + sem, color=MOOD_C, alpha=0.2, lw=0)

    for y in (thr, -thr):
        ax.axhline(y, color="#555555", ls=":", lw=1)
    ax.text(2 * n_ep * 0.99, -thr, f"route-flip threshold \u00b1{thr:.3f}  ",
            va="bottom", ha="right", fontsize=8, color="#555555")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(n_ep, color="black", ls="--", lw=1)
    ax.text(n_ep * 0.5, 1.02, "Phase 1: shield protects", ha="center",
            fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(n_ep * 1.5, 1.02, "Phase 2: shield inert", ha="center",
            fontsize=9, transform=ax.get_xaxis_transform())
    # Headroom above +threshold so the param note has clear space (the early
    # Phase 1 transient sets the lower limit on its own).
    ax.set_ylim(top=thr * 1.45)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mood M")
    ax.set_title("Mood trace over training", pad=18)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    param_note(ax, loc="upper left", eta=eta, lam=lam,
               planning=planning_steps, lower_len=lower_len)
    fig.tight_layout()
    return fig


def fig_lambda_sweep(eta=0.1, planning_steps=0, lower_len=4, n_ep=400,
                     lams=(0.5, 0.7, 0.8, 0.9, 0.95, 0.99), seeds=range(30)):
    """Fig 3: episodes to convergence vs lambda, baseline as a reference."""
    base = {1: [], 2: []}
    mood = {1: {l: [] for l in lams}, 2: {l: [] for l in lams}}
    for seed in seeds:
        # The baseline does not depend on lam, so run it once per seed.
        blog, _ = run_one("baseline", seed, eta, None, planning_steps,
                          lower_len, n_ep)
        for ph, tgt in [(1, "lower"), (2, "upper")]:
            base[ph].append(conv_in_phase(blog, f"phase{ph}", tgt, n_ep))
        for lam in lams:
            mlog, _ = run_one("mood", seed, eta, lam, planning_steps,
                              lower_len, n_ep)
            for ph, tgt in [(1, "lower"), (2, "upper")]:
                mood[ph][lam].append(conv_in_phase(mlog, f"phase{ph}", tgt, n_ep))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, ph, label in [(axes[0], 1, "Phase 1: acquisition"),
                          (axes[1], 2, "Phase 2: adaptation")]:
        arr = np.array(base[ph], dtype=float)
        bm = arr.mean()
        bs = arr.std(ddof=1) / np.sqrt(arr.size)
        ax.axhspan(bm - bs, bm + bs, color=BASE_C, alpha=0.15, lw=0)
        ax.axhline(bm, color=BASE_C, lw=2, label="baseline")
        means = [np.mean(mood[ph][l]) for l in lams]
        sems = [np.std(mood[ph][l], ddof=1) / np.sqrt(len(list(seeds))) for l in lams]
        xs = np.arange(len(lams))
        ax.errorbar(xs, means, yerr=sems, color=MOOD_C, marker="o", lw=2,
                    capsize=3, label="mood")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(l) for l in lams])
        ax.set_xlabel("\u03bb (mood retention per real step)")
        ax.set_ylabel("Episodes to greedy convergence")
        ax.set_title(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    param_note(axes[0], loc="lower right", eta=eta, planning=planning_steps,
               lower_len=lower_len, seeds=len(list(seeds)))
    fig.suptitle("Effect of mood timescale on learning speed (mean \u00b1 SEM)", y=1.02)
    fig.tight_layout()
    return fig


# --- planning_steps: the dilution question (CLAUDE.md §8, open question 2) ---

PLAN_ARMS = [
    ("baseline",          BASE_C,    "-",  dict(kind="baseline")),
    ("mood, bias real",   MOOD_C,    "-",  dict(kind="mood", mood_biases="real")),
    ("mood, bias real+planning", "#d69e2e", "--",
     dict(kind="mood", mood_biases="real+planning")),
]


def fig_planning_sweep(eta=0.1, lam=0.7, lower_len=4, n_ep=400,
                       plans=(0, 1, 2, 5, 10, 20), seeds=range(30)):
    """
    Fig 4: episodes to convergence vs planning_steps, three arms.

    With mood_biases="real" and planning_steps=n, only 1/(1+n) of all Q-updates
    carry the mood term, so mood should dilute toward baseline as n grows.
    The "real+planning" arm removes that dilution and is the test of whether
    the mood effect survives planning at all.
    """
    res = {ph: {name: {n: [] for n in plans} for name, *_ in PLAN_ARMS}
           for ph in (1, 2)}
    for seed in seeds:
        for n in plans:
            for name, _c, _ls, kw in PLAN_ARMS:
                kw = dict(kw)
                kind = kw.pop("kind")
                log, _ = run_one(kind, seed, eta, lam, n, lower_len, n_ep, **kw)
                for ph, tgt in [(1, "lower"), (2, "upper")]:
                    res[ph][name][n].append(
                        conv_in_phase(log, f"phase{ph}", tgt, n_ep))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    xs = np.arange(len(plans))
    for ax, ph, label in [(axes[0], 1, "Phase 1: acquisition"),
                          (axes[1], 2, "Phase 2: adaptation")]:
        for name, color, ls, _kw in PLAN_ARMS:
            means = [np.mean(res[ph][name][n]) for n in plans]
            sems = [np.std(res[ph][name][n], ddof=1) / np.sqrt(len(list(seeds)))
                    for n in plans]
            ax.errorbar(xs, means, yerr=sems, color=color, ls=ls, marker="o",
                        ms=4, lw=2, capsize=3, label=name)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(n) for n in plans])
        ax.set_xlabel("Planning steps per real step")
        ax.set_ylabel("Episodes to greedy convergence")
        ax.set_title(label)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    param_note(axes[1], loc="lower left", eta=eta, lam=lam,
               lower_len=lower_len, seeds=len(list(seeds)))
    fig.suptitle("Does planning dilute the mood effect? (mean \u00b1 SEM, symlog y)",
                 y=1.02)
    fig.tight_layout()
    return fig


def fig_mood_at_switch(eta=0.1, lam=0.7, lower_len=4, n_ep=400,
                       plans=(0, 1, 5, 20), window=40, seeds=range(30)):
    """
    Fig 5: mood M around the phase switch, one curve per planning_steps.

    Indexed by episode relative to the switch, so seeds with different route
    lengths still line up (CLAUDE.md §12).
    """
    shades = plt.get_cmap("Reds")(np.linspace(0.45, 0.95, len(plans)))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    thr = flip_threshold(eta, margin=5.0, dL=lower_len - 2)

    for color, n in zip(shades, plans):
        traces = []
        for seed in seeds:
            log, agent = run_one("mood", seed, eta, lam, n, lower_len, n_ep)
            traces.append(per_episode_mood(agent.mood_trace, log))
        arr = np.array(traces)[:, n_ep - window:n_ep + window]
        m, sem = mean_sem(arr)
        x = np.arange(-window, window)
        ax.plot(x, m, color=color, lw=2, label=f"planning_steps={n}")
        ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.2, lw=0)

    for y in (thr, -thr):
        ax.axhline(y, color="#555555", ls=":", lw=1)
    ax.text(-window * 0.98, -thr, f"  route-flip threshold \u00b1{thr:.3f}",
            va="top", ha="left", fontsize=8, color="#555555")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.text(-window * 0.5, 1.02, "Phase 1: shield protects", ha="center",
            fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(window * 0.5, 1.02, "Phase 2: shield inert", ha="center",
            fontsize=9, transform=ax.get_xaxis_transform())
    # Headroom above the +threshold line so the param note has clear space,
    # and below the -threshold line so the legend does not sit on it.
    ax.set_ylim(-0.34, 0.40)
    ax.set_xlabel("Episode relative to the phase switch")
    ax.set_ylabel("Mood M (per-episode mean)")
    ax.set_title("Mood response to the switch, by planning steps", pad=18)
    ax.legend(frameon=False, fontsize=8, loc="center right",
              title=f"n={len(list(seeds))} seeds", title_fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    param_note(ax, loc="upper left", eta=eta, lam=lam, lower_len=lower_len)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig1_learning_curves", fig_learning_curves),
                     ("fig2_mood_trace", fig_mood_trace),
                     ("fig3_lambda_sweep", fig_lambda_sweep),
                     ("fig4_planning_sweep", fig_planning_sweep),
                     ("fig5_mood_at_switch", fig_mood_at_switch)]:
        fig = fn()
        path = os.path.join(OUT, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
