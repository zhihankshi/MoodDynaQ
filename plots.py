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
from optimal import optimal_q, optimal_route
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


def blended(ax):
    """x in data coords, y in axes coords."""
    return ax.get_xaxis_transform()


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
    Fig 4: does planning dilute the mood effect?

    Two rows, because the question is about the GAP between arms, not about
    either arm on its own:
      top    -- episodes to convergence per arm, linear y
      bottom -- paired mood advantage (baseline minus mood, same seed)

    Y-SCALE: linear, deliberately. An earlier version used symlog(linthresh=1).
    That magnified n>=2, where both agents converge in under one episode and
    the metric is at its floor, and compressed n=0, where the real 18.6-episode
    effect lives. Log scale shows ratios; near a floor of ~0.6 episodes a ratio
    is noise. The floor region is shaded instead of being drawn as signal.

    PAIRING: seeds are paired (CLAUDE.md §7), so the advantage is computed
    per seed and then averaged. The paired SEM is much tighter than the
    independent-sample SEM the previous version showed.
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

    n_seeds = len(list(seeds))
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), sharex="col")
    xs = np.arange(len(plans))

    for col, (ph, label) in enumerate([(1, "Phase 1: acquisition"),
                                       (2, "Phase 2: adaptation")]):
        top, bot = axes[0, col], axes[1, col]

        # where the metric bottoms out: baseline already converges in <1 episode,
        # so neither arm can show anything and the numbers are not comparable
        base_mean = np.array([np.mean(res[ph]["baseline"][n]) for n in plans])
        floor = np.flatnonzero(base_mean < 1.0)
        for ax in (top, bot):
            if floor.size:
                ax.axvspan(floor[0] - 0.5, len(plans) - 0.5, color="#edf2f7",
                           lw=0, zorder=0)

        for name, color, ls, _kw in PLAN_ARMS:
            arr = {n: np.array(res[ph][name][n], dtype=float) for n in plans}
            m = [arr[n].mean() for n in plans]
            e = [arr[n].std(ddof=1) / np.sqrt(n_seeds) for n in plans]
            top.errorbar(xs, m, yerr=e, color=color, ls=ls, marker="o", ms=4,
                         lw=2, capsize=3, label=name, zorder=3)
            if name == "baseline":
                continue
            d = np.array([np.array(res[ph]["baseline"][n], dtype=float)
                          - arr[n] for n in plans])          # (len(plans), seeds)
            dm = d.mean(axis=1)
            de = d.std(axis=1, ddof=1) / np.sqrt(n_seeds)
            bot.errorbar(xs, dm, yerr=de, color=color, ls=ls, marker="o", ms=4,
                         lw=2, capsize=3, label=name, zorder=3)

        # The n=0 advantage is ~10x the rest, so the tail -- which is where
        # "dilutes but does not abolish" actually has to be read -- needs its
        # own axes. Phase 1 has no non-floor tail, so it gets no inset.
        if ph == 2:
            ins = bot.inset_axes([0.45, 0.40, 0.52, 0.50])
            for name, color, ls, _kw in PLAN_ARMS:
                if name == "baseline":
                    continue
                arr = {n: np.array(res[ph][name][n], dtype=float) for n in plans}
                d = np.array([np.array(res[ph]["baseline"][n], dtype=float)
                              - arr[n] for n in plans])
                ins.errorbar(xs[1:], d.mean(axis=1)[1:],
                             yerr=(d.std(axis=1, ddof=1) / np.sqrt(n_seeds))[1:],
                             color=color, ls=ls, marker="o", ms=3, lw=1.5,
                             capsize=2)
            ins.axhline(0, color=BASE_C, lw=1)
            ins.set_xticks(xs[1:])
            ins.set_xticklabels([str(n) for n in plans[1:]], fontsize=7)
            ins.tick_params(labelsize=7)
            ins.set_title("zoom: n\u22651", fontsize=7.5, color="#4a5568", pad=2)
            ins.spines[["top", "right"]].set_visible(False)

        bot.axhline(0, color=BASE_C, lw=1.5, zorder=2)
        bot.text(len(plans) - 0.55, 0, " baseline", color=BASE_C, fontsize=7.5,
                 va="bottom", ha="right")
        if floor.size:
            # centred in the shaded band, mid-height: the legend owns the top
            # corner and the curves sit near zero down below.
            top.text((floor[0] - 0.5 + len(plans) - 0.5) / 2, 0.45,
                     "metric floor\nbaseline <1 episode,\nnothing to measure",
                     transform=blended(top), ha="center", va="center",
                     fontsize=7.5, color="#718096")

        top.set_title(label)
        top.set_ylabel("Episodes to greedy convergence")
        bot.set_ylabel("Mood advantage (episodes saved)")
        bot.set_xlabel("Planning steps per real step")
        bot.set_xticks(xs)
        bot.set_xticklabels([str(n) for n in plans])
        for ax in (top, bot):
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper right")
    # Phase 1 bottom-right is empty (the floor sits at zero); Phase 2 bottom
    # holds the zoom inset, so the note cannot go there.
    param_note(axes[1, 0], loc="upper right", eta=eta, lam=lam,
               lower_len=lower_len, seeds=n_seeds)
    fig.suptitle("Does planning dilute the mood effect?  "
                 "(top: mean \u00b1 SEM;  bottom: paired difference \u00b1 SEM)",
                 y=0.98)
    fig.tight_layout()
    return fig


def fig_planning_sweep_2arm(eta=0.1, lam=0.7, lower_len=4, n_ep=400,
                            plans=(0, 1, 2, 5, 10, 20), seeds=range(30),
                            boot=2000):
    """
    Fig 4b: baseline vs mood (bias "real") only -- the theory-faithful pair.

    Fig 4's third arm ("real+planning") answered its question and now only
    overplots. Dropping it leaves room for the comparison fig 4 cannot make:
    the advantage RELATIVE to baseline. The absolute advantage collapses from
    14.0 to 1.6 episodes between n=0 and n=1 mostly because baseline itself
    collapses from 27.2 to 10.1 -- there is simply less left to save. The
    relative panel separates "the effect went away" from "the room went away".

    Percentage error bars are a paired bootstrap over seeds (the ratio of means
    is not a per-seed quantity: individual seeds hit baseline=0 at high n).
    """
    rng = np.random.default_rng(0)
    arms = [("baseline", BASE_C, "-", dict(kind="baseline")),
            ("mood (bias real)", MOOD_C, "-", dict(kind="mood", mood_biases="real"))]
    res = {ph: {name: {} for name, *_ in arms} for ph in (1, 2)}
    for name, _c, _ls, kw in arms:
        kw = dict(kw)
        kind = kw.pop("kind")
        for n in plans:
            acc = {1: [], 2: []}
            for seed in seeds:
                log, _ = run_one(kind, seed, eta, lam, n, lower_len, n_ep, **kw)
                for ph, tgt in [(1, "lower"), (2, "upper")]:
                    acc[ph].append(conv_in_phase(log, f"phase{ph}", tgt, n_ep))
            for ph in (1, 2):
                res[ph][name][n] = np.array(acc[ph], dtype=float)

    n_seeds = len(list(seeds))
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), sharex="col")
    xs = np.arange(len(plans))

    for col, (ph, label) in enumerate([(1, "Phase 1: acquisition"),
                                       (2, "Phase 2: adaptation")]):
        top, bot = axes[0, col], axes[1, col]
        base = res[ph]["baseline"]
        mood = res[ph]["mood (bias real)"]
        # One criterion for both rows: once baseline converges in ~1 episode,
        # a metric that needs 20 sustained episodes cannot resolve anything,
        # and a ratio with that denominator is unstable. CLAUDE.md §8 says the
        # same in words ("Phase 1 is uninformative past n=1").
        floor = np.flatnonzero(np.array([base[n].mean() for n in plans]) < 2.0)
        for ax in (top, bot):
            if floor.size:
                ax.axvspan(floor[0] - 0.5, len(plans) - 0.5, color="#edf2f7",
                           lw=0, zorder=0)

        for name, color, ls, _kw in arms:
            m = [res[ph][name][n].mean() for n in plans]
            e = [res[ph][name][n].std(ddof=1) / np.sqrt(n_seeds) for n in plans]
            top.errorbar(xs, m, yerr=e, color=color, ls=ls, marker="o", ms=4,
                         lw=2, capsize=3, label=name, zorder=3)

        pct, lo, hi, wins = [], [], [], []
        for n in plans:
            b, m = base[n], mood[n]
            pct.append(100 * (b.mean() - m.mean()) / b.mean())
            idx = rng.integers(0, n_seeds, size=(boot, n_seeds))
            bs = 100 * (b[idx].mean(axis=1) - m[idx].mean(axis=1)) / b[idx].mean(axis=1)
            lo.append(np.percentile(bs, 2.5))
            hi.append(np.percentile(bs, 97.5))
            wins.append(int((b - m > 0).sum()))
        pct = np.array(pct)
        bot.errorbar(xs, pct, yerr=[pct - np.array(lo), np.array(hi) - pct],
                     color=MOOD_C, marker="o", ms=4, lw=2, capsize=3, zorder=3)
        bot.axhline(0, color=BASE_C, lw=1.5, zorder=2)
        # left edge: the right-hand end is where the n=20 point sits at 0%
        bot.text(-0.55, 0, "baseline ", color=BASE_C, fontsize=7.5,
                 va="bottom", ha="left")
        # seeds on which mood won, so a percentage near a small baseline cannot
        # be mistaken for a solid effect
        for x, w in zip(xs, wins):
            bot.text(x, 0.97, f"{w}/{n_seeds}", transform=blended(bot),
                     ha="center", va="top", fontsize=6.5, color="#718096")

        if floor.size:
            top.text((floor[0] - 0.5 + len(plans) - 0.5) / 2, 0.45,
                     "baseline \u22642 episodes\nmetric cannot resolve",
                     transform=blended(top), ha="center", va="center",
                     fontsize=7.5, color="#718096")
        top.set_title(label)
        top.set_ylabel("Episodes to greedy convergence")
        bot.set_ylabel("Mood advantage (% of baseline)")
        bot.set_xlabel("Planning steps per real step")
        bot.set_xticks(xs)
        bot.set_xticklabels([str(n) for n in plans])
        for ax in (top, bot):
            ax.spines[["top", "right"]].set_visible(False)
        top.set_ylim(bottom=0)
        bot.set_xlim(-0.6, len(plans) - 0.45)
        if ph == 1:
            # the n=1 bootstrap CI runs to about -365%; showing it in full
            # would crush the one point that matters (n=0) to a few pixels
            bot.set_ylim(-150, 62)

    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1, 0].annotate("\u2212114%, 95% CI to \u2212365% (clipped):\n"
                        "baseline is ~1 episode, so the ratio\n"
                        "is unstable \u2014 \u00a78 open q.2",
                        xy=(1, -140), xytext=(1.75, -118), fontsize=7,
                        color="#718096", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="#a0aec0"))
    param_note(axes[0, 1], loc="lower left", eta=eta, lam=lam,
               lower_len=lower_len, seeds=n_seeds, boot=boot)
    fig.suptitle("Baseline vs mood (bias real): absolute and relative advantage "
                 "(top: mean \u00b1 SEM;  bottom: paired bootstrap 95% CI)", y=0.98)
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


# --- the environment itself -------------------------------------------------

TRAP_C, SHIELD_C, GOAL_C = "#dd6b20", "#2f855a", "#2d3748"


def fig_maze(upper_len=2, lower_len=4, trap_cost=10.0):
    """
    Fig 0: the maze. Structure, rewards and exact Q* for both phases.

    A schematic, not a data figure, so the blue/red agent colours do not apply
    (there are no agent arms here). Everything drawn is read from
    `make_maze_spec` and `Config.matched`, so it cannot drift from the
    environment the agents actually run in.
    """
    spec = make_maze_spec(upper_len, lower_len)
    cfgs = {ph: Config.matched(trap_cost=trap_cost, upper_len=upper_len,
                               lower_len=lower_len, phase=ph) for ph in (1, 2)}
    c = cfgs[1].step_cost

    upper = ["start"] + [f"u_{i}" for i in range(1, upper_len)] + ["u_goal"]
    lower = ["start", "shield"] + [f"d_{i}" for i in range(1, lower_len - 1)] + ["d_goal"]
    pos = {s: (i, 1.0) for i, s in enumerate(upper)}
    pos.update({s: (i - 1, 0.0) for i, s in enumerate(lower) if i > 0})

    # which states are entered through a trap, and whether that trap is shielded
    trap_at = {ns: prot for (ns, is_trap, prot) in spec["transitions"].values() if is_trap}

    fig, ax = plt.subplots(figsize=(9.5, 4.4))

    # corridor bands, so "which route" is readable at a glance
    for states, y, label in [(upper, 1.0, "UPPER  %d steps, trap unshielded" % upper_len),
                             (lower[1:], 0.0, "LOWER  %d steps, trap shielded in Phase 1" % lower_len)]:
        x0 = min(pos[s][0] for s in states) - 0.45
        x1 = max(pos[s][0] for s in states) + 0.45
        ax.add_patch(plt.Rectangle((x0, y - 0.3), x1 - x0, 0.6, fc="#f7fafc",
                                   ec="#e2e8f0", zorder=0))
        ax.text(x1 + 0.12, y, label, va="center", ha="left", fontsize=8,
                color="#4a5568")

    def node(s):
        x, y = pos[s]
        if s in spec["terminals"]:
            fc, ec, lw, txt = "#ffffff", GOAL_C, 2.0, "GOAL"
        elif s == "shield":
            fc, ec, lw, txt = "#f0fff4", SHIELD_C, 1.8, "shield"
        elif s in trap_at:
            fc, ec, lw, txt = "#fffaf0", TRAP_C, 1.8, "TRAP"
        elif s == "start":
            fc, ec, lw, txt = "#ffffff", "#000000", 2.0, "start"
        else:
            fc, ec, lw, txt = "#ffffff", "#a0aec0", 1.2, ""
        ax.text(x, y, txt or s, ha="center", va="center", fontsize=8.5,
                zorder=3, color=ec if txt else "#718096",
                fontweight="bold" if txt in ("GOAL", "TRAP", "start") else "normal",
                bbox=dict(boxstyle="round,pad=0.30", fc=fc, ec=ec, lw=lw))
        if txt and s not in ("start", "shield"):
            ax.text(x, y - 0.235, s, ha="center", va="top", fontsize=7,
                    color="#718096", zorder=3)

    for s in pos:
        node(s)

    def edge(s, a):
        ns, is_trap, prot = spec["transitions"][(s, a)]
        (x1, y1), (x2, y2) = pos[s], pos[ns]
        dx, dy = x2 - x1, y2 - y1
        n = (dx ** 2 + dy ** 2) ** 0.5
        pad = 0.30
        ax.annotate("", xy=(x2 - dx / n * pad, y2 - dy / n * pad),
                    xytext=(x1 + dx / n * pad, y1 + dy / n * pad),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5,
                                    color=TRAP_C if is_trap else "#4a5568"),
                    zorder=2)
        if is_trap and prot:
            lab = f"$-{c:g}$  (P1)\n$-{c + trap_cost:g}$  (P2)"
        elif is_trap:
            lab = f"$-{c + trap_cost:g}$"
        else:
            lab = f"$-{c:g}$"
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # the vertical start->shield label goes LEFT of the arrow; to the
        # right it collides with the lower trap's two-line (P1/P2) label.
        off = (0.0, 0.17) if dy == 0 else (-0.12, 0.0)
        ax.text(mx + off[0], my + off[1], lab, ha="center" if dy == 0 else "right",
                va="bottom" if dy == 0 else "center", fontsize=7.5,
                color=TRAP_C if is_trap else "#4a5568", zorder=3)

    for (s, a) in spec["transitions"]:
        edge(s, a)

    # the only choice in the whole maze
    ax.annotate("the only decision:\nright (upper) or down (lower)",
                xy=pos["start"], xytext=(pos["start"][0] - 0.30, 1.62),
                ha="center", fontsize=8, color="#000000",
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#000000"))

    # exact Q* from optimal.py -- the ground truth, not re-derived here
    rows = ["            Q*(start,right)   Q*(start,down)   optimal   margin"]
    for ph in (1, 2):
        Q, _ = optimal_q(cfgs[ph])
        route, margin = optimal_route(cfgs[ph])
        rows.append(f"  Phase {ph}     {Q[('start','right')]:>8.1f}      "
                    f"{Q[('start','down')]:>10.1f}      {route:>7}    {margin:>5.1f}")
    ax.text(-0.45, -0.72, "\n".join(rows), fontsize=8, family="monospace",
            va="top", ha="left", color="#2d3748")
    ax.text(-0.45, -1.16,
            "Structural asymmetry (§5): the upper trap is 1 step from start, the lower trap 2, so the\n"
            "Phase 2 cost change must propagate before it reaches Q(start,down). Phase 2 is harder for\n"
            "reasons unrelated to mood.", fontsize=7.5, va="top", ha="left",
            color="#718096", style="italic")

    ax.set_xlim(-0.75, lower_len - 1 + 1.05)
    ax.set_ylim(-1.45, 1.95)
    ax.axis("off")
    ax.set_title("The two-corridor shield/trap maze  (γ=1, no goal bonus, "
                 "no way back)", pad=6)
    param_note(ax, loc="upper right", step_cost=c, trap_cost=trap_cost,
               upper_len=upper_len, lower_len=lower_len)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("fig0_maze", fig_maze),
                     ("fig1_learning_curves", fig_learning_curves),
                     ("fig2_mood_trace", fig_mood_trace),
                     ("fig3_lambda_sweep", fig_lambda_sweep),
                     ("fig4_planning_sweep", fig_planning_sweep),
                     ("fig4b_planning_sweep_2arm", fig_planning_sweep_2arm),
                     ("fig5_mood_at_switch", fig_mood_at_switch)]:
        fig = fn()
        path = os.path.join(OUT, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
