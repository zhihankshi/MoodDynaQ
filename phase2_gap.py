"""
Why do the two Phase 2 measures disagree? (sustained ~27 vs ~13, first
passage ~13 vs ~12, baseline vs mood.) Read from real per-episode logs.

Config: eta=0.1, lam=0.7 (half-life 1.94 real steps), planning_steps=0,
eps=0.1, lower_len=4 (c=2.5, T=10), q_init=0, mood_updates_from=real,
mood_biases=real, 400 episodes/phase, 15 paired seeds.

    python3 phase2_gap.py      -> logs/phase2_gap/*.csv, figures/fig9_phase2_gap.png

Logs are written once and re-read on later runs; delete logs/phase2_gap/ to
regenerate. Every number printed comes from those CSVs.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mood import half_life_steps
from plots import run_one, BASE_C, MOOD_C, mean_sem, param_note, OUT
from train import write_log

ETA, LAM, PLAN, EPS, LOWER, N_EP, SEEDS = 0.1, 0.7, 0, 0.1, 4, 400, range(15)
LOGDIR = os.path.join("logs", "phase2_gap")
KINDS = ("baseline", "mood")
FLOAT = ("return", "q_start_right", "q_start_down", "q_shield_right",
         "q_d1_right", "q_d2_right", "mood_M", "epsilon")


def load_logs():
    os.makedirs(LOGDIR, exist_ok=True)
    logs = {}
    for kind in KINDS:
        for seed in SEEDS:
            path = os.path.join(LOGDIR, f"{kind}_seed{seed:02d}.csv")
            if not os.path.exists(path):
                log, _ = run_one(kind, seed, ETA, LAM, PLAN, LOWER, N_EP, epsilon=EPS)
                write_log(log, path)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                r["episode"] = int(r["episode"]); r["steps"] = int(r["steps"])
                for k in FLOAT:
                    r[k] = float(r[k])
            logs[kind, seed] = rows
    return logs


def phase2_stats(log):
    p1 = [r for r in log if r["phase"] == "phase1"]
    p2 = [r for r in log if r["phase"] == "phase2"]
    g = [r["greedy_route"] for r in p2]
    first = g.index("upper") if "upper" in g else N_EP
    run, sus = 0, N_EP
    for i, x in enumerate(g):
        run = run + 1 if x == "upper" else 0
        if run >= 20:
            sus = i - 19
            break
    changes = [i for i in range(first + 1, N_EP) if g[i] != g[i - 1]]
    # Each flip back to lower: which Q moved in the episode that caused it?
    backs = []
    for i in changes:
        if g[i] == "lower":
            prev, cur = p2[i - 1], p2[i]
            backs.append({"ep": i, "route": cur["route"],
                          "d_right": cur["q_start_right"] - prev["q_start_right"],
                          "d_down": cur["q_start_down"] - prev["q_start_down"],
                          "right": cur["q_start_right"], "down": cur["q_start_down"]})
    return {"end1": p1[-1], "first": first, "sus": sus, "flips": len(changes),
            "backs": backs, "at_first": p2[first] if first < N_EP else None,
            "p2": p2}


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():7.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):5.2f}"


def report(S):
    n = len(SEEDS)
    print(f"eta={ETA} lam={LAM} (half-life {half_life_steps(LAM):.2f} real steps) "
          f"planning_steps={PLAN} eps={EPS} lower_len={LOWER} q_init=0 "
          f"mood_updates_from=real mood_biases=real {N_EP} ep/phase, n={n} paired seeds\n")

    def row(label, f):
        b = np.array([f(S["baseline", s]) for s in SEEDS], float)
        m = np.array([f(S["mood", s]) for s in SEEDS], float)
        print(f"  {label:34s} base {ms(b)} | mood {ms(m)} | base−mood {ms(b - m)}")
        return b, m

    print("End of Phase 1 (Q* right −15, down −10):")
    row("Q(start,down)", lambda s: s["end1"]["q_start_down"])
    rb, rm = row("Q(start,right)", lambda s: s["end1"]["q_start_right"])
    row("Q(shield,right)  [Q* −7.5]", lambda s: s["end1"]["q_shield_right"])
    m_end = [S["mood", s]["end1"]["mood_M"] for s in SEEDS]
    print(f"  {'M':34s} mood {ms(m_end)}")
    print("\nPhase 2 (episodes from the switch; greedy read after each episode):")
    fb, fm = row("first greedy 'upper'", lambda s: s["first"])
    sb, sm = row("sustained (20) convergence", lambda s: s["sus"])
    kb, km = row("greedy flips after first passage", lambda s: s["flips"])
    row("sustained − first", lambda s: s["sus"] - s["first"])
    print(f"  censored (never sustained): base {sum(sb >= N_EP)}, mood {sum(sm >= N_EP)}")

    print("\nAt the first greedy 'upper' episode (Q* right −15, down −20):")
    row("Q(start,right)", lambda s: s["at_first"]["q_start_right"])
    row("Q(start,down)", lambda s: s["at_first"]["q_start_down"])
    row("Q(shield,right)  [Q* −17.5]", lambda s: s["at_first"]["q_shield_right"])
    row("Q(d_1,right)     [Q* −5]", lambda s: s["at_first"]["q_d1_right"])

    print("\nFlips back to lower after first passage — what moved in that episode:")
    for kind in KINDS:
        B = [b for s in SEEDS for b in S[kind, s]["backs"]]
        up = [b for b in B if b["route"] == "upper"]
        lo = [b for b in B if b["route"] == "lower"]
        print(f"  {kind:8s} {len(B):3d} flips back | ran upper, Q(start,right) fell: "
              f"{sum(b['d_right'] < 0 for b in up)}/{len(up)} "
              f"(mean ΔQ_right {np.mean([b['d_right'] for b in up]) if up else np.nan:.2f}, "
              f"Q_right after {np.mean([b['right'] for b in up]) if up else np.nan:.2f}) | "
              f"ran lower, Q(start,down) rose: {sum(b['d_down'] > 0 for b in lo)}/{len(lo)}")

    print("\nHypothesis check — does Phase 1 optimism in Q(start,right) predict flips?")
    for kind, r, k in (("baseline", rb, kb), ("mood", rm, km)):
        c = np.corrcoef(r, k)[0, 1] if np.std(k) > 0 else np.nan
        print(f"  {kind:8s} corr(end-P1 Q(start,right), P2 flips) = {c:+.2f}  "
              f"(seeds with Q_right > −15: {int((r > -15).sum())}/{n})")


def figure(S):
    fig, (ax, axf) = plt.subplots(1, 2, figsize=(11, 4.2),
                                  gridspec_kw={"width_ratios": [3.2, 1]})
    W = 80
    x = np.arange(-10, W)
    for kind, col in (("baseline", BASE_C), ("mood", MOOD_C)):
        for key, ls in (("q_start_right", "-"), ("q_start_down", "--")):
            arr = []
            for s in SEEDS:
                full = [r for r in (S[kind, s]["end_tail"] + S[kind, s]["p2"])]
                arr.append([r[key] for r in full][:len(x)])
            arr = np.array(arr)
            for t in arr:
                ax.plot(x, t, color=col, ls=ls, alpha=0.07, lw=0.7)
            m, sem = mean_sem(arr)
            act = "right" if key.endswith("right") else "down"
            ax.plot(x, m, color=col, ls=ls, lw=2,
                    label=f"{kind} Q(start,{act}) (n={len(SEEDS)})")
            ax.fill_between(x, m - sem, m + sem, color=col, alpha=0.18, lw=0)
    for y, lab in ((-15, "Q* right = −15"), (-20, "Q* down (Phase 2) = −20")):
        ax.axhline(y, color="#555555", ls=":", lw=1)
        ax.text(W - 1, y, lab, ha="right", va="bottom", fontsize=7.5, color="#555555")
    ax.axvline(-0.5, color="black", ls="--", lw=1)
    ax.text(-5.5, 1.02, "Phase 1", ha="center", fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(W / 2, 1.02, "Phase 2: shield inert", ha="center", fontsize=9,
            transform=ax.get_xaxis_transform())
    for kind, col in (("baseline", BASE_C), ("mood", MOOD_C)):
        fs = [S[kind, s]["first"] for s in SEEDS]
        ss = [S[kind, s]["sus"] for s in SEEDS]
        ax.plot([np.mean(fs)], [-8.3], marker="v", color=col, ms=7, clip_on=False)
        ax.plot([np.mean(ss)], [-8.3], marker="v", color=col, ms=7, mfc="white",
                clip_on=False)
    ax.text(32, -8.3, "▼ mean first 'upper'   ▽ mean sustained (20)", fontsize=7.5,
            va="center", color="#555555")
    ax.set_ylim(-22, -7.5)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Episode relative to the switch")
    ax.set_ylabel("Q value (return units)")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9), frameon=False, fontsize=7.5, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    param_note(ax, loc="lower right", eta=ETA, lam=f"{LAM} (t½ {half_life_steps(LAM):.1f} steps)",
               planning=PLAN, eps=EPS, lower_len=LOWER)
    ax.set_title("Phase 2: values at start, per episode", pad=18)

    for i, (kind, col) in enumerate((("baseline", BASE_C), ("mood", MOOD_C))):
        k = np.array([S[kind, s]["flips"] for s in SEEDS], float)
        jit = np.random.default_rng(0).uniform(-0.12, 0.12, len(k))
        axf.scatter(i + jit, k, color=col, alpha=0.5, s=14, lw=0)
        axf.errorbar(i + 0.28, k.mean(), yerr=k.std(ddof=1) / np.sqrt(len(k)),
                     fmt="o", color=col, ms=6, capsize=3)
    for s in SEEDS:
        axf.plot([0, 1], [S["baseline", s]["flips"], S["mood", s]["flips"]],
                 color="#999999", lw=0.5, alpha=0.5, zorder=0)
    axf.set_xticks([0, 1], ["baseline", "mood"])
    axf.set_xlim(-0.4, 1.5)
    axf.set_ylabel("Greedy flips after first 'upper' (count)")
    axf.set_title(f"Flips, Phase 2 (n={len(SEEDS)}, paired)", pad=18, fontsize=10)
    axf.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    logs = load_logs()
    S = {}
    for key, log in logs.items():
        S[key] = phase2_stats(log)
        S[key]["end_tail"] = [r for r in log if r["phase"] == "phase1"][-10:]
    report(S)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig9_phase2_gap.png")
    figure(S).savefig(path, dpi=150)
    print(f"\nwrote {path}")
