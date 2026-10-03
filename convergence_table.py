"""
Episodes to greedy convergence: mean and standard deviation, per agent.

The metric is the one defined in train.episodes_to_greedy_convergence: the
first episode IN THE PHASE where the greedy (epsilon=0) policy picks the
currently-optimal route and keeps picking it for 20 consecutive episodes.

Reported as mean +/- SD across seeds (SEM also shown, since CLAUDE.md figures
and tables elsewhere use SEM -- SD = SEM * sqrt(n)). Seeds are paired: the
same seed drives the baseline and every mood arm.

A seed that never converges within the phase is censored at n_ep, exactly as
plots.conv_in_phase does. `cens` counts those -- a nonzero count means the
mean and SD are lower bounds.

    python3 convergence_table.py
"""

import numpy as np

from plots import run_one, conv_in_phase
from mood import half_life_steps

ETA = 0.1
EPSILON = 0.1
LOWER_LEN = 4
PLANNING = 0
N_EP = 400
SEEDS = range(30)
LAMS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
HEADLINE_LAM = 0.7


def collect(kind, lam):
    """Per-seed episodes to convergence, both phases."""
    out = {1: [], 2: []}
    for seed in SEEDS:
        log, _ = run_one(kind, seed, ETA, lam, PLANNING, LOWER_LEN, N_EP,
                         epsilon=EPSILON)
        for ph, tgt in [(1, "lower"), (2, "upper")]:
            out[ph].append(conv_in_phase(log, f"phase{ph}", tgt, N_EP))
    return out


def stats(v):
    a = np.asarray(v, dtype=float)
    n = a.size
    return dict(mean=a.mean(), sd=a.std(ddof=1), sem=a.std(ddof=1) / np.sqrt(n),
                median=np.median(a), lo=a.min(), hi=a.max(),
                cens=int((a >= N_EP).sum()))


def row(label, s):
    return (f"| {label} | {s['mean']:.2f} | {s['sd']:.2f} | {s['sem']:.2f} | "
            f"{s['median']:.1f} | {s['lo']:.0f}–{s['hi']:.0f} | {s['cens']} |")


HEAD = ("| agent | mean | SD | SEM | median | min–max | cens |\n"
        "|---|---|---|---|---|---|---|")


def main():
    n = len(list(SEEDS))
    print(f"Episodes to sustained greedy convergence (20 consecutive episodes)\n"
          f"eta={ETA}  epsilon={EPSILON}  planning_steps={PLANNING}  "
          f"lower_len={LOWER_LEN}  upper_len=2  episodes/phase={N_EP}  "
          f"seeds={n} (paired)\n")

    base = collect("baseline", None)
    moods = {lam: collect("mood", lam) for lam in LAMS}

    for ph, name, tgt in [(1, "Phase 1 — acquisition", "lower"),
                          (2, "Phase 2 — adaptation", "upper")]:
        print(f"\n### {name} (target route: {tgt})\n")
        print(HEAD)
        print(row("baseline Dyna-Q", stats(base[ph])))
        hl = half_life_steps(HEADLINE_LAM)
        print(row(f"mood, λ={HEADLINE_LAM} (half-life {hl:.1f} real steps)",
                  stats(moods[HEADLINE_LAM][ph])))
        print()
        print(HEAD.replace("| agent |", "| mood λ |"))
        for lam in LAMS:
            hl = half_life_steps(lam)
            print(row(f"{lam} (half-life {hl:.1f} steps)", stats(moods[lam][ph])))

    print("\n### paired difference, baseline − mood "
          f"(λ={HEADLINE_LAM}), same seed\n")
    print("| phase | mean diff | SD | SEM | seeds mood faster |")
    print("|---|---|---|---|---|")
    for ph, name in [(1, "Phase 1"), (2, "Phase 2")]:
        d = np.array(base[ph], dtype=float) - np.array(moods[HEADLINE_LAM][ph], dtype=float)
        print(f"| {name} | {d.mean():.2f} | {d.std(ddof=1):.2f} | "
              f"{d.std(ddof=1)/np.sqrt(d.size):.2f} | {int((d > 0).sum())}/{d.size} |")


if __name__ == "__main__":
    main()
