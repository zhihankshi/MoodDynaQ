"""
Control arms for the mood speedup: is it route discrimination, or acceleration?

The concern this answers: the greedy metric might be a proxy for "Q got large
fast", and mood might win only by adding extra negative pressure to every
update. The metric itself is not that (train.episodes_to_greedy_convergence
reads the argmax at `start`, never Q error) -- but a faster descent out of the
optimistic q_init=0 region would still produce an earlier correct argmax.

Six arms, all eta=0.1, epsilon=0.1, planning_steps=0, lower_len=4, 400 ep/phase:

  baseline    plain Dyna-Q
  mood        MoodyDynaQ, the theory-faithful agent
  const       baseline plus a CONSTANT extra on real updates. Tests whether a
              sustained pessimism offset -- mood's average, with no dynamics --
              is enough. This is the §3.4 per-step-bonus story.
  antimood    mood with the sign of the (1-eta)M term flipped. A sanity check
              on direction, and a warning: it looks superb in Phase 1.
  local       one INDEPENDENT MoodTracker per (s,a). Strips out every bit of
              cross-state coupling. If the effect survives, mood is not acting
              as a global affective signal at all.
  yoked       M replayed from another seed's trace: same magnitude and time
              course, decoupled from this agent's own deltas. If the effect
              vanishes, what matters is correlation with the current delta.

Plus baseline swept over alpha, so each arm's episodes-to-convergence can be
read back as an EFFECTIVE LEARNING RATE.

    python3 control_arms.py
"""

import numpy as np

from agent import DynaQ
from env import Maze, Config, make_maze_spec
from mood import MoodyDynaQ, MoodTracker, half_life_steps
from optimal import optimal_q
from train import run_episode, episodes_to_greedy_convergence

ETA, EPSILON, PLANNING, LOWER_LEN, N_EP, LAM = 0.1, 0.1, 0, 4, 400, 0.7
SEEDS = range(30)
ALPHAS = np.round(np.arange(0.10, 0.211, 0.005), 3)

SPEC = make_maze_spec(2, LOWER_LEN)
CFG = {p: Config.matched(lower_len=LOWER_LEN, phase=p) for p in (1, 2)}
QSTAR = {p: optimal_q(CFG[p])[0] for p in (1, 2)}


class ConstOffset(DynaQ):
    """Baseline plus a fixed additive term on real updates only."""

    def __init__(self, *args, offset=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset
        self._in_planning = False

    def plan(self):
        self._in_planning = True
        try:
            super().plan()
        finally:
            self._in_planning = False

    def update(self, s, a, r, ns, extra=0.0):
        return super().update(s, a, r, ns,
                              extra=0.0 if self._in_planning else self.offset)


class AntiMood(MoodyDynaQ):
    def _mood_term(self):
        return -super()._mood_term()


class LocalMood(MoodyDynaQ):
    """Per-(s,a) mood: same update rule, nothing shared across state-actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._local = {}
        self._key = None

    def _tracker(self, key):
        if key not in self._local:
            self._local[key] = MoodTracker(lam=self.mood.lam)
        return self._local[key]

    def _mood_term(self):
        if self._in_planning and self.mood_biases == "real":
            return 0.0
        return self.mood_weight * self._tracker(self._key).M

    def update(self, s, a, r, ns, extra=0.0):
        self._key = (s, a)
        tracker = self._tracker(self._key)
        delta = DynaQ.update(self, s, a, r, ns, extra=self._mood_term())
        if (not self._in_planning) or self.mood_updates_from == "real+planning":
            tracker.update(delta, self.eta)
            self.mood.M = tracker.M      # so mood_trace still logs something
        return delta


class YokedMood(MoodyDynaQ):
    """M read from a prerecorded trace instead of this agent's own deltas."""

    def __init__(self, *args, trace=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace, self.i = trace, 0

    def update(self, s, a, r, ns, extra=0.0):
        delta = DynaQ.update(self, s, a, r, ns, extra=self._mood_term())
        if not self._in_planning:
            self.mood.M = self.trace[min(self.i, len(self.trace) - 1)]
            self.i += 1
        return delta


def build(kind, seed, alpha=ETA, lam=LAM, offset=0.0, trace=None):
    va = SPEC["valid_actions"]
    common = dict(epsilon=EPSILON, planning_steps=PLANNING, seed=seed)
    if kind == "baseline":
        return DynaQ(va, alpha=alpha, **common)
    if kind == "const":
        return ConstOffset(va, alpha=alpha, offset=offset, **common)
    cls = {"mood": MoodyDynaQ, "antimood": AntiMood,
           "local": LocalMood, "yoked": YokedMood}[kind]
    extra = {"trace": trace} if kind == "yoked" else {}
    return cls(va, eta=alpha, lam=lam, **common, **extra)


def run(agent):
    """Both phases. Per episode: greedy route, M, mean |Q - Q*| over all pairs."""
    log, mood_trace, q_err = [], [], []
    for phase in (1, 2):
        env = Maze(CFG[phase])
        for i in range(N_EP):
            run_episode(env, agent)
            log.append({
                "episode": (phase - 1) * N_EP + i,
                "phase": f"phase{phase}",
                "greedy_route": ("lower" if agent.greedy_action("start") == "down"
                                 else "upper"),
            })
            mood_trace.append(agent.mood.M if hasattr(agent, "mood") else 0.0)
            qs = QSTAR[phase]
            q_err.append(np.mean([abs(agent.q(s, a) - qs[(s, a)]) for (s, a) in qs]))
    return log, np.array(mood_trace), np.array(q_err)


def conv(log, phase, target):
    """Episodes to greedy convergence within the phase; censored at N_EP."""
    c = episodes_to_greedy_convergence(log, phase, target)
    return N_EP if c is None else c - (N_EP if phase == "phase2" else 0)


def first_and_flips(log):
    """Phase 2 first greedy 'upper' (episodes from the switch; censored at
    N_EP) and greedy flips after it."""
    g = [r["greedy_route"] for r in log if r["phase"] == "phase2"]
    first = g.index("upper") if "upper" in g else N_EP
    return first, sum(g[i] != g[i - 1] for i in range(first + 1, len(g)))


def sweep(kind, **kw):
    """Per-seed metrics for one arm."""
    out = {k: [] for k in ("g1", "g2", "f2", "k2", "err1", "err2", "m1")}
    for seed in SEEDS:
        log, m, err = run(build(kind, seed, **kw))
        out["g1"].append(conv(log, "phase1", "lower"))
        out["g2"].append(conv(log, "phase2", "upper"))
        f2, k2 = first_and_flips(log)
        out["f2"].append(f2); out["k2"].append(k2)
        out["err1"].append(err[N_EP - 1])
        out["err2"].append(err[-1])
        out["m1"].append(m[:30].mean())
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def ms(a):
    return f"{a.mean():6.2f} ± {a.std(ddof=1) / np.sqrt(a.size):.2f}"


def main():
    print(f"eta={ETA}  epsilon={EPSILON}  planning_steps={PLANNING}  "
          f"lower_len={LOWER_LEN}  upper_len=2  lam={LAM} "
          f"(half-life {half_life_steps(LAM):.1f} real steps)  "
          f"episodes/phase={N_EP}  seeds={len(list(SEEDS))} (paired)  "
          f"mood_updates_from=real  mood_biases=real\n")

    # baseline's alpha -> episodes curve, used to express each arm as an
    # effective learning rate. Monotone decreasing, so invert by first crossing.
    curve = {a: sweep("baseline", alpha=a) for a in ALPHAS}
    means = {a: (curve[a]["g1"].mean(), curve[a]["g2"].mean()) for a in ALPHAS}

    def eff_alpha(target, idx):
        hits = [a for a in ALPHAS if means[a][idx] <= target]
        return f"{min(hits):.3f}" if hits else f">{ALPHAS[-1]:.3f}"

    arms = [("baseline", {}), ("mood", {}), ("local", {}), ("yoked", {}),
            ("antimood", {}), ("const", {})]
    # learning-rate-matched baselines (2026-09-27): full metrics, not just the
    # sustained-convergence curve used for eff. alpha
    arms += [(f"a{a:.2f}", {"alpha": a}) for a in (0.12, 0.15, 0.19)]
    res = {}
    for name, kw in arms:
        if name == "yoked":
            # Per-STEP trace: YokedMood reads one value per real step. (Before
            # 2026-09-27 this passed run()'s per-episode trace, so the yoked M
            # ran ~3x fast and froze after ~250 episodes.)
            traces = {}
            for s in SEEDS:
                a = build("mood", s)
                run(a)
                traces[s] = a.mood_trace
            res[name] = {k: np.asarray(v, dtype=float) for k, v in
                         _yoked(traces).items()}
            continue
        if name == "const":
            # mood's mean M over the first 30 episodes of phase 1, made constant
            kw = {"offset": (1 - ETA) * res["mood"]["m1"].mean()}
        res[name] = sweep("baseline" if name.startswith("a0") else name, **kw)
        res[name + "_kw"] = kw

    off = res["const_kw"]["offset"]
    print("| arm | P1 episodes | P2 episodes | eff. alpha P1 | eff. alpha P2 | "
          "final |Q−Q*| P1 | P2 |")
    print("|---|---|---|---|---|---|---|")
    labels = {"baseline": "baseline", "mood": "mood (global)",
              "local": "mood, per-(s,a) local", "yoked": "mood, yoked to other seed",
              "antimood": "antimood (sign flipped)",
              "const": f"const offset {off:+.3f}"}
    for name in ("baseline", "mood", "local", "yoked", "antimood", "const"):
        r = res[name]
        print(f"| {labels[name]} | {ms(r['g1'])} | {ms(r['g2'])} | "
              f"{eff_alpha(r['g1'].mean(), 0)} | {eff_alpha(r['g2'].mean(), 1)} | "
              f"{ms(r['err1'])} | {ms(r['err2'])} |")

    print("\nAll four convergence measures (Phase 2 first passage and flips "
          "after it added 2026-09-27):")
    print("| arm | P1 sustained | P2 first passage | P2 sustained | P2 flips |")
    print("|---|---|---|---|---|")
    labels.update({"baseline": "baseline α=0.10", "a0.12": "baseline α=0.12",
                   "a0.15": "baseline α=0.15", "a0.19": "baseline α=0.19"})
    for name in ("baseline", "a0.12", "a0.15", "a0.19", "mood", "yoked", "local"):
        r = res[name]
        print(f"| {labels[name]} | {ms(r['g1'])} | {ms(r['f2'])} | "
              f"{ms(r['g2'])} | {ms(r['k2'])} |")

    print("\nbaseline alpha curve (episodes to convergence):")
    print("| alpha | " + " | ".join(f"{a:.3f}" for a in ALPHAS[::2]) + " |")
    print("|---|" + "---|" * len(ALPHAS[::2]))
    for idx, ph in ((0, "P1"), (1, "P2")):
        print(f"| {ph} | " + " | ".join(f"{means[a][idx]:.1f}" for a in ALPHAS[::2]) + " |")


def _yoked(traces):
    out = {k: [] for k in ("g1", "g2", "f2", "k2", "err1", "err2", "m1")}
    seeds = list(SEEDS)
    for i, seed in enumerate(seeds):
        other = traces[seeds[(i + 7) % len(seeds)]]
        log, m, err = run(build("yoked", seed, trace=other))
        out["g1"].append(conv(log, "phase1", "lower"))
        out["g2"].append(conv(log, "phase2", "upper"))
        f2, k2 = first_and_flips(log)
        out["f2"].append(f2); out["k2"].append(k2)
        out["err1"].append(err[N_EP - 1])
        out["err2"].append(err[-1])
        out["m1"].append(m[:30].mean())
    return out


if __name__ == "__main__":
    main()
