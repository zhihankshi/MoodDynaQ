# CLAUDE.md — Mood in Tabular Dyna-Q

Read this whole file before changing anything. It records the theory, the
design decisions and the reasons for them, and the mistakes this project has
already made once. Most of those mistakes came from choices made implicitly
inside code. The rule here is: decisions live in this file, and code matches it.

---

## 1. What this project is

We test whether a specific computational theory of mood (Emanuel & Eldar, 2023)
produces its predicted effect on learning when built into a reinforcement
learning agent that has to learn from scratch.

The theory has mostly been tested by fitting mood models to human behavior after
the fact. We ask the forward question: implement the mechanism exactly as
specified, and see what it does to learning.

This is **preliminary, scoping work**. The goal is a small, correct, legible
setup that later work can build on. It is not a paper. Keep it simple.

---

## 2. Research question and hypotheses

**Question:** Does the Emanuel & Eldar mood mechanism change how quickly a
learning agent acquires an optimal route, and how quickly it adapts when that
route stops being optimal?

**Primary measure:** episodes until the **greedy policy** (ε=0) selects the
currently-optimal route, sustained for 20 consecutive episodes. See §9 for why
greedy policy rather than a rolling window of executed routes.

**Hypotheses** (stated as three outcomes, because the project did not start
with a committed directional prediction and we are not pretending otherwise):

- **H0:** no difference between mood and baseline agents.
- **H1:** mood speeds convergence. Grounded in the theory itself: mood
  integrates recent value-prediction errors, and Emanuel & Eldar derive the
  biased update as an optimal (Kalman-style) estimator where value is position
  and mood is velocity.
- **H2:** mood slows convergence. Grounded in our own derivation (§3.4): a
  sustained mood acts as a per-step reward bonus that shifts long and short
  routes by different amounts.

Report the measure separately for Phase 1 (acquisition) and Phase 2
(adaptation). They test different things.

---

## 3. Theory

### 3.1 Mood update (Emanuel & Eldar Eq. 3.3.2)

    M <- M + (1 - lam) * (eta * delta - M)

Equivalently `M <- lam*M + (1-lam)*eta*delta`: an exponential moving average of
recent value changes, with retention `lam` per update.

### 3.2 Value update (Emanuel & Eldar Eq. 3.4.1)

    Q <- Q + eta*delta + (1 - eta)*M

**`eta` is both the learning rate and, through `(1-eta)`, the mood weight.**
They are not independent parameters. Eq. 3.4.1 is derived as an optimal
estimator where the weights on the new observation and on the velocity estimate
sum to 1.

Consequence: at the usual tabular learning rate `eta=0.1`, the mood weight is
0.9. Mood is a **major** term here. (In the earlier DQN version `eta=0.9` made
it 0.1, which is part of why nothing happened there.) `mood_weight` can be
overridden for sensitivity analysis, but that departs from the theory, so it is
off by default and any use of it must be stated.

### 3.3 Lambda has units — always state them

`lam` is retention **per mood update**. With `mood_updates_from="real"`, one
mood update = one real environment step.

    half-life (updates) = ln(0.5) / ln(lam)
    lam=0.8 -> 3.1    lam=0.95 -> 13.5    lam=0.99 -> 69

Episodes here are 2–4 real steps, so `lam=0.8` decays **within** an episode.
Emanuel & Eldar describe λ as the dial from momentary emotion (λ near 0) to
lasting mood (λ near 1). Choose λ from the half-life you want, and write the
half-life down, not just λ. `mood.lam_for_half_life()` does the conversion.

### 3.4 Mood cannot move the fixed point

At convergence the expected update is zero:
`eta*E[delta] + (1-eta)*M = 0`, so `E[delta] = -(1-eta)M/eta`. The mood update's
own fixed point is `M = eta*E[delta]`. Substituting gives `M(2-eta) = 0`, so
**M\* = 0** for any `eta < 2`.

Mood is self-annihilating: any sustained M produces a δ of the opposite sign
that drives it back to zero. **Mood can only affect the transient — the path to
convergence, never the destination.** Mood and baseline agents converge to the
same Q-values. Any hypothesis phrased as "mood changes what is learned" is wrong
on the math before anything runs.

During the transient, a sustained M acts like a constant per-step reward bonus

    b = (1 - eta) * M / eta

which shifts the value gap between two routes by `b * dL`. Route preference
flips when `|b| * dL > margin`, i.e. when

    |M| > margin * eta / ((1 - eta) * dL)        (mood.flip_threshold)

At `eta=0.1, margin=5, dL=2` that threshold is **|M| > 0.278**.

---

## 4. Why tabular Dyna-Q, not DQN

Honest answer: the project began with a DQN inherited from an earlier maze where
the agent had to infer causal structure from pixels. This maze has only a few
states and does not need function approximation. DQN also made the mood effect
hard to probe (see §10). A small tabular agent lets us compute exact optimal
values by hand and check the implementation against them.

Do not reintroduce function approximation, CNNs, or pixel observations without
a specific reason recorded here.

---

## 5. Environment (`env.py`)

Two corridors from a single decision point. No way back; corridors never merge.

    start --right--> [trap] ... --> u_goal        upper: upper_len steps, unprotected trap
      |
     down
      |
    shield --> [trap] ... --> d_goal              lower: lower_len steps, protected in Phase 1

- The **only decision is at `start`**. Every other state is a forced corridor.
- Because corridors never merge, shield possession is determined by which
  corridor the agent is in. It is **not** part of the state.
- Illegal actions are masked everywhere: exploration, greedy argmax, and the
  bootstrap max.
- γ = 1. Episodes end only at a goal.

**Rewards** — all negative, no goal bonus. (A constant on every terminal
transition shifts all Q equally and cannot change the argmax.)

| | |
|---|---|
| step cost `c` | every transition |
| trap cost `T` | extra, entering an unprotected trap |
| protected trap `t` | extra, entering the lower trap: `0` in Phase 1, `T` in Phase 2 |

**Margin arithmetic** (`dL = lower_len - upper_len`):

    Phase 1 margin (lower wins) = (T - t) - c*dL
    Phase 2 margin (upper wins) = c*dL
    Matched margins  <=>  c*dL = (T - t)/2

`Config.matched(lower_len=, phase=)` solves for `c` so both margins equal 5.

**Default geometry:** `upper_len=2, lower_len=4, T=10`, so `c=2.5`.

| | Q\*(start, right) | Q\*(start, down) | optimal | margin |
|---|---|---|---|---|
| Phase 1 | −15 | −10 | lower | 5 |
| Phase 2 | −15 | −20 | upper | 5 |

**Known structural asymmetry (not a bug, but it matters):** the upper trap is on
the *first* step from `start`; the lower trap is on the *second*. So in Phase 1,
`Q(start,right)` sees the −10 on its very first update — the agent can reject the
upper route by one-step lookahead, no propagation needed. In Phase 2, the lower
trap's new cost is two steps out and reaches `Q(start,down)` only by
propagation. **Phase 2 is structurally harder than Phase 1, for reasons that
have nothing to do with mood.** Don't read a phase difference as a mood effect.
Moving the upper trap one step later would equalize this; that is an open
design choice, not yet made.

Corridor length and step cost are interchangeable for the *decision* (only
`c*dL` enters the margin) but not for *learning*. Counterintuitively, under
matched margins longer corridors converge faster, because `c = 5/dL` shrinks and
the trap dominates the return more cleanly. `planning_steps` is by far the
stronger difficulty lever (§6).

---

## 6. Agents

### 6.1 Baseline — `agent.DynaQ`

Three parts per real step (Sutton & Barto):
1. **Direct RL:** Q-learning update from the real transition.
2. **Model learning:** `Model[(s,a)] <- (r, s')`. One entry per pair, overwritten.
3. **Planning:** `planning_steps` times, sample a previously-seen `(s,a)`, read
   `(r, s')` from the model, apply the same update.

Planning never touches the environment and never writes to the model. It reads
fixed `(r, s')` from the model and reads the **current** `max Q(s')` from the
Q-table, which may have been changed moments earlier by another planning update.
That is how value propagates backward without new experience. In this
deterministic environment, planning is deduplicated replay.

`agent.update()` takes an `extra` argument, 0 for baseline. It is the single
insertion point for the mood term.

### 6.2 Mood — `mood.MoodyDynaQ`

Identical to baseline except the `(1-eta)*M` term. Two **independent** switches:

- **`mood_updates_from`** — which δs feed the mood tracker. `"real"` is
  theory-faithful: mood integrates *experienced* value updates. Planning δs are
  simulated and temporally scrambled.
- **`mood_biases`** — which Q-updates get the mood term. `"real"` is
  theory-faithful, **but** with `planning_steps=n` only `1/(1+n)` of updates
  then carry mood, so planning dilutes it.

Both default to `"real"`. The effect of `mood_biases="real+planning"` at `n>0`
is **untested** and is the most important open design question (§8).

### 6.3 `planning_steps` must be low to see anything

At `planning_steps=5`, baseline converges in under 2 episodes and the two agents
are identical to the episode. There is no room for an effect. Mean episodes to
Phase 1 convergence:

| planning_steps | 0 | 1 | 5 |
|---|---|---|---|
| episodes (10 seeds) | 60.0 | 1.9 | 0.6 |

Run comparisons at `planning_steps=0` (or 1). This is a design choice, not a
default — state it in every result.

### 6.4 ε matters at planning_steps=0

Whether exploration matters depends on planning. Baseline Phase 2 adaptation,
episodes after the switch, 8 seeds:

| ε | 0.05 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|
| planning_steps=0 | 37.2 | 31.2 | 20.2 | 18.4 |
| planning_steps=5 | 3.9 | 3.5 | 3.5 | 3.2 |

With planning, value propagation is the bottleneck and ε barely matters. With
**no** planning — the setting we actually use — the agent must physically visit
states to learn, so ε roughly halves adaptation time across this range. The
floor's value sets how much room there is for a mood effect: a higher ε makes
baseline faster and may shrink the gap. Choose it deliberately, report it, and
consider sweeping it alongside λ.

---

## 7. Settled design decisions

| Decision | Why |
|---|---|
| Two phases only, 500 or fewer episodes each | A third switch (back to shield-works) is silent: the habitual upper route's return doesn't change, so there is no prediction error for mood to integrate. Dropped. |
| ε held at a constant floor, never reset at the switch | Raising ε at the switch hands every agent an external cue that something changed — exactly what mood is supposed to supply internally. The floor is identical for both agents, so it is not a confound. **But its value matters** — see §6.4. |
| Mood persists across the switch, never reset | Detecting the change is the hypothesis. Resetting M erases the signal at the moment it should fire. |
| No mood clip by default | The paper has none. The old DQN clip of ±1 discarded 87% of the magnitude of trap events — the most informative ones. |
| Shield pickup reward = 0 | A pickup bonus can exceed the detour cost and keep the shield route optimal even when the shield does nothing. Then Phase 2 has nothing to reverse. |
| Both phases pixel-identical / state-identical | Only the lower trap's cost changes. Nothing observable signals the switch. |
| Seed-paired comparison | Same seed for baseline and mood agent in each pair. |

---

## 8. Current results and open questions

**At `eta=0.1, planning_steps=0`, 30 seeds** (`figures/fig3_lambda_sweep.png`):

| | baseline | mood, λ=0.5 | mood, λ=0.7 | mood, λ=0.99 |
|---|---|---|---|---|
| Phase 1 episodes | 62.5 ± 2.8 | 42.2 ± 0.9 | 43.9 ± 1.1 | 60.9 ± 0.9 |
| Phase 2 episodes | 27.2 ± 1.8 | 13.9 ± 0.4 | 13.2 ± 0.3 | 26.4 ± 0.6 |

- Mood speeds both acquisition and adaptation across λ from 0.5 to 0.95.
- The λ curve is **flat across 0.5–0.9** and collapses to baseline by 0.99.
  (An earlier 10-seed run read as a peak at 0.7; at 30 seeds Phase 1 is
  slightly best at 0.5 and Phase 2 at 0.7, within ~2 SEM of each other. Do not
  claim a peak.)
- Mood responds sharply to the switch: per-episode M drops from ~0 to −0.245
  within two episodes and recovers over ~17 (`fig2`, `fig5`). The DQN version
  never showed this.
- Direction is **opposite** to the earlier DQN preliminary result (mood slower).
  Those DQN runs had a mood implementation bug (§10), so they did not test a
  working mood signal.

**Planning steps, 30 seeds** (`fig4_planning_sweep.png`, `fig5_mood_at_switch.png`;
full tables in `PREDICTIONS.md`):

| Phase 2 episodes to converge | n=0 | n=1 | n=2 | n=5 | n=10 | n=20 |
|---|---|---|---|---|---|---|
| baseline | 27.2 | 10.1 | 7.3 | 3.7 | 2.0 | 1.07 |
| mood | 13.2 | 8.5 | 6.3 | 3.5 | 1.9 | 1.07 |
| mood advantage | 52% | 16% | 14% | 5% | 7% | 0% |

- **Planning dilutes the mood effect but does not abolish it** until n=20.
  Phase 1 is uninformative past n=1: baseline already converges in ~1 episode,
  so the metric bottoms out and both agents return identical numbers.
- **Open question 2 is answered, negatively.** `mood_biases="real+planning"`
  matches `"real"` in Phase 2 (inside SEM) and is *worse* in Phase 1 at n≥2
  (1.30 vs 0.80 episodes at n=10). Biasing simulated, temporally scrambled
  planning updates buys nothing. Keep `"real"`.
- **M's response to the switch is monotone in planning**: trough −0.245 → −0.167
  and recovery 17 → 5 episodes as n goes 0 → 20. More planning corrects Q
  faster, so the burst of large negative δ is shorter and the EMA accumulates
  less of it.
- **The trough never reaches the flip threshold** (88% of it at n=0, less at
  every larger n). The speedup is not mood crossing the threshold and flipping
  the greedy action. Mechanism still unknown — see §9 and open question 1.

**Open questions, roughly in priority order:**

1. **What mechanism drives the speedup?** The per-step-bonus route-flipping
   story (§3.4) is *not* it — see §9, and the threshold result above. Work this
   out before scaling up.
2. Phase 1 at `planning_steps=1`: mood reads slower than baseline (2.50 ± 0.87
   vs 1.17 ± 0.40). SEMs overlap; needs more seeds before it is called either
   way. It is the only condition where mood may hurt.
3. Sweep `eta`. The theory couples learning rate and mood weight, so this is a
   sweep over "how much mood," not just "how fast."
4. Equalize the trap-position asymmetry (§5) and check whether the Phase 1 /
   Phase 2 difference survives.
5. More seeds before any claim leaves this repo. Runs are cheap — a full
   `plots.py` pass at 30 seeds is ~9 s, so seed count is not the constraint.

## 9. Predictions that turned out wrong — do not re-derive them

**"At eta=0.1 mood will dominate and block Phase 1 acquisition."** Reasoning:
M should approach `eta*delta ≈ -0.5 to -1.5`, well past the 0.278 flip
threshold, and negative mood penalizes the longer (optimal) lower route. **It
didn't happen.** At `lam=0.95`, M peaked at 0.252 — 91% of the threshold —
because mood is an EMA chasing a δ that shrinks faster than mood can track.
Lowering λ does push M past the threshold (0.756 at λ=0.5), and the agent
**still** converges faster, not slower. So the flip-threshold story does not
explain the observed effect.

**"A longer corridor makes the task harder to learn."** Under matched margins it
makes it *easier*, because the step cost shrinks. See §5.

**Why greedy policy, not an 80% rolling window:** with a constant ε floor, the
*executed* route can never hit 100% even after the agent has fully learned, so a
rolling window conflates "has it learned" with "is exploration noise hiding it."
Reading the greedy action at `start` measures what the Q-table says, free of
exploration noise.

---

## 10. History: what went wrong in the DQN version

Kept so it isn't repeated. The pattern behind all of these: **a consequential
choice made implicitly inside code, never written down or checked.**

- Mood applied 32 updates per gradient step (once per replay-batch element).
  With λ=0.8 that is `0.8^32 ≈ 8e-4` retention per step — mood was wiped every
  step and became near-white noise.
- Mood fed temporally scrambled replay δs instead of experienced δs in order.
- `eta=0.9` — the minimum-mood end of the range, chosen without noticing.
- λ set per step when it was meant per trial; 3-step half-life inside 14-step
  episodes (an emotion, not a mood).
- Mood clipped to ±1 against rewards of ±55.
- DQN target-network sync every 1000 updates gated value propagation; Phase 2
  never trained, and agents collapsed into timeouts.
- Cluster (SLURM) scripts were written against an invented command-line
  interface and never worked. **Never write code that calls another script
  without reading that script's actual argument parser first.**

---

## 11. Rules for working in this repo

1. **Run `python3 test_correctness.py` before and after any change.** It checks
   learned Q-values against hand-computed ones. If it fails, nothing involving
   mood is interpretable.
2. **Hand-computed values in the tests are hardcoded on purpose.** They are an
   independent check on `optimal.py`. If you change the default geometry, redo
   the arithmetic by hand and update them. Do not derive them from the code.
3. **State units and design choices in every result:** `eta`, `lam` *and its
   half-life*, `planning_steps`, `mood_updates_from`, `mood_biases`, geometry,
   seeds. A number without these is not a result.
4. **Predict before you run.** Write the expected outcome down, then check.
   §9 exists because predictions were wrong in instructive ways.
5. **Add the minimum.** No new abstractions, config systems, logging frameworks,
   or dependencies unless something concrete requires them. If a change adds
   more than it clarifies, don't make it.
6. **Surface design changes immediately.** If you change the environment, the
   update rule, a default, or a metric, say so explicitly and update this file.
   Silent redesigns are how this project lost months.
7. **Don't infer, check.** If something depends on how another file behaves,
   read that file.

---

## 12. Plotting rules

`plots.py` is the reference implementation. New figures follow it.

- Baseline **blue `#2b6cb0`**, mood **red `#c53030`**, in every figure.
- Mean as a thick line, **SEM as a shaded band**, individual seeds as faint lines
  underneath. Say `n=` in the legend.
- Mark the phase switch with a **dashed vertical line**, and label both phases.
- **Index time by episode**, not raw step, whenever phases are compared. Seeds
  take routes of different lengths, so the switch lands at a different raw step
  for each; averaging by step blurs it.
- Write the parameters that produced the figure **on the figure** (`param_note`).
- Label every axis, with units. Remove top and right spines.
- Check the output by viewing it. Look for overlapping labels.

Core figures:
- `fig1_learning_curves` — P(greedy policy optimal) over episodes, both phases.
- `fig2_mood_trace` — per-episode mean M, aligned on the switch.
- `fig3_lambda_sweep` — episodes to convergence vs λ, baseline as reference band.
- `fig4_planning_sweep` — episodes to convergence vs `planning_steps`, three
  arms (baseline, mood bias real, mood bias real+planning), both phases.
- `fig5_mood_at_switch` — M aligned on the switch, one curve per
  `planning_steps`.

Two further conventions, learned the hard way:

- **Smooth each phase separately** (`smooth_by_phase`). A window straddling the
  switch averages pre- and post-switch episodes together and blurs the exact
  transition the figure exists to show.
- `param_note` takes a `loc`. The default corner is often occupied — check the
  rendered png and move it rather than leaving text on a line or a curve.

---

## 13. Files

| file | purpose |
|---|---|
| `env.py` | maze, `make_maze_spec`, `Config.matched` |
| `optimal.py` | exact Q\* by backward induction |
| `agent.py` | baseline `DynaQ` |
| `mood.py` | `MoodTracker`, `MoodyDynaQ`, `flip_threshold`, λ helpers |
| `train.py` | two-phase loop, `episodes_to_greedy_convergence` |
| `test_correctness.py` | correctness gate — run first |
| `trace.py` | prints every update with its arithmetic; teaching tool |
| `plots.py` | reference figures and plotting standard |
| `mood_predict.py`, `mood_diag.py`, `lam_sweep.py`, `mood_compare.py` | exploratory scripts behind §8–9 |
| `PREDICTIONS.md` | predictions recorded before a run, with the outcome appended (rule 4) |

---

## 14. References

- Emanuel, A., & Eldar, E. (2023). Emotions as computations. *Neuroscience &
  Biobehavioral Reviews*, 144, 104977. — Eqs. 3.3.2 and 3.4.1.
- Eldar, E., Rutledge, R. B., Dolan, R. J., & Niv, Y. (2016). Mood as
  representation of momentum. *Trends in Cognitive Sciences*, 20(1), 15–24.
- Bennett, D., Davidson, G., & Niv, Y. (2022). A model of mood as integrated
  advantage. *Psychological Review*, 129(3), 513–541.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). — Dyna-Q, Ch. 8.

An abstract based on the earlier DQN work was submitted to the JHU DSAI
symposium (Oct 27, 2026). If accepted, the poster may report these Dyna-Q
results instead; flag any claim that differs from the abstract.
