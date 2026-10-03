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
Moving the upper trap one step later was tried (scratch prototype, 2026-09-23;
`PREDICTIONS.md`) and **does not help**: Phase 1 first passage stays a coin
flip on episode 0 (it is set by tried-vs-untried Q, not trap position), and
Phase 1 flips roughly double (18 → 32 at n=0). Not adopted.

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

- Mood shortens sustained convergence in both phases across λ from 0.5 to
  0.95. **The Phase 2 number is not an adaptation measure on its own:** about
  half of it is inherited from the Phase 1 end state (see "Phase 2 gap split"
  below). Do not describe 28 vs 13 as mood improving adaptation.
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
  the greedy action.

**Mechanism: mood is per-entry momentum — an effective learning rate.**
Answered 2026-09-21 by `control_arms.py`; full tables in `PREDICTIONS.md`.
Settings as above, λ=0.7, 30 paired seeds.

| arm | P1 episodes | P2 episodes | eff. alpha P1 | eff. alpha P2 | final \|Q−Q*\| P1 | P2 |
|---|---|---|---|---|---|---|
| baseline | 62.5 ± 2.8 | 27.2 ± 1.8 | 0.100 | 0.100 | 0.20 | 0.30 |
| mood (global) | 43.9 ± 1.1 | 13.2 ± 0.3 | 0.150 | 0.150 | 0.12 | 0.76 |
| mood, per-(s,a) local | 35.9 ± 0.9 | 10.2 ± 0.1 | 0.170 | 0.165 | 0.02 | 0.03 |
| mood, yoked to another seed ‡ | 44.1 ± 0.8 | 15.1 ± 0.7 | 0.150 | 0.140 | 0.20 | 0.35 |
| antimood (sign flipped) | 0.5 ± 0.1 | 220.4 ± 7.7 | >0.210 | 0.100 | 2.74 | 3.08 |
| const offset −0.146 | 94.7 ± 2.0 | 14.9 ± 1.1 | 0.100 | 0.140 | 3.07 | 2.63 |

‡ Corrected 2026-09-27. The original row (58.0 / 27.0) came from a bug:
`control_arms.py` passed the per-**episode** M trace (800 values) to
`YokedMood`, which reads one value per real **step** (~2400), so the yoked M
ran ~3× fast and froze on its last value after ~250 episodes. Fixed to pass
the per-step `mood_trace`; every other row reproduces unchanged. Of the bullets
below, the "Why" bullet (momentum from correlated δ) and the Kalman bullet
leaned on the old yoked row and are **now in question** — the corrected
yoked arm gets nearly the whole effect without any correlation to its own δ.

- **One effective learning rate explains both phases.** Mood at `eta=0.1`
  matches a baseline at `alpha ≈ 0.15` in Phase 1 *and* Phase 2. Across λ the
  implied alpha stays in 0.125–0.17 for λ=0–0.95 and returns to 0.105 at λ=0.99.
- **Why.** Consecutive δs on the same `(s,a)` are correlated during a transient,
  so `M ≈ eta*delta_recent` and `Q += eta*delta + (1-eta)*M` behaves as
  `Q += eta_eff*delta` with `eta_eff ≈ eta*(1 + (1-eta)*rho)`, capped at 0.19 at
  `eta=0.1`. λ sets `rho`: small λ tracks the current δ, λ→1 averages it away.
  **This is why the λ curve is flat over 0.5–0.9 and collapses at 0.99.**
- **It is not a global affective signal.** Giving every `(s,a)` its own
  independent `MoodTracker` — zero cross-state coupling — makes it *faster*
  (35.9 / 10.2), with a far smaller final Q error.
- **It is not a sustained pessimism offset.** A constant `(1-eta)*M̄ = −0.146`
  makes Phase 1 markedly *worse* and Phase 2 better: exactly the §3.4 per-step
  bonus trade. Mood improves both phases, so it is not that.
- **It does not require M to correlate with the agent's own δ** (corrected
  2026-09-27; the earlier claim that it does came from the yoked bug above).
  Yoking M to another seed's per-step trace — same magnitude and time course,
  uncorrelated with this agent's δ — reproduces mood's Phase 1 result (44.1 vs
  43.9) and most of Phase 2 (15.1 vs 13.2). Together with the constant-offset
  arm (mood's mean M, no time course: much worse in Phase 1), this points to
  **M's time course** — a large early negative pulse that decays — rather than
  per-entry momentum. Not yet tested directly.
- **Consistent with Emanuel & Eldar's Kalman framing** (mood = velocity), but it
  means the result does **not** show mood supplying information the agent did
  not already have in its own last δ.

**Mood contaminates entries whose prediction error is zero.**
Found 2026-09-22 from `fig6_q_snapshots`; numbers are 30 seeds, Phase 2,
`eta=0.1, lam=0.7, planning_steps=0, eps=0.1`, mean learned Q.

| Phase 2, mean Q | `d_1` (Q\*=−5.0) | `d_2` (Q\*=−2.5) | `shield` (Q\*=−17.5) |
|---|---|---|---|
| baseline α=0.10 | −5.00 | −2.50 | −16.27 |
| baseline α=0.15 (rate-matched) | −5.00 | −2.50 | −16.66 |
| mood η=0.10 | −7.49 | −3.40 | −18.86 |

- `d_1` and `d_2` are **downstream of the change**: the switch alters only the
  trap entered *on* the step into `d_1`, so those two entries already hold their
  exact Phase 2 values and their δ is 0. Baseline therefore never moves them —
  at either learning rate. Mood moves them by up to −2.6, because
  `Q += eta*delta + (1-eta)*M` adds `(1-eta)*M` **whether or not there is a
  prediction error**. A negative M after the switch drags every entry the agent
  touches further negative, correct ones included.
- This is the §3.4 per-step bonus acting on converged entries. It is transient
  in theory (M\* = 0), but it decays only through revisits, and after the greedy
  policy has moved to the upper route the lower corridor is visited at rate ε —
  so the error is still there 100 episodes later.
- It is **not** a learning-rate effect: the rate-matched baseline shows none of
  it. This is the clearest thing found so far that a learning rate cannot
  reproduce, and it is a cost, not a benefit. It is the mechanism behind the
  "faster on behaviour, *less* accurate on value" dissociation below.

**Two consequences for how results are reported:**

1. **Every mood-vs-baseline comparison needs a learning-rate-matched baseline
   arm.** Without it, "mood converges faster" and "mood updates Q faster" are
   the same statement. This control was missing from §8's headline numbers.
2. **Never report a Phase 1 number without Phase 2.** The antimood arm converges
   in 0.5 episodes in Phase 1 by pure stickiness — a positive push inflates
   whatever the agent just did — and then takes 220 episodes in Phase 2, with a
   final Q error of 3.1. Phase 1 alone cannot tell learning from lock-in.

**First passage vs sustained convergence.** Checked 2026-09-23 by
`first_passage.py` (settings as above, λ=0.7, 30 paired seeds; tables in
`PREDICTIONS.md`). Proposal was to count learning at the *first* episode the
greedy action is optimal (sustain=1), since the flips are a by-product of
optimistic `q_init` plus on-policy updating (fig8), not unstable learning.

| planning_steps=0 | P1 s=1 | P1 s=20 | P2 s=1 | P2 s=20 | P2 flips after 1st pass |
|---|---|---|---|---|---|
| baseline α=0.10 | 0.5 | 62.5 | 13.3 ± 0.3 | 27.2 ± 1.8 | 4.3 |
| baseline α=0.15 | 0.5 | 39.2 | 9.9 ± 0.2 | 11.4 ± 0.4 | 0.7 |
| mood λ=0.7 | 0.5 | 43.9 | 12.1 ± 0.2 | 13.2 ± 0.3 | 0.7 |

- **Phase 1 first passage is degenerate.** One upper run puts −12.5 into
  `Q(start,right)` (§5 asymmetry), so every arm is "correct" by episode 0–1,
  identical in 30/30 seeds. Do not use sustain=1 for Phase 1.
- **Phase 2 first passage is meaningful and shrinks the mood effect** from
  14 episodes (52%) to 1.3 (10%, 26/30 seeds), and mood is *slower* than the
  rate-matched α=0.15 baseline. Most of the sustain=20 advantage is baseline
  α=0.10 flipping back after first passage (4.3 vs 0.7 flips). Why: see
  "Why baseline flips back" below.
- **Report both** sustain=1 and sustain=20 for Phase 2. The headline metric is
  unchanged; sustain=1 is an added view, not a replacement.
- With planning, the metrics coincide (0 Phase 2 flips at n=1), confirming the
  flips are an on-policy artefact — but planning also shrinks the mood effect
  (§6.3), so it is not a free fix.

**Why baseline flips back: it leaves Phase 1 with an optimistic
`Q(start,right)`.** Checked 2026-09-27 from saved per-episode logs
(`phase2_gap.py` → `logs/phase2_gap/`, `fig9_phase2_gap`). `eta=0.1, lam=0.7`
(half-life 1.94 steps), `planning_steps=0, eps=0.1, lower_len=4, q_init=0`,
mood real/real, 400 ep/phase, 15 paired seeds, mean ± SEM.

| | baseline | mood |
|---|---|---|
| end P1 `Q(start,right)` (Q\* −15) | −13.95 ± 0.11 | −15.04 ± 0.06 |
| end P1 M | — | −0.00 ± 0.00 |
| P2 first greedy upper | 13.7 ± 0.4 | 12.4 ± 0.3 |
| P2 sustained (20) | 28.3 ± 2.9 | 13.1 ± 0.4 |
| P2 flips after first passage | 4.1 ± 0.7 | 0.5 ± 0.3 |
| at first passage: `Q(start,right)` / `Q(start,down)` | −14.00 / −14.19 | −15.11 / −15.37 |
| at first passage: `Q(d_1,right)` (Q\* −5) | −5.00 | −7.63 |

- The upper route is visited mostly at ε in Phase 1, so at α=0.1 baseline's
  `Q(start,right)` is still ~1 above −15 when Phase 2 starts (15/15 seeds).
  Mood's higher effective rate has it at Q\* already (11/15 seeds ≤ −15).
- **Every flip back is an upper run lowering `Q(start,right)`**: 31/31
  baseline, 4/4 mood, mean step −0.06. None came from a lower run raising
  `Q(start,down)`. Baseline crosses at ~−14.1, so each upper run pushes
  `Q(start,right)` back under `Q(start,down)`, until `Q(start,down)` falls past
  −15 — around episode 28, which is the sustained-convergence episode. Mood
  crosses at ≤ −15, so there is nothing left to undo.
- End-P1 `Q(start,right)` predicts P2 flip count across seeds: r = +0.71
  (baseline), +0.91 (mood; its 3 flipping seeds are all among the 4 above −15).
- **This revises the earlier "untested link".** Part of the sustained Phase 2
  advantage is set up **before** the switch (M is 0 at the end of Phase 1), by
  mood's faster learning of the rarely-visited upper route. The post-switch
  contamination (`Q(d_1,right)` −7.63 vs Q\* −5) also helps `Q(start,down)`
  fall past −15 sooner. The split is measured below.

**Phase 2 gap split: inherited table vs post-switch agent.** 2026-09-27,
`transplant.py` → `logs/transplant/`, same config as above (15 paired seeds).
Phase 2 is run by a fresh agent given a donor's end-of-Phase-1 Q table, model
and RNG state, with M=0. Own-table cells reproduce the ordinary runs exactly.

| Phase 2 starts from | agent | first upper | sustained (20) | flips |
|---|---|---|---|---|
| baseline's table | baseline | 13.7 ± 0.4 | 28.3 ± 2.9 | 4.1 ± 0.7 |
| baseline's table | mood | 10.1 ± 0.4 | 20.5 ± 1.5 | 3.6 ± 0.5 |
| mood's table | baseline | 17.0 ± 0.3 | 19.7 ± 0.7 | 1.3 ± 0.3 |
| mood's table | mood | 12.4 ± 0.3 | 13.1 ± 0.4 | 0.5 ± 0.3 |

- **The 15-episode sustained gap splits roughly in half.** Inherited table,
  agent held fixed: 8.5 episodes (baseline), 7.4 (mood). Post-switch agent,
  table held fixed: 7.7 ± 2.9 and 6.6 ± 0.6 (paired).
- **The inherited half is not mood-specific.** The corrected yoked agent ends
  Phase 1 with `Q(start,right)` −14.50 ± 0.05, between baseline (−13.95) and
  mood (−15.04), and flips 3.3 ± 0.5 times in Phase 2 against mood's 0.5 —
  which is why its sustained Phase 2 (15.5 at 15 seeds) is slower than real
  mood's (13.1) while its first passage (10.0) is the fastest.
- **The post-switch effect is ~4 episodes of first passage, not 1.3.** From a
  shared table mood reaches first passage 3.7 ± 0.2 / 4.6 ± 0.2 episodes
  earlier. The natural-run 1.3 hides this: mood's accurate `Q(start,right)`
  means `Q(start,down)` must fall further before the first crossing (baseline
  from mood's table: 17.0 vs 13.7 from its own). Whether this post-switch part
  is mood-specific or a learning-rate effect is untested (no α=0.15 arm here).

**The metric itself is sound.** It reads the greedy argmax at `start` and
nothing else (`train.run_phase` logs `greedy_route` — plus, since
2026-09-27, `q_shield_right`, `q_d1_right`, `q_d2_right` and end-of-episode
`mood_M` (NaN for baseline), logging only;
`episodes_to_greedy_convergence` reads only that field). It is not a Q-error
proxy, and the two dissociate: in Phase 2 mood converges behaviourally at
episode 13 while its final `|Q−Q*|` is 0.76 against baseline's 0.30 — faster on
behaviour, *less* accurate on value. What the concern gets right is upstream of
the metric: a faster descent produces an earlier correct argmax.

**Open questions, roughly in priority order:**

1. **Is there a task where mood does something a learning rate cannot?** The
   mechanism answer above ("mood is momentum") is **reopened** by the
   corrected yoked arm (2026-09-27): an M uncorrelated with the agent's own δ
   gets nearly the whole effect, so what matters may be M's time course. Test
   that first (e.g. a yoked trace time-shuffled within phase). Finding a setting where the *global*, signed,
   history-integrating character of M earns its keep — reward noise, multiple
   decision points, state aliasing — is now the central question. Work this out
   before scaling up.
2. Phase 1 at `planning_steps=1`: mood reads slower than baseline (2.50 ± 0.87
   vs 1.17 ± 0.40). SEMs overlap; needs more seeds before it is called either
   way. It is the only condition where mood may hurt.
3. Sweep `eta`. The theory couples learning rate and mood weight, so this is a
   sweep over "how much mood," not just "how fast." Now also the cleanest way to
   test the `eta_eff` prediction above.
4. Equalize the trap-position asymmetry (§5) and check whether the Phase 1 /
   Phase 2 difference survives.
5. Record `q_init` as a decision and justify it. Sweep 2026-09-23
   (`PREDICTIONS.md`): below 0, Phase 2 post-passage flips vanish but Phase 2
   slows sharply (baseline 13 → 29 → 88 episodes at q_init 0/−15/−25, n=0);
   Phase 1 first passage is 0.5 at every q_init. At the default `q_init=0` with
   all-negative rewards the init is optimistic, and the mood advantage depends
   on it: Phase 2 advantage is 52% at `q_init=0`, 26% at −12.5, 19% at −25.
6. More seeds before any claim leaves this repo. Runs are cheap — a full
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
- `fig0_maze` — the environment itself: structure, per-transition rewards,
  exact Q\* for both phases. A schematic, so the blue/red agent colours do not
  apply; everything drawn is read from `make_maze_spec`/`Config.matched`/
  `optimal.py` so it cannot drift from what the agents run in.
- `fig1_learning_curves` — P(greedy policy optimal) over episodes, both phases.
- `fig2_mood_trace` — per-episode mean M, aligned on the switch.
- `fig3_lambda_sweep` — episodes to convergence vs λ, baseline as reference band.
- `fig4_planning_sweep` — episodes to convergence vs `planning_steps`, three
  arms (baseline, mood bias real, mood bias real+planning), both phases.
  Two rows: absolute episodes on top, **paired** mood advantage (baseline minus
  mood, same seed) below, with the metric-floor region shaded. Linear y, not
  symlog: symlog magnified the `n>=2` floor, where both agents converge in under
  one episode and nothing is measurable, and compressed the `n=0` gap that is
  the actual result. Phase 2's tail gets a zoom inset, since "dilutes but does
  not abolish" has to be read at `n=1..10`.
- `fig4b_planning_sweep_2arm` — the same sweep with the answered
  `real+planning` arm dropped: baseline vs mood (bias real) only. Top row
  absolute episodes, bottom row advantage as a **percentage of baseline**
  (paired bootstrap 95% CI), because the absolute advantage shrinks partly
  because baseline itself shrinks. Region where baseline ≤2 episodes is
  shaded in both rows — the 20-episode sustain metric cannot resolve
  differences there, and the ratio's denominator is unstable.
- `fig5_mood_at_switch` — M aligned on the switch, one curve per
  `planning_steps`.
- `fig6_q_snapshots` — the Q table itself at fixed episodes (1, 10, 50, 100 of
  each phase), baseline and mood, **one seed** (a Q table averaged over seeds is
  not a table any agent held). One cell per state, coloured on a single scale
  shared by every panel; `start` is split in half (upper = `Q(start,right)`,
  lower = `Q(start,down)`) because it is the only two-action state, and the
  greedy half is outlined. The last column is the exact `Q*` for that phase,
  drawn identically — **the panels are meant to be read against it**, which is
  what makes the Phase 2 overshoot in §8 visible. A never-visited `(s,a)` is
  left blank rather than printed as `0.0`, which at `q_init=0` would read as a
  learned value.
  The first version of this figure had per-edge Q labels, state names in every
  panel, three-line titles and a footnote block, and was unreadable. Labels are
  not free: **a figure should carry one quantity, and the key belongs in one
  line at the foot.**
  **A snapshot tick is not convergence.** The ✓ under a panel is the greedy
  argmax in that one episode. The argmax is correct at episode 10 in 30/30
  seeds and still flips ~15 times over the next 90, because the two values at
  `start` stay within ~1 of each other for tens of episodes (read them in the
  figure: −2.4 vs −2.0 at ep 10, −8.8 vs −8.1 at ep 50). The row label carries
  the sustained-convergence episode so the figure cannot be read as
  contradicting §8. The flips are the metric's whole reason for existing (§9):
  the agent must visit the upper route often enough for `Q(start,right)` to
  fall below `Q(start,down)` and stay there. **Most of those visits are not ε.**
  `fig8` shows the flips are an alternation driven by the optimistic
  `q_init=0`: a greedy lower run lowers `Q(start,down)` below `Q(start,right)`,
  the argmax flips to upper, the next *greedy* upper run hits the trap and flips
  it back, and both values ratchet down together. Episodes 11→convergence,
  seeds 0–4 pooled: 33 upper runs greedy vs 12 ε (baseline), 27 vs 9 (mood). (An earlier
  version of this note said "at ε only"; that was wrong.) This ties the Phase 1
  metric to open question 5 (`q_init`).
- `fig8_q_flips` — Phase 1 only, one seed: the Q table at episode 10 and at
  every episode where the greedy argmax at `start` flips, until sustained
  convergence (the last flip is the convergence episode). Episodes between
  flips are omitted, because the argmax does not change there. Flips alternate,
  so each column is one pair: top = flips to upper, bottom = flips back. Under
  each panel is the route actually run in that episode, and whether it was
  greedy or ε. Built by `fig_q_flips` in `q_snapshots.py`, reusing `draw_panel`.
- `fig8a_q_first_flip` — the simple version of fig8: every episode from 10 to
  each agent's first flip away from lower (10, 11, 12 for both at seed 0), one
  row per agent, same captions as fig8. Its colour scale is fitted to these
  panels (−5 to 0), not shared with fig8. Built by `fig_q_first_flip`.
- `fig9_phase2_gap` — Phase 2 `Q(start,right)` (solid) and `Q(start,down)`
  (dashed) per episode, episodes −10 to +80 around the switch, both arms, mean
  ± SEM over seeds, with markers at mean first-passage and sustained episodes;
  side panel is the paired per-seed count of greedy flips after first passage.
  Built by `phase2_gap.py` from the saved logs in `logs/phase2_gap/`.
- `fig7_steps_per_episode` — steps per episode, both arms, both phases.
  **Degenerate by construction:** no way back and one decision at `start`, so an
  episode is exactly `upper_len` (2) or `lower_len` (4) steps and
  `mean steps = upper_len + dL * P(lower executed)`. It is the executed route
  rescaled, not an independent measure, and it carries the ε floor — which is
  why convergence is read off the greedy argmax instead (§9).

Two further conventions, learned the hard way:

- **Plot the quantity the title asks about.** If the question is about the gap
  between two arms, plot the paired difference, not two curves and an eyeball.
  Seeds are paired, so the paired SEM is the honest one.
- **Never let a log scale give floor noise the same ink as a real effect.**
  Shade the region where the metric has bottomed out and say so on the figure.
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
| `control_arms.py` | control arms for the mood effect: constant offset, sign-flipped, per-(s,a) local, yoked to another seed, plus a baseline alpha sweep to read each arm back as an effective learning rate |
| `q_snapshots.py` | Q-table snapshots at fixed episodes (`fig6`) and steps per episode (`fig7`); `steps_table()` prints the snapshot-episode step counts |
| `first_passage.py` | first-passage (sustain=1) vs sustained (20) greedy convergence and post-passage flip counts, baseline α=0.10/0.15 vs mood, planning 0/1/5 |
| `phase2_gap.py` | why the Phase 2 sustained and first-passage measures disagree: generates/saves per-episode logs to `logs/phase2_gap/` (15 paired seeds) and re-reads them; end-of-Phase-1 Q, first passage, sustained, flips, flip-back attribution; `fig9` |
| `transplant.py` | Phase 2 from a transplanted end-of-Phase-1 Q table (baseline's or mood's, M=0), both agents; plus the corrected yoked arm's end-P1 Q and P2 flips; logs to `logs/transplant/` |
| `convergence_table.py` | episodes-to-greedy-convergence as mean ± SD (and SEM, median, range, censored count), baseline vs mood, per phase |
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
