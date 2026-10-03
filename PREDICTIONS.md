# Predictions recorded before the planning-steps runs (2026-09-20)

Written before executing, per CLAUDE.md rule 4. Settings: eta=0.1, lam=0.7
(half-life 1.9 real steps), lower_len=4, epsilon=0.1, 30 seeds, 400 ep/phase.

## Fig 4 — episodes to convergence vs planning_steps

1. **Baseline collapses immediately.** Phase 1 ~60 episodes at n=0, under 1 at
   n>=1 (test_correctness already shows 60.0 -> 0.2). Phase 2 ~30 at n=0, ~3-4
   at n=5. Past n=1 there is no headroom left for any mood effect.
2. **`mood_biases="real"` dilutes toward baseline.** Only 1/(1+n) of updates
   carry the mood term, so the mood advantage seen at n=0 (~60->42 in P1,
   ~30->13 in P2) should shrink monotonically and be gone by n=5.
3. **`mood_biases="real+planning"` is the interesting arm.** Mood then rides
   every update, so it does not dilute. Either it preserves the speedup at
   n>0 (mood is doing real work), or it degrades (mood integrated over
   temporally scrambled simulated deltas is noise). Genuinely unsure which.
   This is CLAUDE.md §8 open question 2.

## Fig 5 — mood at the switch, by planning_steps

4. M dips negative at the switch (fig2 already shows ~-0.24 at n=0) and
   recovers.
5. **The dip should get shallower and briefer as planning_steps rises**, because
   planning corrects Q faster, so the burst of large negative delta is shorter.
   M is an EMA of eta*delta, so a shorter burst means less accumulation.
6. M returns to ~0 in every condition. §3.4 proves M* = 0; a sustained nonzero
   M at the end of a phase would mean a bug.

---

# Outcome (30 seeds, 400 episodes/phase)

## Fig 4 — episodes to greedy convergence vs planning_steps

Phase 1 (acquisition, target = lower):

| planning_steps | 0 | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| baseline | 62.5±2.8 | 1.17±0.40 | 0.73±0.18 | 0.57±0.12 | 0.80±0.19 | 0.60±0.13 |
| mood, bias real | 43.9±1.1 | 2.50±0.87 | 0.73±0.19 | 0.57±0.12 | 0.80±0.19 | 0.60±0.13 |
| mood, bias real+planning | 43.9±1.1 | 2.70±1.00 | 0.97±0.24 | 0.67±0.17 | 1.30±0.33 | 1.03±0.23 |

Phase 2 (adaptation, target = upper):

| planning_steps | 0 | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| baseline | 27.2±1.8 | 10.07±0.20 | 7.27±0.20 | 3.73±0.19 | 2.00±0.12 | 1.07±0.05 |
| mood, bias real | 13.2±0.3 | 8.47±0.20 | 6.27±0.16 | 3.53±0.19 | 1.87±0.10 | 1.07±0.05 |
| mood, bias real+planning | 13.2±0.3 | 8.20±0.24 | 6.10±0.25 | 3.40±0.20 | 1.87±0.11 | 1.03±0.08 |

**Prediction 1 — confirmed.** Baseline collapses from 62.5 to 1.2 episodes
between n=0 and n=1 in Phase 1. Past n=1 there is no headroom in Phase 1.

**Prediction 2 — wrong for Phase 2.** Mood's adaptation advantage does *not*
dilute away. It survives every n below 20, shrinking only in proportion:
52% faster at n=0, 16% at n=1, 14% at n=2, 5% at n=5, 7% at n=10, 0% at n=20.
Note that `mood_biases="real"` means only `1/(1+n)` of updates carry mood —
4.8% of them at n=20, which is where the effect finally disappears.

**Prediction 3 — answered, negatively.** `mood_biases="real+planning"` is
indistinguishable from `"real"` in Phase 2 (differences well inside SEM) and
consistently *worse* in Phase 1 at n>=2 (1.30 vs 0.80 at n=10, 1.03 vs 0.60
at n=20). Letting mood ride simulated, temporally scrambled planning updates
buys nothing and mildly hurts acquisition. This closes CLAUDE.md §8 open
question 2.

**Two things the table cannot resolve.** At n=1 in Phase 1 mood looks *slower*
than baseline (2.50 vs 1.17) but the SEMs overlap (±0.87 vs ±0.40) — needs more
seeds before it is called. At n>=2 in Phase 1 mood/real is numerically identical
to baseline to two decimals, SEM included; that is the metric hitting its floor
below one episode, not evidence of no effect.

## Fig 5 — mood at the switch, by planning_steps

| planning_steps | M at ep −1 | trough M | trough episode | episodes to \|M\|<0.01 | % of flip threshold |
|---|---|---|---|---|---|
| 0 | −0.0003 | −0.245 | +2 | 17 | 88% |
| 1 | 0.0000 | −0.228 | +2 | 12 | 82% |
| 5 | −0.0000 | −0.210 | +1 | 8 | 76% |
| 20 | 0.0000 | −0.167 | +1 | 5 | 60% |

**Prediction 4 — confirmed.** M dips sharply negative within 1–2 episodes of
the switch.

**Prediction 5 — confirmed on both counts, monotonically.** More planning gives
a shallower trough (−0.245 → −0.167) and a faster recovery (17 → 5 episodes).
Planning corrects Q faster, so the burst of large negative delta is shorter and
the EMA accumulates less of it.

**Prediction 6 — confirmed.** M at the last episode before the switch is 0.0000
to four decimals in every condition, as §3.4 requires.

**Unpredicted:** the trough never reaches the route-flip threshold — 88% of it
at n=0, less at every larger n. Whatever makes mood speed adaptation, it is not
mood crossing the threshold and flipping the greedy action directly. Consistent
with §9, and still the open mechanism question.

## Revision to the λ story (§8) at 30 seeds

The earlier 10-seed claim "best near λ=0.7" is weaker than stated. Phase 1 is
flat across λ=0.5–0.9 (42.2 to 43.9, best at 0.5) and Phase 2 is best at 0.7
(13.2) with 0.5 (13.9) inside ~2 SEM. Better description: **flat across
λ=0.5–0.9, collapsing to baseline by λ=0.99.**

Separately, `fig2` at λ=0.7 shows M reaching about −0.45 (mean) and −0.58
(individual seeds) during early Phase 1 — well past the ±0.278 threshold — and
the agent still converges faster than baseline. This corroborates §9 rather
than contradicting it.

---

# Predictions before the control-arm runs (2026-09-21)

Prompted by a reviewer's concern: *"check what your definition of greedy policy
is. It should be that as long as the agent will choose the right route, it's
considered learned — not how long it takes the Q-value to converge. Your result
might be measuring the latter, which would explain the high mood advantage,
because mood simply provides extra negative values to make Q updates faster."*

Settings: eta=0.1, lam=0.7 (half-life 1.9 real steps), planning_steps=0,
epsilon=0.1, lower_len=4, upper_len=2, 30 paired seeds, 400 ep/phase,
`mood_updates_from="real"`, `mood_biases="real"`.

1. **The metric is the greedy argmax, not Q error.** Expect confirmed from code
   alone (`train.run_phase` logs `greedy_route`; `episodes_to_greedy_convergence`
   reads only that field).
2. **But the two are confounded by the optimistic init.** `q_init=0` with all
   rewards negative means the greedy argmax at `start` is partly "which action
   has descended less", so anything that speeds the descent buys an earlier
   correct argmax. Expect mood's |Q−Q*| to fall faster than baseline's.
3. **A constant offset reproduces most of the speedup.** If mood is just
   sustained negative pressure, replacing M with its mean should work.
4. **Sign-flipped mood slows convergence** in both phases.
5. **A non-optimistic `q_init` shrinks the mood advantage markedly.**

---

# Outcome — the concern is half right, and the half that is right is the answer
# to §8 open question 1

**Prediction 1 — confirmed.** The metric reads the argmax and nothing else.

**Prediction 2 — confirmed, and it dissociates.** Mood is faster on behaviour
*and* slower on accuracy in Phase 2: it converges behaviourally at episode 13.2
while its final Phase 2 |Q−Q*| is 0.76 ± 0.01 against baseline's 0.30 ± 0.01.
So the metric is not a Q-accuracy proxy — but the descent speed is what moves
it.

**Prediction 3 — WRONG, informatively.** A constant offset equal to mood's mean
M over the first 30 episodes ((1−eta)·M̄ = −0.146) makes Phase 1 much *worse*
(94.7 ± 2.0 vs baseline 62.5 ± 2.8) while helping Phase 2 (14.9 ± 1.1). That is
exactly the §3.4 per-step-bonus trade: a negative bonus penalises the longer
route, which is optimal in Phase 1 and not in Phase 2. Mood improves **both**
phases, so mood is not a sustained pessimism offset. Full offset sweep:

| offset | equiv. M | P1 episodes | P2 episodes |
|---|---|---|---|
| 0 (baseline) | 0 | 62.5 ± 2.8 | 27.2 ± 1.8 |
| −0.02 | −0.022 | 62.2 ± 2.9 | 29.2 ± 2.0 |
| −0.05 | −0.056 | 69.8 ± 2.0 | 20.1 ± 1.3 |
| −0.10 | −0.111 | 82.4 ± 2.1 | 15.4 ± 0.8 |
| −0.20 | −0.222 | 120.3 ± 1.2 | 8.3 ± 0.5 |
| −0.40 | −0.444 | never | 0.0 ± 0.0 |

**Prediction 4 — WRONG in Phase 1, right in Phase 2, and a cautionary tale.**
Antimood converges in **0.47 ± 0.09** episodes in Phase 1 and takes **220 ± 8**
in Phase 2. A positive push inflates whatever the agent just did; the lower
route's small per-step deltas are outweighed by the bump, so Q(start,down) is
driven up and locked in. That is correct in Phase 1 by luck of the geometry and
catastrophic in Phase 2. Its final |Q−Q*| is 2.7–3.1. **A Phase 1 number on its
own can be produced by pure stickiness; never report Phase 1 without Phase 2.**

**Prediction 5 — confirmed.** The advantage is initialisation-dependent:

| q_init | baseline P1 | mood P1 | baseline P2 | mood P2 |
|---|---|---|---|---|
| 0 (optimistic, the default) | 62.5 ± 2.8 | 43.9 ± 1.1 | 27.2 ± 1.8 | 13.2 ± 0.3 |
| −12.5 (between the two Q*) | 16.9 ± 0.8 | 8.5 ± 1.0 | 23.8 ± 0.4 | 17.7 ± 0.3 |
| −25 (pessimistic) | 10.3 ± 0.6 | 0.5 ± 0.1 | 87.7 ± 10.9 | 71.2 ± 13.5 |

The Phase 2 advantage falls from 52% to 26% to 19%. The headline number depends
on `q_init=0`, which was never a recorded decision.

## The mechanism: mood is per-entry momentum, i.e. an effective learning rate

Baseline swept over alpha gives a calibration curve; each arm's episodes to
convergence then reads back as an effective alpha (`control_arms.py`):

| arm | P1 episodes | P2 episodes | eff. alpha P1 | eff. alpha P2 | final \|Q−Q*\| P1 | P2 |
|---|---|---|---|---|---|---|
| baseline | 62.5 ± 2.8 | 27.2 ± 1.8 | 0.100 | 0.100 | 0.20 ± 0.01 | 0.30 ± 0.01 |
| mood (global) | 43.9 ± 1.1 | 13.2 ± 0.3 | 0.150 | 0.150 | 0.12 ± 0.01 | 0.76 ± 0.01 |
| mood, per-(s,a) local | 35.9 ± 0.9 | 10.2 ± 0.1 | 0.170 | 0.165 | 0.02 ± 0.00 | 0.03 ± 0.00 |
| mood, yoked to another seed | 58.0 ± 2.0 | 27.0 ± 1.4 | 0.110 | 0.105 | 0.23 ± 0.02 | 0.33 ± 0.02 |
| ↳ corrected 2026-09-27 (per-step trace; bug below) | 44.07 ± 0.77 | 15.07 ± 0.73 | 0.150 | 0.140 | 0.20 ± 0.03 | 0.35 ± 0.02 |
| antimood | 0.5 ± 0.1 | 220.4 ± 7.7 | >0.210 | 0.100 | 2.74 ± 0.02 | 3.08 ± 0.01 |
| const offset −0.146 | 94.7 ± 2.0 | 14.9 ± 1.1 | 0.100 | 0.140 | 3.07 ± 0.01 | 2.63 ± 0.03 |

Three things fall out.

- **One effective alpha explains both phases.** Mood at eta=0.1 matches baseline
  at alpha ≈ 0.15 in Phase 1 *and* Phase 2 — a single ~1.5× learning-rate
  multiplier, not a phase-specific effect. Across the whole λ sweep the implied
  alpha stays in 0.125–0.17 for λ=0–0.95 and returns to 0.105 at λ=0.99.
- **Removing all cross-state coupling does not remove the effect — it
  strengthens it.** A separate `MoodTracker` per (s,a) gives 35.9 / 10.2, better
  than the global tracker, with a far smaller final Q error. Whatever mood is
  doing here, it is not acting as a global affective signal.
- **Decoupling M from the agent's own deltas abolishes it.** Yoking M to another
  seed's trace — identical magnitude and time course — returns baseline numbers
  (58.0 / 27.0 vs 62.5 / 27.2). What matters is that M correlates with the
  *current* delta.

Together: during a transient, consecutive deltas on the same (s,a) are
correlated, so `M ≈ eta·delta_recent` and the update
`Q += eta·delta + (1−eta)·M` behaves as `Q += eta_eff·delta` with
`eta_eff ≈ eta·(1 + (1−eta)·rho)`, bounded above by 0.19 at eta=0.1. Observed
0.125–0.17 sits under that bound. λ controls rho: small λ tracks the current
delta (large multiplier), λ→1 averages it away (multiplier → 1). **That is why
the λ curve is flat over 0.5–0.9 and collapses at 0.99** — a fact §8 recorded
without explaining.

This is a coherent reading of Emanuel & Eldar's own Kalman framing: mood is the
velocity term, and a velocity term on a monotone transient is momentum. But it
means the current result does **not** show mood contributing information the
agent did not already have in its own last delta.

## What this changes

The reviewer's literal claim is wrong — the metric is behavioural. The
underlying worry is right: on this task, the mood advantage is an accelerated
value update, and a learning-rate-matched baseline is the control that was
missing. Every future mood-vs-baseline comparison should carry one.

---

# 2026-09-23 — First-passage metric (sustain=1)

Proposal (grad student): count the agent as having learned at the **first**
episode its greedy action at `start` picks the optimal route, instead of 20
consecutive episodes, since the post-acquisition flips are a by-product of the
optimistic `q_init=0` and on-policy updating (fig8), not unstable learning.

Settings: eta=0.1, lam=0.7 (half-life 1.9 real steps), eps=0.1,
planning_steps=0 (also 1, 5), mood_updates_from=real, mood_biases=real,
upper_len=2, lower_len=4, T=10, c=2.5, q_init=0, 400 episodes/phase, 30 paired seeds.

**Predictions, written before running:**

1. Phase 1 first-passage will be ~0–2 episodes for every arm and will not
   separate them. Reason: `Q(start,right)` absorbs the −12.5 trap on its first
   update (§5 asymmetry), so one upper run makes down greedy. The metric then
   measures "time until the first upper visit", not learning.
2. Phase 2 first-passage will still separate the arms, mood faster than
   baseline α=0.10, and roughly matched by baseline α=0.15 — the
   effective-learning-rate reading should not depend on the sustain length.
3. At planning_steps=5 the sustain=1 and sustain=20 numbers will nearly
   coincide (few flips once replay propagates values).

**Outcome** (`first_passage.py`), mean ± SEM episodes into the phase, 30 seeds:

| planning_steps=0 | P1 s=1 | P1 s=20 | P2 s=1 | P2 s=5 | P2 s=20 | P2 flips after 1st pass |
|---|---|---|---|---|---|---|
| baseline α=0.10 | 0.5 ± 0.1 | 62.5 ± 2.8 | 13.3 ± 0.3 | 16.3 ± 0.5 | 27.2 ± 1.8 | 4.3 |
| baseline α=0.15 | 0.5 ± 0.1 | 39.2 ± 1.3 | 9.9 ± 0.2 | 10.9 ± 0.3 | 11.4 ± 0.4 | 0.7 |
| mood λ=0.7 | 0.5 ± 0.1 | 43.9 ± 1.1 | 12.1 ± 0.2 | 13.2 ± 0.3 | 13.2 ± 0.3 | 0.7 |

Phase 2 paired baseline(α=0.10) − mood, s=1: 1.27 ± 0.14 episodes, mood faster
in 26/30 seeds, 4 ties (n=0); 1.73 ± 0.16, 29/30 (n=1); 0.27 ± 0.08, 8/30 with
22 ties (n=5). Phase 1 s=1: identical in 30/30 seeds at every n.

1. **Confirmed.** Phase 1 first passage is 0.5 episodes in every arm, every
   seed paired identically: it only records whether episode 0 happened to go
   upper. It measures nothing about learning.
2. **Half right.** Mood is still faster than baseline α=0.10 in Phase 2, but by
   1.3 episodes (10%), not 14 (52%). And mood is *slower* than the rate-matched
   α=0.15 baseline (12.1 vs 9.9). Most of the headline Phase 2 advantage came
   from what happens *after* first passage: baseline α=0.10 flips back 4.3
   times, mood 0.7 times.
3. **Confirmed.** At n=5 all sustain lengths coincide; at n=1 Phase 2 already
   does (0 flips).

---

# 2026-09-23 — Scratch prototype: upper trap moved to step 2

Not a design change; a monkeypatched spec in the scratchpad. The upper trap moves
from `start->u_1` to `u_1->u_goal`. Route returns are unchanged (2c+T either way),
so c, margins and Q*(start,·) are unchanged; only Q*(u_1,right) goes −2.5 → −12.5.
Both traps are then 2 steps from `start`.

Settings as before: eta=0.1, lam=0.7, eps=0.1, lower_len=4, 30 seeds, 400 ep/phase.

**Predictions:**
1. Phase 1 first passage stops being a 0/1 coin flip. After one visit to each
   route both `start` values are −0.25 (a tie); the trap reaches
   `Q(start,right)` only by backup through `u_1`. Expect several episodes at
   n=0, and the arms to separate.
2. Flipping in Phase 1 at n=0 persists: `q_init=0` is still optimistic, so the
   20-episode number stays large.
3. Phase 2 numbers change little (the lower trap is untouched).

**Outcome.** Prediction 1 wrong: Phase 1 first passage is still 0.5–0.7 at
n=0 after the move. The 0/1 coin flip is not caused by the trap position but by
`q_init=0`: any untried action (0) beats a tried one (≤ −0.25), so the greedy
action after episode 0 is always the route *not* taken. Prediction 2 right, more
so than expected: Phase 1 flips roughly double (18 → 32 at n=0) because a single
visit no longer settles the ranking. Prediction 3 right (P2 s=1 within ~1 ep).

Follow-up prediction, q_init sweep on the CURRENT maze, n=0: with q_init below
both Q*(start,·) (≤ −15), whichever route runs first becomes sticky, so Phase 1
first passage is again a coin flip on episode 0 but in the opposite direction,
and flips drop sharply.

**Outcome of q_init sweep** (current maze, 30 seeds, scratch script; total greedy
flips per phase include the one onto the optimal route):

| n=0 | P1 s=1 | P1 s=20 base / α=.15 / mood | P2 s=1 base / α=.15 / mood | P2 flips (base) |
|---|---|---|---|---|
| q_init=0 | 0.5 all | 62.5 / 39.2 / 43.9 | 13.3 / 9.9 / 12.1 | 5.3 |
| q_init=−10 | 0.5 all | 16.7 / 10.4 / 11.6 | 20.7 / 12.0 / 15.8 | 1.0 |
| q_init=−15 | 0.5 all | 10.3 / 6.3 / 0.5 | 28.5 / 14.2 / 20.1 | 1.0 |
| q_init=−25 | 0.5 all | 10.3 / 6.3 / 0.5 | 87.7 / 21.9 / 71.2 | 1.0 |

Follow-up prediction wrong: the tried route does not become sticky, because the
bootstrapped target also starts at q_init, so a tried entry still falls below
an untried one. **Phase 1 first passage is 0.5 at every q_init and both trap
positions**: after one episode the greedy action is the route not taken, in
every setting tested. Below q_init=0, Phase 2 has no post-passage flips (s=1 =
s=20) but is much slower, and α=0.15 beats mood at every q_init.

---

## 2026-09-27 — Phase 2 from a transplanted end-of-Phase-1 Q-table

Config: eta=0.1, lam=0.7 (half-life 1.94 steps), planning_steps=0, eps=0.1,
lower_len=4, q_init=0, mood real/real, 400 ep/phase, 15 paired seeds.
Each Phase 2 agent receives the donor's end-of-Phase-1 Q, model and RNG state,
M=0. A = baseline's table, B = mood's table. Also the `control_arms.YokedMood`
agent (trace from seed (i+7)%15).

Prediction (written before running): Phase 2 behaviour is set mostly by the
table. A: both agents flip (baseline ~4, mood ~2), mood sustained well above
13 (~18–22), first passage both ~13–14. B: both agents flip rarely (<1),
sustained ≈ first passage; baseline ~14–15, mood ~12–13. Yoked: end-P1
Q(start,right) between baseline and mood (~−14.3), P2 flips ~2–3.

Outcome (`transplant.py`, logs in `logs/transplant/`). Own-table cells
reproduce the ordinary runs exactly (transplant check).

| Phase 2 from | agent | first 'upper' | sustained (20) | flips after 1st |
|---|---|---|---|---|
| A: baseline's Q | baseline | 13.73 ± 0.43 | 28.27 ± 2.86 | 4.13 ± 0.69 |
| A: baseline's Q | mood | 10.07 ± 0.38 | 20.53 ± 1.54 | 3.60 ± 0.45 |
| A | paired base−mood | 3.67 ± 0.21 | 7.73 ± 2.94 | 0.53 ± 0.50 |
| B: mood's Q | baseline | 17.00 ± 0.34 | 19.73 ± 0.67 | 1.33 ± 0.32 |
| B: mood's Q | mood | 12.40 ± 0.34 | 13.13 ± 0.39 | 0.53 ± 0.31 |
| B | paired base−mood | 4.60 ± 0.19 | 6.60 ± 0.62 | 0.80 ± 0.26 |

| agent | end-P1 Q(start,right) | P1 sustained | P2 first | P2 sustained | P2 flips |
|---|---|---|---|---|---|
| baseline | −13.95 ± 0.11 | — | 13.73 ± 0.43 | 28.27 ± 2.86 | 4.13 ± 0.69 |
| mood | −15.04 ± 0.06 | — | 12.40 ± 0.34 | 13.13 ± 0.39 | 0.53 ± 0.31 |
| yoked (per-step trace) | −14.50 ± 0.05 | 43.60 ± 1.19 | 10.00 ± 0.49 | 15.53 ± 1.43 | 3.33 ± 0.54 |

Prediction partly wrong. The table explains about half the sustained gap, not
most of it: holding the agent fixed, mood's table saves baseline 8.5 episodes
and mood 7.4; holding the table fixed, mood saves 7.7 (A) and 6.6 (B). With a
shared table, mood's first passage is 3.7–4.6 episodes earlier, not 1.3. The
natural 1.3 understates the post-switch effect because mood's accurate
Q(start,right)=−15 means Q(start,down) must fall further before the first
crossing (baseline from mood's table: 17.0 vs 13.7 from its own).
Yoked flips (3.3) as predicted; its end-P1 Q(start,right) (−14.5) sits
between baseline and mood.

Bug found: `control_arms.py` passes a per-EPISODE M trace (800 values) to
`YokedMood`, which indexes it per REAL STEP (~2400). The yoked M runs ~3×
fast and freezes at its last value after ~250 episodes. At 15 seeds:
per-episode trace P1 59.3 ± 2.7, P2 26.3 ± 1.6 (≈ the §8 row 58.0 / 27.0);
per-step trace P1 43.6 ± 1.2, P2 15.5 ± 1.4.

---

## 2026-09-27 — Learning-rate-matched baselines vs mood, all four measures

Config: eta=0.1, lam=0.7 (half-life 1.94 steps), planning_steps=0, eps=0.1,
lower_len=4, q_init=0, mood real/real, 400 ep/phase, 30 paired seeds.
Baseline alpha in {0.10, 0.12, 0.15, 0.19}. Measures: P1 sustained, P2 first
passage, P2 sustained, P2 flips after first passage. `transplant.py` also runs
each recipient from the α=0.10, α=0.15 and mood end-of-Phase-1 tables.

Prediction (before running), from the existing alpha curve and
`first_passage.py`: no single alpha matches mood on all four. α≈0.13–0.15
matches P1 (43.9) and P2 sustained (13.2); α=0.15 is faster than mood on P2
first passage (9.9 vs 12.1); α≈0.12 matches mood's first passage but is slower
on sustained. Flips: α≥0.15 ≈ mood (<1). From mood's table, α=0.15 reaches first
passage faster than mood (i.e. mood's post-switch edge is a learning-rate effect).

Outcome (`control_arms.py` and `transplant.py` agree to the decimal on shared cells):

| agent | end-P1 Q(start,right) | P1 sustained | P2 first | P2 sustained | P2 flips |
|---|---|---|---|---|---|
| baseline α=0.10 | −13.89 ± 0.07 | 62.47 ± 2.83 | 13.33 ± 0.28 | 27.17 ± 1.84 | 4.27 ± 0.45 |
| baseline α=0.12 | −14.25 ± 0.06 | 50.33 ± 1.58 | 11.90 ± 0.20 | 19.40 ± 0.97 | 2.07 ± 0.28 |
| baseline α=0.15 | −14.58 ± 0.04 | 39.23 ± 1.33 | 9.93 ± 0.18 | 11.40 ± 0.36 | 0.73 ± 0.20 |
| baseline α=0.19 | −14.81 ± 0.03 | 33.47 ± 1.65 | 8.17 ± 0.11 | 8.43 ± 0.16 | 0.13 ± 0.09 |
| mood | −15.02 ± 0.04 | 43.87 ± 1.07 | 12.07 ± 0.22 | 13.17 ± 0.28 | 0.73 ± 0.22 |
| yoked (per-step) | −14.49 ± 0.04 | 44.07 ± 0.77 | 9.83 ± 0.31 | 15.07 ± 0.73 | 3.20 ± 0.31 |

Paired P2 first passage minus mood's, from a shared table (α=0.10 / α=0.15 /
mood tables): α=0.12 +1.27 / +1.53 / +1.67; α=0.15 −1.10 / −1.13 / −1.17
(all SEM ≤ 0.12). P2 sustained from the α=0.10 table: mood 20.7, α=0.12 21.2,
α=0.15 14.9.

Prediction right on the main point: no alpha matches mood on all four. α=0.15
matches flips and is faster than mood on the other three; α=0.12 matches first
passage only. From every shared table, mood's post-switch first passage sits
between α=0.12 and α=0.15 (≈ α 0.135), so the post-switch half of the gap is
a learning-rate effect. Not predicted: mood ends Phase 1 with Q(start,right)
at −15.02, beyond even α=0.19 (−14.81), while converging in Phase 1 like
α≈0.14. That pairing is the one thing no alpha reproduces; yoked (−14.49)
does not reproduce it either.

Premise check for the mechanism bullet: M trace of seed i vs seed i+7,
per-episode correlation 0.77 (P1 ep 0–49), 0.07 (P1 ep 50–399, where M ≈ 0),
0.92 (P2 ep 0–49); per-step, first 150 steps, 0.73.
