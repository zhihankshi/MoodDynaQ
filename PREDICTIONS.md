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

---

# 2026-10-03 — Redesign: gamma=0.9, decaying epsilon, Phase 2 first-passage metrics

Design changes (code done, before any run): gamma=0.9 in `optimal.py`/`env.Config`
(agent default now 0.9 too); matched step cost c = T(2-γ)/(2D) = 3.5737,
D = S(4) − S(2) = 1.539, margins 4.5 both phases. Q*(start,right) = −16.790,
Q*(start,down) = −12.290 (P1) / −21.290 (P2). ε linear 1.0 → 0.1 over the first
`N_decay` Phase 1 episodes, 0.1 for the rest of Phase 1 and all of Phase 2.
Arms: baseline α ∈ {0.10, 0.12, 0.15, 0.19}; mood η=0.1, λ=0.7 (half-life 1.94
real steps), mood real/real. planning_steps=0, lower_len=4, q_init=0.
Episode-count pilot on held-out seeds 100–129 (the study uses 0–29).

## Pilot predictions (design numbers)

1. **Q(start,right) at the switch.** Error shrinks ~×(1−α) per upper visit
   (with a lag through `u_1`), so <0.1 needs ~50 upper visits at α=0.1. Upper
   is run at ~ε/2 per episode once greedy settles on lower. With N_decay=100,
   ~30–40 upper visits come during the decay, the rest at 0.05/episode, so the
   slowest seeds of α=0.10 need Phase 1 ≈ 300–400. N_decay=200 roughly halves
   the tail. α≥0.15 and mood (eff. α≈0.15) get there ~1/3 sooner.
2. Q(start,down) is on the greedy path and within 0.1 of Q* by episode ~100
   in every arm. Mood's M at the switch is ~0 (|M| < 0.01), per §3.4.
3. **Phase 2 length.** With Q(start,right) already at Q*, first passage is set
   by Q(start,down) falling 4.5 through a two-link chain (shield → start):
   ~17 episodes at α=0.10, ~9 at α=0.19 (hand estimate of the lagged filter,
   ignoring ε). Last flip-back by episode ~40 in every arm; 100 episodes
   censors nothing.

## Study predictions (to check once the full run is approved)

4. **Flip-backs ≈ 0 for every baseline arm.** At γ=1 the flips came from an
   optimistic Q(start,right) left by Phase 1 (§8). The new Phase 1 length is
   chosen to remove exactly that, so nothing is left to undo.
5. **Mood flip-backs ≥ baseline, mean < 1.** Negative M after the switch drags
   Q(start,right) below Q* on every upper run (the §8 contamination), which
   can push it back under Q(start,down) while that is still near the crossing.
6. **First passage: mood between α=0.12 and α=0.15**, as in the transplant
   result (post-switch effect ≈ α 0.135). The inherited-table half of the old
   gap is gone by design, so the mood vs α=0.10 gap should be ~4 episodes,
   not 14.
7. **Secondary |Q−Q*| at start.** Q(start,right): ~0 for baselines at all four
   checkpoints; mood > 0 (contamination, a few tenths) at 5–10, decaying.
   Q(start,down): large at 5 (~3–4), falling while it is greedy, then nearly
   frozen once the greedy route is upper — the 25 and 50 values (~0.5–1.5)
   mostly reflect ε visits, not learning speed.

## Pilot outcome (`python3 study.py pilot`, seeds 100–129)

Seeds (of 30) with |Q(start,right) − Q*| < 0.1 at Phase 1 episode N (worst error):

| decay | arm | N=200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|
| 100 | α=0.10 | 0 (2.02) | 0 (1.40) | 2 (0.80) | 7 (0.50) | 11 (0.35) |
| 100 | mood | 1 (0.49) | 2 (0.44) | 5 (0.43) | 11 (0.31) | 18 (0.26) |
| 200 | α=0.10 | 16 (0.29) | 22 (0.22) | 27 (0.12) | 30 (0.08) | 30 (0.06) |
| 200 | α=0.12 | 29 (0.12) | 30 (0.08) | 30 (0.04) | 30 (0.02) | 30 (0.02) |
| 200 | α=0.15 | 30 (0.04) | 30 | 30 | 30 | 30 |
| 200 | α=0.19 | 30 (0.01) | 30 | 30 | 30 | 30 |
| 200 | mood | 21 (0.15) | 28 (0.13) | 30 (0.09) | 30 (0.05) | 30 (0.03) |

Chosen: decay over 200, Phase 1 = 500 (first N where all arms are 30/30).
At the switch (N=500), mean Q(start,right) vs Q* −16.790: α=0.10 −16.762 (max
err 0.076), 0.12 −16.783, 0.15 −16.789, 0.19 −16.790, mood −16.804 (max err
0.049, overshoots *below* Q*). Q(start,down) = −12.290 in every arm (max err
0.002). Mood |M| ≤ 0.0003.
Phase 2 (run 300 to check censoring): max first passage 19 / 15 / 12 / 10 / 14
(α 0.10 / 0.12 / 0.15 / 0.19 / mood); nothing censored at 100. Baseline arms:
**zero flip-backs on every seed**; mood: some, the last at Phase 2 episode 15.

Prediction 1 right on the order of magnitude but N_decay=100 was much worse
than predicted (α=0.10 still 7/30 at N=500): the floor-rate tail is slower than
the visit count suggested. Mood was predicted to be among the fast arms; it
is closer to α=0.12, and its error is a negative overshoot (contamination),
not lag. Prediction 2 right. Prediction 3 right on length (first passage ≤ 19);
"last flip-back by ~40" moot for baselines — they have none.

## 2026-10-04 — Study predictions, restated before the seeds 0–29 run

Written after the held-out pilot (seeds 100–129), before running `study.py` on
seeds 0–29. Design as above: Phase 1 = 500 (ε 1.0 → 0.1 over 200), Phase 2 = 100.
Primary = Phase 2 first greedy switch to upper, all arms; flip-backs are a
mood-specific cost, not primary (CLAUDE.md §2). Changes from predictions 4–6:
prediction 5 sharpened to "only mood".

A. **Mood's Phase 2 first switch falls between the α=0.12 and α=0.15 arms.**
B. **α=0.19 is fastest** of all five arms; order α=0.19 < 0.15 < mood < 0.12 < 0.10.
C. **Only mood has flip-backs.** All four baseline arms: 0 on every seed.
   Mood: mean > 0 and < 1.
D. Secondary (unchanged from 7): Q(start,right) error ~0 for baselines at
   every checkpoint, a few tenths for mood at k=5–10 and decaying;
   Q(start,down) error large at k=5, small and roughly frozen by k=25–50.

### Outcome (`python3 study.py`, seeds 0–29, 2026-10-04)

| arm | P2 first switch | flip-backs | Q(start,right) err k=5/10/25/50 | Q(start,down) err k=5/10/25/50 |
|---|---|---|---|---|
| α=0.10 | 17.00 ± 0.20 | 0 | 0.02 / 0.02 / 0.01 / 0.00 | 8.40 / 6.86 / 4.30 / 4.03 |
| α=0.12 | 13.87 ± 0.20 | 0 | 0.01 / 0.01 / 0.00 / 0.00 | 8.17 / 6.20 / 4.26 / 3.93 |
| α=0.15 | 10.73 ± 0.17 | 0 | 0.00 at all k | 7.78 / 5.23 / 4.14 / 3.78 |
| α=0.19 | 8.67 ± 0.17 | 0 | 0.00 at all k | 7.18 / 4.30 / 3.89 / 3.43 |
| mood | 12.97 ± 0.24 | 0.03 ± 0.03 | 0.06 / 0.07 / 0.05 / 0.00 | 7.92 / 5.79 / 3.96 / 3.60 |

Paired first switch, arm − mood: α=0.10 +4.03 ± 0.09, α=0.12 +0.90 ± 0.06,
α=0.15 −2.23 ± 0.10, α=0.19 −4.30 ± 0.13.

A. **Right.** Mood sits between α=0.12 and 0.15, nearer 0.12 (≈ α 0.13 by
   linear interpolation), as in the γ=1 transplant result (≈ 0.135).
B. **Right.** Order α=0.19 < 0.15 < mood < 0.12 < 0.10, all paired gaps ≥ 9 SEM.
C. **Right, barely.** Baselines 0 on every seed; mood has one flip-back on one
   seed out of 30.
D. **Half wrong.** Mood's Q(start,right) error is a few *hundredths*, not
   tenths. Q(start,down) error does not get small: it stays ~3.4–4.3 at
   k=25–50 in every arm. When the greedy route flips to upper,
   Q(start,down) has just fallen past Q(start,right) = −16.79, so it is still
   ~4.5 from its Q* (−21.29), and after that it moves only on ε visits. The
   k=25/50 values are that leftover gap, slowly eroded by exploration — the
   caveat in CLAUDE.md §2, but much larger than predicted (prediction 7 said
   0.5–1.5).

### Not predicted: mood slows Phase 1 acquisition (seen in `fig_study`, 2026-10-04)

Same run, seeds 0–29. Phase 1 episodes until the greedy route is lower for good
(last wrong greedy episode + 1): every baseline arm 1.40 ± 0.37, identical
across α (1/30 seeds past episode 5); mood 5.80 ± 1.48 (11/30 past episode 5,
last at episode 27). Mood's M bottoms out at −0.56 on average in episodes 0–29,
past the 0.325 flip threshold (§3.4): the per-step bonus penalizes the longer
lower route, as H2 predicts. The cost is gone by episode 30, long before the
switch at 500. No prediction covered Phase 1, because §8 had called Phase 1
first passage degenerate. Under decaying ε it no longer is for mood.

Mechanism check (scratch diagnostic, same settings, seeds 0–29, Phase 1
episodes 0–39). Every wrong-way greedy flip in both arms is a `Q(start,down)`
update pushing it below `Q(start,right)` (baseline 10/10, mood 26/26). That's
the optimistic-`q_init` alternation of §12/fig8, not something new. Mood just
makes more of them (10 vs 1 after episode 5). At the late mood flips the TD
part is always negative (10/10). The mood part `(1−η)M` is −0.22 against
`ηδ` −0.30. It is the larger of the two in only 2/10, so mood adds to the
descent rather than reversing it. Correction, same day: an earlier reading here put
it down to `Q(start,down)` getting more updates. That is weak: ε is still
0.83–1.0 in episodes 0–39, so executed routes are 55% lower / 45% upper in
both arms. The flips follow the trap instead. Of mood's 24 wrong-way flips in
episodes 3–39, 17 come on a lower episode that directly follows an upper
episode (291 such episodes) and 7 after a lower episode (321 such). M entering
the episode is −0.214 after an upper episode and −0.134 after a lower one.
Baseline: 1 vs 2, no dependence. Reading: M persists across episodes, so the
negative surprise of the upper trap is still in M when the next episode's first
update lands on `Q(start,down)`. Mood moves the trap's punishment onto the
action taken next. In Phase 1 that drags down the correct route. Gone by
~episode 30 as δ shrinks and M → 0 (§3.4).

## 2026-10-04 — Reversed phase order: predictions, written before running

Grad-student design: Phase 1 = shield does NOT work (`Config.matched(phase=2)`,
upper optimal), Phase 2 = shield works (`phase=1`, lower optimal). Everything
else identical to the study: γ=0.9, ε 1.0 → 0.1 over P1 episodes 0–199 then
0.1, P1=500 / P2=100, planning_steps=0, lower_len=4, q_init=0, mood η=0.1
λ=0.7 (half-life 1.94 real steps) real/real, baseline α 0.10/0.12/0.15/0.19,
seeds 0–29 paired. Question: is mood again slower in Phase 1 and faster in Phase 2?

R1. **Phase 1: mood's cost shrinks or vanishes.** The carry-over found above
    (a trap surprise left in M lands on the next episode's first update) now
    cuts both ways: both routes have a trap, so lower episodes leave M just
    as negative as upper ones. Predict mood ≤ 2 episodes slower than baseline
    on Phase 1 "last wrong greedy + 1" (original design: 5.8 vs 1.4).
R2. **Phase 2 is silent and slow; many seeds censored at 100.** The greedy
    upper route's return does not change (§7), so learning only happens on
    ε-driven lower runs (~5% of episodes). `Q(start,down)` must rise by
    ~4.5 through two levels of propagation at α=0.1. Predict median baseline
    α=0.10 first passage > 100 (censored).
R3. **Mood is not faster in Phase 2, and may be slower.** Surprises are now
    positive (lower route better than expected), so M goes positive after a
    lower run, and the carry-over puts it on the next episode's first update,
    usually `Q(start,right)` (greedy upper). That props up the route to be
    abandoned. Against this, mood's higher effective learning rate speeds up the
    lower-route updates. Lean: mood no faster than baseline α=0.10.

### Outcome (`python3 study.py reversed`, seeds 0–29, 2026-10-04)

| arm | P1 settled on upper | P2 first lower @100 | P2 first lower, P2 run 3000 (median, range) |
|---|---|---|---|
| α=0.10 | 50.53 ± 1.66 | censored 30/30 | 349.6 ± 15.9 (331, 189–550) |
| α=0.12 | 39.03 ± 2.43 | censored 30/30 | 289.7 ± 15.4 (268, 132–499) |
| α=0.15 | 32.20 ± 2.05 | censored 30/30 | 221.6 ± 12.7 (219, 103–458) |
| α=0.19 | 26.60 ± 1.52 | censored 29/30 | 180.8 ± 12.9 (170, 99–442) |
| mood | 34.30 ± 2.13 | censored 30/30 | 295.1 ± 15.9 (286, 132–499) |

Paired α=0.10 − mood: Phase 1 +16.2 ± 1.9 (mood faster on 28/30 seeds),
Phase 2 (3000-episode run) +54.5 ± 5.8 (30/30). No flip-backs in any arm. At the
switch, |Q(start,down) − Q*| (the abandoned route) 0.20 for α=0.10 and 0.12 for mood.
The 3000-episode Phase 2 is a supplementary run; nothing else changed.

R1. **Right in direction, wrong in size.** Mood's Phase 1 cost is gone. Mood
    is clearly *faster* (34 vs 51), between α=0.12 and 0.15. Phase 1 is
    much harder for every arm than in the original order (51 vs 1.4 at
    α=0.10): the optimal upper route's trap is on step 1, so `Q(start,right)`
    takes the −10 at once, and the lower trap must propagate before the
    lower route looks worse.
R2. **Right.** Every arm censored at 100. Real Phase 2 durations 99–550 episodes,
    ~20× the original order. The silent switch (§7) is learned only from
    ε-driven lower runs.
R3. **Wrong.** Mood is faster in Phase 2 too, on every seed, about like
    α=0.12. Any carry-over of positive M onto `Q(start,right)` is too small to
    show. In this order mood behaves like a higher learning rate in both phases
    (≈ α 0.13–0.14 in Phase 1, ≈ α 0.12 in Phase 2). The original order's Phase 1
    cost does not reappear. That fits the carry-over reading above: it hurts
    only when one route alone carries the trap surprise and the other is the
    correct one.

## 2026-10-04 — Reversed order, Phase 2 = 1000 episodes (design change)

Reversed-order Phase 2 lengthened from 100 to 1000 episodes; the original order
keeps 100. Raising ε at the switch was rejected (§7: an external change cue).
Prediction: the first 1000 Phase 2 episodes are the same as in the 3000-episode
supplementary run (same seeds, same RNG stream), so first passage should
reproduce it exactly: α=0.10 349.6, mood 295.1, nothing censored, no flip-backs.
Outcome: **right, exactly.** `python3 study.py reversed` now gives α=0.10
349.63 ± 15.94, mood 295.13 ± 15.88, 0/30 censored in every arm, no flip-backs.
Original order unchanged (17.00 / 12.97).

### Phase 1 Q-table audit, seed 19 (`q_audit.py`, original order, 2026-10-04)

Same routes in both agents through episode 20. Baseline greedy is correct from
episode 0; mood's is wrong at episode 2 and from 9 to 19 (settles at 20). Each episode's change in
the updated `Q(start,·)` splits into a TD part `ηδ` and a mood part `(1−η)M`.
The TD parts are almost identical in the two agents (lower runs −0.27 to −0.36,
upper runs −0.89 to −1.36). The mood part is −0.10 to −0.47 per update. It is
added whichever route was run and is largest on the first update after an
upper (trap) episode (−0.25 to −0.47, vs −0.14 to −0.23 after a lower one).
Episodes 0–20, summed: mood part −3.75 on `Q(start,down)` (14 lower runs)
vs −1.07 on `Q(start,right)` (7 upper runs). Per visit, baseline's upper
route costs ~3.4× its lower route (−1.15 vs −0.34); with mood it is ~2.3×
(−1.4 vs −0.6). Mood narrows the per-visit difference between the trap route
and the cheap one, so a streak of lower runs (episodes 6–9, `Q(start,right)`
frozen at −4.464) pulls `Q(start,down)` below it. This seed's flip at 9 comes
in a streak of lower runs, not right after an upper episode. The carry-over
reading above is one part of a broader effect: the mood term does not
depend on which route earned it.

### Phase 1 Q-table audit, reversed order, seed 12 (`q_audit.py reversed`)

Seeds with identical routes until both settle: 4, 12, 16, 19. Seed 12 (baseline
settles at 47, mood at 24) is closest to the means (50.5 / 34.3). Both agents
first prefer the wrong (lower) route: the upper trap hits `Q(start,right)` on
step 1, while the lower trap is on step 2 and reaches `Q(start,down)` only via
`Q(shield,right)`. Episodes 0–47, summed: the mood part is −4.18 on
`Q(start,down)` and −3.71 on `Q(start,right)`, nearly even (original order,
seed 19: −3.75 vs −1.07). Both routes earned it. Mood also pushes the
lower-corridor entries, so the trap's cost travels back faster:
`Q(shield,right)` at episode 18 is −14.98 (mood) vs −10.55 (baseline),
Q\* −19.685. The TD part of mood's lower-run updates is therefore larger: −15.40
vs baseline's −14.58 summed. Mood overshoots `Q(shield,right)` (−20.30 at
episode 32 vs Q\* −19.685); that is the route being abandoned, so it costs nothing here.
After first passage baseline flips back three times (43, 44, 46: values within
0.4); mood's values are 2–3 apart and do not flip back.

## 2026-10-05 — Reversed order with an ε reset at the switch: predictions, written before running

Exploratory arm, not a design change: the 2026-10-04 reversed study (no reset,
P2 = 1000) stays the reference. Phase 2 reuses the Phase 1 schedule: ε 1.0 →
0.1 linear over Phase 2 episodes 0–199, then 0.1. Phase 2 = 500 episodes.
Same for every arm. Everything else as `study.py reversed`. Why test it: our
mood never touches action selection, so a reset cannot mask a mood-driven
exploration effect; it changes the regime (visit-limited → update-limited)
and narrows the claim to "given exploration, does mood adapt faster".
`python3 study.py reversed reset`.

P1. Phase 1 identical to the reversed study (same seeds, RNG and schedule up
    to the switch): α=0.10 50.53, mood 34.30.
P2. Phase 2 first passage drops from hundreds of episodes to tens: α=0.10
    ~30–50, mood ~20–35, nothing censored at 500. With ε near 1 about half the
    early episodes go lower, so visits stop being the bottleneck; the trap
    removal must propagate shield → start, so it is not immediate.
P3. Mood still faster on most seeds (≥ 24/30), ordered like a learning rate
    (between α=0.12 and 0.15). The absolute gap shrinks well below the 54.5
    of the no-reset run; the relative gap is similar or larger.
P4. Flip-backs near zero in every arm: the abandoned upper route keeps
    `Q(start,right)` at Q*, and lower runs only raise `Q(start,down)`.

### Outcome (`python3 study.py reversed reset`, seeds 0–29, 2026-10-05)

| arm | P1 settled on upper | P2 first lower (reset, P2=500) | no reset (2026-10-04) |
|---|---|---|---|
| α=0.10 | 50.53 ± 1.66 | 37.77 ± 1.53 (median 36.5) | 349.6 ± 15.9 |
| α=0.12 | 39.03 ± 2.43 | 30.40 ± 1.31 (29.5) | 289.7 ± 15.4 |
| α=0.15 | 32.20 ± 2.05 | 23.30 ± 1.04 (23.0) | 221.6 ± 12.7 |
| α=0.19 | 26.60 ± 1.52 | 18.17 ± 0.97 (17.0) | 180.8 ± 12.9 |
| mood | 34.30 ± 2.13 | 30.97 ± 1.35 (30.0) | 295.1 ± 15.9 |

0/30 censored and 0 flip-backs in every arm. Paired α=0.10 − mood: +6.80 ±
0.83 (mood faster on 29/30 seeds); α=0.12 − mood: −0.57 ± 0.38. Mood's
abandoned `Q(start,right)` drifts off Q* (0.33 / 0.41 at k=10 / 25, back to
0.15 at k=50); every baseline holds it at 0.00. That is the §8 contamination:
the upper route is visited often under high ε, and each visit gets `(1−η)M`.

P1 **right**, identical. P2 **right** (37.8 / 31.0). P3 **right in direction,
slightly off in placement**: 29/30 seeds, but mood sits at α≈0.12, not between
0.12 and 0.15, which is the same place it sits without the reset. Absolute gap
54.5 → 6.8; relative gap 15.6% → 18.0%. P4 **right**.

Reading: the reset shortens Phase 2 ~9× for every arm and leaves the ranking
of the arms, and mood's equivalent learning rate, unchanged. In this agent the
reset changes how long Phase 2 takes, not what the comparison says. Expected,
since mood has no path to action selection.
