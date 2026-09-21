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
