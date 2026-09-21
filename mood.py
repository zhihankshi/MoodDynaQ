"""
Mood-biased Dyna-Q, following Emanuel & Eldar (2023).

MOOD UPDATE (their Eq. 3.3.2)
    M <- M + (1 - lam) * (eta * delta - M)

    Equivalently M <- lam*M + (1-lam)*eta*delta, i.e. an exponential moving
    average of recent value changes with retention `lam` per update.

VALUE UPDATE (their Eq. 3.4.1)
    Q <- Q + eta*delta + (1 - eta)*M

    NOTE the coupling: eta is BOTH the learning rate and (via 1-eta) the mood
    weight. They are not independent -- Eq. 3.4.1 is derived as a Kalman-style
    optimal estimator where value is position and mood is velocity, so the
    weights on the new observation and the velocity estimate sum to 1.

    Consequence for tabular work: a typical tabular learning rate eta=0.1 puts
    the mood weight at 0.9. Mood is a MAJOR term here, unlike the DQN version
    where eta=0.9 left it at 0.1. Decoupling is available for sensitivity
    analysis but is a departure from the theory, so it is off by default.

TWO INDEPENDENT DESIGN CHOICES, both exposed as flags:

  mood_updates_from : 'real' | 'real+planning'
      Which deltas feed the mood tracker. The theory says mood integrates the
      agent's *experienced* value updates, so 'real' is theory-faithful.
      Planning deltas are simulated and temporally scrambled.

  mood_biases : 'real' | 'real+planning'
      Which Q-updates get the (1-eta)*M term added. 'real' is theory-faithful,
      but note that with planning_steps=n, only 1/(1+n) of all updates would
      carry mood -- so mood's total influence is diluted by planning.

LAMBDA UNITS
    lam is per mood update. With mood_updates_from='real', one update = one
    real env step. Half-life in real steps = ln(0.5)/ln(lam).
    lam=0.8 -> 3.1 steps;  lam=0.95 -> 13.5 steps;  lam=0.99 -> 69 steps.
    Episodes here are 2-4 steps, so lam=0.8 decays WITHIN an episode (an
    emotion, not a mood). Pick lam from the half-life you want, not by habit.
"""

import math
import random

from agent import DynaQ


def half_life_steps(lam):
    """Mood updates for M to decay to half its value."""
    return math.log(0.5) / math.log(lam) if 0 < lam < 1 else float("inf")


def lam_for_half_life(steps):
    """Inverse: the lam giving a target half-life in mood updates."""
    return 0.5 ** (1.0 / steps)


class MoodTracker:
    """Exponential moving average of eta*delta. Emanuel & Eldar Eq. 3.3.2."""

    def __init__(self, lam=0.95, clip=None, init=0.0):
        self.lam = lam
        self.clip = clip          # None = no clip (the paper has none)
        self.M = init
        self.n_updates = 0
        self.n_clipped = 0

    def update(self, delta, eta):
        self.M = self.M + (1 - self.lam) * (eta * delta - self.M)
        self.n_updates += 1
        if self.clip is not None:
            lo, hi = -self.clip, self.clip
            if self.M < lo or self.M > hi:
                self.n_clipped += 1
                self.M = max(lo, min(hi, self.M))
        return self.M

    @property
    def clip_fraction(self):
        return self.n_clipped / self.n_updates if self.n_updates else 0.0


class MoodyDynaQ(DynaQ):
    """
    Dyna-Q with the Emanuel & Eldar mood term on the value update.

    Identical to DynaQ in every other respect: same model learning, same
    planning, same action selection, same masking.
    """

    def __init__(self, valid_actions, eta=0.1, lam=0.95, mood_clip=None,
                 mood_updates_from="real", mood_biases="real",
                 mood_weight=None, **kwargs):
        # eta IS the learning rate (theory couples them)
        kwargs.pop("alpha", None)
        super().__init__(valid_actions, alpha=eta, **kwargs)

        self.eta = eta
        # (1 - eta) per Eq. 3.4.1; override only for sensitivity analysis
        self.mood_weight = (1 - eta) if mood_weight is None else mood_weight
        self.mood = MoodTracker(lam=lam, clip=mood_clip)

        if mood_updates_from not in ("real", "real+planning"):
            raise ValueError(mood_updates_from)
        if mood_biases not in ("real", "real+planning"):
            raise ValueError(mood_biases)
        self.mood_updates_from = mood_updates_from
        self.mood_biases = mood_biases

        self._in_planning = False
        self.mood_trace = []       # M after every real step

    def _mood_term(self):
        """The (1-eta)*M added to the target, if mood biases this update."""
        if self._in_planning and self.mood_biases == "real":
            return 0.0
        return self.mood_weight * self.mood.M

    def update(self, state, action, reward, next_state, extra=0.0):
        delta = super().update(state, action, reward, next_state,
                               extra=self._mood_term())
        feed = (not self._in_planning) or self.mood_updates_from == "real+planning"
        if feed:
            self.mood.update(delta, self.eta)
        return delta

    def plan(self):
        self._in_planning = True
        try:
            super().plan()
        finally:
            self._in_planning = False

    def step(self, state, action, reward, next_state):
        delta = super().step(state, action, reward, next_state)
        self.mood_trace.append(self.mood.M)
        return delta


def per_step_bonus(eta, M):
    """
    A sustained mood M acts like a constant per-step reward bonus.

    At the fixed point of the biased update, eta*E[delta] + (1-eta)*M = 0, so
    E[delta] = -(1-eta)M/eta -- i.e. Q behaves as if every step paid an extra
        b = (1 - eta) * M / eta

    Because b accrues per step, it shifts long and short routes by different
    amounts: the value gap between two routes moves by b * (L_long - L_short).
    Route preference flips when |b| * dL exceeds the route margin.
    """
    return (1 - eta) * M / eta


def flip_threshold(eta, margin, dL):
    """|M| needed for the per-step bonus to flip route preference."""
    return abs(margin) * eta / ((1 - eta) * dL)
