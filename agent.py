"""
Tabular Dyna-Q.

Three components, per Sutton & Barto:
  1. Direct RL   -- Q-learning update from the real transition
  2. Model learning -- Model[(s,a)] <- (r, s'); one entry per pair, overwritten.
                       In a deterministic environment this loses nothing.
  3. Planning    -- n times per real step, sample a previously-visited (s,a)
                    from the model and apply the SAME Q-update to the
                    simulated transition.

Note on planning vs replay: a replay buffer stores individual transitions and
resamples them; Dyna's model stores one entry per (s,a). Here, deterministic,
they are arithmetically identical -- planning is deduplicated replay. This
matters later: mood should integrate *experienced* value updates, so when mood
is added it updates on real steps only, and whether it biases planning updates
is an explicit design choice (see `mood_in_planning` hook below).

Illegal actions are masked everywhere: exploration, greedy argmax, and the
bootstrap max. The agent never attempts to walk into a wall.
"""

import random


class DynaQ:
    def __init__(self, valid_actions, alpha=0.1, gamma=1.0, epsilon=0.1,
                 planning_steps=5, q_init=0.0, seed=None):
        self.valid_actions = valid_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.q_init = q_init
        self.rng = random.Random(seed)

        self.Q = {}
        self.model = {}          # (s,a) -> (r, s')
        self.visited = []        # (s,a) pairs seen at least once, for planning

    # ---- Q table -----------------------------------------------------------

    def q(self, state, action):
        return self.Q.get((state, action), self.q_init)

    def max_q(self, state):
        """Bootstrap value. Terminal states have no valid actions -> 0."""
        actions = self.valid_actions.get(state, [])
        if not actions:
            return 0.0
        return max(self.q(state, a) for a in actions)

    def greedy_action(self, state):
        actions = self.valid_actions[state]
        best = max(self.q(state, a) for a in actions)
        # tie-break randomly so an all-zero table doesn't bias toward one action
        return self.rng.choice([a for a in actions if self.q(state, a) == best])

    def select_action(self, state):
        actions = self.valid_actions[state]
        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)
        return self.greedy_action(state)

    # ---- learning ----------------------------------------------------------

    def td_error(self, state, action, reward, next_state):
        return reward + self.gamma * self.max_q(next_state) - self.q(state, action)

    def update(self, state, action, reward, next_state, extra=0.0):
        """
        One Q-learning update. `extra` is an additive term on the update,
        unused by baseline Dyna-Q -- it is the hook where a mood term
        (1-eta)*M would later enter, per Emanuel & Eldar Eq. 3.4.1.
        """
        delta = self.td_error(state, action, reward, next_state)
        self.Q[(state, action)] = self.q(state, action) + self.alpha * delta + extra
        return delta

    def learn_model(self, state, action, reward, next_state):
        if (state, action) not in self.model:
            self.visited.append((state, action))
        self.model[(state, action)] = (reward, next_state)

    def plan(self):
        """n simulated updates from the learned model."""
        for _ in range(self.planning_steps):
            if not self.visited:
                return
            state, action = self.rng.choice(self.visited)
            reward, next_state = self.model[(state, action)]
            self.update(state, action, reward, next_state)

    def step(self, state, action, reward, next_state):
        """Full Dyna-Q step: direct RL, model learning, then planning."""
        delta = self.update(state, action, reward, next_state)
        self.learn_model(state, action, reward, next_state)
        self.plan()
        return delta
