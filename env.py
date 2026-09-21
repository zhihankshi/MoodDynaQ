"""
Two-corridor shield/trap maze, with configurable corridor lengths.

    start --right--> [u_1 ... u_{Lu-1}] --> u_goal          Lu steps, unprotected
      |              (trap on the first upper step)
    down
      |
    shield --right--> [d_1 ... d_{Ll-1}] --> d_goal         Ll steps, protected in phase 1
                      (trap on the step after the shield)

The only decision is at `start`. Corridors never merge and there is no way back,
so shield possession is fully determined by which corridor the agent is in and
does NOT need to be part of the state.

Rewards (all negative; no goal bonus -- a constant on every terminal transition
shifts all Q-values equally and cannot change the argmax):
    step_cost      c  charged on every transition
    trap_cost      T  charged additionally on entering an unprotected trap
    trap_protected t  charged additionally on entering a protected trap

Phase 1: t = 0  (shield works)  -> lower corridor optimal
Phase 2: t = T  (shield inert)  -> upper corridor optimal

MARGIN ARITHMETIC
    Phase 1 margin = (T - t) - c * dL
    Phase 2 margin = c * dL          where dL = lower_len - upper_len
    Matched margins  <=>  c * dL = (T - t) / 2

At gamma=1 there is no discounting, so corridor length and step cost are
interchangeable levers on the same quantity c*dL. They are NOT interchangeable
for the learning problem: a longer corridor means a longer backward-propagation
chain, so value takes more passes to reach `start`. Use length, not step cost,
to make the task harder to learn.
"""


def make_maze_spec(upper_len=2, lower_len=4):
    """Build states, valid actions and transitions for the given corridor lengths."""
    if upper_len < 2 or lower_len < 2:
        raise ValueError("each corridor needs at least 2 steps (trap + goal)")

    valid, trans = {}, {}

    # Upper corridor: start -> u_1 -> ... -> u_goal. Trap entered on step 1.
    upper = ["start"] + [f"u_{i}" for i in range(1, upper_len)] + ["u_goal"]
    for i in range(upper_len):
        s, ns = upper[i], upper[i + 1]
        valid.setdefault(s, []).append("right")
        trans[(s, "right")] = (ns, i == 0, False)   # unprotected trap on first step

    # Lower corridor: start -> shield -> d_1 -> ... -> d_goal. Trap after shield.
    lower = ["start", "shield"] + [f"d_{i}" for i in range(1, lower_len - 1)] + ["d_goal"]
    for i in range(lower_len):
        s, ns = lower[i], lower[i + 1]
        action = "down" if i == 0 else "right"
        valid.setdefault(s, []).append(action)
        trans[(s, action)] = (ns, i == 1, True)     # protected trap on step after shield

    return {
        "states": upper + lower[1:],
        "valid_actions": valid,
        "transitions": trans,
        "terminals": {"u_goal", "d_goal"},
        "upper_len": upper_len,
        "lower_len": lower_len,
    }


SPEC = make_maze_spec(upper_len=2, lower_len=4)
VALID_ACTIONS = SPEC["valid_actions"]
TERMINALS = SPEC["terminals"]
STATES = SPEC["states"]
_TRANSITIONS = SPEC["transitions"]


class Config:
    def __init__(self, step_cost=2.5, trap_cost=10.0, trap_protected=0.0, spec=None):
        self.step_cost = step_cost
        self.trap_cost = trap_cost
        self.trap_protected = trap_protected
        self.spec = spec or SPEC

    @classmethod
    def matched(cls, trap_cost=10.0, upper_len=2, lower_len=4, phase=1):
        """
        Config with equal margins in both phases, for the given geometry.
        Solves c = (T - t) / (2 * dL) so both margins equal (T-t)/2.
        """
        spec = make_maze_spec(upper_len, lower_len)
        dL = lower_len - upper_len
        if dL <= 0:
            raise ValueError("lower corridor must be longer than upper")
        step_cost = trap_cost / (2 * dL)
        t = 0.0 if phase == 1 else trap_cost
        return cls(step_cost, trap_cost, t, spec)

    @classmethod
    def phase1(cls, step_cost=2.5, trap_cost=10.0, spec=None):
        return cls(step_cost, trap_cost, 0.0, spec)

    @classmethod
    def phase2(cls, step_cost=2.5, trap_cost=10.0, spec=None):
        return cls(step_cost, trap_cost, trap_cost, spec)

    def margins(self):
        dL = self.spec["lower_len"] - self.spec["upper_len"]
        protection = self.trap_cost - self.trap_protected
        return {"phase1_style": protection - self.step_cost * dL,
                "phase2_style": self.step_cost * dL}


class Maze:
    def __init__(self, config=None):
        self.config = config or Config.phase1()
        self.spec = self.config.spec
        self.state = "start"

    def reset(self):
        self.state = "start"
        return self.state

    def valid_actions(self, state=None):
        return self.spec["valid_actions"].get(self.state if state is None else state, [])

    def step(self, action):
        if action not in self.valid_actions():
            raise ValueError(f"illegal action {action!r} in state {self.state!r}")
        next_state, is_trap, is_protected = self.spec["transitions"][(self.state, action)]
        reward = -self.config.step_cost
        if is_trap:
            reward -= (self.config.trap_protected if is_protected
                       else self.config.trap_cost)
        self.state = next_state
        return next_state, reward, next_state in self.spec["terminals"]

    def route_taken(self):
        return "lower" if self.state == "d_goal" else "upper"
