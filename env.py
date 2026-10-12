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

MARGIN ARITHMETIC (discount gamma, default 0.9)
    S(L) = 1 + gamma + ... + gamma^(L-1)       discounted step count of a route
    D    = S(lower_len) - S(upper_len)         discounted length difference
    The upper trap is entered on step 1 (undiscounted), the lower trap on
    step 2 (discounted by gamma).

    Q*(start,right) = -c*S(upper_len) - T
    Q*(start,down)  = -c*S(lower_len) - gamma*t
    Phase 1 margin (t=0, lower wins) = T - c*D
    Phase 2 margin (t=T, upper wins) = c*D - (1 - gamma)*T
    Matched margins  <=>  c = T*(2 - gamma) / (2*D),  margin = gamma*T/2

At gamma=1 this reduces to the old c*dL = T/2 and margin T/2. With gamma<1
corridor length and step cost are no longer interchangeable even for the
decision: only the discounted difference D enters.
"""

GAMMA = 0.9


def discounted_steps(L, gamma=GAMMA):
    """S(L) = sum_{k<L} gamma^k: what a per-step cost of 1 costs over L steps."""
    return sum(gamma ** k for k in range(L))


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
    """
    Reward structure plus the discount. gamma lives here, not only on the
    agent, because Q* and the matched step cost both depend on it; the agent
    must be built with the same gamma (DynaQ's default matches GAMMA).
    """

    def __init__(self, step_cost, trap_cost=10.0, trap_protected=0.0, spec=None,
                 gamma=GAMMA):
        self.step_cost = step_cost
        self.trap_cost = trap_cost
        self.trap_protected = trap_protected
        self.spec = spec or SPEC
        self.gamma = gamma

    @classmethod
    def matched(cls, trap_cost=10.0, upper_len=2, lower_len=4, phase=1,
                gamma=GAMMA):
        """
        Config with equal margins in both phases, for the given geometry.
        Solves c = T*(2 - gamma) / (2*D) so both margins equal gamma*T/2.
        """
        spec = make_maze_spec(upper_len, lower_len)
        D = discounted_steps(lower_len, gamma) - discounted_steps(upper_len, gamma)
        if D <= 0:
            raise ValueError("lower corridor must be longer than upper")
        step_cost = trap_cost * (2 - gamma) / (2 * D)
        t = 0.0 if phase == 1 else trap_cost
        return cls(step_cost, trap_cost, t, spec, gamma)

    @classmethod
    def phase1(cls, gamma=GAMMA):
        """Default geometry, matched margins, shield works."""
        return cls.matched(phase=1, gamma=gamma)

    @classmethod
    def phase2(cls, gamma=GAMMA):
        """Default geometry, matched margins, shield inert."""
        return cls.matched(phase=2, gamma=gamma)

    def margins(self):
        """Margins of this reward structure with t=0 and with t=T."""
        g = self.gamma
        D = (discounted_steps(self.spec["lower_len"], g)
             - discounted_steps(self.spec["upper_len"], g))
        return {"phase1_style": self.trap_cost - self.step_cost * D,
                "phase2_style": self.step_cost * D - (1 - g) * self.trap_cost}


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
