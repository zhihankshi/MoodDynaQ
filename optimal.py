"""
Exact optimal Q-values by backward induction.

This is the ground truth the learned agent must converge to. With ~9 states,
gamma=1 and a deterministic environment, tabular Q-learning should land on
these values essentially exactly -- so any mismatch is a bug in the agent,
not an approximation error.

Q*(s,a) = r(s,a) + max_a' Q*(s',a'),  with V(terminal) = 0.
"""

from env import VALID_ACTIONS, TERMINALS, _TRANSITIONS


def optimal_q(config):
    """Return {(state, action): Q*} computed backward from the goals."""
    Q = {}
    V = {t: 0.0 for t in TERMINALS}

    # Reverse topological order: each state's value depends only on states
    # already solved (corridors are acyclic and never merge).
    order = ["u2", "u1", "d4", "d3", "d2", "d1", "start"]

    for state in order:
        for action in VALID_ACTIONS[state]:
            next_state, is_trap, is_protected = _TRANSITIONS[(state, action)]
            reward = -config.step_cost
            if is_trap:
                reward -= (config.trap_protected if is_protected
                           else config.trap_cost)
            Q[(state, action)] = reward + V[next_state]
        V[state] = max(Q[(state, a)] for a in VALID_ACTIONS[state])

    return Q, V


def optimal_route(config):
    """Which corridor is optimal, and by how much."""
    Q, _ = optimal_q(config)
    up, down = Q[("start", "right")], Q[("start", "down")]
    if down > up:
        return "lower", down - up
    return "upper", up - down


def report(config, label=""):
    Q, V = optimal_q(config)
    route, margin = optimal_route(config)
    print(f"--- {label} (c={config.step_cost}, T={config.trap_cost}, "
          f"t={config.trap_protected}) ---")
    print(f"  Q*(start, right) = {Q[('start','right')]:>7.2f}   [upper, 3 steps]")
    print(f"  Q*(start, down)  = {Q[('start','down')]:>7.2f}   [lower, 5 steps]")
    print(f"  optimal route: {route}, margin {margin:.2f}")
    return Q, V


if __name__ == "__main__":
    from env import Config
    report(Config.phase1(), "PHASE 1  shield protects")
    print()
    report(Config.phase2(), "PHASE 2  shield inert")
