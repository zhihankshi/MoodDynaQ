"""
Exact optimal Q-values by backward induction.

This is the ground truth the learned agent must converge to. The environment is
small, deterministic and acyclic with gamma=1, so tabular Q-learning should land
on these values essentially exactly -- any mismatch is a bug, not approximation
error.

    Q*(s,a) = r(s,a) + max_a' Q*(s',a'),   V(terminal) = 0
"""


def optimal_q(config):
    """Return ({(state, action): Q*}, {state: V*}) by backward induction."""
    spec = config.spec
    trans, valid = spec["transitions"], spec["valid_actions"]
    Q, V = {}, {t: 0.0 for t in spec["terminals"]}

    # Solve states in reverse order of distance from a goal, so each state's
    # value depends only on states already solved.
    remaining = [s for s in spec["states"] if s not in spec["terminals"]]
    while remaining:
        progressed = False
        for state in list(remaining):
            successors = [trans[(state, a)][0] for a in valid[state]]
            if not all(ns in V for ns in successors):
                continue
            for action in valid[state]:
                next_state, is_trap, is_protected = trans[(state, action)]
                reward = -config.step_cost
                if is_trap:
                    reward -= (config.trap_protected if is_protected
                               else config.trap_cost)
                Q[(state, action)] = reward + V[next_state]
            V[state] = max(Q[(state, a)] for a in valid[state])
            remaining.remove(state)
            progressed = True
        if not progressed:
            raise RuntimeError("maze graph has a cycle; backward induction failed")

    return Q, V


def optimal_route(config):
    """Which corridor is optimal, and by how much."""
    Q, _ = optimal_q(config)
    up, down = Q[("start", "right")], Q[("start", "down")]
    return ("lower", down - up) if down > up else ("upper", up - down)


def report(config, label=""):
    Q, V = optimal_q(config)
    route, margin = optimal_route(config)
    s = config.spec
    print(f"--- {label} (c={config.step_cost}, T={config.trap_cost}, "
          f"t={config.trap_protected}) ---")
    print(f"  Q*(start, right) = {Q[('start','right')]:>7.2f}   "
          f"[upper, {s['upper_len']} steps]")
    print(f"  Q*(start, down)  = {Q[('start','down')]:>7.2f}   "
          f"[lower, {s['lower_len']} steps]")
    print(f"  optimal route: {route}, margin {margin:.2f}")
    return Q, V


if __name__ == "__main__":
    from env import Config

    print("Geometry options, all with matched margins of 5 (T=10):\n")
    for lower in [3, 4, 6, 8]:
        c1 = Config.matched(trap_cost=10.0, upper_len=2, lower_len=lower, phase=1)
        c2 = Config.matched(trap_cost=10.0, upper_len=2, lower_len=lower, phase=2)
        Q1, _ = optimal_q(c1)
        Q2, _ = optimal_q(c2)
        _, m1 = optimal_route(c1)
        _, m2 = optimal_route(c2)
        print(f"  lower_len={lower}, dL={lower-2}, c={c1.step_cost:>5.2f}   "
              f"P1: {Q1[('start','right')]:>7.2f}/{Q1[('start','down')]:>7.2f} (margin {m1:.1f})   "
              f"P2: {Q2[('start','right')]:>7.2f}/{Q2[('start','down')]:>7.2f} (margin {m2:.1f})")

    print()
    report(Config.matched(lower_len=4, phase=1), "PHASE 1  shield protects")
    print()
    report(Config.matched(lower_len=4, phase=2), "PHASE 2  shield inert")
