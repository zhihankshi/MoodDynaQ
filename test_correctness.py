"""
Correctness tests: the learned Q table must match the analytic optimum.

Run this before touching mood. If these fail, the bug is in the baseline
agent or the environment, and no result involving mood would be interpretable.

    python3 test_correctness.py
"""

from env import Maze, Config, VALID_ACTIONS
from agent import DynaQ
from optimal import optimal_q, optimal_route
from train import run_phase, episodes_to_greedy_convergence

TOL = 0.01


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  {detail}" if detail else ""))
    return condition


def test_hand_computed_values():
    """The analytic values must match the numbers derived by hand."""
    print("\ntest: analytic values match hand calculation")
    Q1, _ = optimal_q(Config.phase1())
    Q2, _ = optimal_q(Config.phase2())
    ok = True
    ok &= check("phase1 Q*(start,right) = -17.5",
                abs(Q1[("start", "right")] + 17.5) < TOL,
                f"got {Q1[('start','right')]}")
    ok &= check("phase1 Q*(start,down)  = -12.5",
                abs(Q1[("start", "down")] + 12.5) < TOL,
                f"got {Q1[('start','down')]}")
    ok &= check("phase2 Q*(start,down)  = -22.5",
                abs(Q2[("start", "down")] + 22.5) < TOL,
                f"got {Q2[('start','down')]}")
    ok &= check("phase1 optimal is lower, margin 5",
                optimal_route(Config.phase1()) == ("lower", 5.0))
    ok &= check("phase2 optimal is upper, margin 5",
                optimal_route(Config.phase2()) == ("upper", 5.0))
    return ok


def test_convergence_all_pairs(seed=0, episodes=400):
    """Every visited (s,a) must converge to Q*, not just the start state."""
    print(f"\ntest: full Q table converges to Q* (seed {seed})")
    agent = DynaQ(VALID_ACTIONS, alpha=0.1, epsilon=0.1,
                  planning_steps=5, seed=seed)
    log = []
    env = Maze(Config.phase1())
    run_phase(env, agent, episodes, "phase1", log, 0)

    Q_star, _ = optimal_q(Config.phase1())
    worst, worst_pair = 0.0, None
    for pair, exact in Q_star.items():
        learned = agent.q(*pair)
        err = abs(learned - exact)
        if err > worst:
            worst, worst_pair = err, pair
    return check("max |Q_learned - Q*| < 0.01 across all pairs",
                 worst < TOL, f"worst {worst:.5f} at {worst_pair}")


def test_seed_robustness(n_seeds=10, episodes=400):
    """Convergence must not depend on a lucky seed."""
    print(f"\ntest: converges across {n_seeds} seeds")
    ok = True
    for seed in range(n_seeds):
        agent = DynaQ(VALID_ACTIONS, alpha=0.1, epsilon=0.1,
                      planning_steps=5, seed=seed)
        log = []
        run_phase(Maze(Config.phase1()), agent, episodes, "phase1", log, 0)
        run_phase(Maze(Config.phase2()), agent, episodes, "phase2", log, episodes)

        c1 = episodes_to_greedy_convergence(log, "phase1", "lower")
        c2 = episodes_to_greedy_convergence(log, "phase2", "upper")
        good = c1 is not None and c2 is not None
        ok &= check(f"seed {seed}: phase1 conv {c1}, phase2 conv {c2}", good)
    return ok


def test_planning_speeds_propagation():
    """
    More planning steps should propagate value backward faster.
    This is the Dyna knob that replaces the DQN target-network sync as the
    thing gating how fast value reaches the start state.
    """
    print("\ntest: planning_steps accelerates convergence")
    results = {}
    for n in [0, 1, 5, 20]:
        convs = []
        for seed in range(5):
            agent = DynaQ(VALID_ACTIONS, alpha=0.1, epsilon=0.1,
                          planning_steps=n, seed=seed)
            log = []
            run_phase(Maze(Config.phase1()), agent, 400, "phase1", log, 0)
            c = episodes_to_greedy_convergence(log, "phase1", "lower")
            convs.append(c if c is not None else 400)
        results[n] = sum(convs) / len(convs)
        print(f"    planning_steps={n:>2}: mean episodes to converge {results[n]:.1f}")
    return check("n=20 converges no slower than n=0",
                 results[20] <= results[0])


def test_no_illegal_actions():
    """Masking must hold: the agent never selects an action the env rejects."""
    print("\ntest: action masking")
    agent = DynaQ(VALID_ACTIONS, epsilon=1.0, seed=0)  # pure random
    env = Maze(Config.phase1())
    try:
        for _ in range(200):
            state = env.reset()
            for _ in range(50):
                action = agent.select_action(state)
                state, _, done = env.step(action)
                if done:
                    break
        return check("200 fully-random episodes, no illegal action", True)
    except ValueError as e:
        return check("200 fully-random episodes, no illegal action", False, str(e))


if __name__ == "__main__":
    results = [
        test_hand_computed_values(),
        test_convergence_all_pairs(),
        test_no_illegal_actions(),
        test_seed_robustness(),
        test_planning_speeds_propagation(),
    ]
    print(f"\n{'='*55}")
    print("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED")
