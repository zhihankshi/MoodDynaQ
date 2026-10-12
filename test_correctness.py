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
    # Default geometry: upper=2 steps (trap on step 1), lower=4 steps (trap on
    # step 2), T=10, gamma=0.9, step cost c solved for matched margins.
    # Hand calculation:
    #   S(2) = 1 + 0.9                    = 1.9
    #   S(4) = 1 + 0.9 + 0.81 + 0.729     = 3.439
    #   D    = S(4) - S(2) = 0.81 + 0.729 = 1.539
    #   c    = T(2 - gamma) / (2D) = 11 / 3.078 = 5500/1539 = 3.573749
    #   upper      = -c - T - 0.9c = -10 - 1.9c     = -10 - 6.790123 = -16.790123
    #   lower (P1) = -3.439c                         = -12.290123
    #   lower (P2) = -3.439c - 0.9*10                = -21.290123
    #   margin P1  = -12.290123 - (-16.790123)       = 4.5
    #   margin P2  = -16.790123 - (-21.290123)       = 4.5   (= gamma*T/2)
    # Inside the corridors (V(goal) = 0):
    #   Q*(u_1,right) = Q*(d_2,right) = -c          = -3.573749
    #   Q*(d_1,right) = -c - 0.9c = -1.9c           = -6.790123
    #   Q*(shield,right) = -c - t - 0.9*1.9c = -2.71c - t
    #                    P1: -9.684860   P2: -19.684860
    # These are HARDCODED on purpose: this test is an independent check on
    # optimal.py, so it must not derive its expected values from the same code.
    # If you change the default geometry, redo the arithmetic by hand.
    c1, c2 = Config.phase1(), Config.phase2()
    Q1, _ = optimal_q(c1)
    Q2, _ = optimal_q(c2)
    ok = check("step cost c = 3.573749", abs(c1.step_cost - 3.573749) < 1e-5,
               f"got {c1.step_cost}")
    expected = [
        (Q1, "phase1", ("start", "right"), -16.790123),
        (Q1, "phase1", ("start", "down"), -12.290123),
        (Q2, "phase2", ("start", "right"), -16.790123),
        (Q2, "phase2", ("start", "down"), -21.290123),
        (Q1, "phase1", ("u_1", "right"), -3.573749),
        (Q1, "phase1", ("shield", "right"), -9.684860),
        (Q2, "phase2", ("shield", "right"), -19.684860),
        (Q1, "phase1", ("d_1", "right"), -6.790123),
        (Q2, "phase2", ("d_1", "right"), -6.790123),
        (Q2, "phase2", ("d_2", "right"), -3.573749),
    ]
    for Q, ph, pair, want in expected:
        ok &= check(f"{ph} Q*{pair} = {want}", abs(Q[pair] - want) < 1e-5,
                    f"got {Q[pair]:.6f}")
    r1, m1 = optimal_route(c1)
    r2, m2 = optimal_route(c2)
    ok &= check("phase1 optimal is lower, margin 4.5",
                r1 == "lower" and abs(m1 - 4.5) < 1e-9, f"got {r1} {m1}")
    ok &= check("phase2 optimal is upper, margin 4.5",
                r2 == "upper" and abs(m2 - 4.5) < 1e-9, f"got {r2} {m2}")
    ok &= check("agent default gamma = env gamma = 0.9",
                DynaQ(VALID_ACTIONS).gamma == c1.gamma == 0.9)

    # Margin formula holds for every geometry, not just the default:
    # matched margin = gamma*T/2 = 4.5 regardless of lower_len.
    for lower in [3, 4, 6, 8]:
        for ph in [1, 2]:
            cfg = Config.matched(lower_len=lower, phase=ph)
            _, m = optimal_route(cfg)
            ok &= check(f"matched geometry lower_len={lower} phase{ph}: margin 4.5",
                        abs(m - 4.5) < TOL, f"got {m}")
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
