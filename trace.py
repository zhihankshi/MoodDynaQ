"""
Instrumented walkthrough: prints every Q-update with the arithmetic shown.

Run this, read the output, and work out what is happening before reading any
explanation. The questions at the bottom of this file are the ones worth being
able to answer.

    python3 trace.py
"""

from env import Maze, Config, VALID_ACTIONS
from agent import DynaQ
from optimal import optimal_q


class TracingDynaQ(DynaQ):
    """Same agent, but narrates every update."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_on = True
        self.update_count = 0

    def update(self, state, action, reward, next_state, extra=0.0):
        q_before = self.q(state, action)
        bootstrap = self.max_q(next_state)
        delta = reward + self.gamma * bootstrap - q_before
        q_after = q_before + self.alpha * delta + extra
        self.Q[(state, action)] = q_after
        self.update_count += 1

        if self.trace_on:
            print(f"      Q({state},{action}): {q_before:>7.3f} -> {q_after:>7.3f}")
            print(f"        target = r + gamma*max_a' Q({next_state},a')"
                  f" = {reward:.2f} + {self.gamma}*{bootstrap:.3f}"
                  f" = {reward + self.gamma*bootstrap:.3f}")
            print(f"        delta  = target - Q_before"
                  f" = {reward + self.gamma*bootstrap:.3f} - {q_before:.3f}"
                  f" = {delta:.3f}")
            print(f"        Q_new  = Q_before + alpha*delta"
                  f" = {q_before:.3f} + {self.alpha}*{delta:.3f}"
                  f" = {q_after:.3f}")
        return delta

    def plan(self):
        if self.trace_on and self.planning_steps > 0 and self.visited:
            print(f"    -- planning ({self.planning_steps} simulated updates "
                  f"from a model with {len(self.visited)} known pairs) --")
        super().plan()


def trace_episodes(n_episodes=3, planning_steps=2, alpha=0.1, epsilon=0.0, seed=0):
    agent = TracingDynaQ(VALID_ACTIONS, alpha=alpha, gamma=1.0,
                         epsilon=epsilon, planning_steps=planning_steps, seed=seed)
    env = Maze(Config.phase1())

    for ep in range(n_episodes):
        print(f"\n{'='*70}")
        print(f"EPISODE {ep}")
        print(f"{'='*70}")
        state = env.reset()
        step_i = 0

        while True:
            actions = VALID_ACTIONS[state]
            q_vals = {a: round(agent.q(state, a), 3) for a in actions}
            action = agent.select_action(state)

            print(f"\n  step {step_i}: at '{state}'")
            print(f"    Q-values here: {q_vals}")
            print(f"    chose '{action}'" +
                  (" (only legal action)" if len(actions) == 1
                   else " (greedy over the values above)"))

            next_state, reward, done = env.step(action)
            print(f"    -> '{next_state}', reward {reward:.2f}"
                  + ("  [TERMINAL]" if done else ""))
            print(f"    -- direct RL --")
            agent.update(state, action, reward, next_state)
            agent.learn_model(state, action, reward, next_state)
            agent.plan()

            state = next_state
            step_i += 1
            if done:
                break

        print(f"\n  end of episode {ep}. Q(start,*) now: "
              f"right={agent.q('start','right'):.3f}, "
              f"down={agent.q('start','down'):.3f}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    Q_star, _ = optimal_q(Config.phase1())
    print(f"{'(state, action)':<22} {'learned':>10} {'exact Q*':>10} {'gap':>8}")
    for pair in sorted(Q_star, key=lambda p: (p[0], p[1])):
        learned, exact = agent.q(*pair), Q_star[pair]
        print(f"{str(pair):<22} {learned:>10.3f} {exact:>10.3f} {exact-learned:>8.3f}")
    print(f"\ntotal Q-updates performed: {agent.update_count}")


def show_backward_propagation(planning_steps=0, alpha=0.1, n_episodes=12, seed=0):
    """
    Value propagates backward one link per pass. Watch the lower corridor fill
    in from the goal end. Compare planning_steps=0 against a larger value.
    """
    agent = DynaQ(VALID_ACTIONS, alpha=alpha, gamma=1.0, epsilon=0.0,
                  planning_steps=planning_steps, seed=seed)
    agent.Q[("start", "down")] = 0.01  # force it down the lower corridor
    env = Maze(Config.phase1())

    chain = [("d4", "down"), ("d3", "down"), ("d2", "down"),
             ("d1", "down"), ("start", "down")]
    print(f"\nplanning_steps={planning_steps}: value propagating backward "
          f"from the goal (alpha={alpha})")
    print(f"{'ep':>3} " + " ".join(f"{s:>8}" for s, _ in chain))
    for ep in range(n_episodes):
        state = env.reset()
        for _ in range(50):
            a = agent.select_action(state)
            ns, r, done = env.step(a)
            agent.step(state, a, r, ns)
            state = ns
            if done:
                break
        print(f"{ep:>3} " + " ".join(f"{agent.q(s,a):>8.3f}" for s, a in chain))
    print("     exact: " + " ".join(f"{v:>8.2f}" for v in
                                    [-2.5, -5.0, -7.5, -10.0, -12.5]))


if __name__ == "__main__":
    print("#" * 70)
    print("# PART 1: three episodes, every update shown")
    print("#" * 70)
    trace_episodes(n_episodes=3, planning_steps=2, epsilon=0.0)

    print("\n\n" + "#" * 70)
    print("# PART 2: backward propagation, with and without planning")
    print("#" * 70)
    show_backward_propagation(planning_steps=0)
    show_backward_propagation(planning_steps=10)

    print("""

{0}
QUESTIONS TO WORK THROUGH
{0}

1. In episode 0, every Q starts at 0. Why is the FIRST update's delta just the
   reward, with nothing added from the next state?

2. In episode 0, the agent walks the whole corridor but Q(start,down) barely
   moves. Why can't the trap's cost reach the start state in one episode?

3. Look at PART 2. With planning_steps=0, how many episodes does it take for
   a non-zero value to appear in the 'start' column? Relate that number to the
   corridor's length.

4. With planning_steps=10, the whole chain fills in almost immediately. The
   agent had exactly the same real experience in both runs. Where did the extra
   learning come from?

5. alpha=0.1 means each update moves Q one tenth of the way to its target.
   After k updates with a FIXED target y, Q_k = y*(1 - (1-alpha)^k). Verify this
   against one of the traced sequences by hand.

6. Why does the bootstrap term use max_a' Q(s',a') rather than the Q-value of
   the action actually taken next? What would change if it used the latter?

7. Q(u1,right) should converge to -15: that is -2.5 (the step) -10 (the trap)
   -2.5 (the remaining step to the goal). With epsilon=0 the agent never takes
   the upper corridor at all. So how does Q(u1,right) ever get updated?
   (Check the SUMMARY table -- is it still 0?)

8. In the SUMMARY, which pairs have converged and which have not? What do the
   unconverged ones have in common?
""".format("=" * 70))
