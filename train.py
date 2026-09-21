"""
Two-phase training loop.

Phase 1: shield protects  -> lower corridor optimal
Phase 2: shield inert     -> upper corridor optimal

The switch changes ONLY the reward structure. The same agent object continues
across the boundary: the Q table, the model, and epsilon are all carried over
untouched. Nothing about the switch is signalled to the agent -- it must
discover the change from experienced outcomes alone.

Epsilon is held at a constant floor throughout both phases rather than being
reset or raised at the switch. Raising it at the boundary would hand the agent
an external cue that something changed, which is precisely the information an
internal mood signal is hypothesised to supply.
"""

import csv

from env import Maze, Config, VALID_ACTIONS, make_maze_spec
from agent import DynaQ
from optimal import optimal_q, optimal_route


def run_episode(env, agent, max_steps=50):
    state = env.reset()
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.step(state, action, reward, next_state)
        total_reward += reward
        steps += 1
        state = next_state
        if done:
            break

    return {
        "return": total_reward,
        "steps": steps,
        "route": env.route_taken(),
    }


def run_phase(env, agent, n_episodes, phase_label, log, episode_offset=0):
    for i in range(n_episodes):
        result = run_episode(env, agent)
        log.append({
            "episode": episode_offset + i,
            "phase": phase_label,
            "route": result["route"],
            "return": result["return"],
            "steps": result["steps"],
            "q_start_right": agent.q("start", "right"),
            "q_start_down": agent.q("start", "down"),
            "greedy_route": "lower" if agent.greedy_action("start") == "down" else "upper",
            "epsilon": agent.epsilon,
        })
    return episode_offset + n_episodes


def episodes_to_greedy_convergence(log, phase, target_route, sustain=20):
    """
    First episode in `phase` where the GREEDY policy picks `target_route` and
    keeps picking it for `sustain` consecutive episodes.

    This measures what the agent has learned, independent of exploration noise
    -- unlike a rolling window over executed routes, which is bounded below by
    the epsilon floor and so can never reach 100%.
    """
    rows = [r for r in log if r["phase"] == phase]
    run = 0
    for r in rows:
        if r["greedy_route"] == target_route:
            run += 1
            if run >= sustain:
                return r["episode"] - sustain + 1
        else:
            run = 0
    return None


def main(seed=0, episodes_per_phase=300, alpha=0.1, epsilon=0.1,
         planning_steps=5, lower_len=4, verbose=True):
    spec = make_maze_spec(2, lower_len)
    agent = DynaQ(spec["valid_actions"], alpha=alpha, epsilon=epsilon,
                  planning_steps=planning_steps, seed=seed)
    log = []

    # --- Phase 1 ---
    cfg1 = Config.matched(lower_len=lower_len, phase=1)
    env = Maze(cfg1)
    offset = run_phase(env, agent, episodes_per_phase, "phase1", log, 0)

    # --- Phase 2: same agent, new reward structure, nothing reset ---
    cfg2 = Config.matched(lower_len=lower_len, phase=2)
    env = Maze(cfg2)
    run_phase(env, agent, episodes_per_phase, "phase2", log, offset)

    if verbose:
        for phase, cfg, target in [("phase1", cfg1, "lower"),
                                   ("phase2", cfg2, "upper")]:
            Q_star, _ = optimal_q(cfg)
            route, margin = optimal_route(cfg)
            conv = episodes_to_greedy_convergence(log, phase, target)
            final = [r for r in log if r["phase"] == phase][-1]
            print(f"--- {phase} (optimal: {route}, margin {margin:.2f}) ---")
            print(f"  episodes to sustained greedy {target}: {conv}")
            print(f"  Q(start,right) learned {final['q_start_right']:>8.3f}   "
                  f"exact {Q_star[('start','right')]:>8.3f}")
            print(f"  Q(start,down)  learned {final['q_start_down']:>8.3f}   "
                  f"exact {Q_star[('start','down')]:>8.3f}")

    return agent, log


def write_log(log, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        writer.writeheader()
        writer.writerows(log)


if __name__ == "__main__":
    agent, log = main()
    write_log(log, "run_log.csv")
