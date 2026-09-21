"""Baseline vs mood-biased Dyna-Q, seed-paired."""
from env import Maze, Config, make_maze_spec
from agent import DynaQ
from mood import MoodyDynaQ
from train import run_phase, episodes_to_greedy_convergence

def run(agent, lower_len, n_ep):
    log = []
    run_phase(Maze(Config.matched(lower_len=lower_len, phase=1)), agent, n_ep, "phase1", log, 0)
    run_phase(Maze(Config.matched(lower_len=lower_len, phase=2)), agent, n_ep, "phase2", log, n_ep)
    return log

def summarize(log, phase, target):
    rows = [r for r in log if r["phase"] == phase]
    conv = episodes_to_greedy_convergence(log, phase, target)
    frac = sum(1 for r in rows if r["greedy_route"] == target) / len(rows)
    return conv, frac

LOWER, N_EP, SEEDS = 4, 400, range(8)
spec = make_maze_spec(2, LOWER)

for n_plan in [0, 5]:
    for eta in [0.1, 0.5, 0.9]:
        print(f"\n### planning_steps={n_plan}, eta={eta} (mood weight {1-eta:.1f}), lam=0.95")
        print(f"{'seed':>5} {'base P1':>9} {'mood P1':>9} {'base P2':>9} {'mood P2':>9} {'final M':>9}")
        for seed in SEEDS:
            b = DynaQ(spec["valid_actions"], alpha=eta, epsilon=0.1,
                      planning_steps=n_plan, seed=seed)
            m = MoodyDynaQ(spec["valid_actions"], eta=eta, lam=0.95, epsilon=0.1,
                           planning_steps=n_plan, seed=seed,
                           mood_updates_from="real", mood_biases="real")
            lb, lm = run(b, LOWER, N_EP), run(m, LOWER, N_EP)
            b1, _ = summarize(lb, "phase1", "lower"); m1, _ = summarize(lm, "phase1", "lower")
            b2, _ = summarize(lb, "phase2", "upper"); m2, _ = summarize(lm, "phase2", "upper")
            f = lambda x: "never" if x is None else str(x)
            print(f"{seed:>5} {f(b1):>9} {f(m1):>9} {f(b2):>9} {f(m2):>9} {m.mood.M:>9.3f}")
