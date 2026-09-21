"""What does M actually do over training?"""
from env import Maze, Config, make_maze_spec
from mood import MoodyDynaQ, per_step_bonus, flip_threshold
from train import run_phase

spec = make_maze_spec(2, 4)
for eta in [0.1, 0.5]:
    m = MoodyDynaQ(spec["valid_actions"], eta=eta, lam=0.95, epsilon=0.1,
                   planning_steps=0, seed=0)
    log = []
    run_phase(Maze(Config.matched(lower_len=4, phase=1)), m, 400, "phase1", log, 0)
    tr = m.mood_trace
    thr = flip_threshold(eta, 5.0, 2)
    print(f"\neta={eta}: flip threshold |M| > {thr:.3f}")
    print(f"  peak |M|      = {max(abs(x) for x in tr):.4f}")
    print(f"  mean |M| first 50 steps = {sum(abs(x) for x in tr[:50])/50:.4f}")
    print(f"  steps with |M| > threshold: {sum(1 for x in tr if abs(x)>thr)} / {len(tr)}")
    print(f"  M at steps 0,5,10,20,50,100: " +
          " ".join(f"{tr[i]:.3f}" for i in [0,5,10,20,50,100] if i < len(tr)))
    print(f"  implied peak bonus b = {per_step_bonus(eta, max(tr, key=abs)):.3f}"
          f"  (need |b*dL| > 5, dL=2 -> |b|>2.5)")

# sanity: does mood actually change anything at all?
print("\n\n=== sanity: mood term is actually being applied ===")
m = MoodyDynaQ(spec["valid_actions"], eta=0.1, lam=0.95, epsilon=0.0,
               planning_steps=0, seed=0)
m.mood.M = -1.0   # force a large sustained mood
q_before = m.q("start","down")
m.update("start","down",-2.5,"shield")
print(f"forced M=-1.0, mood_weight={m.mood_weight}")
print(f"  expected extra term = {m.mood_weight * -1.0:.3f}")
print(f"  Q went {q_before:.4f} -> {m.q('start','down'):.4f}")
