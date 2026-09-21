from env import Maze, Config, make_maze_spec
from agent import DynaQ
from mood import MoodyDynaQ, flip_threshold
from train import run_phase, episodes_to_greedy_convergence

spec = make_maze_spec(2, 4)
ETA, N_EP = 0.1, 400
thr = flip_threshold(ETA, 5.0, 2)
print(f"eta={ETA}, flip threshold |M| > {thr:.3f}, planning_steps=0\n")
print(f"{'lam':>6} {'peak|M|':>9} {'>thr?':>6} {'base P1':>9} {'mood P1':>9} {'base P2':>9} {'mood P2':>9}")

for lam in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    bp1=mp1=bp2=mp2=0; peaks=[]; nb=nm=0
    for seed in range(8):
        b = DynaQ(spec["valid_actions"], alpha=ETA, epsilon=0.1, planning_steps=0, seed=seed)
        m = MoodyDynaQ(spec["valid_actions"], eta=ETA, lam=lam, epsilon=0.1,
                       planning_steps=0, seed=seed)
        for a, tag in [(b,'b'), (m,'m')]:
            log=[]
            run_phase(Maze(Config.matched(lower_len=4, phase=1)), a, N_EP, "phase1", log, 0)
            run_phase(Maze(Config.matched(lower_len=4, phase=2)), a, N_EP, "phase2", log, N_EP)
            c1 = episodes_to_greedy_convergence(log,"phase1","lower")
            c2 = episodes_to_greedy_convergence(log,"phase2","upper")
            c1 = N_EP if c1 is None else c1; c2 = 2*N_EP if c2 is None else c2-N_EP
            if tag=='b': bp1+=c1; bp2+=c2; nb+=1
            else: mp1+=c1; mp2+=c2; nm+=1
        peaks.append(max(abs(x) for x in m.mood_trace))
    pk = sum(peaks)/len(peaks)
    print(f"{lam:>6} {pk:>9.3f} {'YES' if pk>thr else 'no':>6} "
          f"{bp1/nb:>9.1f} {mp1/nm:>9.1f} {bp2/nb:>9.1f} {mp2/nm:>9.1f}")
