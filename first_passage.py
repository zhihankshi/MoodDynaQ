"""
First-passage vs sustained greedy convergence (CLAUDE.md §8, 2026-09-23).

sustain=1 counts the agent as having learned at the FIRST episode its greedy
action at `start` picks the optimal route; sustain=20 is the headline metric.
Also reports how often the greedy argmax flips after that first passage.
Arms: baseline alpha=0.10, rate-matched baseline alpha=0.15, mood eta=0.1
lam=0.7. eps=0.1, lower_len=4, 400 episodes/phase, 30 paired seeds,
planning_steps 0, 1, 5.

    python3 first_passage.py
"""
import numpy as np

from plots import run_one
from train import episodes_to_greedy_convergence
N_EP=400; SEEDS=range(30)
def conv(log, ph, tgt, k):
    c = episodes_to_greedy_convergence(log, f"phase{ph}", tgt, sustain=k)
    return N_EP if c is None else c - (N_EP if ph==2 else 0)
def flips(log, ph, tgt, start):
    rows=[r for r in log if r["phase"]==f"phase{ph}"][start:]
    g=[r["greedy_route"] for r in rows]
    return sum(a!=b for a,b in zip(g,g[1:]))
arms=[("baseline α=0.10","baseline",0.1,None),("baseline α=0.15","baseline",0.15,None),("mood η=0.1 λ=0.7","mood",0.1,0.7)]
for n in (0,1,5):
    print(f"\n## planning_steps={n}")
    res={}
    for lab,kind,eta,lam in arms:
        d={}
        for seed in SEEDS:
            log,_=run_one(kind,seed,eta,lam,n,4,N_EP,epsilon=0.1)
            for ph,t in ((1,"lower"),(2,"upper")):
                for k in (1,5,20):
                    d.setdefault((ph,k),[]).append(conv(log,ph,t,k))
                f1=conv(log,ph,t,1)
                d.setdefault((ph,"flips"),[]).append(flips(log,ph,t,f1) if f1<N_EP else 0)
        res[lab]=d
    for ph in (1,2):
        print(f"Phase {ph}: | arm | sustain=1 | sustain=5 | sustain=20 | flips after 1st pass | censored(s=1)")
        for lab in res:
            d=res[lab]; s=lambda v: f"{np.mean(v):6.1f} ± {np.std(v,ddof=1)/np.sqrt(len(v)):4.1f}"
            print(f"  {lab:18s} {s(d[(ph,1)])} | {s(d[(ph,5)])} | {s(d[(ph,20)])} | {np.mean(d[(ph,'flips')]):5.1f} | {sum(x>=N_EP for x in d[(ph,1)])}")
    b=np.array(res["baseline α=0.10"][(2,1)],float); m=np.array(res["mood η=0.1 λ=0.7"][(2,1)],float)
    dd=b-m; print(f"  P2 paired base−mood sustain=1: {dd.mean():.2f} ± {dd.std(ddof=1)/np.sqrt(30):.2f}, mood faster {int((dd>0).sum())}/30, ties {int((dd==0).sum())}")
    b=np.array(res["baseline α=0.10"][(1,1)],float); m=np.array(res["mood η=0.1 λ=0.7"][(1,1)],float)
    dd=b-m; print(f"  P1 paired base−mood sustain=1: {dd.mean():.2f} ± {dd.std(ddof=1)/np.sqrt(30):.2f}, mood faster {int((dd>0).sum())}/30, ties {int((dd==0).sum())}")
