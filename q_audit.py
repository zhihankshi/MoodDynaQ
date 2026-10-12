"""
Phase 1 Q-table audit (γ=0.9 study design): Q(start,·) at the end of every
episode, baseline α=0.10 vs mood η=0.10, for one seed on which both agents run
the same route every episode until both have settled on the correct greedy
route. Shows where their values part ways.

    python3 q_audit.py [seed]            -> figures/fig_q_audit_p1.png (original order, default seed 19)
    python3 q_audit.py reversed [seed]   -> figures/fig_q_audit_p1_reversed.png (default seed 12)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from study import run, p1_settled, N_DECAY, LAM, OPT, QSTAR, REVERSED
from mood import half_life_steps

HEAD, ROW = ("#2c5f7c", "white"), ("#d5d9de", "#eceef1")
WRONG = "#f6d5d5"           # greedy route wrong (prefers upper)


def audit(seed=19, order=(1, 2)):
    target = OPT[order[0]]
    b, m = (run(k, 0.10, seed, 120, 0, N_DECAY, order) for k in ("baseline", "mood"))
    sb, sm = p1_settled(b, target), p1_settled(m, target)
    last = max(sb, sm)
    assert all(b[i]["route"] == m[i]["route"] for i in range(last + 1)), \
        "routes differ before both settle; pick another seed"
    for name, log in (("baseline", b), ("mood", m)):
        print(f"\n{name}  (settled on {target} from episode {p1_settled(log, target)})")
        for r in log[:last + 1]:
            print(f"  {r['episode']:3d} {r['route']:6s} right {r['q_start_right']:8.3f}  "
                  f"down {r['q_start_down']:8.3f}  greedy {r['greedy_route']:6s}"
                  + ("" if name == "baseline" else f"  M {r['mood_M']:+.3f}"))
    return b[:last + 1], m[:last + 1], sb, sm


def draw_table(ax, log, cols, title, target):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, pad=6)
    rows = []
    for r in log:
        row = [str(r["episode"]), r["route"].capitalize(),
               f"{r['q_start_right']:.3f}", f"{r['q_start_down']:.3f}"]
        if len(cols) == 5:
            row.append(f"{r['mood_M']:+.3f}")
        rows.append(row)
    t = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="left",
                 colLoc="left", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor("white")
        if i == 0:
            cell.set_facecolor(HEAD[0])
            cell.get_text().set_color(HEAD[1])
            cell.get_text().set_weight("bold")
            continue
        r = log[i - 1]
        wrong = r["greedy_route"] != target
        cell.set_facecolor(WRONG if wrong else ROW[i % 2])
        greedy_col = 2 if r["greedy_route"] == "upper" else 3
        if j == greedy_col:
            cell.get_text().set_weight("bold")


def figure(seed=19, order=(1, 2)):
    target, wrong = OPT[order[0]], OPT[order[1]]
    q = QSTAR[order[0]]
    b, m, sb, sm = audit(seed, order)
    # every episode to 10, then every 5th, plus each agent's settle episode;
    # routes are identical, so both tables show the same rows
    show = sorted(set(range(11)) | set(range(15, len(b), 5)) | {sb, sm})
    wb, wm = (sum(r["greedy_route"] != target for r in log) for log in (b, m))
    b, m = [b[i] for i in show], [m[i] for i in show]
    n = len(b)
    h = 0.27 * (n + 1) + 1.6                 # inches; margins below are fixed in inches too
    fig, (axB, axM) = plt.subplots(1, 2, figsize=(10, h),
                                   gridspec_kw=dict(width_ratios=[4, 5], wspace=0.12))
    draw_table(axB, b, ["Ep", "Route", "B Right", "B Down"],
               f"Baseline α=0.10\ngreedy correct from ep {sb}; wrong in {wb} eps", target)
    draw_table(axM, m, ["Ep", "Route", "M Right", "M Down", "Mood"],
               f"Mood η=0.10\ngreedy correct from ep {sm}; wrong in {wm} eps", target)
    fig.suptitle(f"Phase 1 Q-table audit{', reversed order (shield broken)' if order == REVERSED else ''}"
                 f", seed {seed}: Q(start, ·) at end of episode",
                 x=0.125, ha="left", fontsize=12, y=1 - 0.15 / h)
    fig.subplots_adjust(top=1 - 0.95 / h, bottom=0.62 / h)
    fig.text(0.125, 0.5 / h,
             f"Bold = the greedy choice. Red rows = greedy route wrong (prefers {wrong}). Rows: every "
             "episode to 10, then every 5th, plus each agent's settle episode;\nthe greedy route can "
             "flip between rows shown. Both agents run the same route every episode. "
             "Mood = M at the end of the episode.\n"
             f"Q*: right {q[('start', 'right')]:.3f}, down {q[('start', 'down')]:.3f}.   γ=0.9   λ={LAM} (half-life "
             f"{half_life_steps(LAM):.2f} real steps)   planning_steps=0   q_init=0   "
             f"ε 1.0→0.1 over P1 ep 0–{N_DECAY - 1}   lower_len=4",
             fontsize=7.5, color="#555555", va="top")
    return fig


if __name__ == "__main__":
    args = sys.argv[1:]
    rev = "reversed" in args
    nums = [int(x) for x in args if x.isdigit()]
    seed = nums[0] if nums else (12 if rev else 19)
    path = "figures/fig_q_audit_p1%s.png" % ("_reversed" if rev else "")
    figure(seed, REVERSED if rev else (1, 2)).savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
