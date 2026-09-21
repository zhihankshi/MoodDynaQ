"""Analytic predictions before running anything."""
from mood import half_life_steps, lam_for_half_life, per_step_bonus, flip_threshold
from env import Config

print("=== 1. Lambda units ===")
print("half-life in MOOD UPDATES (= real env steps if mood_updates_from='real')")
for lam in [0.8, 0.9, 0.95, 0.98, 0.99]:
    print(f"  lam={lam:<5} -> {half_life_steps(lam):>6.1f} updates")
print("\nEpisodes here are 2-4 real steps. For mood to span EPISODES:")
for eps in [2, 5, 10, 20]:
    steps = eps * 3  # ~3 steps/episode
    print(f"  half-life of ~{eps:>2} episodes (~{steps:>2} steps) -> lam={lam_for_half_life(steps):.4f}")

print("\n=== 2. eta coupling: mood weight is (1-eta) ===")
for eta in [0.1, 0.3, 0.5, 0.9]:
    print(f"  eta={eta:<4} learning rate {eta:<5} mood weight {1-eta:.2f}"
          f"   bonus per unit M = {per_step_bonus(eta, 1.0):>6.2f}")

print("\n=== 3. Can mood flip route preference? ===")
print("margin=5, dL=2 (lower_len=4). |M| needed to flip:")
for eta in [0.1, 0.3, 0.5, 0.9]:
    thr = flip_threshold(eta, margin=5.0, dL=2)
    print(f"  eta={eta:<4} need |M| > {thr:>7.3f}")

print("\n=== 4. What M will actually reach ===")
print("M tracks eta*delta. Early transient: delta ~ -(route cost), eta*delta:")
for eta in [0.1, 0.3, 0.5, 0.9]:
    for delta in [-15.0, -5.0]:
        print(f"  eta={eta:<4} delta={delta:>6} -> M approaches {eta*delta:>7.3f}"
              f"   bonus b={per_step_bonus(eta, eta*delta):>8.3f}")
