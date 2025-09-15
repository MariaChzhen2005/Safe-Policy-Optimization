#!/usr/bin/env python3
# collect_pointgoal2_dataset.py

import argparse
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import gymnasium as gym
import safety_gymnasium  # Registers Safety-Gymnasium envs

def collect_pairs(
    env_name: str,
    n_samples: int,
    seed: int = 0,
    max_ep_len: int = 1000,
    balance_ratio: float = 0.5,
    exploration_strategy: str = "adaptive",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out a random policy until `n_samples` balanced state–action pairs are stored.
    This version only reflects TRUE hazards (as defined by the environment).
    """
    env = gym.make(env_name)
    env.reset(seed=seed)
    np.random.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    target_feasible = int(n_samples * balance_ratio)
    target_infeasible = n_samples - target_feasible
    
    safe:   List[np.ndarray] = []
    unsafe: List[np.ndarray] = []
    obs, _ = env.reset()
    steps_left = max_ep_len
    total_steps = 0

    feasible_seen = 0
    infeasible_seen = 0
    max_attempts = n_samples * 50
    while (len(safe) < target_feasible or len(unsafe) < target_infeasible) and total_steps < max_attempts:
        if total_steps % 5000 == 0:
            print(f"Step {total_steps}: safe={len(safe)}, unsafe={len(unsafe)}")
        
        # Adaptive action sampling (unchanged)
        infeasible_ratio = len(unsafe) / max(1, len(safe) + len(unsafe))
        need_infeasible = len(unsafe) < target_infeasible

        act = env.action_space.sample()

        next_obs, _, cost, terminated, truncated, _ = env.step(act)
        total_steps += 1
        pair_vec = np.concatenate([obs, act]).astype(np.float32)

        # Diagnostics: Print hazardous lidar features when a cost is detected
        if cost == 0.0:
            feasible_seen += 1
            if len(safe) < target_feasible:
                safe.append(pair_vec)
        else:
            infeasible_seen += 1
            if len(unsafe) < target_infeasible:
                unsafe.append(pair_vec)
        steps_left -= 1
        if terminated or truncated or steps_left == 0:
            obs, _ = env.reset()
            steps_left = max_ep_len
        else:
            obs = next_obs

    print(f"Collection complete! Total steps: {total_steps:,}")
    print(f"Final counts: {len(safe)} feasible, {len(unsafe)} infeasible")

    X_f = np.stack(safe)   if safe   else np.empty((0, obs_dim + act_dim), np.float32)
    X_i = np.stack(unsafe) if unsafe else np.empty((0, obs_dim + act_dim), np.float32)
    X_all = np.concatenate([X_f, X_i], axis=0)
    feasible_mask = np.concatenate([
        np.ones (len(X_f), dtype=bool),
        np.zeros(len(X_i), dtype=bool)
    ])
    return X_f, X_i, X_all, feasible_mask

def collect_pairs_reservoir(
    env_name: str,
    n_samples: int,
    seed: int = 0,
    max_ep_len: int = 1000,
    balance_ratio: float = 0.5,
    max_steps: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-efficient collection using reservoir sampling for very large datasets.
    """
    env = gym.make(env_name)
    env.reset(seed=seed)
    np.random.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    target_feasible = int(n_samples * balance_ratio)
    target_infeasible = n_samples - target_feasible

    feasible_reservoir = np.zeros((target_feasible, obs_dim + act_dim), dtype=np.float32)
    infeasible_reservoir = np.zeros((target_infeasible, obs_dim + act_dim), dtype=np.float32)
    feasible_count = 0
    infeasible_count = 0
    feasible_seen = 0
    infeasible_seen = 0
    obs, _ = env.reset()
    steps_left = max_ep_len
    total_steps = 0
    while total_steps < (max_steps or float('inf')):
        act = env.action_space.sample()
        next_obs, _, cost, terminated, truncated, _ = env.step(act)
        total_steps += 1
        pair_vec = np.concatenate([obs, act]).astype(np.float32)

        if cost == 0.0:
            feasible_seen += 1
            if feasible_count < target_feasible:
                feasible_reservoir[feasible_count] = pair_vec
                feasible_count += 1
            else:
                j = np.random.randint(0, feasible_seen)
                if j < target_feasible:
                    feasible_reservoir[j] = pair_vec
        else:
            infeasible_seen += 1
            if infeasible_count < target_infeasible:
                infeasible_reservoir[infeasible_count] = pair_vec
                infeasible_count += 1
            else:
                j = np.random.randint(0, infeasible_seen)
                if j < target_infeasible:
                    infeasible_reservoir[j] = pair_vec
        steps_left -= 1
        if terminated or truncated or steps_left == 0:
            obs, _ = env.reset()
            steps_left = max_ep_len
        else:
            obs = next_obs
        if not max_steps and feasible_count >= target_feasible and infeasible_count >= target_infeasible:
            break
        print(f"Counts: {feasible_count} feasible, {infeasible_count} infeasible, steps: {total_steps}")

    X_f = feasible_reservoir[:min(feasible_count, target_feasible)]
    X_i = infeasible_reservoir[:min(infeasible_count, target_infeasible)]
    X_all = np.concatenate([X_f, X_i], axis=0)
    feasible_mask = np.concatenate([
        np.ones (len(X_f), dtype=bool),
        np.zeros(len(X_i), dtype=bool)
    ])
    return X_f, X_i, X_all, feasible_mask

def main():
    parser = argparse.ArgumentParser(description="Collect balanced dataset for Safety Gym environments")
    parser.add_argument("--env",        default="SafetyPointGoal2-v0",
                        help="Safety Gym environment name")
    parser.add_argument("--n_samples",  type=int, default=60_000,
                        help="total number of (obs, action) pairs to save")
    parser.add_argument("--seed",       type=int, default=0,
                        help="random seed for reproducibility")
    parser.add_argument("--out",        default="dataset_pointgoal2.npz",
                        help="output .npz path")
    parser.add_argument("--max-ep-len", type=int, default=1000,
                        help="episode length before forced reset")
    parser.add_argument("--balance-ratio", type=float, default=0.5,
                        help="ratio of feasible samples (0.5 = 50%% feasible, 50%% infeasible)")
    parser.add_argument("--use-reservoir", action="store_true",
                        help="use reservoir sampling for memory-efficient collection")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="maximum environment steps (only with --use-reservoir)")
    parser.add_argument("--exploration-strategy", default="adaptive", 
                        choices=["adaptive", "random"],
                        help="exploration strategy for finding infeasible samples")
    args = parser.parse_args()
    print(f"Collecting {args.n_samples:,} balanced samples from {args.env} ...")
    print(f"Target balance: {args.balance_ratio*100:.1f}% feasible, {(1-args.balance_ratio)*100:.1f}% infeasible")
    print(f"Method: {'Reservoir sampling' if args.use_reservoir else 'Adaptive sampling'}")
    
    if args.use_reservoir:
        X_f, X_i, X_all, feasible = collect_pairs_reservoir(
            env_name     = args.env,
            n_samples    = args.n_samples,
            seed         = args.seed,
            max_ep_len   = args.max_ep_len,
            balance_ratio = args.balance_ratio,
            max_steps    = args.max_steps,
        )
    else:
        X_f, X_i, X_all, feasible = collect_pairs(
            env_name     = args.env,
            n_samples    = args.n_samples,
            seed         = args.seed,
            max_ep_len   = args.max_ep_len,
            balance_ratio = args.balance_ratio,
            exploration_strategy = args.exploration_strategy,
        )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_feasible    = X_f,
        X_infeasible  = X_i,
        X_all         = X_all,
        feasible_mask = feasible,
    )

    print("Saved dataset to", out_path)
    print(f"  Feasible   : {len(X_f):7d}")
    print(f"  Infeasible : {len(X_i):7d}")
    frac = len(X_f) / (len(X_f) + len(X_i)) * 100 if (len(X_f)+len(X_i)) > 0 else float('nan')
    print(f"  Feasible ratio: {frac:5.1f} %")

if __name__ == "__main__":
    main()
