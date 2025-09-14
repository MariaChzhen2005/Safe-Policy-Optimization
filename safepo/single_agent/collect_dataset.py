#!/usr/bin/env python3
# -------------------------------------------------------------------------
# collect_pointgoal2_dataset.py
# -------------------------------------------------------------------------
# Generate a numpy dataset for training a state–action safety-projection
# auto-encoder on SafetyPointGoal2-v0.
#
# Saves four arrays that plug straight into your framework
#
#   X_feasible      : np.ndarray  (N_f , obs_dim + act_dim)
#   X_infeasible    : np.ndarray  (N_i , obs_dim + act_dim)
#   X_all           : np.ndarray  (N_f+N_i , obs_dim + act_dim)
#   feasible_mask   : np.ndarray  (N_f+N_i ,)   – boolean
#
# Each row is the concatenation  [ observation , action ].
# A pair is labelled feasible iff  cost == 0  when executing the action.
#
# -------------------------------------------------------------------------
# Usage
#   python collect_dataset.py --n_samples 250000 --out data/dataset_pointgoal2.npz
# -------------------------------------------------------------------------

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import gymnasium as gym                                 # Gym API
import safety_gymnasium                                  # Registers Safety-Gymnasium envs


# -------------------------------------------------------------------------#
# Optional wrapper: terminate episode when the robot exits the 21×21 arena #
# -------------------------------------------------------------------------#
class ResetOnBoundary(gym.Wrapper):
    """End the episode if the Point robot leaves the square [-10.5,10.5]^2."""
    def step(self, action):
        obs, rew, cost, terminated, truncated, info = super().step(action)
        # Point's first two observation entries are x, y
        x, y = obs[0], obs[1]
        if abs(x) > 10.5 or abs(y) > 10.5:
            terminated = True
        return obs, rew, cost, terminated, truncated, info


# -------------------------------------------------------------------------#
# Collection function                                                      #
# -------------------------------------------------------------------------#
def collect_pairs(
    env_name: str,
    n_samples: int,
    seed: int = 0,
    max_ep_len: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out a random policy until `n_samples` state–action pairs are stored.

    Returns
    -------
    X_feasible, X_infeasible, X_all, feasible_mask
    """
    env = ResetOnBoundary(gym.make(env_name))
    env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    safe:   List[np.ndarray] = []
    unsafe: List[np.ndarray] = []

    obs, _ = env.reset()
    steps_left = max_ep_len

    while (len(safe) + len(unsafe) < n_samples):
        act = env.action_space.sample()
        next_obs, _, cost, terminated, truncated, _ = env.step(act)

        pair_vec = np.concatenate([obs, act]).astype(np.float32)
        save_not_save = np.random.randint(0, 2)
        if cost == 0.0 and save_not_save != 1:
            safe.append(pair_vec)
        else:
            unsafe.append(pair_vec)

        # episode bookkeeping
        steps_left -= 1
        if terminated or truncated or steps_left == 0:
            obs, _ = env.reset()
            steps_left = max_ep_len
        else:
            obs = next_obs
        
        if steps_left % 1000 == 0:
            print(f"Collecting {len(safe)} safe and {len(unsafe)} unsafe samples from {env_name} ...")
    # -----------------------  convert to arrays  --------------------------
    X_f = np.stack(safe)   if safe   else np.empty((0, obs_dim + act_dim), np.float32)
    X_i = np.stack(unsafe) if unsafe else np.empty((0, obs_dim + act_dim), np.float32)

    X_all = np.concatenate([X_f, X_i], axis=0)
    feasible_mask = np.concatenate([
        np.ones (len(X_f), dtype=bool),
        np.zeros(len(X_i), dtype=bool)
    ])

    return X_f, X_i, X_all, feasible_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="SafetyPointGoal2-v0")
    parser.add_argument("--n_samples",  type=int, default=250_000,
                        help="total number of (obs, action) pairs to save")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--out",        default="dataset_pointgoal2.npz",
                        help="output .npz path")
    parser.add_argument("--max-ep-len", type=int, default=1000,
                        help="episode length before forced reset")
    args = parser.parse_args()

    print(f"Collecting {args.n_samples:,} samples from {args.env} ...")
    X_f, X_i, X_all, feasible = collect_pairs(
        env_name     = args.env,
        n_samples    = args.n_samples,
        seed         = args.seed,
        max_ep_len   = args.max_ep_len,
    )

    # ----------------------------  save  ---------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_feasible    = X_f,
        X_infeasible  = X_i,
        X_all         = X_all,
        feasible_mask = feasible,
    )

    # ----------------------------  stats ---------------------------------
    print("Saved dataset to", out_path)
    print(f"  Feasible   : {len(X_f):7d}")
    print(f"  Infeasible : {len(X_i):7d}")
    frac = len(X_f) / (len(X_f) + len(X_i)) * 100
    print(f"  Feasible ratio: {frac:5.1f} %")


if __name__ == "__main__":
    main()