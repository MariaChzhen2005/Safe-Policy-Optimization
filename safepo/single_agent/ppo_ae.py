# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

import os
import random
import sys
import time
from collections import deque

import numpy as np
try: 
    from isaacgym import gymutil
except ImportError:
    pass
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, TensorDataset

from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.env import make_sa_mujoco_env, make_sa_isaac_env
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVCritic
from safepo.utils.config import single_agent_args, isaac_gym_map, parse_sim_params
from safepo.single_agent.autoencoder import ConditionalConstraintAwareAutoencoder
print("imported everything")


default_cfg = {
    'total_steps': 1024000,  # Moderate training: ~500 epochs (1024000 / 2048 = 500)
    'steps_per_epoch': 2048,
    'hidden_sizes': [64, 64],
    'gamma': 0.99,
    'target_kl': 0.02,
    'batch_size': 64,
    'learning_iters': 30,
    'max_grad_norm': 40.0,
    'proj_action_penalty_coef': 0.05,
}

isaac_gym_specific_cfg = {
    'total_steps': 3000000,
    'steps_per_epoch': 32768,
    'hidden_sizes': [1024, 1024, 512],
    'gamma': 0.96,
    'target_kl': 0.016,
    'num_mini_batch': 4,
    'use_value_coefficient': True,
    'learning_iters': 8,
    'max_grad_norm': 1.0,
    'use_critic_norm': False,
    'proj_action_penalty_coef': 0.05,
}

def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f'{args.device}:{args.device_id}')
    is_cuda = device.type == 'cuda'

    if args.task not in isaac_gym_map.keys():
        env, obs_space, act_space = make_sa_mujoco_env(
            num_envs=args.num_envs, env_id=args.task, seed=args.seed
        )
        eval_env, _, _ = make_sa_mujoco_env(num_envs=1, env_id=args.task, seed=None)
        config = default_cfg

    else:
        sim_params = parse_sim_params(args, cfg_env, None)
        env = make_sa_isaac_env(args=args, cfg=cfg_env, sim_params=sim_params)
        eval_env = env
        obs_space = env.observation_space
        act_space = env.action_space
        args.num_envs = env.num_envs
        config = isaac_gym_specific_cfg

    # set training steps
    steps_per_epoch = config.get("steps_per_epoch", args.steps_per_epoch)
    total_steps = config.get("total_steps", args.total_steps)
    local_steps_per_epoch = steps_per_epoch // args.num_envs
    epochs = total_steps // steps_per_epoch
    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)

    # Action bounds for clamping projected actions
    act_low = torch.as_tensor(act_space.low, dtype=torch.float32, device=device)
    act_high = torch.as_tensor(act_space.high, dtype=torch.float32, device=device)

    # load the trained autoencoder for action projection
    autoencoder_path = "/workspace/Safe-Policy-Optimization/safepo/single_agent/data/conditional_phase2_safety_gym_1_decoders_2_2_absolute_Adam.pt"
    print(f"Loading autoencoder from: {autoencoder_path}")
    print(f"Observation space shape: {obs_space.shape[0]}D")
    print(f"Action space shape: {act_space.shape[0]}D")
    
    # Check if autoencoder file exists
    if not os.path.exists(autoencoder_path):
        print(f"ERROR: Autoencoder file not found at {autoencoder_path}")
        print("Continuing without autoencoder (actions won't be projected)")
        autoencoder = None
    else:
        print(f"Autoencoder file found at {autoencoder_path}")
        
        # Initialize autoencoder with same architecture as the saved model
        print("Initializing autoencoder...")
        try:
            print("Creating ConditionalConstraintAwareAutoencoder instance...")
            autoencoder = ConditionalConstraintAwareAutoencoder(
                action_dim=act_space.shape[0],
                state_dim=obs_space.shape[0],
                latent_dim=act_space.shape[0],  # assuming latent_dim matches action_dim
                hidden_dim=64,
                num_decoders=1,  # Fixed: saved model has 1 decoder, not 2
                latent_geom="hypersphere",
                norm_params_path=None,
                ieee37_model_instance_in=None
            )
            print("Moving autoencoder to device...")
            autoencoder = autoencoder.to(device)
            print("Autoencoder initialized, loading weights...")

            # Load the trained weights
            print("Loading state dict...")
            autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
            print("Setting to eval mode...")
            autoencoder.eval()  # Set to evaluation mode
            print("Autoencoder loaded successfully!")
        except Exception as e:
            print(f"Error loading autoencoder: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            print("Continuing without autoencoder (actions won't be projected)")
            autoencoder = None
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs
    )
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=3e-4
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=3e-4
    )

    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=args.num_envs,
        gamma=config["gamma"],
    )

    # set up the logger
    dict_args = vars(args)
    dict_args.update(config)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    rew_deque = deque(maxlen=50)
    cost_deque = deque(maxlen=50)
    len_deque = deque(maxlen=50)
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    logger.save_config(dict_args)
    logger.setup_torch_saver(policy.actor)
    print(f"Model checkpoints will be saved to: {args.log_dir}")
    logger.log("Start with training.")
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret, ep_cost, ep_len = (
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
    )
    # training loop
    for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch", file=sys.stdout):
        rollout_start_time = time.time()
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(obs, deterministic=False)

                # Use autoencoder's project_action method if available
                if autoencoder is not None:
                    projected_act = autoencoder.project_action(act, obs)
                else:
                    projected_act = act  # Use original action if no autoencoder

                # Clamp projected action to env bounds before stepping and log-prob computation
                projected_act = torch.clamp(projected_act, act_low, act_high)

                # Compute log-prob of the executed (projected) action under the old policy
                old_dist = policy.actor(obs)
                log_prob_exec = old_dist.log_prob(projected_act).sum(-1)

            action = projected_act.detach().squeeze() if args.task in isaac_gym_map.keys() else projected_act.detach().squeeze().cpu().numpy()
            next_obs, reward, cost, terminated, truncated, info = env.step(action)

            ep_ret += reward.cpu().numpy() if args.task in isaac_gym_map.keys() else reward
            ep_cost += cost.cpu().numpy() if args.task in isaac_gym_map.keys() else cost
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            if "final_observation" in info:
                info["final_observation"] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info["final_observation"]
                    ],
                )
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"],
                    dtype=torch.float32,
                    device=device,
                )
            # Store executed (projected and clamped) action and its log-prob
            buffer.store(
                obs=obs,
                act=projected_act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob_exec,
            )

            obs = next_obs
            epoch_end = steps >= local_steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obs[idx], deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    info["final_observation"][idx], deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        rew_deque.append(ep_ret[idx])
                        cost_deque.append(ep_cost[idx])
                        len_deque.append(ep_len[idx])
                        logger.store(
                            **{
                                "Metrics/EpRet": np.mean(rew_deque),
                                "Metrics/EpCost": np.mean(cost_deque),
                                "Metrics/EpLen": np.mean(len_deque),
                            }
                        )
                        ep_ret[idx] = 0.0
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0
                        logger.logged = False

                    buffer.finish_path(
                        last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                    )
        rollout_end_time = time.time()
        logger.log(f"Epoch {epoch}: rollout done in {rollout_end_time - rollout_start_time:.2f}s")

        eval_start_time = time.time()

        eval_episodes = 1 if epoch < epochs - 1 else 10
        if args.use_eval:
            for _ in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device)
                eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
                while not eval_done:
                    with torch.no_grad():
                        act, log_prob, value_r, value_c = policy.step(eval_obs, deterministic=True)

                        # Project evaluation action through autoencoder for safety
                        projected_act = autoencoder.project_action(act, eval_obs) if autoencoder is not None else act
                        projected_act = torch.clamp(projected_act, act_low, act_high)

                    next_obs, reward, cost, terminated, truncated, info = eval_env.step(
                        projected_act.detach().squeeze().cpu().numpy()
                    )
                    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    eval_rew += reward
                    eval_cost += cost
                    eval_len += 1
                    eval_done = terminated[0] or truncated[0]
                    eval_obs = next_obs
                eval_rew_deque.append(eval_rew)
                eval_cost_deque.append(eval_cost)
                eval_len_deque.append(eval_len)
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew_deque),
                    "Metrics/EvalEpCost": np.mean(eval_cost_deque),
                    "Metrics/EvalEpLen": np.mean(eval_len_deque),
                }
            )

        eval_end_time = time.time()
        logger.log(f"Epoch {epoch}: eval done in {eval_end_time - eval_start_time:.2f}s")

        # update lagrange multiplier
        ep_costs = logger.get_stats("Metrics/EpCost")

        # update policy
        logger.log(f"Epoch {epoch}: starting buffer.get()")
        data = buffer.get()
        logger.log(f"Epoch {epoch}: buffer.get() done; dataset_size={data['obs'].shape[0]}")
        logger.log(f"Epoch {epoch}: computing old_distribution baseline for KL")
        old_distribution = policy.actor(data["obs"])
        logger.log(f"Epoch {epoch}: old_distribution ready")

        # comnpute advantage + normalize
        advantage = data["adv_r"]
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Manual mini-batching to avoid DataLoader overhead/hangs with GPU tensors
        # batch_size = config.get("batch_size", args.steps_per_epoch//config.get("num_mini_batch", 1))
        batch_size = 128
        num_samples = data["obs"].shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        logger.log(f"Epoch {epoch}: update using manual batching with {num_batches} batches x {config['learning_iters']} iters")
        update_counts = 0
        final_kl = 0.0
        for update_iter in range(config["learning_iters"]):
            logger.log(f"Epoch {epoch}: update iter {update_iter+1}/{config['learning_iters']} starting")
            perm = torch.randperm(num_samples, device=data["obs"].device)
            for batch_idx in range(num_batches):
                idx_start = batch_idx * batch_size
                idx_end = min(idx_start + batch_size, num_samples)
                batch_idxes = perm[idx_start:idx_end]
                obs_b = data["obs"][batch_idxes]
                act_b = data["act"][batch_idxes]
                log_prob_b = data["log_prob"][batch_idxes]
                target_value_r_b = data["target_value_r"][batch_idxes]
                target_value_c_b = data["target_value_c"][batch_idxes]
                adv_b = advantage[batch_idxes]
                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                    logger.log(f"Epoch {epoch}: batch {batch_idx+1}/{num_batches} forward start")
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                distribution = policy.actor(obs_b)
                log_prob = distribution.log_prob(act_b).sum(dim=-1)
                ratio = torch.exp(log_prob - log_prob_b)
                ratio_cliped = torch.clamp(ratio, 0.8, 1.2)
                loss_pi = -torch.min(ratio * adv_b, ratio_cliped * adv_b).mean()
                # Projected-action distance penalty (encourage original actions to be close to their projections)
                proj_penalty_coef = config.get("proj_action_penalty_coef", 0.0)
                proj_penalty = torch.tensor(0.0, device=obs_b.device)
                if autoencoder is not None and proj_penalty_coef > 0.0:
                    with torch.no_grad():
                        projected_action = autoencoder.project_action(distribution.loc, obs_b)
                    proj_penalty = nn.functional.mse_loss(distribution.loc, projected_action)
                actor_optimizer.zero_grad()
                total_loss = loss_pi + 2*loss_r + loss_c \
                    if config.get("use_value_coefficient", False) \
                    else loss_pi + loss_r + loss_c
                # Add projection penalty
                total_loss = total_loss + proj_penalty_coef * proj_penalty
                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                    logger.log(f"Epoch {epoch}: batch {batch_idx+1}/{num_batches} forward ok; starting backward")
                if is_cuda:
                    torch.cuda.synchronize()
                total_loss.backward()
                if is_cuda:
                    torch.cuda.synchronize()
                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                    logger.log(f"Epoch {epoch}: batch {batch_idx+1}/{num_batches} backward ok; stepping optimizers")
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()
                actor_optimizer.step()
                if is_cuda:
                    torch.cuda.synchronize()
                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                    logger.log(f"Epoch {epoch}: batch {batch_idx+1}/{num_batches} optimizer step ok")

                # Heartbeat within update loop
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
                    logger.log(
                        f"Epoch {epoch}: update iter {update_iter+1}/{config['learning_iters']} "
                        f"batch {batch_idx+1}/{num_batches}"
                    )

                logger.store(
                    **{
                        "Loss/Loss_reward_critic": loss_r.mean().item(),
                        "Loss/Loss_cost_critic": loss_c.mean().item(),
                        "Loss/Loss_actor": loss_pi.mean().item(),
                        "Loss/Loss_proj_penalty": proj_penalty.item() if torch.is_tensor(proj_penalty) else float(proj_penalty),
                    }
                )

            new_distribution = policy.actor(data["obs"])
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            final_kl = kl
            update_counts += 1
            if kl > config["target_kl"]:
                break
        update_end_time = time.time()
        logger.log(f"Epoch {epoch}: update done in {update_end_time - eval_end_time:.2f}s (KL={final_kl:.4f})")
        actor_scheduler.step()
        if epoch == 0 or epoch == 1 or (epoch + 1) % 15 == 0:
            logger.torch_save(itr=epoch)
            if args.task not in isaac_gym_map.keys():
                logger.save_state(
                    state_dict={
                        "Normalizer": env.obs_rms,
                    },
                    itr=epoch,
                )
        if not logger.logged:
            # log data
            logger.log_tabular("Metrics/EpRet")
            logger.log_tabular("Metrics/EpCost")
            logger.log_tabular("Metrics/EpLen")
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalEpLen")
            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Train/TotalSteps", (epoch + 1) * args.steps_per_epoch)
            logger.log_tabular("Train/StopIter", update_counts)
            logger.log_tabular("Train/KL", final_kl)
            logger.log_tabular("Train/LR", actor_scheduler.get_last_lr()[0])
            logger.log_tabular("Loss/Loss_reward_critic")
            logger.log_tabular("Loss/Loss_cost_critic")
            logger.log_tabular("Loss/Loss_actor")
            logger.log_tabular("Loss/Loss_proj_penalty")
            logger.log_tabular("Time/Rollout", rollout_end_time - rollout_start_time)
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular("Time/Update", update_end_time - eval_end_time)
            logger.log_tabular("Time/Total", update_end_time - rollout_start_time)
            logger.log_tabular("Value/RewardAdv", data["adv_r"].mean().item())
            logger.log_tabular("Value/CostAdv", data["adv_c"].mean().item())

            logger.dump_tabular()
    logger.close()


if __name__ == "__main__":
    args, cfg_env = single_agent_args()
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_log_dir = os.path.join(project_root, "runs")
    args.log_dir = os.path.join(base_log_dir, args.experiment, args.task, algo, relpath)
    
    # Save terminal and error logs to files for full reproducibility
    args.write_terminal = False
    
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args, cfg_env)
    else:
        main(args, cfg_env)

