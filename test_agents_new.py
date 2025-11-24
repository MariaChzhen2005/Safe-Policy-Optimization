#!/usr/bin/env python3

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import joblib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import imageio
except ImportError:
    imageio = None

# Add safepo to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'safepo'))

from safepo.common.env import make_sa_mujoco_env
from safepo.common.model import ActorVCritic
from safepo.single_agent.autoencoder import ConditionalConstraintAwareAutoencoder


class AgentTester:
    def __init__(self, model_dirs: Dict[str, str], env_name: str = "SafetyPointGoal2-v0", num_episodes: int = 10):
        """
        Initialize the agent tester.

        Args:
            model_dirs: Dictionary mapping algorithm names to their model directories
            env_name: Name of the environment to test on
            num_episodes: Number of episodes to run per seed
        """
        self.model_dirs = model_dirs
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)

        # Initialize environment to get spaces
        self.eval_env, self.obs_space, self.act_space = make_sa_mujoco_env(
            num_envs=1, env_id=env_name, seed=None
        )

        # Results storage
        self.results = defaultdict(lambda: defaultdict(list))
        self.episode_data = defaultdict(lambda: defaultdict(list))

    def load_model(self, model_path: str, algo: str) -> Tuple[Optional[ActorVCritic], Optional[object], Optional[ConditionalConstraintAwareAutoencoder]]:
        """Load a trained model from the given directory or file path.

        For 'ppo_ae', also load an autoencoder to project actions at inference.
        """
        try:
            # Check if it's a direct file path or directory
            if model_path.endswith('.pt'):
                # Direct model file path
                if not os.path.exists(model_path):
                    print(f"Model file not found at {model_path}")
                    return None, None, None

                model_dir = os.path.dirname(model_path)
                actual_model_path = model_path
                print(f"Loading model directly from file: {actual_model_path}")
            else:
                # Directory path - look for config and models
                model_dir = model_path
                config_path = os.path.join(model_dir, 'config.json')
                if not os.path.exists(config_path):
                    print(f"Config not found at {config_path}")
                    return None, None, None

                # Find the latest model
                torch_save_dir = os.path.join(model_dir, 'torch_save')
                if not os.path.exists(torch_save_dir):
                    print(f"Torch save directory not found at {torch_save_dir}")
                    return None, None, None

                models = [f for f in os.listdir(torch_save_dir) if f.endswith('.pt')]
                if not models:
                    print(f"No models found in {torch_save_dir}")
                    return None, None, None

                latest_model = sorted(models)[-1]
                actual_model_path = os.path.join(torch_save_dir, latest_model)
                print(f"Loading latest model from directory: {actual_model_path}")

            # Try to load config for model architecture
            config = {'hidden_sizes': [64, 64]}
            config_path = os.path.join(model_dir, 'config.json')
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        loaded_config = json.load(f)
                        config.update(loaded_config)
                except Exception as e:
                    print(f"Could not load config, using defaults: {e}")

            # Load model
            model = ActorVCritic(
                obs_dim=self.obs_space.shape[0],
                act_dim=self.act_space.shape[0],
                hidden_sizes=config.get('hidden_sizes', [64, 64])
            )

            # Load the state dict - handle both full model and actor-only saves
            try:
                checkpoint = torch.load(actual_model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif hasattr(checkpoint, 'state_dict'):
                    model.actor = checkpoint
                else:
                    model.actor.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Error loading model weights: {e}")
                # Try loading as full model
                try:
                    full_model = torch.load(actual_model_path, map_location=self.device)
                    if hasattr(full_model, 'actor'):
                        model = full_model
                    else:
                        model.actor = full_model
                except Exception as e2:
                    print(f"Could not load model: {e2}")
                    return None, None, None

            model.eval()

            # Load normalizer if available
            normalizer = None
            pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            if pkl_files:
                latest_pkl = sorted(pkl_files)[-1]
                pkl_path = os.path.join(model_dir, latest_pkl)
                try:
                    norm_data = joblib.load(pkl_path)
                    if 'Normalizer' in norm_data:
                        normalizer = norm_data['Normalizer']
                        print(f"Loaded normalizer from {pkl_path}")
                except Exception as e:
                    print(f"Could not load normalizer from {pkl_path}: {e}")

            # Load autoencoder for PPO_AE inference-time projection
            autoencoder = None
            if algo == 'ppo_ae':
                try:
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    data_dir = os.path.join(base_dir, 'safepo', 'single_agent', 'data')
                    env_token_map = {
                        'SafetyPointGoal2-v0': 'goal',
                        'SafetyPointPush2-v0': 'push',
                        'SafetyPointButton2-v0': 'button',
                    }
                    token = env_token_map.get(self.env_name, None)
                    candidate_filenames = []
                    if token:
                        candidate_filenames.append(f"conditional_phase2_safety_gym_{token}_1_decoders_2_2_absolute_Adam.pt")
                        if token and os.path.isdir(data_dir):
                            for fname in sorted(os.listdir(data_dir)):
                                if fname.endswith('.pt') and ('safety_gym' in fname) and (token in fname) and (fname not in candidate_filenames):
                                    candidate_filenames.append(fname)
                except Exception:
                    pass
                # candidate_filenames.append("conditional_phase2_safety_gym_1_decoders_2_2_absolute_Adam.pt")

                    candidate_paths = []
                    for fname in candidate_filenames:
                        candidate_paths.append(os.path.join(data_dir, fname))
                        candidate_paths.append(os.path.join(os.getcwd(), "safepo", "single_agent", "data", fname))
                        candidate_paths.append(os.path.join("safepo", "single_agent", "data", fname))

                    for autoencoder_path in candidate_paths:
                        if os.path.exists(autoencoder_path):
                            try:
                                autoencoder = ConditionalConstraintAwareAutoencoder(
                                    action_dim=self.act_space.shape[0],
                                    state_dim=self.obs_space.shape[0],
                                    latent_dim=self.act_space.shape[0],
                                    hidden_dim=64,
                                    num_decoders=1,
                                    latent_geom="hypersphere",
                                    norm_params_path=None,
                                    ieee37_model_instance_in=None,
                                )
                                autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device))
                                autoencoder.eval()
                                print(f"Loaded autoencoder for PPO_AE inference from {autoencoder_path}")
                                break
                            except Exception as e:
                                print(f"Could not load autoencoder from {autoencoder_path}: {e}")
                                autoencoder = None
                                continue
                    if autoencoder is None:
                        print("Warning: Could not load an environment-specific autoencoder for PPO_AE - actions won't be projected")
                except Exception as e:
                    print(f"Autoencoder selection failed: {e}")

            print(f"Successfully loaded {algo} model from {actual_model_path}")
            return model, normalizer, autoencoder

        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def run_episode(self, model: ActorVCritic, normalizer: Optional[object], autoencoder: Optional[ConditionalConstraintAwareAutoencoder], seed: int) -> Dict:
        """Run a single episode and return metrics."""
        # Reset environment with seed
        env, _, _ = make_sa_mujoco_env(num_envs=1, env_id=self.env_name, seed=seed)
        if normalizer is not None:
            env.obs_rms = normalizer

        obs, _ = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        episode_reward = 0.0
        episode_cost = 0.0
        episode_length = 0
        done = False

        rewards = []
        costs = []
        actions = []
        observations = []

        while not done:
            with torch.no_grad():
                # Get action from policy
                act, log_prob, value_r, value_c = model.step(obs, deterministic=True)

                # Apply autoencoder projection if available (for PPO inference)
                if autoencoder is not None:
                    projected_act = autoencoder.project_action(act, obs)
                    final_act = projected_act
                else:
                    final_act = act

                # Store data
                observations.append(obs.cpu().numpy())
                actions.append(final_act.cpu().numpy())

            # Step environment
            next_obs, reward, cost, terminated, truncated, info = env.step(
                final_act.detach().squeeze().cpu().numpy()
            )

            episode_reward += reward[0]
            episode_cost += cost[0]
            episode_length += 1

            rewards.append(reward[0])
            costs.append(cost[0])

            done = terminated[0] or truncated[0]
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

        env.close()

        return {
            'episode_reward': episode_reward,
            'episode_cost': episode_cost,
            'episode_length': episode_length,
            'rewards': rewards,
            'costs': costs,
            'actions': actions,
            'observations': observations,
        }

    def test_algorithm(self, algo: str, model_path: str, seeds: List[int]) -> Dict:
        """Test an algorithm across multiple seeds."""
        print(f"\n=== Testing {algo.upper()} ===")

        # Load model
        model, normalizer, autoencoder = self.load_model(model_path, algo)
        if model is None:
            print(f"Failed to load model for {algo}")
            return {}

        algo_results = {
            'episode_rewards': [],
            'episode_costs': [],
            'episode_lengths': [],
            'all_episode_data': [],
        }

        print(f"Running {len(seeds)} seeds with {self.num_episodes} episodes each...")

        for i, seed in enumerate(seeds):
            print(f"Seed {seed} ({i+1}/{len(seeds)})", end=' ')

            seed_rewards = []
            seed_costs = []
            seed_lengths = []
            seed_episodes = []

            for ep in range(self.num_episodes):
                episode_data = self.run_episode(model, normalizer, autoencoder, seed + ep)

                seed_rewards.append(episode_data['episode_reward'])
                seed_costs.append(episode_data['episode_cost'])
                seed_lengths.append(episode_data['episode_length'])
                seed_episodes.append(episode_data)

                if (ep + 1) % 5 == 0:
                    print(f".", end='')

            # Store seed results
            algo_results['episode_rewards'].extend(seed_rewards)
            algo_results['episode_costs'].extend(seed_costs)
            algo_results['episode_lengths'].extend(seed_lengths)
            algo_results['all_episode_data'].extend(seed_episodes)

            print(f" Avg Reward: {np.mean(seed_rewards):.2f}, Avg Cost: {np.mean(seed_costs):.2f}")

        # Compute statistics
        rewards = np.array(algo_results['episode_rewards'])
        costs = np.array(algo_results['episode_costs'])
        lengths = np.array(algo_results['episode_lengths'])

        algo_results['reward_mean'] = np.mean(rewards)
        algo_results['reward_std'] = np.std(rewards)
        algo_results['cost_mean'] = np.mean(costs)
        algo_results['cost_std'] = np.std(costs)
        algo_results['length_mean'] = np.mean(lengths)
        algo_results['length_std'] = np.std(lengths)

        # Find best episode by reward
        best_idx = np.argmax(rewards)
        algo_results['best_episode'] = algo_results['all_episode_data'][best_idx]
        algo_results['best_reward'] = rewards[best_idx]

        print(f"Results for {algo.upper()}:")
        print(f"  Reward: {algo_results['reward_mean']:.2f} ± {algo_results['reward_std']:.2f}")
        print(f"  Cost: {algo_results['cost_mean']:.2f} ± {algo_results['cost_std']:.2f}")
        print(f"  Length: {algo_results['length_mean']:.2f} ± {algo_results['length_std']:.2f}")
        print(f"  Best Episode Reward: {algo_results['best_reward']:.2f}")

        return algo_results

    def run_all_tests(self, seeds: List[int]) -> Dict:
        """Run tests for all algorithms."""
        print(f"Testing {len(self.model_dirs)} algorithms on {self.env_name}")
        print(f"Seeds: {len(seeds)}, Episodes per seed: {self.num_episodes}")

        all_results = {}

        for algo, model_path in self.model_dirs.items():
            # Check if path exists (either file or directory)
            path_exists = os.path.exists(model_path) or (not model_path.endswith('.pt') and os.path.exists(model_path))
            if path_exists:
                all_results[algo] = self.test_algorithm(algo, model_path, seeds)
            else:
                print(f"Model path not found for {algo}: {model_path}")

        return all_results

    def create_mujoco_video(self, algo: str, episode_data: Dict, output_path: str):
        """Create visualization for the best episode."""
        print(f"Creating visualization for {algo} best episode...")

        try:
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Create comprehensive episode analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"{algo.upper()} - Best Episode Analysis", fontsize=16, fontweight='bold')

            # Plot 1: Reward and Cost over time
            rewards = episode_data.get('rewards', [])
            costs = episode_data.get('costs', [])
            steps = range(len(rewards)) if rewards else []

            if rewards:
                axes[0, 0].plot(steps, np.cumsum(rewards), 'g-', linewidth=2, label='Cumulative Reward')
                axes[0, 0].plot(steps, rewards, 'g--', alpha=0.6, label='Step Reward')
                if costs:
                    axes[0, 0].plot(steps, np.cumsum(costs), 'r-', linewidth=2, label='Cumulative Cost')
                    axes[0, 0].plot(steps, costs, 'r--', alpha=0.6, label='Step Cost')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Value')
                axes[0, 0].set_title('Rewards and Costs Over Time')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Action analysis
            actions = episode_data.get('actions', [])
            if actions and len(actions) > 0:
                actions_array = np.array(actions).squeeze()
                if actions_array.ndim > 1:
                    for i in range(min(4, actions_array.shape[-1])):
                        axes[0, 1].plot(actions_array[:, i], label=f'Action {i+1}', alpha=0.7)
                else:
                    axes[0, 1].plot(actions_array, 'b-', linewidth=2)
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Action Value')
                axes[0, 1].set_title('Action Trajectory')
                if actions_array.ndim > 1:
                    axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Episode Statistics
            axes[1, 0].text(0.1, 0.8, f"Total Reward: {episode_data['episode_reward']:.3f}",
                           fontsize=14, transform=axes[1, 0].transAxes)
            axes[1, 0].text(0.1, 0.7, f"Total Cost: {episode_data['episode_cost']:.3f}",
                           fontsize=14, transform=axes[1, 0].transAxes)
            axes[1, 0].text(0.1, 0.6, f"Episode Length: {episode_data['episode_length']}",
                           fontsize=14, transform=axes[1, 0].transAxes)
            if rewards:
                axes[1, 0].text(0.1, 0.5, f"Avg Step Reward: {np.mean(rewards):.3f}",
                               fontsize=14, transform=axes[1, 0].transAxes)
                axes[1, 0].text(0.1, 0.4, f"Max Step Reward: {np.max(rewards):.3f}",
                               fontsize=14, transform=axes[1, 0].transAxes)
            if costs:
                axes[1, 0].text(0.1, 0.3, f"Avg Step Cost: {np.mean(costs):.3f}",
                               fontsize=14, transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Episode Statistics')
            axes[1, 0].axis('off')

            # Plot 4: Reward/Cost distribution histogram
            if rewards:
                axes[1, 1].hist(rewards, bins=20, alpha=0.7, color='green', label='Rewards', density=True)
            if costs:
                axes[1, 1].hist(costs, bins=20, alpha=0.7, color='red', label='Costs', density=True)
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Step Reward/Cost Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # Save the analysis plot
            analysis_file = f"{output_path}_analysis.png"
            plt.tight_layout()
            plt.savefig(analysis_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved episode analysis for {algo}: {analysis_file}")

            # Create a simpler summary plot as well
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.8, f"{algo.upper()} - Best Episode",
                   fontsize=24, ha='center', va='center', weight='bold')
            ax.text(0.5, 0.6, f"Reward: {episode_data['episode_reward']:.3f}",
                   fontsize=18, ha='center', va='center', color='green')
            ax.text(0.5, 0.5, f"Cost: {episode_data['episode_cost']:.3f}",
                   fontsize=18, ha='center', va='center', color='red')
            ax.text(0.5, 0.4, f"Length: {episode_data['episode_length']} steps",
                   fontsize=18, ha='center', va='center')

            # Safety rating
            if episode_data['episode_cost'] == 0:
                safety_rating = "Perfect Safety"
                safety_color = "green"
            elif episode_data['episode_cost'] < 1:
                safety_rating = "Good Safety"
                safety_color = "orange"
            else:
                safety_rating = "Poor Safety"
                safety_color = "red"

            ax.text(0.5, 0.2, f"Safety Rating: {safety_rating}",
                   fontsize=16, ha='center', va='center', color=safety_color, style='italic')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            # Save summary plot
            summary_file = f"{output_path}_summary.png"
            plt.savefig(summary_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved episode summary for {algo}: {summary_file}")

            # Create a basic analysis video from images
            if imageio is not None:
                try:
                    video_frames = []
                    analysis_img = plt.imread(analysis_file)
                    summary_img = plt.imread(summary_file)
                    for _ in range(30):
                        video_frames.append(summary_img)
                    for _ in range(90):
                        video_frames.append(analysis_img)
                    video_file = f"{output_path}.mp4"
                    imageio.mimsave(video_file, video_frames, fps=30)
                    print(f"Created analysis video: {video_file}")
                except Exception as e:
                    print(f"Could not create video: {e}")

        except Exception as e:
            print(f"Error creating visualization for {algo}: {e}")
            import traceback
            traceback.print_exc()

    def save_results(self, results: Dict, output_file: str):
        """Save all results to a text file."""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RL AGENT TESTING RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Environment: {self.env_name}\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Seeds: {len([s for r in results.values() if r for s in range(len(r.get('episode_rewards', [])) // self.num_episodes)])}\n")
            f.write(f"Episodes per Seed: {self.num_episodes}\n")
            f.write("\n")

            # Summary table
            f.write("SUMMARY RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Algorithm':<15} {'Reward Mean':<12} {'Reward Std':<12} {'Cost Mean':<12} {'Cost Std':<12} {'Length Mean':<12}\n")
            f.write("-" * 80 + "\n")

            for algo, result in results.items():
                if result:
                    f.write(f"{algo.upper():<15} {result['reward_mean']:<12.2f} {result['reward_std']:<12.2f} "
                           f"{result['cost_mean']:<12.2f} {result['cost_std']:<12.2f} {result['length_mean']:<12.2f}\n")

            f.write("\n")

            # Detailed results for each algorithm
            for algo, result in results.items():
                if not result:
                    continue

                f.write(f"\nDETAILED RESULTS - {algo.upper()}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total Episodes: {len(result['episode_rewards'])}\n")
                f.write(f"Reward: {result['reward_mean']:.3f} ± {result['reward_std']:.3f}\n")
                f.write(f"Cost: {result['cost_mean']:.3f} ± {result['cost_std']:.3f}\n")
                f.write(f"Episode Length: {result['length_mean']:.3f} ± {result['length_std']:.3f}\n")
                f.write(f"Best Episode Reward: {result['best_reward']:.3f}\n")

                # Reward distribution
                rewards = np.array(result['episode_rewards'])
                f.write(f"Reward Min: {np.min(rewards):.3f}\n")
                f.write(f"Reward Max: {np.max(rewards):.3f}\n")
                f.write(f"Reward Median: {np.median(rewards):.3f}\n")
                f.write(f"Reward 25th percentile: {np.percentile(rewards, 25):.3f}\n")
                f.write(f"Reward 75th percentile: {np.percentile(rewards, 75):.3f}\n")

                # Cost distribution
                costs = np.array(result['episode_costs'])
                f.write(f"Cost Min: {np.min(costs):.3f}\n")
                f.write(f"Cost Max: {np.max(costs):.3f}\n")
                f.write(f"Cost Median: {np.median(costs):.3f}\n")

                # Best episode details
                best_ep = result['best_episode']
                f.write(f"\nBest Episode Details:\n")
                f.write(f"  Reward: {best_ep['episode_reward']:.3f}\n")
                f.write(f"  Cost: {best_ep['episode_cost']:.3f}\n")
                f.write(f"  Length: {best_ep['episode_length']}\n")
                f.write(f"  Average reward per step: {best_ep['episode_reward']/best_ep['episode_length']:.3f}\n")

                f.write("\n")

            # Save individual episode data
            f.write("\nRAW EPISODE DATA\n")
            f.write("-" * 50 + "\n")
            for algo, result in results.items():
                if not result:
                    continue
                f.write(f"\n{algo.upper()} - All Episode Rewards:\n")
                rewards = result['episode_rewards']
                for i, reward in enumerate(rewards):
                    f.write(f"{reward:.3f}")
                    if (i + 1) % 10 == 0:
                        f.write("\n")
                    else:
                        f.write(", ")
                f.write("\n")

                f.write(f"\n{algo.upper()} - All Episode Costs:\n")
                costs = result['episode_costs']
                for i, cost in enumerate(costs):
                    f.write(f"{cost:.3f}")
                    if (i + 1) % 10 == 0:
                        f.write("\n")
                    else:
                        f.write(", ")
                f.write("\n")

        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test trained RL agents")
    parser.add_argument("--model-dirs", type=str, default="example_model_dirs_goal.json",
                       help="JSON file containing algorithm to model directory mapping")
    parser.add_argument("--env", type=str, default="SafetyPointGoal2-v0",
                       help="Environment to test on")
    parser.add_argument("--num-seeds", type=int, default=50,
                       help="Number of seeds to test")
    parser.add_argument("--episodes-per-seed", type=int, default=1,
                       help="Number of episodes per seed")
    parser.add_argument("--output-dir", type=str, default="./test_results",
                       help="Output directory for results")
    parser.add_argument("--create-videos", action="store_true",
                       help="Create simulation videos for best episodes")

    args = parser.parse_args()

    # Load model directories
    if not os.path.exists(args.model_dirs):
        print(f"Model directories file not found: {args.model_dirs}")
        return

    with open(args.model_dirs, 'r') as f:
        model_dirs = json.load(f)

    # Generate seeds
    seeds = list(range(args.num_seeds))

    # Create output directory, organized by environment
    env_output_dir = os.path.join(args.output_dir, args.env)
    os.makedirs(env_output_dir, exist_ok=True)

    # Initialize tester
    tester = AgentTester(model_dirs, args.env, args.episodes_per_seed)

    # Run tests
    print("Starting agent testing...")
    start_time = time.time()

    results = tester.run_all_tests(seeds)

    end_time = time.time()
    print(f"\nTesting completed in {end_time - start_time:.2f} seconds")

    # Save results
    output_file = os.path.join(env_output_dir, f"agent_test_results_{args.env}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    tester.save_results(results, output_file)

    # Create videos for best episodes
    if args.create_videos:
        print("\nCreating simulation frames for best episodes...")
        for algo, result in results.items():
            if result:
                video_path = os.path.join(env_output_dir, f"best_episode_{algo}")
                tester.create_mujoco_video(algo, result['best_episode'], video_path)

    print("\nAll tests completed!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()


