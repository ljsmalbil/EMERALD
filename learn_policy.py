import os
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from configs.configs import config
from wrappers import make_multi_env
from modules.worldmodel import WorldModel
from policy_eval import policy_eval_no_render


class RewardLoggerCallback(BaseCallback):  # Placeholder if custom logging is needed
    def _on_step(self) -> bool:
        return True


def evaluate_policy_on_factors(model, world_model, factors, label):
    rewards = [policy_eval_no_render(model, world_model, mass_scale=f) for f in factors]
    avg_reward = np.mean(rewards)
    print(f"{label} rewards: {avg_reward}")
    return avg_reward


def train_policy():
    # === Load world model ===
    world_model = WorldModel(config=config)
    world_model.load_state_dict(torch.load(f'models/world_model_{config.environment_id}.pth', weights_only=True))
    world_model.eval()

    # === Logging path ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = f"data/logs/{config.environment_id}_{config.policy}_{timestamp}_rewards.csv"

    # === Vectorized training environments ===
    envs = DummyVecEnv([
        make_multi_env(world_model=world_model, volume_factor=f, with_FORT=config.with_FORT)
        for f in config.training_factors
    ])

    # === Policy selection ===
    policy_cls = PPO if config.policy == 'ppo' else SAC

    if config.policy == 'ppo':
        model = PPO(
            "MlpPolicy",
            envs,
            verbose=0,
            learning_rate=config.policy_network_lr,
            gamma=config.gamma,
            n_steps=4096,
            batch_size=512,
            n_epochs=10,
            ent_coef=0.0,
            clip_range=0.2,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            vf_coef=0.5,
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                log_std_init=-1.0
            )
        )
    else:  # SAC
        model = SAC(
            "MlpPolicy",
            envs,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto_0.3",
            use_sde=True,
            policy_kwargs=dict(
                net_arch=[256, 256],
                log_std_init=-2,
                use_sde=True
            ),
            target_update_interval=1,
        )

    # === Training Loop ===
    intermed_rewards_train = []
    intermed_rewards_moderate = []
    intermed_rewards_extreme = []

    steps_per_iter = 5_000_000 #if config.policy == 'ppo' else 100_000  # adjustable
    total_iters = 1# config.policy_iterations_learn // steps_per_iter

    for i in range(total_iters):
        print(f"\n=== Iteration {i + 1}/{total_iters} ===")
        model.learn(total_timesteps=steps_per_iter, progress_bar=True)

        train_reward = evaluate_policy_on_factors(model, world_model, config.training_factors, "Train")
        moderate_reward = evaluate_policy_on_factors(model, world_model, config.moderate_factors, "Moderate")
        extreme_reward = evaluate_policy_on_factors(model, world_model, config.extreme_factors, "Extreme")

        intermed_rewards_train.append(train_reward)
        intermed_rewards_moderate.append(moderate_reward)
        intermed_rewards_extreme.append(extreme_reward)

        # === Logging ===
        rewards_df = pd.DataFrame({
            "Timestep": [i + 1],
            "Train_Rewards": [train_reward],
            "Moderate_Rewards": [moderate_reward],
            "Extreme_Rewards": [extreme_reward]
        })
        rewards_df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
        print(f"Saved rewards to {csv_path}")

    # === Final save ===
    model.save(f"models/{config.policy}_{config.environment_id}")
    print("Training completed.")
    
