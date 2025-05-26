import os
import gc
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from gymnasium.wrappers import RecordEpisodeStatistics

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from modules.worldmodel import *
from configs.configs import config


class DirectionRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, direction=1):
        super().__init__(env)
        self.direction = direction  # 1 for forward, -1 for backward
        self.prev_x = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = self.env.unwrapped.data.qpos[0].copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        xpos = self.env.unwrapped.data.qpos[0]
        velocity = xpos - self.prev_x
        self.prev_x = xpos
        directional_reward = self.direction * velocity
        return obs, directional_reward, terminated, truncated, info

# class DirectionRewardWrapper(gym.Wrapper):
#     def __init__(self, env, direction_vector):
#         super(DirectionRewardWrapper, self).__init__(env)
#         self.direction_vector = direction_vector / np.linalg.norm(direction_vector)

#     def step(self, action):
#         obs, _, done, truncated, info = self.env.step(action)

#         # Get actual XY velocity of the torso from MuJoCo
#         velocity = self.unwrapped.data.qvel[0:2]  # But this is less reliable!


#         reward = np.dot(velocity, self.direction_vector)

#         return obs, reward, done, truncated, info

# ====================
# Action Selection
# ====================
def select_best_action_with_policy_guidance(model, world_model, x, planning_horizon, num_candidates):
    policy_action, _ = model.predict(x, deterministic=True)
    policy_action = torch.tensor(policy_action, dtype=torch.float32)

    candidate_actions = torch.stack([
        policy_action + 0.1 * torch.randn_like(policy_action)
        for _ in range(num_candidates)
    ])

    best_return = -float('inf')
    best_action = None

    for a in candidate_actions:
        x_rollout = torch.tensor(x, dtype=torch.float32)
        total_reward = 0.0
        for _ in range(planning_horizon):
            r = world_model.rho(x_rollout.view(1, -1), a.view(1, -1))
            total_reward += r.item()
            x_rollout = x_rollout + world_model.tau(x_rollout.view(1, -1), a.view(1, -1))
        if total_reward > best_return:
            best_return = total_reward
            best_action = a

    return best_action.squeeze().numpy()

# ====================
# Callbacks
# ====================
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.mean_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.mean_rewards.append(info['episode']['r'])
        return True

# ====================
# Wrappers
# ====================
class PsiObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, world_model, config, cartpole=False):#, max_buffer_size=20):
        super().__init__(env)
        self.world_model = world_model
        self.config = config
        self.cartpole = cartpole
        self.max_buffer_size = config.max_buffer_size
        self.state_action_reward_buffer = deque()
        self.context_vector = torch.zeros(1, config.context_dim)
        self.x_rec = None

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(config.latent_dim,), dtype=np.float32
        )

    def observation(self, observation):
        x = self.world_model.psi(observation, self.context_vector).detach().numpy()
        self.x_rec = x
        return x

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if len(self.state_action_reward_buffer) < self.max_buffer_size:
            self.state_action_reward_buffer.append((obs, action, reward))
            self.update_context()
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.state_action_reward_buffer.clear()
        return self.observation(obs), info

    def update_context(self):
        if len(self.state_action_reward_buffer) > 0:
            states, actions, rewards = zip(*self.state_action_reward_buffer)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            x_seq = self.world_model.psi(states, self.context_vector).detach()
            self.context_vector, _ = self.world_model.lstm(x_seq, actions, rewards, self.cartpole)
            self.context_vector = self.context_vector.detach()

class StateActionTrackingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_obs = None
        self.last_action = None

    def reset(self, **kwargs):
        self.last_obs = self.env.reset(**kwargs)
        self.last_action = None
        return self.last_obs

    def step(self, action):
        self.last_action = action
        obs, reward, done, truncated, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, done, truncated, info

class PredictedRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, world_model):
        super().__init__(env)
        self.world_model = world_model
        self.last_obs = None
        self.last_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_action = None
        return obs, info

    def step(self, action):
        self.last_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        return obs, self.reward(reward), terminated, truncated, info

    def reward(self, reward):
        if self.last_obs is None or self.last_action is None:
            return reward
        x = torch.tensor(self.last_obs, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(self.last_action, dtype=torch.float32).unsqueeze(0)
        predicted_reward = self.world_model.rho(x, a).detach().item()
        return predicted_reward



# ====================
# Utility
# ====================
def psi(state, world_model):
    return world_model.psi(state)

def e_f(state, action, reward, world_model):
    return world_model.lstm(state, action, reward)

# ====================
# Environment Factory
# ====================
def make_multi_env(world_model, volume_factor=1.0, with_FORT=True, seed=0):
    def _init():
        env_id = config.environment_id 
        env_id = "Ant" if config.environment_id in ['Ant', 'AntDir'] else env_id# else 'Ant'
        env_id = "HalfCheetah" if config.environment_id in ['HalfCheetahDir'] else env_id# else 'Ant'
        env_version = "v1" if env_id in ['Pendulum', 'CartPole'] else "v5"
        env = gym.make(f"{env_id}-{env_version}")
        env.reset(seed=seed)
        unwrapped = env.unwrapped

        # Modify environment parameters
        if env_id == 'HalfCheetah' or env_id == 'Humanoid':
            unwrapped.model.body_mass[:] *= volume_factor
            #unwrapped.model.dof_damping[:] *= volume_factor

        elif env_id == 'Pendulum':
            unwrapped.m *= volume_factor
            unwrapped.l *= volume_factor

        elif env_id == 'CartPole':
            unwrapped.length *= volume_factor

        elif config.environment_id == 'Ant':
            legs = [3, 4, 5, 6]
            unwrapped.model.body_mass[legs[:2]] *= volume_factor
            unwrapped.model.body_mass[legs[2:]] *= (1 / volume_factor)

        elif config.environment_id == 'AntDir':
            direction = volume_factor
            env = DirectionRewardWrapper(env, direction)

        elif config.environment_id == "HalfCheetahDir":
            env = DirectionRewardWrapper(env, direction)

        # Wrap with FORT components
        if with_FORT:
            if env_id in ['HalfCheetah', 'Humanoid', 'Ant']:
                env = TimeLimit(env, max_episode_steps=1000)
            elif env_id == 'CartPole':
                env = TimeLimit(env, max_episode_steps=200)

            env = PsiObservationWrapper(env, world_model=world_model, config=config, cartpole=(env_id == 'CartPole'))
            env = RecordEpisodeStatistics(env)
            # env = StateActionTrackingWrapper(env)
            # if env_id == 'Ant':
            #     env = ForceMovementWrapper(env)
            # env = PredictedRewardWrapper(env, world_model=world_model)

        return env

    return _init
