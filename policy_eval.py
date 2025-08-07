import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from collections import deque

from stable_baselines3 import PPO, SAC, TD3

from configs.configs import config
from modules.worldmodel import *
from wrappers import *


class PsiObservationWrapperEval(gym.ObservationWrapper):
    def __init__(self, env, world_model, config, cartpole=False):
        super().__init__(env)
        self.world_model = world_model
        self.config = config
        self.cartpole = cartpole
        self.use_x = config.use_x
        self.buffer = deque(maxlen=config.max_buffer_size)
        self.context_vector = torch.zeros(1, config.context_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(config.latent_dim,), dtype=np.float32
        )

    def observation(self, obs):
        x = self.world_model.psi(obs, self.context_vector).detach().numpy()
        self.x_rec = x
        return x

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.buffer.append((obs, action, reward))
        self.update_context()
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.buffer.clear()
        return self.observation(obs), info

    def update_context(self):
        if not self.buffer:
            return
        states, actions, rewards = zip(*self.buffer)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if self.use_x:
            x_seq = self.world_model.psi(states, self.context_vector).detach()
            self.context_vector = self.world_model.lstm(x_seq, actions, rewards, self.cartpole).detach()
        else:
            self.context_vector = self.world_model.lstm(states, actions, rewards, self.cartpole).detach()


def select_best_action_with_policy_guidance(model, world_model, x, planning_horizon=3, num_candidates=10):
    policy_action, _ = model.predict(x, deterministic=True)
    policy_action = torch.tensor(policy_action, dtype=torch.float32)

    candidate_actions = policy_action + 0.1 * torch.randn((num_candidates,) + policy_action.shape)

    best_action, best_return = None, -float('inf')
    x_tensor = torch.tensor(x, dtype=torch.float32)

    for a in candidate_actions:
        x_rollout = x_tensor.clone()
        total_reward = sum(
            world_model.rho(x_rollout.view(1, -1), a.view(1, -1)).item()
            for _ in range(planning_horizon)
        )
        x_rollout += world_model.tau(x_rollout.view(1, -1), a.view(1, -1)).squeeze(0)

        if total_reward > best_return:
            best_return = total_reward
            best_action = a

    return best_action.numpy()


class DirectionRewardWrapper(gym.Wrapper):
    def __init__(self, env, direction_vector):
        super().__init__(env)
        self.direction_vector = direction_vector / np.linalg.norm(direction_vector)

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        velocity = self.unwrapped.data.qvel[0:2]
        reward = np.dot(velocity, self.direction_vector)
        return obs, reward, done, truncated, info


def load_env_and_adjust_masses(render=False, mass_scale=1):
    env_id = config.environment_id
    version = '-v1' if env_id in ['Pendulum', 'CartPole'] else '-v5'
    env = gym.make(f'{env_id}{version}', render_mode="human" if render else None)
    unwrapped_env = env.unwrapped
    cartpole = False

    if env_id == 'Ant':
        indices = [3, 4, 5, 6]
        unwrapped_env.model.body_mass[indices[:2]] *= mass_scale
        unwrapped_env.model.body_mass[indices[2:]] /= mass_scale
    elif env_id == 'AntDir':
        env = DirectionRewardWrapper(env, mass_scale)
    elif env_id in ['HalfCheetah', 'Humanoid']:
        unwrapped_env.model.body_mass[:] *= mass_scale
    elif env_id == 'Pendulum':
        unwrapped_env.max_torque = 2.0
        unwrapped_env.m *= mass_scale
        unwrapped_env.l *= mass_scale
    elif env_id == 'CartPole':
        unwrapped_env.length *= mass_scale
        cartpole = True

    return env, cartpole


def load_policy(policy_path=None, finetune=False):
    path = policy_path if finetune else f"models/{config.policy}_{config.environment_id}"
    if config.policy == 'ppo':
        return PPO.load(path)
    elif config.policy == 'sac':
        return SAC.load(path)
    elif config.policy == 'td3':
        return TD3.load(path)
    raise ValueError(f"Unsupported policy: {config.policy}")


def policy_eval(mass_scale=1, finetune=False, policy_path=None):
    world_model = WorldModel(config=config)
    world_model.load_state_dict(torch.load(f'models/world_model_{config.environment_id}.pth', weights_only=True))
    world_model.eval()

    env, _ = load_env_and_adjust_masses(render=True, mass_scale=mass_scale)
    if config.with_FORT:
        env = PsiObservationWrapper(env, world_model, config=config, max_buffer_size=config.max_buffer_size)

    model = load_policy(policy_path, finetune)

    episode_rewards, saved_obs = [], []
    for _ in range(10):
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(config.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            saved_obs.append(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            env.render()
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)

    env.close()
    pd.DataFrame(saved_obs).to_csv(f'data/x_test_{mass_scale}_{config.policy}_{config.environment_id}.csv', index=False)
    return np.mean(episode_rewards)


def policy_eval_no_render(model, world_model, mass_scale=1, policy_guidance_enabled=False):
    env, cartpole = load_env_and_adjust_masses(render=False, mass_scale=mass_scale)
    if config.with_FORT:
        env = PsiObservationWrapper(env, world_model=world_model, config=config, cartpole=cartpole)

    episode_rewards, saved_obs = [], []

    for _ in range(3):
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(config.max_episode_steps):
            if policy_guidance_enabled:
                action = select_best_action_with_policy_guidance(model, world_model, obs)
            else:
                action, _ = model.predict(obs, deterministic=True)

            step_action = int(action) if config.environment_id == 'CartPole' else action.flatten()
            obs, reward, terminated, truncated, _ = env.step(step_action)
            saved_obs.append(obs[0] if isinstance(obs, np.ndarray) else obs)
            total_reward += reward
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)

    env.close()
    pd.DataFrame(saved_obs).to_csv(f'data/x_test_{mass_scale}_{config.policy}_{config.environment_id}.csv', index=False)
    return np.mean(episode_rewards)
