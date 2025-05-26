import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from configs.configs import config


class DirectionRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, direction=1):
        super().__init__(env)
        self.direction = direction
        self.prev_x = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = self.env.unwrapped.data.qpos[0].copy()
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        xpos = self.env.unwrapped.data.qpos[0]
        velocity = xpos - self.prev_x
        self.prev_x = xpos
        reward = self.direction * velocity
        return obs, reward, terminated, truncated, info


def _save_data(data, filename, state, next_state, action):
    state_cols = [f"state_{i}" for i in range(len(state))]
    next_state_cols = [f"next_state_{i}" for i in range(len(next_state))]
    action_cols = [f"action_{i}" for i in range(len(action))]
    columns = state_cols + action_cols + next_state_cols + ["reward", "terminal", "context", "episode"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Data collection complete. Saved to '{filename}'")


def generate_halfcheetah_data_dir_for_environments(samples=100000, samples_per_ep=1000):
    directions = [1, -1]
    data, episode = [], 0

    for context, direction in enumerate(directions):
        env = gym.make('HalfCheetah-v5')
        env = TimeLimit(env, max_episode_steps=1000)
        env = DirectionRewardWrapper(env, direction=direction)
        env.reset(seed=context)

        state, _ = env.reset()
        tracker = 0

        for _ in range(samples):
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            data.append(state.tolist() + action.tolist() + next_state.tolist() + [reward, done, context, episode])
            state = next_state

            tracker += 1
            if tracker > samples_per_ep:
                tracker = 0
                state, _ = env.reset()
                episode += 1

        episode += 1

    _save_data(data, 'data/HalfCheetahDir_data.csv', state, next_state, action)


def generate_half_cheetah_data_for_environments(modifications=[0.75, 0.85, 1, 1.15, 1.25], samples=100000):
    data, episode = [], 0
    samples_per_ep = 1000

    for context, scale in enumerate(modifications):
        env = gym.make('HalfCheetah-v5')
        unwrapped = env.unwrapped
        unwrapped.model.body_mass *= scale
        state, _ = unwrapped.reset()
        tracker = 0

        for _ in range(samples):
            action = unwrapped.action_space.sample()
            next_state, reward, done, _, _ = unwrapped.step(action)
            data.append(state.tolist() + action.tolist() + next_state.tolist() + [reward, done, context, episode])
            state = next_state

            tracker += 1
            if tracker > samples_per_ep:
                tracker = 0
                state, _ = unwrapped.reset()
                episode += 1

        episode += 1

    _save_data(data, 'data/HalfCheetah_data.csv', state, next_state, action)


def generate_ant_data_for_environments(modifications=[0.85, 0.90, 0.95, 1.0], samples=10000):
    data, episode = [], 0
    samples_per_ep = 2000
    leg_indices = [3, 4, 5, 6]

    for context, scale in enumerate(modifications):
        env = gym.make('Ant-v5')
        unwrapped = env.unwrapped
        unwrapped.model.body_mass[leg_indices[:2]] *= scale
        unwrapped.model.body_mass[leg_indices[2:]] /= scale

        state, _ = unwrapped.reset()
        tracker = 0

        for _ in range(samples):
            action = unwrapped.action_space.sample()
            next_state, reward, done, _, _ = unwrapped.step(action)
            data.append(state.tolist() + action.tolist() + next_state.tolist() + [reward, done, context, episode])
            state = next_state

            tracker += 1
            if tracker > samples_per_ep:
                tracker = 0
                state, _ = unwrapped.reset()
                episode += 1

        episode += 1

    _save_data(data, 'data/Ant_data.csv', state, next_state, action)


def generate_slim_humanoid_data_for_environments(modifications=[0.80, 0.90, 1.0, 1.15, 1.25], samples=100000):
    data, episode = [], 0
    samples_per_ep, context = 2000, 0

    for scale in modifications:
        env = gym.make('Humanoid-v5')
        unwrapped = env.unwrapped
        unwrapped.model.body_mass *= scale
        unwrapped.model.dof_damping *= scale

        state, _ = unwrapped.reset()
        tracker = 0

        for _ in range(samples):
            action = unwrapped.action_space.sample()
            next_state, reward, done, _, _ = unwrapped.step(action)
            data.append(state.tolist() + action.tolist() + next_state.tolist() + [reward, done, context, episode])
            state = next_state

            tracker += 1
            if tracker > samples_per_ep:
                tracker = 0
                state, _ = unwrapped.reset()
                episode += 1

        episode += 1
        context += 1

    _save_data(data, 'data/Humanoid_data.csv', state, next_state, action)


def generate_pendulum_data_for_environments(mass_modifications=None, length_modifications=None, samples=100000):
    data, episode = [], 0
    samples_per_ep = 2000

    for context, (m_scale, l_scale) in enumerate(zip(mass_modifications, length_modifications)):
        env = gym.make('Pendulum-v1')
        unwrapped = env.unwrapped
        unwrapped.m *= m_scale
        unwrapped.l *= l_scale

        state, _ = unwrapped.reset()
        tracker = 0

        for _ in range(samples):
            action = unwrapped.action_space.sample()
            next_state, reward, done, _, _ = unwrapped.step(action)
            data.append(state.tolist() + action.tolist() + next_state.tolist() + [reward, done, context, episode])
            state = next_state

            tracker += 1
            if tracker > samples_per_ep:
                tracker = 0
                state, _ = unwrapped.reset()
                episode += 1

        episode += 1

    _save_data(data, 'data/Pendulum_data.csv', state, next_state, action)


def generate_cartpole_data(lengths=config.training_factors, samples=100000):
    data, episode = [], 0
    samples_per_ep = 200

    for context, length_factor in enumerate(lengths):
        env = gym.make('CartPole-v1')
        unwrapped = env.unwrapped
        unwrapped.length *= length_factor

        state, _ = unwrapped.reset()
        tracker = 0

        for _ in range(samples):
            action = unwrapped.action_space.sample()
            next_state, reward, done, _, _ = unwrapped.step(action)
            data.append(state.tolist() + [action] + next_state.tolist() + [reward, done, context, episode])
            state = next_state

            tracker += 1
            if tracker > samples_per_ep:
                tracker = 0
                state, _ = unwrapped.reset()
                episode += 1

        episode += 1

    _save_data(data, 'data/CartPole_data.csv', state, next_state, [action])


def generate_frozenlake_data(samples=100000, size=4):
    map1 = generate_random_map(size)
    map2 = generate_random_map(size)
    while map1 == map2:
        map2 = generate_random_map(size)

    envs = [
        gym.make("FrozenLake-v1", desc=map1, is_slippery=False),
        gym.make("FrozenLake-v1", desc=map2, is_slippery=False)
    ]

    data, episode = [], 0
    samples_per_ep = 100

    for context, env in enumerate(envs):
        state, _ = env.reset()
        tracker = 0

        for _ in range(samples // len(envs)):
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            data.append([state, action, next_state, reward, done, context, episode])
            state = next_state

            tracker += 1
            if done or tracker >= samples_per_ep:
                tracker = 0
                state, _ = env.reset()
                episode += 1

        episode += 1

    columns = ["state", "action", "next_state", "reward", "terminal", "context", "episode"]
    pd.DataFrame(data, columns=columns).to_csv("data/FrozenLake_data.csv", index=False)
    print("Data collection complete. Saved to 'data/FrozenLake_data.csv'")


def generate_ant_data_dir_for_environments(samples=30000, samples_per_ep=1000, num_envs=5):
    angles = np.linspace(0, 2 * np.pi, num_envs, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    data, episode = [], 0

    for context, direction in enumerate(directions):
        env = gym.make('Ant-v5')
        env = TimeLimit(env, max_episode_steps=1000)
        env = DirectionRewardWrapper(env, direction)
        env.reset(seed=context)

        state, _ = env.reset()
        tracker = 0

        for _ in range(samples):
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            data.append(state.tolist() + action.tolist() + next_state.tolist() + [reward, done, context, episode])
            state = next_state

            tracker += 1
            if tracker > samples_per_ep:
                tracker = 0
                state, _ = env.reset()
                episode += 1

        episode += 1

    _save_data(data, 'data/AntDir_data.csv', state, next_state, action)
    np.save("data/Ant_direction_vectors.npy", directions)
    print("Direction vectors saved to 'data/Ant_direction_vectors.npy'")


def get_env_data():
    if config.environment_id == "HalfCheetah":
        generate_half_cheetah_data_for_environments(modifications=config.training_factors, samples=config.samples_per_env)
    elif config.environment_id == "Pendulum":
        generate_pendulum_data_for_environments(mass_modifications=config.training_factors, length_modifications=config.training_factors)
    elif config.environment_id == "Ant":
        generate_ant_data_for_environments()
    elif config.environment_id == "Humanoid":
        generate_slim_humanoid_data_for_environments()
    elif config.environment_id == "CartPole":
        generate_cartpole_data()
    elif config.environment_id == "AntDir":
        generate_ant_data_dir_for_environments()
    elif config.environment_id == "HalfCheetahDir":
        generate_halfcheetah_data_dir_for_environments()
    else:
        print("Not yet implemented")


if __name__ == "__main__":
    get_env_data()
