import torch

from envs import *
from configs.configs import config
from utils import *
from emerald_wm import emerald_wm
from processing import *

# Set computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_abstract_model(obs, act, r, obs_next, done_env, context):
    """
    Train the world model using FORT (Factorized Observation Representation Training).
    
    Args:
        obs (np.ndarray or torch.Tensor): Observations
        act (np.ndarray or torch.Tensor): Actions
        r (np.ndarray or torch.Tensor): Rewards
        obs_next (np.ndarray or torch.Tensor): Next observations
        done_env (np.ndarray or torch.Tensor): Episode done flags
        context (Any): Additional context or prior knowledge
    """
    agent = emerald_wm(config)
    agent.train_wm(obs, act, r, obs_next, done_env, context)
