import numpy as np
import torch

from configs.configs import config
from envs import get_env_data
from utils import *
from train_abstract_model import train_abstract_model
from learn_policy import train_policy

# Optional: specific modules (in case needed for expansion)
from emerald_wm import emerald_wm
from processing import *

# Set computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Apple Silicon fallback notice
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print("MPS device not found.")

def main():
    # === 1. Generate Environment Data ===
    get_env_data()

    # === 2. Prepare and Process Data ===
    (
        data_obj, obs, obs_env, obs_next, obs_next_env,
        act_env, r_env, r, done_env, act, context
    ) = prepare_data()

    # === 3. Train Abstract (World) Model ===
    train_abstract_model(obs, act, r, obs_next, done_env, context)
    print("Training world model completed.")

    # === 4. Train Policy on Top of World Model ===
    train_policy()


if __name__ == "__main__":
    main()
 
