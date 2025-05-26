import pandas as pd
from configs.configs import *

from utils import *

def prepare_data():
    # Obtai# obtain data
    data = pd.read_csv(f"data/{config.environment_id}_data.csv")

    # Shuffle the data
    #data = data.sample(frac=1).reset_index(drop=True)

    print("Processing data...")

    if config.int_state:
        data = data[['state', 'action', 'reward', 'next_state', 'context', 'terminal']]

        num_classes_s = len(data['state'].unique())
        num_classes_s_prime = len(data['next_state'].unique())
        num_clases = np.max([num_classes_s, num_classes_s_prime])      # ensure that all states are accounted for
        num_actions = config.act_dim

        # set data object
        data_obj = DataProcessor(data, n_envs=config.n_envs)                       # set data object
        obs = data_obj._s_current_dummy_all(num_clases)                  # get all s in one hot
        obs_next = data_obj._s_next_dummy_all(num_clases)                # get all s next in one hot
        act = data_obj._a_current_dummy_all()                            # get all a in one hot
        r = data_obj._r()
        done_env = data_obj._done_e()                                  # get all env dones in one hot

        return data_obj, obs, obs_env, obs_next, obs_next_env, act_env, r_env, r, done_env, act

    elif config.cont_act:
        data_obj = DataProcessor(data, n_envs=config.n_envs)                       # set data object
        obs = data_obj._obs_to_tensor(n_cols=config.obs_shape)

        obs_env = data_obj._obs_to_tensor_e(n_cols=config.obs_shape)
        obs_next = data_obj._obs_next_to_tensor(n_cols=config.obs_shape)
        obs_next_env = data_obj._obs_next_to_tensor_e(n_cols=config.obs_shape)
        act_env = data_obj._acts_to_tensor_e(n_cols=config.act_dim)
        r_env = data_obj._r_e()    
        r = data_obj._r()
        context = data_obj._get_context()

        done_env = data_obj._done_e()            
        act = data_obj._acts_to_tensor(n_cols=config.act_dim)#torch.cat(act_env)
        #episodes = data_obj._get_episodes(n_eps=25)#10*config.n_envs)

        return data_obj, obs, obs_env, obs_next, obs_next_env, act_env, r_env, r, done_env, act, context

    else:
        data_obj = DataProcessor(data, n_envs=config.n_envs)                       # set data object
        obs = data_obj._obs_to_tensor(n_cols=4)
        obs_env = data_obj._obs_to_tensor_e(n_cols=config.obs_shape)
        obs_next = data_obj._obs_next_to_tensor(n_cols=config.obs_shape)
        obs_next_env = data_obj._obs_next_to_tensor_e(n_cols=config.obs_shape)
        act_env = data_obj._a_current_dummy_e(num_classes = 1) # get env a in one hot
        r_env = data_obj._r_e()    
        done_env = data_obj._done_e()                                  # get all env dones in one hot  
        act = data_obj._a_current_dummy_all()                            # get all a in one hot  

        return data_obj, obs, obs_env, obs_next, obs_next_env, act_env, r_env, r, done_env, act

    print("Processing data completed.")
