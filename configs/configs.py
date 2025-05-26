from types import SimpleNamespace
import numpy as np

# config = {
#     'weight_decay': 0.00001,
#     'hidden_dim': 128,
#     'latent_dim': 32,
#     'obs_shape': 17,
#     'policy_network_lr': 0.001,       # maybe make sure the max steps ep is small, otherwise the errors compound too much.
#     'world_lr': 0.0001,
#     'act_dim': 6,
#     'gamma': 0.99,
#     'hidden_dim_rho': 128,
#     'output_dim_rho': 1,
#     'context_dim': 4,
#     'max_buffer_size': 100,
#     'n_envs': 2,
#     'environment_id': 'HalfCheetahDir',
#     'int_state': False,
#     'done_weight': 0, # <----- set to 0, cont. control env.
#     'r_weight': 1,
#     'tau_loss_weight': 1,
#     'train_epochs': 10,#0, 
#     'cont_act': True,
#     'int_reward': False,
#     'with_FORT': True,
#     'max_iter': 1000,
#     'policy': 'sac',
#     'max_episode_steps': 1000,                
#     'policy_iterations_learn': 2000000, # 4000000 - (300000 * 4)<- Change when running experiments. 
#     'samples_per_env': 1000000,
#     'batch_size': 250,
#     'temperature': 0.1,
#     'policy_hidden_dim': 256,
#     'info_nce_weight': 0.2,
#     'num_info_nce': 3,
#     'training_factors': [1, -1],
#     'moderate_factors': [1, -1],
#     'extreme_factors': [1, -1]
# }



config = {
    'weight_decay': 0.00001,
    'hidden_dim': 128,
    'latent_dim': 32,
    'obs_shape': 17,
    'policy_network_lr': 0.001,       # maybe make sure the max steps ep is small, otherwise the errors compound too much.
    'world_lr': 0.0001,
    'act_dim': 6,
    'gamma': 0.99,
    'hidden_dim_rho': 128,
    'output_dim_rho': 1,
    'context_dim': 4,
    'max_buffer_size': 100,
    'n_envs': 5,
    'environment_id': 'HalfCheetah',
    'int_state': False,
    'done_weight': 0, # <----- set to 0, cont. control env.
    'r_weight': 1,
    'tau_loss_weight': 1,
    'train_epochs': 30,#0, 
    'cont_act': True,
    'int_reward': False,
    'with_FORT': True,
    'max_iter': 1000,
    'policy': 'sac',
    'max_episode_steps': 1000,                
    'policy_iterations_learn': 2000000, # 4000000 - (300000 * 4)<- Change when running experiments. 
    'samples_per_env': 100000,
    'batch_size': 250,
    'temperature': 0.1,
    'policy_hidden_dim': 256,
    'info_nce_weight': 0.2,
    'num_info_nce': 3,
    'training_factors': [0.75,0.85,1.0,1.15,1.25],
    'moderate_factors': [0.40,0.50,1.50,1.60],
    'extreme_factors': [0.20,0.30,1.70,1.80]
}


# config = {
#     'weight_decay': 0.00001,
#     'hidden_dim': 256,
#     'latent_dim': 60,
#     'obs_shape': 105,
#     'policy_network_lr': 0.0001,       # maybe make sure the max steps ep is small, otherwise the errors compound too much.
#     'world_lr': 0.0001,
#     'act_dim': 8,
#     'gamma': 0.99,
#     'policy_hidden_dim': 64,
#     'hidden_dim_rho': 128,
#     'output_dim_rho': 1,
#     'context_dim': 4,
#     'n_envs': 4,
#     'with_FORT': True,
#     'environment_id': 'Ant',
#     'int_state': False,
#     'done_weight': 0, # <----- set to 0, cont. control env.
#     'r_weight': 1,
#     'tau_loss_weight': 1,
#     'train_epochs': 200,  
#     'cont_act': True,
#     'int_reward': False,
#     'max_iter': 1000,
#     'policy': 'sac',
#     'max_episode_steps': 1000,                ## <- MAKE SURE THIS NUMBER DOES NOT EXCEED THE SAMPLES_PER_EP HYPERPAR. FROM ENVS.
#     'policy_iterations_learn': 3000000, #1000000#3000000,
#     'samples_per_env': 300000,
#     'batch_size': 256,
#     'temperature': 0.1,
#      'info_nce_weight': 0.5,
#      'num_info_nce': 2,
#      'training_factors': [0.85, 0.9, 0.95, 1],
#      'moderate_factors': [0.20, 0.25, 0.30, 0.35, 0.40],
#      'extreme_factors': [0.45, 0.50, 0.55, 0.6]
# }



# config = {
#     'weight_decay': 0.00001,
#     'hidden_dim': 256,
#     'latent_dim': 32,
#     'obs_shape': 105,
#     'policy_network_lr': 3e-4,
#     'world_lr': 0.0001,
#     'act_dim': 8,
#     'gamma': 0.99,
#     'policy_hidden_dim': 64,
#     'hidden_dim_rho': 128,
#     'output_dim_rho': 1,
#     'context_dim': 1,
#     'n_envs': 5,
#     'with_FORT': False,
#     'environment_id': 'AntDir',
#     'int_state': False,
#     'done_weight': 0,
#     'r_weight': 1,
#     'tau_loss_weight': 1,
#     'train_epochs': 20,#0,
#     'cont_act': True,
#     'int_reward': False,
#     'max_iter': 1000,
#     'policy': 'sac',
#     'max_episode_steps': 1000,
#     'policy_iterations_learn': 3000000,
#     'samples_per_env': 300000,
#     'batch_size': 250,
#     'temperature': 0.1,
#     'info_nce_weight': 0.5,
#     'num_info_nce': 2,

#     # Direction vectors
#     'training_factors': [
#         np.array([1.0, 0.0]),
#         np.array([0.30901699, 0.95105652]),
#         np.array([-0.80901699, 0.58778525]),
#         np.array([-0.80901699, -0.58778525]),
#         np.array([0.30901699, -0.95105652])
#     ],
#     'moderate_factors': [
#         np.array([0.98480775, 0.17364818]),
#         np.array([0.17364818, 0.98480775]),
#         np.array([-0.93969262, 0.34202014]),
#         np.array([-0.93969262, -0.34202014]),
#         np.array([0.17364818, -0.98480775])
#     ],
#     'extreme_factors': [
#         np.array([0.80901699, 0.58778525]),
#         np.array([-0.30901699, 0.95105652]),
#         np.array([-1.0, 0.0]),
#         np.array([-0.30901699, -0.95105652]),
#         np.array([0.80901699, -0.58778525])
#     ]
# }




# config = {
#     'weight_decay': 0.00001,
#     'hidden_dim': 256,
#     'latent_dim': 300,
#     'obs_shape': 348,
#     'policy_network_lr': 0.0003,       # maybe make sure the max steps ep is small, otherwise the errors compound too much.
#     'world_lr': 0.0001,
#     'act_dim': 17,
#     'gamma': 0.99,
#     'policy_hidden_dim': 64,
#     'hidden_dim_rho': 64,
#     'output_dim_rho': 1,
#     'context_dim': 4,
#     'n_envs': 5,
#     'config.with_FORT': True,
#     'environment_id': 'Humanoid',
#     'int_state': False,
#     'done_weight': 0, # <----- set to 0, cont. control env.
#     'r_weight': 1,
#     'tau_loss_weight': 1,
#     'train_epochs': 1000,  
#     'cont_act': True,
#     'int_reward': False,
#     'max_iter': 1000,
#     'policy': 'ppo',
#     'max_episode_steps': 1000,                ## <- MAKE SURE THIS NUMBER DOES NOT EXCEED THE SAMPLES_PER_EP HYPERPAR. FROM ENVS.
#     'policy_iterations_learn': 3*1000000, #1000000#3000000,
#     'samples_per_env': 100000,
#     'batch_size': 256,
#     'temperature': 1,#0.1,
#     'learning_rate': 0.0003,
#     'info_nce_weight': 0.05,
#     'num_info_nce': 3,
#      'training_factors': [0.80, 0.90, 1.0, 1.15, 1.25],
#      'moderate_factors': [0.60, 0.70, 1.50, 1.60],
#      'extreme_factors': [0.40, 0.50, 1.70, 1.80]
# }




# config = {
#     'weight_decay': 0.00001,
#     'hidden_dim': 32,
#     'latent_dim': 8,
#     'obs_shape': 4,
#     'policy_network_lr': 0.0003,       # maybe make sure the max steps ep is small, otherwise the errors compound too much.
#     'world_lr': 0.0001,
#     'act_dim': 1,
#     'gamma': 0.99,
#     'policy_hidden_dim': 64,
#     'hidden_dim_rho': 64,
#     'output_dim_rho': 1,
#     'context_dim': 4,
#     'n_envs': 5,
#     'with_FORT': True,
#     'max_buffer_size': 0,
#     'environment_id': 'CartPole',
#     'int_state': False,
#     'done_weight': 0, # <----- set to 0, cont. control env.
#     'r_weight': 1,
#     'tau_loss_weight': 1,
#     'train_epochs': 1,
#     'cont_act': True,
#     'int_reward': False,
#     'max_iter': 200,
#     'policy': 'ppo',
#     'max_episode_steps': 200,                ## <- MAKE SURE THIS NUMBER DOES NOT EXCEED THE SAMPLES_PER_EP HYPERPAR. FROM ENVS.
#     'policy_iterations_learn': 2000000,#2000000, #1000000#3000000,
#     'samples_per_env': 10000,
#     'batch_size': 250,
#     'temperature': 1,#0.1,
#     'learning_rate': 0.0003,
#     'info_nce_weight': 0.05,
#     'num_info_nce': 3,
#      'training_factors': [1],#[0.40,0.45,0.50,0.55,0.60],
#      'moderate_factors': [0.25,0.30,0.70,0.75],
#      'extreme_factors': [0.15,0.20,0.80,0.85]
# }


# config = {
#     'weight_decay': 0.00001,
#     'hidden_dim': 8,
#     'latent_dim': 8,
#     'obs_shape': 3,
#     'policy_network_lr': 0.0001,       # maybe make sure the max steps ep is small, otherwise the errors compound too much.
#     'world_lr': 0.0001,
#     'act_dim': 1,
#     'gamma': 0.95,
#     'policy_hidden_dim': 12,
#     'hidden_dim_rho': 4,
#     'output_dim_rho': 1,
#     'context_dim': 2,
#     'n_envs': 11,
#     'environment_id': 'Pendulum',
#     'int_state': False,
#     'done_weight': 0, # <----- set to 0, cont. control env.
#     'r_weight': 1,
#     'tau_loss_weight': 1,
#     'train_epochs': 10,  
#     'cont_act': True,
#     'int_reward': False,
#     'max_iter': 200,
#     'policy': 'sac',
#     'max_episode_steps':200,                ## <- MAKE SURE THIS NUMBER DOES NOT EXCEED THE SAMPLES_PER_EP HYPERPAR. FROM ENVS.
#     'policy_iterations_learn': 2*500000,#3000000,#0000, #1000000#3000000,
#     'samples_per_env': 100000,
#     'batch_size': 250,
#     'max_buffer_size': 100,
#     'temperature': 0.1,
#     'num_info_nce': 10,
#     'info_nce_weight':  0.2,
#     'training_factors': [0.75,0.80,0.85,0.90,0.95,1.0,1.05,1.10,1.15,1.20,1.25],
#     'moderate_factors': [0.50,0.70,1.30,1.50],
#     'extreme_factors': [0.20,0.40,1.60,1.80],
#     'with_FORT': True
# }





# config = {
#     'weight_decay': 0.00001,
#     'hidden_dim': 4,
#     'latent_dim': 2,
#     'obs_shape': 16,
#     'policy_network_lr': 0.0001,       # maybe make sure the max steps ep is small, otherwise the errors compound too much.
#     'world_lr': 0.0001,
#     'act_dim': 4,
#     'gamma': 0.99,
#     'policy_hidden_dim': 64,
#     'hidden_dim_rho': 2,
#     'output_dim_rho': 1,
#     'context_dim': 4,
#     'n_envs': 2,
#     'environment_id': 'FrozenLake',
#     'int_state': True,
#     'done_weight': 0, # <----- set to 0, cont. control env.
#     'r_weight': 1,
#     'tau_loss_weight': 1,
#     'train_epochs': 3000,  
#     'cont_act': True,
#     'int_reward': False,
#     'max_iter': 1000,
#     'policy': 'sac',
#     'max_episode_steps':200,                ## <- MAKE SURE THIS NUMBER DOES NOT EXCEED THE SAMPLES_PER_EP HYPERPAR. FROM ENVS.
#     'policy_iterations_learn': 1000000, #1000000#3000000,
#     'samples_per_env': 100000,
#     'batch_size': 3000,
#     'temperature': 0.1,
#     'info_nce_weight': 0.01
# }

config = SimpleNamespace(**config)
