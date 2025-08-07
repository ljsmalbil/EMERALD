from modules.worldmodel import *
from modules.networks import *
from configs.configs import config

from torch.distributions import Categorical
from utils import *
from utils import contrastive_loss

import torch
import pandas as pd
import numpy as np

from batch_to_context import create_context_dicts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class emerald_wm():
    def __init__(self, config):
        super(emerald_wm, self).__init__()
        self.config = config
        self.x_all = None
        self.C_d = 0.1
        self.num_passes = 0

        self.world_model = WorldModel(config=config).to(device)
        
        self.world_optimizer = optim.AdamW(
                self.world_model.parameters(),
                lr=config.world_lr,
                weight_decay=config.weight_decay,
            )

    def update_world_model(self, obs, act, r, obs_next, done_env, context):
        done_criterion = nn.CrossEntropyLoss()
        reward_criterion = nn.MSELoss()
        transition_criterion = nn.MSELoss()
        
        terminal = done_env #torch.cat(done_env).float()

        batch_size = config.batch_size
        num_batches = len(obs) // batch_size
        total_loss = 0
        envs_pred = []

        # if True:
        #     raise "Errr"

        first_batch = 1

        for batch_idx in range(num_batches):
            batch_obs = obs[batch_idx * batch_size:(batch_idx + 1) * batch_size]  
            batch_act = act[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_reward = r[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_obs_next = obs_next[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_done = terminal[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_context = context[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            self.world_optimizer.zero_grad()

            # Forward pass
            predictions = self.world_model(batch_obs.float(), batch_act, batch_reward, batch_obs_next, batch_done, batch_context, first_batch)

            x_current = predictions['x_current'].to(device)
            x_next_psi = predictions['x_next_psi'].to(device)
            x_next_tau = predictions['x_next_tau'].to(device)
            pred_reward = predictions['reward_prediction'].to(device)
            terminal_prediction = predictions['terminal_prediction'].to(device)
            e_est = predictions['envs_est'].to(device)


            envs_pred.append(e_est)

            #sample_current = torch.cat((x_current, batch_reward.unsqueeze(1), x_next_psi), dim=1) #torch.cat((x_current, batch_act, batch_reward.unsqueeze(1), x_next_psi), dim=1)

            info_nce = 0#contrastive_loss(sample_current, positives_cat, negatives_cat, config.temperature)

            # Compute tau loss
            x_next = x_current + x_next_tau
            transition_loss = transition_criterion(x_next, x_next_psi)
            # Compute reward loss
            reward_loss = reward_criterion(pred_reward, batch_reward.unsqueeze(1))
            # Compute entropy loss
            current_abstract_all_random = shuffle_tensor_rows(x_current)
            l2_norm = torch.norm(x_current - current_abstract_all_random, p=2)
            entropy_loss = torch.exp(-self.C_d * l2_norm)
            # Compute total loss 
            loss = transition_loss + reward_loss + (0.01 * entropy_loss) + (config.info_nce_weight * info_nce)

            # Backward pass
            loss.backward()
            self.world_optimizer.step()

            total_loss += loss.item()

            first_batch = 0

        self.num_passes += 1

        avg_loss = total_loss / num_batches
        print(f"Average loss: {avg_loss}.")
        print(f"Reward loss: {reward_loss}.")

        #self.update_world_model(obs, act, r, obs_next, done_env)
        torch.save(self.world_model.state_dict(), f"models/world_model_{config.environment_id}.pth")

        e_all = torch.cat(envs_pred)
        self.x_all = self.world_model.psi(obs, e_all).detach().cpu().numpy()
        
        return None
    
    def train_wm(self, obs, act, r, obs_next, done_env, context):

        data_path = f"data/{config.environment_id}_data.csv"
        # contexts = [i for i in range(config.n_envs)]
        # pos_dict, neg_dict = create_context_dicts(data_path, contexts, batch_size=config.batch_size)

        for update in range(config.train_epochs):
            print(f"Current iteration: {update}.")
            self.update_world_model(obs, act, r, obs_next, done_env, context)
            torch.save(self.world_model.state_dict(), f"models/world_model_{config.environment_id}.pth")

            if update % 1 == 0:    
                df = pd.DataFrame(self.x_all)
                df.to_csv(f"data/x_all_{config.environment_id}.csv", index=False)

        return None
    
     
