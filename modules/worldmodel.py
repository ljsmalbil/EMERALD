import torch
import torch.nn as nn
from modules.networks import *

class WorldModel(nn.Module):
    def __init__(self, config):
        super(WorldModel, self).__init__()
        self.obs_shape = config.obs_shape
        self.act_dim = config.act_dim
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.context_dim = config.context_dim
        self.use_x = config.use_x
        self.hidden_state = None
        self.current_context = None
        self.prev_context = None

        # Context-aware encoder: Psi(s, e)
        self.psi = Psi(state_dim=self.obs_shape, hidden_dim=self.hidden_dim,
                       latent_dim=self.latent_dim, context_dim=self.context_dim)

        if self.use_x:
            # Context encoder from (x, a, r) history
            self.lstm = ContextEncoder(latent_dim=self.latent_dim, act_dim=self.act_dim,
                                    hidden_dim=self.hidden_dim, e_dim=self.context_dim)
        else:
            # Context encoder from (s, a, r) history
            self.lstm = ContextEncoder(latent_dim=self.obs_shape, act_dim=self.act_dim,
                                          hidden_dim=self.hidden_dim, e_dim=self.context_dim)

        # Transition and reward models (now context-free)
        self.tau = Tau(action_dim=self.act_dim, hidden_dim=self.hidden_dim,
                       latent_dim=self.latent_dim)

        self.rho = Rho(action_dim=self.act_dim, hidden_dim=self.hidden_dim,
                       latent_dim=self.latent_dim, output_dim=1)

        self.discount_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.act_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.terminal = TerminalNet(input_dim=self.act_dim + self.latent_dim,
                                    hidden_dim=64, output_dim=config.output_dim_rho)

    def forward(self, s, action, reward, s_next, terminal, context, first_batch=None):
        # Compute context from history of (x, a, r)
        self.current_context = context[0]

        #print("Current context: ", self.current_context)

        batch_size = s.size(0)
        if (first_batch == 1) or (self.current_context != self.prev_context):
            self.lstm.reset()
            e = torch.zeros(batch_size, self.context_dim).to(s.device)
        else:
            if self.use_x:
                with torch.no_grad():
                    # print(s.size())
                    # print(torch.zeros_like(s[:, :self.context_dim]))
                    x_current = self.psi(s, e=torch.zeros_like(s[:, :self.context_dim]))
                e, self.hidden_state = self.lstm(x_current, action, reward, self.hidden_state)
                e = e.detach()
            else:
                e, self.hidden_state = self.lstm(s, action, reward, self.hidden_state)
                e = e.detach()




        # Context-aware encoding of current and next state
        x_current = self.psi(s, e)


        x_next_psi = self.psi(s_next, e)

        # Predict next latent state without e (already embedded)
        x_next_tau = self.tau(x_current, action)

        # Predict reward without e (already embedded)
        reward_prediction = self.rho(x_current, action)

        # Predict terminal state
        terminal_prediction = self.terminal(torch.cat((x_current, action), dim=1))


        self.prev_context = self.current_context

        env_est = e.expand(s.size(0), self.context_dim)

        return {
            'x_current': x_current,
            'x_next_psi': x_next_psi,
            'x_next_tau': x_next_tau,
            'reward_prediction': reward_prediction,
            'terminal_prediction': terminal_prediction,
            'envs_est': env_est,
        }


        
        
