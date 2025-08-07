import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    def __init__(self, latent_dim, act_dim, hidden_dim, e_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=latent_dim + act_dim + 1,
                            hidden_size=hidden_dim,
                            batch_first=True)
        self.to_context = nn.Linear(hidden_dim, e_dim)
        self.hidden = None  # Track hidden state

    def reset(self):
        self.hidden = None

    def forward(self, x_seq, a_seq, r_seq, cartpole=False):
        # if True:
        #     z_seq = torch.cat([x_seq, a_seq.unsqueeze(-1), r_seq.unsqueeze(-1)], dim=-1)
        # else:      
        z_seq = torch.cat([x_seq, a_seq, r_seq.unsqueeze(-1)], dim=-1)

        _, self.hidden = self.lstm(z_seq, self.hidden)
        e = self.to_context(self.hidden[0][-1])  # h_n[-1]

        return e, self.hidden

class Psi(nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim=2, context_dim=16, n_layers=3):
        super(Psi, self).__init__()
        self.context_dim = context_dim

        self.input_layer = nn.Linear(state_dim + context_dim, hidden_dim)

        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, latent_dim)

        self.add = nn.Linear(state_dim, latent_dim)  # Identity-like addition

    def forward(self, s, e):
        s = torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s
        e = torch.tensor(e, dtype=torch.float32) if not isinstance(e, torch.Tensor) else e

        # Convert to tensors if needed
        s = torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s
        e = torch.tensor(e, dtype=torch.float32) if not isinstance(e, torch.Tensor) else e

        # Ensure both are at least 2D
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if e.dim() == 1:
            e = e.unsqueeze(0)

        # If batch sizes don’t match, broadcast the smaller one
        if s.size(0) != e.size(0):
            if s.size(0) == 1:
                s = s.expand(e.size(0), -1)
            elif e.size(0) == 1:
                e = e.expand(s.size(0), -1)
            else:
                raise ValueError(f"Batch size mismatch: s has {s.size(0)}, e has {e.size(0)}")

        x = torch.cat([s, e], dim=-1)
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        identity = self.add(s)
        x = x + identity

        # if True:
        #     raise "error"
        return x

class Tau(nn.Module):
    def __init__(self, action_dim, hidden_dim, latent_dim=2, n_layers=7):
        super(Tau, self).__init__()
        self.action_dim = action_dim

        self.fc_state = nn.Linear(latent_dim, hidden_dim)
        self.fc_action = nn.Linear(action_dim, hidden_dim)

        self.relu = nn.ReLU()

        self.state_layers = self._create_hidden_layers(hidden_dim, n_layers)
        self.action_layers = self._create_hidden_layers(hidden_dim, n_layers)

        self.fc_combined = nn.Linear(hidden_dim * 2, hidden_dim)
        self.combined_layers = self._create_hidden_layers(hidden_dim, n_layers)

        self.decoder_net = self._create_hidden_layers(hidden_dim, n_layers, output_dim=latent_dim)

        self._initialize_weights()

    def _create_hidden_layers(self, hidden_dim, n_layers, output_dim=None):
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        if output_dim:
            layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, a):
        x = self.relu(self.fc_state(x))
        x = self.state_layers(x)

        a = self.relu(self.fc_action(a))
        a = self.action_layers(a)

        combined = torch.cat((x, a), dim=1)
        combined = self.relu(self.fc_combined(combined))
        combined = self.combined_layers(combined)
        x_next = self.decoder_net(combined)

        return x_next


class Rho(nn.Module):
    def __init__(self, action_dim, latent_dim, hidden_dim, output_dim=1):
        super(Rho, self).__init__()
        self.fc_input = nn.Linear(action_dim + latent_dim, hidden_dim)
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, a):
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim > 2:
            x = x.view(x.size(0), -1)

        inp = torch.cat((x, a), dim=1)
        x = self.fc_input(inp)
        return self.network(x)



# Define the MLP model
class TerminalNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(TerminalNet, self).__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        x = self.fc_input(x)
        return self.network(x)

    
