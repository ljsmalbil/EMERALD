import pandas as pd
import numpy as np
import seaborn as sns

import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import networkx as nx


def contrastive_loss(x, x_pos, x_neg, temperature=0.1):
    """
    InfoNCE loss for contrastive learning with multiple positive and negative examples.
    
    x: Query embeddings (batch_size, d)
    x_pos: Positive embeddings (batch_size, d, num_pos)  # Multiple positives per query
    x_neg: Negative embeddings (batch_size, d, num_neg)  # Multiple negatives per query
    
    Returns:
        Scalar contrastive loss.
    """
    # Normalize embeddings
    x = F.normalize(x, dim=-1).unsqueeze(-1)  # (batch_size, d, 1)
    x_pos = F.normalize(x_pos, dim=-2)  # (batch_size, d, num_pos)
    x_neg = F.normalize(x_neg, dim=-2)  # (batch_size, d, num_neg)

    # Compute positive similarities
    pos_sim = torch.exp(torch.sum(x * x_pos, dim=1) / temperature)  # (batch_size, num_pos)

    # Compute negative similarities
    neg_sim = torch.exp(torch.sum(x * x_neg, dim=1) / temperature)  # (batch_size, num_neg)

    # Compute denominator: sum over all similarities (positives + negatives)
    denominator = pos_sim.sum(dim=-1) + neg_sim.sum(dim=-1)  # (batch_size,)

    # Compute loss for each positive example and then average over positives
    loss = -torch.log(pos_sim / denominator.unsqueeze(-1))  # (batch_size, num_pos)
    
    return loss.mean()


def draw_comparison(env1, env2):
    def draw_environment(ax, env, title):
        G = nx.DiGraph()

        # Add nodes and edges based on the environment dictionary
        for state, transitions in env.items():
            for action, next_state in transitions.items():
                # Determine the direction of the arrow based on the action
                if action == 0:
                    direction = "↑"
                elif action == 1:
                    direction = "↓"
                elif action == 2:
                    direction = "←"
                elif action == 3:
                    direction = "→"
                G.add_edge(state, next_state, label=direction)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=12, font_weight="bold", arrows=True, ax=ax)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, ax=ax)
        ax.set_title(title)

    def draw_common_transitions(ax, env1, env2):
        G = nx.DiGraph()

        # Add nodes and edges that are common between env1 and env2
        for state, transitions in env1.items():
            for action, next_state in transitions.items():
                if state in env2 and action in env2[state] and env2[state][action] == next_state:
                    G.add_edge(state, next_state, label=f"S{state} -> S{next_state}")

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightgreen", font_size=12, font_weight="bold", arrows=True, ax=ax)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, ax=ax)
        ax.set_title("Common Transitions")

    def draw_overlap(ax, env1, env2):
        G = nx.DiGraph()

        # Add all nodes from both environments
        all_nodes = set(env1.keys()).union(set(env2.keys()))

        for state in all_nodes:
            if state in env1:
                for action, next_state in env1[state].items():
                    G.add_edge(state, next_state, label=f"S{state} -> S{next_state}")
            if state in env2:
                for action, next_state in env2[state].items():
                    G.add_edge(state, next_state, label=f"S{state} -> S{next_state}")

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightcoral", font_size=12, font_weight="bold", arrows=True, ax=ax)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, ax=ax)
        ax.set_title("Overlap in Environments")

    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Plot the first environment
    draw_environment(axes[0], env1, "Environment 1")

    # Plot the second environment
    draw_environment(axes[1], env2, "Environment 2")

    # Plot the common transitions
    draw_common_transitions(axes[2], env1, env2)

    # Plot the overlap (including shared nodes)
    draw_overlap(axes[3], env1, env2)

    plt.tight_layout()
    plt.show()

def shuffle_tensor_rows(tensor):
    perm = torch.randperm(tensor.size(0))
    return tensor[perm]



def plot_state_transitions(current_abstract_all_psi, next_abstract_psi_all, data, env="all", show=True, save_plot=False):
    """
    This function generates a scatter plot with arrows depicting state transitions 
    based on the starting and next state coordinates and optionally saves the plot as an image file.
    
    Parameters:
    - current_abstract_all_psi: Tensor with the current abstract states
    - next_abstract_psi: List or tensor of the next abstract states
    - data: A dictionary containing 'State' and 'Next State' data
    - save_plot: Boolean, if True, the plot is saved as a PNG file (default: False)
    
    The plot will display the transition between starting and next states with arrows, and label each state.
    """
    # Set seaborn style for grey grid background
    sns.set(style="darkgrid")
    
    # Define the tensors with the state column included in the dataframe
    df = pd.DataFrame({
        'x_start': current_abstract_all_psi[:, 0].detach().numpy(),
        'y_start': current_abstract_all_psi[:, 1].detach().numpy(),
        'x_next': next_abstract_psi_all[:, 0].detach().numpy(),
        'y_next': next_abstract_psi_all[:, 1].detach().numpy()
    })

    if env == "all":
        # Extract coordinates and states
        x_start, y_start = df['x_start'], df['y_start']
        x_next, y_next = df['x_next'], df['y_next']

        states = data['state']
        nextstates = data['next_state']

        print(len(df))
        print(len(states))

    else:
        # Extract coordinates and states
        x_start, y_start = df['x_start'], df['y_start']
        x_next, y_next = df['x_next'], df['y_next']

        states = data[data['context']==env]['state']
        nextstates = data[data['context']==env]['next_state']

        print(len(df))
        print(len(states))


    states = data['state']
    nextstates = data['next_state']

    # Add states and next states to the dataframe
    df['state'] = states
    df['next_state'] = nextstates
    df['action'] = data['action']

    # Remove duplicates from the dataframe
    df = df.drop_duplicates()

    # Drop rows with NaNs
    df = df.dropna()

    print(df.head(100))

    print(f"Len df {len(df)}")

    # Plotting the scatter plot with arrows for state transitions and labeling points
    plt.figure(figsize=(10, 6))
    plt.scatter(x_start, y_start, c='blue', label='Starting States', s=10)
    plt.scatter(x_next, y_next, c='red', label='Next States', s=10)
    
    # Adding arrows for each state transition and labeling points
    for i in range(len(x_start)):
        plt.arrow(x_start[i], y_start[i], x_next[i] - x_start[i], y_next[i] - y_start[i], 
                    head_width=0.01, head_length=0.01, fc='green', ec='green', alpha=0.5)
        plt.text(x_start[i], y_start[i], f'{states[i]}', fontsize=8, ha='right')
        plt.text(x_next[i], y_next[i], f'{nextstates[i]}', fontsize=8, ha='left')

    # Adding labels and title
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Starting States and Next States Scatter Plot with Transitions and State Labels')
    #plt.legend()

    # Save plot conditionally
    if save_plot:
        # Generate file name with current time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f'images/plot_{current_time}.png'
        plt.savefig(file_name)
        print(f'Plot saved as {file_name}')
    else:
        file_name = None  # No file generated

    # Show plot
    if show:
        plt.show()

    return file_name  # Return the file name or None if plot is not saved


def sa_to_one_hot_torch(item, num_classes):
    return torch.nn.functional.one_hot(torch.tensor(item), num_classes).unsqueeze(dim=0)

class DataProcessor():
    def __init__(self, dataset, n_envs):
        super(DataProcessor, self).__init__()
        self.dataset = dataset
        self.n_envs = n_envs
        
    def _tensor(self):
        tensor = torch.tensor(self.dataset.values, dtype=torch.float32)
        return tensor
    
    def _obs_to_tensor(self, n_cols):
        obs = torch.empty((len(self.dataset), n_cols), dtype=torch.float32)

        for i in range(n_cols):
            col = torch.tensor(self.dataset[f'state_{i}'].values, dtype=torch.float32)
            obs[:,i] = col

        return obs
    
    def _obs_next_to_tensor(self, n_cols):
        obs = torch.empty((len(self.dataset), n_cols), dtype=torch.float32)

        for i in range(n_cols):
            col = torch.tensor(self.dataset[f'next_state_{i}'].values, dtype=torch.float32)
            obs[:,i] = col

        return obs
    
    def _obs_next_to_tensor_e(self, n_cols):
        bundle = []
        for env in range(self.n_envs):
            obs = torch.empty((len(self.dataset[self.dataset['context'] == env]), n_cols), dtype=torch.float32)
            for i in range(n_cols):
                col = torch.tensor(self.dataset[self.dataset['context'] == env][f'next_state_{i}'].values, dtype=torch.float32)
                obs[:, i] = col
            
            bundle.append(obs)
        return bundle
    
    def _acts_to_tensor_e(self, n_cols):
        bundle = []
        for env in range(self.n_envs):
            obs = torch.empty((len(self.dataset[self.dataset['context'] == env]), n_cols), dtype=torch.float32)
            for i in range(n_cols):
                col = torch.tensor(self.dataset[self.dataset['context'] == env][f'action_{i}'].values, dtype=torch.float32)
                obs[:, i] = col
            
            bundle.append(obs)
        return bundle
    

    def _acts_to_tensor(self, n_cols):
        obs = torch.empty((len(self.dataset), n_cols), dtype=torch.float32)
        for i in range(n_cols):
            col = torch.tensor(self.dataset[f'action_{i}'].values, dtype=torch.float32)
            obs[:, i] = col
        return obs
    
    def _get_context(self):
        context = torch.tensor(self.dataset[f'context'].values, dtype=torch.float32)
        return context
    
    def _obs_to_tensor_e(self, n_cols):
        bundle = []
        for env in range(self.n_envs):
            obs = torch.empty((len(self.dataset[self.dataset['context'] == env]), n_cols), dtype=torch.float32)
            for i in range(n_cols):
                col = torch.tensor(self.dataset[self.dataset['context'] == env][f'state_{i}'].values, dtype=torch.float32)
                obs[:, i] = col
            
            bundle.append(obs)
        return bundle

        
    def _env_select(self, env):
        tensor = torch.tensor(self.dataset.values, dtype=torch.float32)
        tensor_env = tensor[tensor[:, -1] == env]
        return tensor_env
    
    def _s_current_dummy_e(self, num_classes=25):
        bundle = []
        for env in range(self.n_envs):
            intermed = torch.tensor(np.array(self.dataset[self.dataset['context']==env]['state']))
            one_hot_encoded = torch.nn.functional.one_hot(intermed, num_classes)
            bundle.append(one_hot_encoded)
            
        return bundle
    
    def _a_current_dummy_e(self, num_classes=4):
        bundle = []
        for env in range(self.n_envs):
            intermed = torch.tensor(np.array(self.dataset[self.dataset['context']==env]['action']))
            one_hot_encoded = torch.nn.functional.one_hot(intermed, num_classes)
            bundle.append(one_hot_encoded)
            
        return bundle
    
    def _s_next_dummy_e(self, num_classes=25):
        bundle = []
        for env in range(self.n_envs):
            intermed = torch.tensor(np.array(self.dataset[self.dataset['context']==env]['next_state']))
            one_hot_encoded = torch.nn.functional.one_hot(intermed, num_classes)
            bundle.append(one_hot_encoded)
            
        return bundle
    
    def _s_current_dummy_all(self, num_classes=25):
        s = torch.tensor(np.array(self.dataset['state']))
        one_hot_encoded = torch.nn.functional.one_hot(s, num_classes)
#         intermed = pd.get_dummies(self.dataset['State'])
#         tensor = torch.tensor(intermed.values, dtype=torch.float32)
        return one_hot_encoded
    
    def _a_current_dummy_all(self):
        intermed = pd.get_dummies(self.dataset['action'])
        tensor = torch.tensor(intermed.values, dtype=torch.float32)
        return tensor
    
    def _s_next_dummy_all(self, num_classes=25):
        s_next = torch.tensor(np.array(self.dataset['next_state']))
        one_hot_encoded = torch.nn.functional.one_hot(s_next, num_classes)
        return one_hot_encoded
    
    def _r_e(self):
        bundle = []
        for env in range(self.n_envs):
            bundle.append(torch.tensor(np.array(self.dataset[self.dataset['context']==env]['reward']), dtype=torch.float32))
        return bundle
    
    def _r(self):
        r = torch.tensor(np.array(self.dataset['reward']), dtype=torch.float32)
        return r
    
    def _done_e(self):
        bundle = []
        for env in range(self.n_envs):
            bundle.append(torch.tensor(np.array(self.dataset[self.dataset['context']==env]['terminal']), dtype=torch.float32))
        return bundle
    
    def _s_df_to_tensor(self, state_cols):
        bundle = []
        for env in range(self.n_envs):
            bundle.append(torch.tensor(np.array(self.dataset[self.dataset['context']==env][state_cols])))
            
        return bundle
    
    def _s_df_to_tensor_eps(self, state_cols):
        bundle = []
        for env in range(self.n_envs):
            bundle.append(torch.tensor(np.array(self.dataset[self.dataset['context']==env][state_cols])))
            
        return bundle
    
    def _s_next_df_to_tensor(self, next_state_cols):
        bundle = []
        for env in range(self.n_envs):
            bundle.append(torch.tensor(np.array(self.dataset[self.dataset['context']==env][next_state_cols])))
            
        return bundle
    

    def _get_episodes(self, n_eps):
        bundle = []
        for ep in range(n_eps):
            data_all = self.dataset[self.dataset['episode']==ep]
            data_all = data_all.drop(columns=['episode', 'terminal', 'context'])
            bundle.append(torch.tensor(np.array(data_all)))
            
        return bundle
    


        
    

