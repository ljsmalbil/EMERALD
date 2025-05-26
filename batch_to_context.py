import pandas as pd
from configs.configs import config
from collections import defaultdict
import numpy as np

from itertools import chain
import random

def create_context_dicts(data_path, all_contexts, batch_size):
    # Load data
    data = pd.read_csv(data_path)

    print("Processing context batches...")
    
    # Create batch index
    data['batch_idx'] = data.index // batch_size
    
    # Create dictionaries for positive and negative indices
    pos_dict = defaultdict(dict)  # Maps batch_idx -> {row_index -> pos_indices}
    neg_dict = defaultdict(dict)  # Maps batch_idx -> {row_index -> neg_indices}
    
    # Iterate over each batch
    for batch_idx in data['batch_idx'].unique():
        neg_indices = []
        batch_data = data[data['batch_idx'] == batch_idx]
        
        # get current context
        context = batch_data['context'].values[0]
        
        # get positive and negative indices
        positive_indices = set(data[data['context'] == context]['batch_idx'].values)
        
        pos_dict[batch_idx] = positive_indices
        
        for oth_context in all_contexts:
            if oth_context == context:
                pass
            else:
                negative_indices = set(data[data['context'] == oth_context]['batch_idx'].values)
                neg_indices.append(negative_indices)
        
        neg_dict[batch_idx] = list(chain.from_iterable(neg_indices))

    print("Context batches processed.")
    
    return pos_dict, neg_dict
