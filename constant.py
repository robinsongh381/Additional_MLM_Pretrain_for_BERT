from __future__ import absolute_import
import torch

# GPU device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model
hidden_size = 768
maxlen = 32
epoch = 50
batch_size = 32
dropout = 0.1
learning_rate = 2e-5
vocab_size = 8002

# Optimizer
gradient_accumulation_steps = 1
max_grad_norm = 5

# Indicies
pad_idx = 0

