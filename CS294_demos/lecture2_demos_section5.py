import torch.nn as nn
import torch
from os.path import join
import numpy as np
from deepul_helper.models import Transformer
from deepul_helper.visualize import visualize_batch

device = torch.device('cuda')

model = Transformer(device, mode='pixel_location').to(device)
test_losses = np.load(join('pretrained_models', 'transformer_loc', 'test_losses.npy'))
for epoch in [-1, 0, 1, 2, 8, 19]:
    model.load_state_dict(torch.load(join('pretrained_models', 'transformer_loc', 'checkpoints', f'epoch{epoch}_state_dict')))
    samples = model.sample(64)
    if epoch == -1:
        visualize_batch(samples, title=f'Intialization')
    else:
        print(f'Epoch {epoch} Test Loss: {test_losses[epoch] / np.log(2):.4f} bits/dim')
        visualize_batch(samples, title=f'Epoch {epoch}')
