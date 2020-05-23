import numpy as np
import torch
import torch.nn as nn
from os.path import join
from deepul_helper.models import RNN, MADE, PixelCNN, WaveNet
from deepul_helper.visualize import visualize_batch

import sys
sys.path.append("../../deepul-master")
sys.path.append("../demos")

device = torch.device('cuda')
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# RNN
model = RNN(device, append_loc=False).to(device)
test_losses = np.load(join('pretrained_models', 'rnn', 'test_losses.npy'))
for epoch in [-1, 0, 1, 2, 8, 19]:
    model.load_state_dict(torch.load(join('pretrained_models', 'rnn', 'checkpoints', f'epoch{epoch}_state_dict')))
    samples = model.sample(64)
    if epoch == -1:
        visualize_batch(samples, title=f'Intialization')
    else:
        print(f'Epoch {epoch} Test Loss: {test_losses[epoch] / np.log(2):.4f} bits/dim')
        visualize_batch(samples, title=f'Epoch {epoch}')

model = RNN(device, append_loc=True).to(device)
test_losses = np.load(join('pretrained_models', 'rnn_loc', 'test_losses.npy'))
for epoch in [-1, 0, 1, 2, 8, 19]:
    model.load_state_dict(torch.load(join('pretrained_models', 'rnn_loc', 'checkpoints', f'epoch{epoch}_state_dict')))
    samples = model.sample(64)
    if epoch == -1:
        visualize_batch(samples, title=f'Intialization')
    else:
        print(f'Epoch {epoch} Test Loss: {test_losses[epoch] / np.log(2):.4f} bits/dim')
        visualize_batch(samples, title=f'Epoch {epoch}')

# MADE
model = MADE(device).to(device)
test_losses = np.load(join('pretrained_models', 'made_0', 'test_losses.npy'))
for epoch in [-1, 0, 1, 2, 8, 19]:
    model.load_state_dict(torch.load(join('pretrained_models', 'made_0', 'checkpoints', f'epoch{epoch}_state_dict')))
    samples = model.sample(64)
    if epoch == -1:
        visualize_batch(samples, title=f'Intialization')
    else:
        print(f'Epoch {epoch} Test Loss: {test_losses[epoch] / np.log(2):.4f} bits/dim')
        visualize_batch(samples, title=f'Epoch {epoch}')


# WaveNet
model = WaveNet(device, append_loc=False).to(device)
test_losses = np.load(join('pretrained_models', 'wavenet', 'test_losses.npy'))
for epoch in [-1, 0, 1, 2, 8, 19]:
    model.load_state_dict(torch.load(join('pretrained_models', 'wavenet', 'checkpoints', f'epoch{epoch}_state_dict')))
    samples = model.sample(64)
    if epoch == -1:
        visualize_batch(samples, title=f'Intialization')
    else:
        print(f'Epoch {epoch} Test Loss: {test_losses[epoch] / np.log(2):.4f} bits/dim')
        visualize_batch(samples, title=f'Epoch {epoch}')

model = WaveNet(device, append_loc=True).to(device)
test_losses = np.load(join('pretrained_models', 'wavenet_loc', 'test_losses.npy'))
for epoch in [-1, 0, 1, 2, 8, 19]:
    model.load_state_dict(torch.load(join('pretrained_models', 'wavenet_loc', 'checkpoints', f'epoch{epoch}_state_dict')))
    samples = model.sample(64)
    if epoch == -1:
        visualize_batch(samples, title=f'Intialization')
    else:
        print(f'Epoch {epoch} Test Loss: {test_losses[epoch] / np.log(2):.4f} bits/dim')
        visualize_batch(samples, title=f'Epoch {epoch}')


# PixelCNN
model = PixelCNN(device).to(device)
test_losses = np.load(join('pretrained_models', 'pixelcnn', 'test_losses.npy'))
for epoch in [-1, 0, 1, 2, 8, 19]:
    model.load_state_dict(torch.load(join('pretrained_models', 'pixelcnn', 'checkpoints', f'epoch{epoch}_state_dict')))
    samples = model.sample(64)
    if epoch == -1:
        visualize_batch(samples, title=f'Intialization')
    else:
        print(f'Epoch {epoch} Test Loss: {test_losses[epoch] / np.log(2):.4f} bits/dim')
        visualize_batch(samples, title=f'Epoch {epoch}')