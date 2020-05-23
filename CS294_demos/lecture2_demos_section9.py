from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepul_helper.models import ParallelPixelCNN
from deepul_helper.visualize import visualize_batch

device = torch.device('cuda')


def expand_subsampled(samples):
    images = []
    for i in range(5):
        stride = 2 ** (5 - i)
        img = samples[:, :, ::stride, ::stride]
        img = F.interpolate(img, size=(224, 224))
        images.append(img)
    images.append(samples)
    images = torch.stack(images, dim=1).view(-1, 1, 224, 224)
    return images


model = ParallelPixelCNN(device).to(device)
model.load_state_dict(torch.load(join('pretrained_models', 'parallel_pixelcnn', 'checkpoints', 'epoch15_state_dict')))
samples = model.sample(10)
images = expand_subsampled(samples)
visualize_batch(images, nrow=6, figsize=(10, 10))