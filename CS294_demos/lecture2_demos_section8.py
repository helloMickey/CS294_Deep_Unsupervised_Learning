from os.path import join
import torch
import torch.nn as nn
from deepul_helper.models import GrayscalePixelCNN
from deepul_helper.visualize import visualize_batch

device = torch.device('cuda')

model = GrayscalePixelCNN(device).cuda()
model.load_state_dict(torch.load(join('pretrained_models', 'grayscale_pixelcnn', 'checkpoints', 'epoch17_state_dict')))
gray_samples, color_samples = model.sample(32)
gray_samples = gray_samples.repeat(1, 3, 1, 1)
samples = torch.stack((gray_samples, color_samples), dim=1).view(-1, 3, 28, 28)
visualize_batch(samples, figsize=(10, 10))