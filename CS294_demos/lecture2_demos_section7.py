from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepul_helper.visualize import visualize_batch
from deepul_helper.utils import to_one_hot
from deepul_helper.models import PixelCNN

device = torch.device('cuda')


def load_data():
    from torchvision import transforms
    from torchvision.datasets import MNIST
    import torch.utils.data as data

    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x > 0.5).float()
    ])
    train_dset = MNIST('data', transform=transform, train=True, download=True)
    test_dset = MNIST('data', transform=transform, train=False, download=True)

    # train_loader = data.DataLoader(train_dset, batch_size=128, shuffle=True,
    #                                pin_memory=True, num_workers=2)
    # test_loader = data.DataLoader(test_dset, batch_size=128, shuffle=True,
    #                               pin_memory=True, num_workers=2)
    # 为了可以设断点调试，删掉 num_workers 这个参数
    train_loader = data.DataLoader(train_dset, batch_size=128, shuffle=True,
                                   pin_memory=True)
    test_loader = data.DataLoader(test_dset, batch_size=128, shuffle=True,
                                  pin_memory=True)

    return train_loader, test_loader


"""
1. Class-Conditional PixelCNN
2. Image Super-Resolution with a PixelCNN
两个的条件不同，一个是 one-hot 的类别 condition，另一个是缩小四分之一后的low resolution作为 condition
"""

# Class-Conditional PixelCNN
model = PixelCNN(device, conditional_size=(10,), n_layers=5).to(device)
model.load_state_dict(torch.load(join('pretrained_models', 'pixelcnn_class', 'checkpoints', 'epoch9_state_dict')))

cond = torch.arange(10).unsqueeze(1).repeat(1, 10).view(-1).to(device)
cond = to_one_hot(cond, 10, device)
samples = model.sample(100, cond=cond)
visualize_batch(samples, nrow=10, figsize=(10, 10))


# Image Super-Resolution with a PixelCNN
model = PixelCNN(device, conditional_size=(1, 7, 7), n_layers=5).to(device)
model.load_state_dict(torch.load(join('pretrained_models', 'pixelcnn_image', 'checkpoints', 'epoch9_state_dict')))
_, test_loader = load_data()
x = next(iter(test_loader))[0][:32]
cond = F.interpolate(x, scale_factor=0.25, mode='bilinear').to(device)
samples = model.sample(32, cond=cond)
cond = F.interpolate(cond, scale_factor=4).cpu()
images = torch.stack((cond, samples), dim=1)
images = images.view(-1, *images.shape[2:])
visualize_batch(images, nrow=8, figsize=(10, 10))