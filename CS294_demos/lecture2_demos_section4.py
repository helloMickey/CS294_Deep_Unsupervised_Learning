import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from deepul_helper.models import MaskConv2d

import sys
sys.path.append("../../deepul-master")
sys.path.append("../demos")


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class PixelCNN(nn.Module):
    name = 'PixelCNN'
    def __init__(self, n_layers):
        super().__init__()
        model = [MaskConv2d('A', 1, 1, 3, padding=1)]
        for _ in range(n_layers - 2):
            model.extend([MaskConv2d('B', 1, 1, 3, padding=1)])
        model.append(MaskConv2d('B', 1, 1, 3,padding=1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x)


class HoriVertStackConv2d(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, k=3, padding=1):
        super().__init__()
        self.vertical = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                                  padding=padding, bias=False)
        self.horizontal = nn.Conv2d(in_channels, out_channels, kernel_size=(1, k),
                                    padding=(0, padding), bias=False)
        self.vtohori = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 + 1:, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2 + 1:] = 0
        if mask_type == 'A':
            self.hmask[:, :, :, k // 2] = 0

    def down_shift(self, x):
        x = x[:, :, :-1, :]
        pad = nn.ZeroPad2d((0, 0, 1, 0))
        return pad(x)

    def forward(self, x):
        vx, hx = x.chunk(2, dim=1)

        self.vertical.weight.data *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx = self.vertical(vx)
        hx = self.horizontal(hx)
        # Allow horizontal stack to see information from vertical stack
        hx = hx + self.vtohori(self.down_shift(vx))

        return torch.cat((vx, hx), dim=1)


# PixelCNN using horizontal and vertical stacks to fix blind-spot
class HoriVertStackPixelCNN(nn.Module):
    name = 'HoriVertStackPixelCNN'
    def __init__(self, n_layers):
        super().__init__()
        model = [HoriVertStackConv2d('A', 1, 1, 3, padding=1)]
        for _ in range(n_layers - 2):
            model.extend([HoriVertStackConv2d('B', 1, 1, 3, padding=1)])
        model.append(HoriVertStackConv2d('B', 1, 1, 3, padding=1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(torch.cat((x, x), dim=1)).chunk(2, dim=1)[1]


def plot_receptive_field(model, data):
    out = model(data)
    # 从顶层的map上的某一点（变量）进行反向传播
    out[0, 0, 5, 5].backward()
    # 获取out[0, 0, 5, 5]变量对应输出层上的感受野
    grad = data.grad.detach().cpu().numpy()[0, 0]
    grad = np.abs(grad)
    grad = (grad > 1e-8).astype('float32')
    grad[5, 5] = 0.5

    plt.figure()
    plt.imshow(grad)
    plt.title(f'Receptive field from pixel (5, 5), {model.name} {n_layers} layers')
    plt.show()

    
x = torch.randn(1, 1, 10, 10).cuda()
x.requires_grad = True

# PixelCNN Blindspot
# 随着层数的增加 顶层的map中某一点对应的输入层上的感受野逐渐增大，blindspot的区域逐渐增大
for i, n_layers in enumerate([2, 3, 5]):
    pixelcnn = PixelCNN(n_layers=n_layers).cuda()
    plot_receptive_field(pixelcnn, x)
    x.grad.zero_()

# PixelCNN with Horizontal and Vertical Stacked Convolutions (No Blindspot)
for i, n_layers in enumerate([2, 3, 5]):
    gated_pixelcnn = HoriVertStackPixelCNN(n_layers=n_layers).cuda()
    plot_receptive_field(gated_pixelcnn, x)
    x.grad.zero_()