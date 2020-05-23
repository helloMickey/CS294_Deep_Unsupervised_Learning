from os.path import join
import numpy as np
import torch

from deepul_helper.models import MADE
from deepul_helper.visualize import visualize_batch

device = torch.device('cuda')

def run_demo(ordering, order_id):
    model = MADE(device, ordering=ordering).to(device)
    model.load_state_dict(torch.load(join('pretrained_models', f'made_{order_id}', 'checkpoints', f'epoch19_state_dict')))
    test_losses = np.load(join('pretrained_models', f'made_{order_id}', 'test_losses.npy'))
    samples = model.sample(64)
    print('Test Loss', test_losses[-1])
    visualize_batch(samples, title=f'Samples')

# 随机顺序
ordering = np.random.permutation(784)
run_demo(ordering, order_id=1)


# 先偶再奇
ordering = np.concatenate((np.arange(0, 784, 2), np.arange(1, 784, 2)))
run_demo(ordering, order_id=2)

# 行扫
ordering = np.arange(784)
run_demo(ordering, order_id=0)

# 列扫
ordering = np.arange(784).reshape(28, 28).T.reshape(-1)
run_demo(ordering, order_id=3)


# Top to Middle then Bottom to Middle
ordering = np.concatenate((np.arange(784 // 2), np.arange(784 // 2, 784)[::-1]))
run_demo(ordering, order_id=4)