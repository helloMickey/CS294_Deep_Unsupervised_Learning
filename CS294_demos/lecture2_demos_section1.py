import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from deepul_helper.data import load_demo_1
from deepul_helper.visualize import plot_hist, plot_train_curves


SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

n_train, d = 1000, 100
loader_args = dict(batch_size=128, shuffle=True)
train_loader = load_demo_1(n_train, 1, d, loader_args, visualize=True, train_only=True)


def train(model, train_loader, optimizer):
    model.train()
    for x in train_loader:
        loss = model.nll(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            loss = model.nll(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss.item()


def train_epochs(model, train_loader, test_loader, train_args):
    epochs, lr = train_args['epochs'], train_args['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()

        train(model, train_loader, optimizer)
        train_loss = eval_loss(model, train_loader)
        train_losses.append(train_loss)

        if test_loader is not None:
            test_loss = eval_loss(model, test_loader)
            test_losses.append(test_loss)

        if epoch in [0, 2, 10, 50, 99]:
            plot_hist(train_loader.dataset.array, bins=d,
                      title=f'Epoch {epoch}', density=model.get_density())
    if test_loader is not None:
      print('Test Loss', test_loss)

    plot_train_curves(epochs, train_losses, test_losses, title='Training Curve')


class Histogram(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.logits = nn.Parameter(torch.zeros(d), requires_grad=True)

    # Compute loss as negative log-likelihood
    def nll(self, x):
        logits = self.logits.unsqueeze(0).repeat(x.shape[0], 1) # batch_size x d
        return F.cross_entropy(logits, x.long())

    def get_density(self):
        x = np.linspace(-0.5, self.d - 0.5, 1000)
        with torch.no_grad():
            y = F.softmax(self.logits, dim=0).unsqueeze(1).repeat(1, 1000 // self.d).view(-1).numpy()
        return x, y


hist = Histogram(d)
train_epochs(hist, train_loader, None, dict(epochs=100, lr=2.5e-2))



class MixtureOfLogistics(nn.Module):
    def __init__(self, d, n_mix=4):
        super().__init__()
        self.d = d
        self.n_mix = n_mix

        self.logits = nn.Parameter(torch.zeros(n_mix), requires_grad=True)
        self.means = nn.Parameter(torch.arange(n_mix).float() / (n_mix - 1) * d, requires_grad=True)
        self.log_scales = nn.Parameter(torch.randn(n_mix), requires_grad=True)

    def nll(self, x):
        x = x.unsqueeze(1).repeat(1, self.n_mix) # b x n_mix
        means, log_scales = self.means.unsqueeze(0), self.log_scales.unsqueeze(0) # 1 x n_mix
        inv_scales = torch.exp(-log_scales)

        plus_in = inv_scales * (x + 0.5 - means)
        min_in = inv_scales * (x - 0.5 - means)

        cdf_plus = torch.sigmoid(plus_in) # CDF of logistics at x + 0.5
        cdf_min = torch.sigmoid(min_in) # CDF of logistics at x - 0.5

        cdf_delta = cdf_plus - cdf_min # probability of x in bin [x - 0.5, x + 0.5]
        log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=1e-12))
        log_cdf_plus = torch.sigmoid(inv_scales * (0.5 - means))
        log_cdf_min = torch.sigmoid(inv_scales * (d - 1.5 - means))

        x_log_probs = torch.where(x < 0.001, log_cdf_plus,
                                  torch.where(x > d - 1 - 1e-3,
                                              log_cdf_min, log_cdf_delta))
        pi_log_probs = F.log_softmax(self.logits, dim=0).unsqueeze(0)
        log_probs = x_log_probs + pi_log_probs
        return -torch.mean(torch.logsumexp(log_probs, dim=1))

    def get_density(self):
        x = np.linspace(-0.5, self.d - 1 + 0.5, 1000)

        with torch.no_grad():
            x_pt = torch.FloatTensor(x).unsqueeze(1)
            means, log_scales = self.means.unsqueeze(0), self.log_scales.unsqueeze(0)
            pi_log_probs = F.log_softmax(self.logits, dim=0).unsqueeze(0)

            # Calculate pdf of logistic distributions and weight them
            # according to learned mixture probabilities
            x_in = (x_pt - means) * torch.exp(-log_scales)
            log_pdf = x_in - log_scales - 2 * F.softplus(x_in)
            log_pdf = log_pdf + pi_log_probs
            log_pdf = torch.logsumexp(log_pdf, dim=1)
            pdf = log_pdf.exp()

        return x, pdf.numpy()


discretized = MixtureOfLogistics(d, n_mix=4)
train_epochs(discretized, train_loader, None, dict(epochs=100, lr=1e-1))


n_train, n_test, d = 1000, 500, 100
loader_args = dict(batch_size=128, shuffle=True)
train_loader, test_loader = load_demo_1(n_train, n_test, d, loader_args, visualize=True)


hist = Histogram(d)
train_epochs(hist, train_loader, test_loader, dict(epochs=100, lr=2.5e-2))


discretized = MixtureOfLogistics(d, n_mix=4)
train_epochs(discretized, train_loader, test_loader, dict(epochs=100, lr=1e-1))
