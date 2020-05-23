import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import append_location, sample_multinomial, quantize, to_grayscale

##################################    RNN    ################################################

class RNN(nn.Module):

    def __init__(self, device, append_loc=False, input_shape=(1, 28, 28), hidden_size=256):
        super().__init__()
        self.device = device
        self.append_loc = append_loc
        self.input_channels = input_shape[0] + 2 if append_loc else input_shape[0]
        self.hidden_size = hidden_size
        self.input_shape = input_shape
        self.canvas_size = input_shape[1] * input_shape[2]

        self.lstm = nn.LSTM(self.input_channels, self.hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, input_shape[0])

    def nll(self, x):
        batch_size = x.shape[0]
        x_inp = append_location(x, self.device) if self.append_loc else x

        # Shift input by one to the right
        x_inp = x_inp.permute(0, 2, 3, 1).contiguous().view(batch_size, self.canvas_size, self.input_channels)
        x_inp = torch.cat((torch.zeros(batch_size, 1, self.input_channels).to(self.device), x_inp[:, :-1]), dim=1)

        h0 = torch.zeros(1, x_inp.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(1, x_inp.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x_inp, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out).squeeze(-1)  # b x 784

        return F.binary_cross_entropy_with_logits(out, x.view(batch_size, -1))


    def sample(self, n):
        with torch.no_grad():
            samples = torch.zeros(n, 1, self.input_channels).to(self.device)
            h = torch.zeros(1, n, self.hidden_size).to(self.device)
            c = torch.zeros(1, n, self.hidden_size).to(self.device)

            for i in range(self.canvas_size):
                x_inp = samples[:, [i]]
                out, (h, c) = self.lstm(x_inp, (h, c))
                out = self.fc(out[:, 0, :])
                prob = torch.sigmoid(out)
                sample_pixel = torch.bernoulli(prob).unsqueeze(-1)  # n x 1 x 1
                if self.append_loc:
                    loc = np.array([i // 28, i % 28]) / 27
                    loc = torch.FloatTensor(loc).to(self.device)
                    loc = loc.view(1, 1, 2).repeat(n, 1, 1)
                    sample_pixel = torch.cat((sample_pixel, loc), dim=-1)
                samples = torch.cat((samples, sample_pixel), dim=1)


            if self.append_loc:
                samples = samples[:, 1:, 0] # only get sampled pixels, ignore location
            else:
                samples = samples[:, 1:].squeeze(-1) # n x 784
            samples = samples.view(n, *self.input_shape)
            return samples.cpu()


##################################    MADE    ################################################

# Code based one Andrej Karpathy's implementation: https://github.com/karpathy/pytorch-made
class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, device, input_shape=(1, 28, 28), hidden_size=[512, 512, 512], ordering=np.arange(784)):
        super().__init__()
        self.nin = input_shape[1] * input_shape[2]
        self.nout = input_shape[1] * input_shape[2]
        self.hidden_sizes = hidden_size
        self.ordering = ordering
        self.device = device
        self.input_shape = input_shape

        # define a simple MLP neural net
        self.net = []
        hs = [self.nin] + self.hidden_sizes + [self.nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        self.m = {}
        self.create_mask()  # builds the initial self.m connectivity

    def create_mask(self):
        L = len(self.hidden_sizes)

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = self.ordering
        for l in range(L):
            self.m[l] = np.random.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def nll(self, x):
        x = x.view(-1, self.nin) # Flatten image
        logits = self.net(x)
        return F.binary_cross_entropy_with_logits(logits, x)

    def sample(self, n):
        samples = torch.zeros(n, self.nin).to(self.device)
        with torch.no_grad():
            for i in range(self.nin):
                logits = self.net(samples)[:, self.ordering[i]]
                probs = torch.sigmoid(logits)
                samples[:, self.ordering[i]] = torch.bernoulli(probs)
            samples = samples.view(n, *self.input_shape)
        return samples.cpu()


##################################    PixelCNN    ################################################

class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, conditional_size=None, **kwargs):
        assert mask_type == 'A' or mask_type == 'B'
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)
        self.conditional_size = conditional_size
        if self.conditional_size:
            if len(conditional_size) == 1:
                self.cond_op = nn.Linear(conditional_size[0], self.out_channels)
            elif len(conditional_size) == 3:
                self.cond_op = nn.Conv2d(conditional_size[0], self.out_channels,
                                         kernel_size=3, padding=1)
            else:
                raise Exception('Invalid conditional_size', conditional_size)

    def forward(self, input, cond=None):
        batch_size = input.shape[0]
        out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        if self.conditional_size:
            if len(self.conditional_size) == 1:
                # Broadcast across height and width of image and add as conditional bias
                out = out + self.cond_op(cond).view(batch_size, -1, 1, 1)
            elif len(self.conditional_size) == 3:
                out = out + self.cond_op(cond)
            else:
                raise Exception()
        return out

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1
        if mask_type == 'B':
            self.mask[:, :, k // 2, k // 2] = 1


class PixelCNN(nn.Module):
    # If conditional_size is None, then the model is not conditional
    def __init__(self, device, input_shape=(1, 28, 28), kernel_size=7, n_layers=7, conditional_size=None):
        super().__init__()
        assert n_layers >= 2
        assert conditional_size is None or isinstance(conditional_size, tuple)
        if conditional_size is not None:
            # 1D conditional (e.g. one-hot vectors for class-conditional)
            if len(conditional_size) == 1:
                self.cond_op = lambda x: x # Identity (no preprocessing)
            # 3D conditional (e.g. subsampled image when performing super-resolution)
            elif len(conditional_size) == 3:
                self.cond_op = nn.Sequential(
                    nn.Conv2d(conditional_size[0], 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU()
                )
                conditional_size = (64,) + conditional_size[1:]

        model = nn.ModuleList([MaskConv2d('A', input_shape[0], 64, kernel_size, padding=kernel_size // 2,
                                          conditional_size=conditional_size), nn.ReLU()])
        for _ in range(n_layers - 2):
            model.extend([MaskConv2d('B', 64, 64, kernel_size, padding=kernel_size // 2,
                                     conditional_size=conditional_size), nn.ReLU()])
        model.append(MaskConv2d('B', 64, input_shape[0], kernel_size, padding=kernel_size // 2,
                                conditional_size=conditional_size))
        self.net = model
        self.device = device
        self.input_shape = input_shape

    def forward(self, x, cond=None):
        if cond is not None:
            cond = self.cond_op(cond)
        out = x
        for layer in self.net:
            if isinstance(layer, MaskConv2d):
                out = layer(out, cond=cond)
            else:
                out = layer(out)
        return out

    def nll(self, x, cond=None):
        return F.binary_cross_entropy_with_logits(self(x, cond=cond), x)

    def sample(self, n, cond=None):
        samples = torch.zeros(n, *self.input_shape).to(self.device)
        with torch.no_grad():
            for r in range(self.input_shape[1]):
                for c in range(self.input_shape[2]):
                    logits = self(samples, cond=cond)[:, :, r, c]
                    probs = torch.sigmoid(logits)
                    samples[:, :, r, c] = torch.bernoulli(probs)
        return samples.cpu()

##################################    WaveNet    ################################################

# Implementation pulled from https://github.com/ryujaehun/wavenet
# Type 'B' Conv
class DilatedCausalConv1d(nn.Module):
    """Dilated Causal Convolution for WaveNet"""
    def __init__(self, mask_type, in_channels, out_channels, dilation=1):
        super(DilatedCausalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=2, dilation=dilation, padding=0)
        self.dilation = dilation
        self.mask_type = mask_type
        assert mask_type in ['A', 'B']

    def forward(self, x):
        if self.mask_type == 'A':
            return self.conv(F.pad(x, [2, 0]))[:, :, :-1]
        else:
            return self.conv(F.pad(x, [self.dilation, 0]))


class ResidualBlock(nn.Module):
    def __init__(self, res_channels, dilation):
        super(ResidualBlock, self).__init__()

        self.dilated = DilatedCausalConv1d('B', res_channels, 2 * res_channels, dilation=dilation)
        self.conv_res = nn.Conv1d(res_channels, res_channels, 1)

    def forward(self, x):
        output = self.dilated(x)

        # PixelCNN gate
        o1, o2 = output.chunk(2, dim=1)
        output = torch.tanh(o1) * torch.sigmoid(o2)
        output = x + self.conv_res(output) # Residual network

        return output


class WaveNet(nn.Module):
    def __init__(self, device, append_loc):
        super(WaveNet, self).__init__()

        in_channels = 3 if append_loc else 1
        out_channels = 1
        res_channels = 32
        layer_size = 5 # Largest dilation is 16
        stack_size = 2

        self.causal = DilatedCausalConv1d('A', in_channels, res_channels, dilation=1)
        self.res_stack = nn.Sequential(*sum([[ResidualBlock(res_channels, 2 ** i)
                                         for i in range(layer_size)] for _ in range(stack_size)], []))
        self.out_conv = nn.Conv1d(res_channels, out_channels, 1)
        self.append_loc = append_loc
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        if self.append_loc:
            x = append_location(x, self.device)
        output = x.view(batch_size, -1, 784)
        output = self.causal(output)
        output = self.res_stack(output)
        output = self.out_conv(output)
        return output.view(batch_size, 1, 28, 28)

    def nll(self, x):
        logits = self(x)
        return F.binary_cross_entropy_with_logits(logits, x)

    def sample(self, n):
        with torch.no_grad():
            samples = torch.zeros(n, 1, 28, 28).to(self.device)
            for r in range(28):
                for c in range(28):
                    logits = self(samples)[:, :, r, c]
                    probs = torch.sigmoid(logits)
                    samples[:, :, r, c] = torch.bernoulli(probs)
        return samples.cpu()


##################################    Self-Attention    ################################################
# Implemented from https://github.com/jadore801120/attention-is-all-you-need-pytorch/

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=784):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0) * 0.1

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, mask=None):
        dec_output = self.slf_attn(dec_input, dec_input, dec_input, mask=mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output

class Transformer(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, device, mode='none'):

        super().__init__()
        n_layers = 2
        self.input_size = 3 if mode == 'pixel_location' else 1

        if mode == 'pos_encoding':
            self.pos_enc = PositionalEncoding(1, n_position=784)
        self.fc_in = nn.Linear(self.input_size, 64)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(64, 64, 1, 16, 64, dropout=0.1)
            for _ in range(n_layers)])
        self.fc_out = nn.Linear(64, 1)

        self.register_buffer('mask', torch.zeros(784, 784))
        for i in range(784):
            self.mask[i, :i] = 1

        self.mode = mode
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        if self.mode == 'pixel_location':
            x = append_location(x, self.device)
            x = x.permute(0, 2, 3, 1).view(batch_size, 784, self.input_size)
        elif self.mode == 'pos_encoding':
            x = x.view(batch_size, 784, self.input_size)
            x = self.pos_enc(x)
        else:
            x = x.view(batch_size, 784, self.input_size)
        x = torch.cat((torch.zeros(batch_size, 1, self.input_size).to(self.device), x[:, :-1]), dim=1)
        # -- Forward
        x = F.relu(self.fc_in(x))
        for i, dec_layer in enumerate(self.layer_stack):
            x = dec_layer(x, mask=self.mask)
        x = self.fc_out(x)
        x = x.view(batch_size, 1, 28, 28)
        return x

    def nll(self, x):
        logits = self(x)
        return F.binary_cross_entropy_with_logits(logits, x)

    def sample(self, n):
        samples = torch.zeros(n, 1, 28, 28).to(self.device)
        with torch.no_grad():
            for r in range(28):
                for c in range(28):
                    logits = self(samples)[:, :, r, c]
                    probs = torch.sigmoid(logits)
                    samples[:, :, r, c] = torch.bernoulli(probs)
        return samples.cpu()


##################################    Grayscale PixelCNN    ################################################

class PixelCNN2(nn.Module):
    def __init__(self, device, n_layers, n_channels, n_color_bits, conditional=False):
        super().__init__()
        assert n_layers >= 2
        if conditional:
            # Condition on grayscale image which is (1, 28, 28)
            self.cond_op = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()
            )
            conditional_size = (64, 28, 28)
        else:
            conditional_size = None

        # Build model
        output_dim = n_channels * 2 ** n_color_bits
        model = nn.ModuleList([MaskConv2d('A', n_channels, 64, 7, padding=3, conditional_size=conditional_size), nn.ReLU()])
        for _ in range(n_layers - 2):
            model.extend([MaskConv2d('B', 64, 64, 7, padding=3, conditional_size=conditional_size), nn.ReLU()])
        model.append(MaskConv2d('B', 64, output_dim, 7,padding=3, conditional_size=conditional_size))
        self.net = model

        self.device = device
        self.n_channels = n_channels
        self.color_dim_size = 2 ** n_color_bits
        self.conditional = conditional

    def forward(self, x, cond=None):
        if cond is not None:
            cond = self.cond_op(cond)
        out = x
        for layer in self.net:
            if isinstance(layer, MaskConv2d):
                out = layer(out, cond=cond)
            else:
                out = layer(out)

        batch_size = out.shape[0]
        out = out.view(batch_size, self.color_dim_size, self.n_channels, 28, 28)
        return out

    def nll(self, x, cond=None):
        # Scale x from [0, 1] to [0, self.color_dim_size - 1]
        target = (x * (self.color_dim_size - 1)).long()
        return F.cross_entropy(self(x, cond=cond), target)

    def sample(self, n, cond=None):
        samples = torch.zeros(n, self.n_channels, 28, 28).to(self.device)
        with torch.no_grad():
            for r in range(28):
                for c in range(28):
                    logits = self(samples, cond=cond)[:, :, :, r, c]
                    sample = sample_multinomial(logits, dim=1)
                    sample /= self.color_dim_size -1 # Scale to [0, 1]
                    samples[:, :, r, c] = sample
        return samples.cpu()

class GrayscalePixelCNN(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.input_size = (3, 28, 28) # Colored MNIST
        self.device = device

        self.grayscale_prior = PixelCNN2(device, n_layers=8, n_channels=1,
                                         n_color_bits=1, conditional=False)
        self.color_conditional = PixelCNN2(device, n_layers=8, n_channels=3,
                                           n_color_bits=2, conditional=True)

    def nll(self, x):
        x_gray = quantize(to_grayscale(x), 1) # convert to grayscale and quantize to 1 bits
        x_color = quantize(x, 2) # quantize to 2 bits

        nll_gray = self.grayscale_prior.nll(x_gray)
        nll_color = self.color_conditional.nll(x_color, cond=x_gray)

        return nll_gray + nll_color

    def sample(self, n):
        gray_samples = self.grayscale_prior.sample(n)
        color_samples = self.color_conditional.sample(n, cond=gray_samples.to(self.device))
        return gray_samples, color_samples


##################################    Parallel PixelCNN    ################################################

class CNN(nn.Module):

    def __init__(self, in_channels, out_channels, n_layers, filter_size=64, kernel_size=3):
        super().__init__()
        model = []
        h_prev = in_channels
        for h in [filter_size] * (n_layers - 1) + [out_channels]:
            model.append(nn.Conv2d(h_prev, h, kernel_size=kernel_size, padding=kernel_size // 2))
            model.append(nn.ReLU())
            h_prev = h
        model.pop()
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x)



class ParallelPixelCNN(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.base_pixelcnn = PixelCNN(device, input_shape=(1, 7, 7), kernel_size=3, n_layers=7)
        self.group_nets = nn.ModuleList([CNN(1, 1, 4, kernel_size=7),
                                         CNN(2, 1, 4, kernel_size=7),
                                         CNN(3, 1, 4, kernel_size=7)])

        self.base_size = 7
        self.input_shape = (1, 224, 224)

    def nll(self, x):
        nll = 0
        n_scale_iter = int(np.log2(x.shape[2] // self.base_size))
        for i in range(n_scale_iter):
            stride = 2 ** (n_scale_iter - i)
            offset = stride // 2
            current_x = x[:, :, ::stride, ::stride] # Top-left group 1
            if i == 0:
                nll = nll + self.base_pixelcnn.nll(current_x)

            for j in range(3):
                if j == 0:
                    target = x[:, :, ::stride, offset::stride] # Top-right group 2
                elif j == 1:
                    target = x[:, :, offset::stride, ::stride] # Bottom-left group 3
                else:
                    target = x[:, :, offset::stride, offset::stride] # Bottom-right group 4
                logits = self.group_nets[j](current_x)
                nll = nll + F.binary_cross_entropy_with_logits(logits, target)
                current_x = torch.cat((current_x, target), dim=1)
        return nll


    def sample(self, n):
        with torch.no_grad():
            samples = torch.zeros(n, *self.input_shape).to(self.device)
            n_scale_iter = int(np.log2(samples.shape[2] // self.base_size))
            for i in range(n_scale_iter):
                stride = 2 ** (n_scale_iter - i)
                offset = stride // 2
                if i == 0:
                    samples[:, :, ::stride, ::stride] = self.base_pixelcnn.sample(n)

                current_input = samples[:, :, ::stride, ::stride]
                for j in range(3):
                    logits = self.group_nets[j](current_input)
                    probs = torch.sigmoid(logits)
                    s = torch.bernoulli(probs)
                    current_input = torch.cat((current_input, s), dim=1)

                    if j == 0:
                        samples[:, :, ::stride, offset::stride] = s
                    elif j == 1:
                        samples[:, :, offset::stride, ::stride] = s
                    else:
                        samples[:, :, offset::stride, offset::stride] = s
            return samples.cpu()



