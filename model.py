import torch
from torch import nn

class LogDecoder(nn.Module):
    def __init__(self,in_features, out_features, scale_range = 10, scale_shift = 1):

        self.in_features = in_features
        self.out_features = out_features
        self.scale_range = scale_range
        self.scale_shift = scale_shift
        self.fcs = [nn.Linear(in_features, scale_range)] * out_features

    def forward(self, x):
        out = torch.zeros((self.out_features, x.size()[1]), dtype= torch.double)

        for i in range(self.out_features):
            mags = torch.tanh(self.fcs[i](x))

            for j in range(self.scale_range):
                out[i] = out[i] + mags[i] * 10 ** -(j - self.scale_shift)

        return out

class LogEncoder(nn.Module):
    def __init__(self, in_features, out_features, scale_range = 32, scale_shift = 1):

        self.in_features = in_features
        self.out_features = out_features
        self.scale_range = scale_range
        self.scale_shift = scale_shift
        self.fcs = [nn.Linear(scale_range, out_features)] * out_features

    def forward(self, x):

        out = torch.zeros(size = (self.out_features, x.size()[1]))

        for i in range(self.in_features):
            mags = torch.zeros((self.scale_range, x.size()[1]))

            x[i] = x[i] * 10 ** (-self.scale_shift)
            for j in range(self.scale_range):
                mags[i] = x[i]
                x[i] = x[i] - x[i] // 1
                x[i] = x[i] * 10

            out += self.fcs[i](mags)

        return out

class FractalApproximator(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = LogEncoder(2,64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.decoder = LogDecoder(64,2)

    def forward(self, x):

        x = torch.tanh(self.encoder(x))
        x = torch.selu(self.fc1(x))
        x = torch.selu(self.fc2(x))
        x = self.decoder(x)