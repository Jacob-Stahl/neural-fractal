import torch
from torch import nn

class LogDecoder(nn.Module):
    def init(self,in_features, out_features, scale_range = 32, scale_shift = 1):
        super.__init__()

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
                out[i] = out[i] + mags[i] * 2 ** -(j + self.scale_shift)

        return outs
        