import numpy
import torch
from torch.utils import data
from torch import distributions

class JuliaSampler(data.Dataset):
    def __init__(self, seed = [0.285, 0.01], real_range = [-2, 2], imag_range = [-2, 2]):
        self.seed = seed
        self.real_range = real_range
        self.imag_range = imag_range
        self.real_dis = distributions.Uniform(real_range[0], real_range[1])
        self.imag_dis = distributions.Uniform(imag_range[0], imag_range[1])

    def __len__(self):
        return 1

    def __getitem__(self, index):
        
        Z0 = self.real_dis.sample(2)

        Z1 = torch.zeros(2)
        Z1[0] = Z0[0] ** 2 - Z0[1] ** 2 + seed[0]
        Z1[1] = 2 * Z0[0] * Z0[1] + seed[1]

        return Z0, Z1