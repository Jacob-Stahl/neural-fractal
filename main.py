import numpy
import torch
from model import FractalApproximator


model = FractalApproximator()

X = torch.tensor([0.3544324, 7.32423432])
X = X.unsqueeze_(0)
Y = model(X)
print(Y)