import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# layer for learning CFA
class LCFA(nn.Module):
    def __init__(self):
        super(LCFA, self).__init__()

        # learnable softmax
        self.pi = nn.Parameter(torch.normal(0, 0.1, (1,4,8,8)))
        self.pi.requires_grad = True

        # (alpha temperature)
        self.alpha = 0
        self.fac = torch.sqrt(torch.tensor(399.0))/8e5

    def forward(self, x):
        self.lcfa = 1. + (self.alpha*self.fac)**2
        self.alpha += 1
        pi = self.pi - torch.max(self.pi, axis=1, keepdims=True)[0]
        pi = torch.exp(pi*self.lcfa)
        pi = pi / torch.sum(pi, axis=1, keepdims=True)
        return torch.sum( pi.tile((1,1,3,3)) * x, axis=1, keepdims=True ) 

# layer for learning to reconstruct 
class Demosaic(nn.Module):
    def __init__(self):
        super(Demosaic, self).__init__()

        # Interpolation layers
        self.fc0 = nn.Linear(576, 4608, bias=False)
        nn.init.normal_(self.fc0.weight, std=0.001)
        self.conv1x1 = nn.Conv2d(72, 72, (1,1))

        # Convolution layers
        self.c0 = nn.Conv2d(1, 128, (8,8), 8)
        self.c1 = nn.Conv2d(128, 128, (2,2), 1)
        self.c2 = nn.Conv2d(128, 128, (2,2), 1)
        self.fc1 = nn.Linear(128, 4608)
        
    def forward(self, x):

        # Compute interpolation path
        flat = torch.log(x.view(-1, 24*24) + 1e-8)
        fc0 = torch.exp(self.fc0(flat))
        fc0 = fc0.view(-1, 72, 8, 8)
        f = self.conv1x1(fc0)

        # Compute convolution path 
        c0 = F.relu(self.c0(x))
        c1 = F.relu(self.c1(c0))
        c2 = F.relu(self.c2(c1))
        gate = F.relu(self.fc1(c2.view(-1, 128))).view(-1, 72, 8, 8)

        # Reshape to preserve color channels
        # Compute inner product
        f = f.view(-1, 3, 24, 8, 8)
        gate = gate.view(-1, 3, 24, 8, 8)
        y = torch.sum( f * gate, axis=2, keepdims=True ).squeeze(2)
        return y









        
