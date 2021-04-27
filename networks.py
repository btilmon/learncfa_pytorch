import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import aux
import sys

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
        return torch.sum( pi.tile((1,1,3,3)) * x, axis=1, keepdims=True ), pi[0]

# layer for learning to reconstruct 
class Demosaic(nn.Module):
    def __init__(self):
        super(Demosaic, self).__init__()

        # Interpolation layers
        self.fc0 = nn.Linear(576, 4608, bias=False)
        nn.init.normal_(self.fc0.weight, std=0.001)
        self.conv1x1 = nn.Conv2d(72, 72, (1,1))
        nn.init.xavier_uniform_(self.conv1x1.weight)
        
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
        y = torch.sigmoid(y)
        return y


# Sense as bayer pattern
class Bayer(nn.Module):
    def __init__(self):
        super(Bayer, self).__init__()
        
    def forward(self, x, device):

        y = torch.zeros(x.shape[0],1,x.shape[2],x.shape[3], device=device)
    
        y[:,0,0:24:2,0:24:2] = x[:,1,0:24:2,0:24:2]
        y[:,0,1:24:2,1:24:2] = x[:,1,1:24:2,1:24:2]
        y[:,0,1:24:2,0:24:2] = x[:,0,1:24:2,0:24:2]
        y[:,0,0:24:2,1:24:2] = x[:,2,0:24:2,1:24:2]
        return y


# Sense as CFZ pattern
class CFZ(nn.Module):
    def __init__(self):
        super(CFZ, self).__init__()
        
    def forward(self, x, device):
        y = torch.zeros(x.shape[0],1,x.shape[2],x.shape[3], device=device)
        y[:,0,:,:] = x[:,3,:,:]
        y[:,0,0:24:4,0:24:4] = x[:,1,0:24:4,0:24:4]
        y[:,0,1:24:4,1:24:4] = x[:,1,1:24:4,1:24:4]
        y[:,0,1:24:4,0:24:4] = x[:,0,1:24:4,0:24:4]
        y[:,0,0:24:4,1:24:4] = x[:,2,0:24:4,1:24:4]
        return y

'''    
# layer for learning to reconstruct 
class Demosaic_runDM(nn.Module):
    """for test data"""
    def __init__(self, device):
        super(Demosaic_runDM, self).__init__()

        # gpu or cpu
        self.device = device
        
        # Interpolation layers
        self.fc0 = nn.Linear(576, 4608, bias=False)
        nn.init.normal_(self.fc0.weight, std=0.001)
        self.conv1x1 = nn.Conv2d(72, 72, (1,1))
        nn.init.xavier_uniform_(self.conv1x1.weight)
        
        # Convolution layers
        self.c0 = nn.Conv2d(1, 128, (8,8), 8)
        self.c1 = nn.Conv2d(128, 128, (2,2), 1)
        self.c2 = nn.Conv2d(128, 128, (2,2), 1)
        self.fc1 = nn.Linear(128, 4608)
        
        
    def p2im(self, img):
        w = img.shape[0]; h = img.shape[1]
        out = torch.zeros((w*8,h,8*3)).to(self.device)
        for i in range(8):
            out[i::8,...] = img[:,:,(i*24):((i+1)*24)]
        out = out.view(w*8,h*8,3)
        return out

        
        
    def forward(self, x):
        a = aux.im2col_I(x.shape, 24, 8)
        b = aux.im2col_I(x.shape, 8, 8)
        
        x1 = aux.im2col(x, a)
        x2 = aux.im2col(x, b)

        wA = x1.shape[0]; wB = x2.shape[0]
        hA = x1.shape[1]; hB = x2.shape[1]

        x1 = x1.unsqueeze(0).reshape(1,1,-1,576).to(self.device)
        x2 = x2.unsqueeze(0).reshape(1,1,-1,64).to(self.device)


        print('a',x1.shape, x2.shape)

        # Compute interpolation path
        #flat = torch.log(x.view(-1, 24*24) + 1e-8)
        flat = torch.log(x1 + 1e-8)
        fc0 = torch.exp(self.fc0(flat))
        print('b',fc0.shape)
        #fc0 = fc0.view(-1, 72, 8, 8)
        fc0 = fc0.view(1, 72, -1, 1)
        print('c',fc0.shape)
        f = self.conv1x1(fc0)
        print('d',f.shape)
        f = f.view(wA,hA,64,3,24)
        print('e', f.shape)


        # Compute convolution path
        
        print('f', x2.shape)
        c0 = F.relu(self.c0(x2))
        c0 = c0.view(1, 128, hB, wB)
        print('g', c0.shape)
        c1 = F.relu(self.c1(c0))
        print('h', c1.shape)
        c2 = F.relu(self.c2(c1))
        print('i',c2.shape)

        #gate = F.relu(self.fc1(c2.view(-1, 128))).view(-1, 72, 8, 8)
        gate = F.relu(self.fc1(c2.view(1,46,30,128)))#.view(-1, 72, 8, 8)
        print('j', gate.shape)

        gate = gate.view(f.size())
        print('k',gate.shape)

        product = torch.sum(f*gate, axis=4)
        product = product.view(wA, hA, 64*3)
        product = self.p2im(product)
        product = torch.sigmoid(product)
        print('L', product.shape)
        return product
'''





        
