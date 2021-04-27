# auxiliary functions

import torch
import numpy as np

def trunc(img):
    w = img.shape[1]; h = img.shape[0]
    w = (w//8)*8
    h = (h//8)*8
    return img[0:w,0:h,...]


def cfa2code(cfa):
    """converts predicted color filter array to code for reconstruction tests"""
    s = torch.argmax(cfa, axis=0).flatten()
    st = [str(c.data.item()) for c in s]
    return "".join(st), s.float()

def clip(img):
    return torch.maximum(torch.zeros_like(img),
                         torch.minimum(torch.ones_like(img), img))

def bayer(img, noise):
    v = np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
    v[0::2,0::2] = img[0::2,0::2,1]
    v[1::2,1::2] = img[1::2,1::2,1]
    v[1::2,0::2] = img[1::2,0::2,0]
    v[0::2,1::2] = img[0::2,1::2,2]

    v = v/3.0 + np.float32(np.random.normal(0,1,v.shape))*noise

    return clip(v)

def cfz(img, noise):
    v = np.sum(img,axis=2)
    v[0::4,0::4] = img[0::4,0::4,1]
    v[1::4,1::4] = img[1::4,1::4,1]
    v[1::4,0::4] = img[1::4,0::4,0]
    v[0::4,1::4] = img[0::4,1::4,2]

    v = v/3.0 + np.float32(np.random.normal(0,1,v.shape))*noise

    return clip(v)

def lcfa(img, noise, code_str, device):
    code = np.asarray([int(s) for s in code_str],dtype=np.int)
    code.shape = (8,8)

    v = torch.sum(img,axis=2,keepdims=True)
    for i in range(8):
        for j in range(8):
            if code[i,j] < 3:
                v[i::8,j::8,0] = img[i::8,j::8,code[i,j]]

    #v = v/3.0 + np.random.normal(0,1,v.shape))*noise
    v = v/3.0 + torch.normal(0,1,v.size()).to(device)*noise
    return clip(v)[:,:,0]



# Set up index calculations for im2col
def im2col_I(ishape,block,stride):
    bidx = np.array(range(block)).reshape(block,1)*ishape[1]
    bidx = bidx + np.array(range(block))
    bidx.shape = (1,1,block*block)


    idx = np.array(range(ishape[0]*ishape[1]))
    idx.shape = ishape
    idx = idx[0:1-block:stride,0:1-block:stride].copy()
    idx.shape = (idx.shape[0],idx.shape[1],1)
    
    idx = idx + bidx;oshape = idx.shape
    idx.shape = (oshape[0]*oshape[1]*oshape[2],)


    i2c = [(ishape[0]*ishape[1]), idx, oshape]
    return i2c

# Do im2col with pre-computed indices
def im2col(img,i2c):
    out = img.reshape(i2c[0])[i2c[1]]
    out = out.view(i2c[2])
    return out
