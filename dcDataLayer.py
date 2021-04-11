'''
Transform data for input to sensor network
'''

import torch
from torch.utils.data import Dataset
from glob import glob
from skimage.io import imread
import numpy as np
import sys

class ToTensor(object):
    def __call__(self, sample):
        light, gt = sample['light'], sample['gt']
        return {'light': torch.from_numpy(light),
                'gt': torch.from_numpy(gt)}

class dcDataLayer(Dataset):
    """transform images for sensor network"""
    def __init__(self):
        self.transform = ToTensor()
        self.flist = sorted(glob('data/processed/*.png'))

        # create dimension matrix for altering input to 4D
        self.cspace = np.zeros((4,3),dtype=np.float32)
        self.cspace[0:3, 0:3] = np.eye(3,dtype=np.float32)
        self.cspace[3, :] = np.ones((1,3),dtype=np.float32)
        self.cspace = self.cspace/3.

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        img = np.float32(imread(self.flist[idx]))/255.
        
        # random crop
        x = np.random.randint(img.shape[0]-24)
        y = np.random.randint(img.shape[1]-24)
        
        # Permute array dims
        im = img[x:(x+24), y:(y+24), :].transpose(2,0,1).copy()

        # Crop center patch as gt output
        gt = im[:, 8:16, 8:16].reshape(3,8,8)

        # Creat input to sensor, add noise
        im = im.reshape(3, 24*24); im = np.dot(self.cspace, im)
        im += np.random.normal(0, 0.01, im.shape)
        np.maximum(0.,im,im); np.minimum(1.,im,im);
        im = im.reshape(4, 24, 24)
        return self.transform({'light': im, 'gt': gt})
        


        
