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
        light, gt = sample['input'], sample['gt']
        return {'input': torch.from_numpy(light),
                'gt': torch.from_numpy(gt)}

class dcDataLayer(Dataset):
    """transform images for sensor network"""
    def __init__(self, txt_file, test=None):
        self.transform = ToTensor()
                # Read list of files
        self.flist = ['data/processed/00'+line.rstrip('\n') for
                      line in open( "data/" + txt_file)]

        #self.flist = sorted(glob('data/processed/*.png'))

        # create dimension matrix for altering input to 4D
        self.cspace = np.zeros((4,3),dtype=np.float32)
        self.cspace[0:3, 0:3] = np.eye(3,dtype=np.float32)
        self.cspace[3, :] = np.ones((1,3),dtype=np.float32)
        self.cspace = self.cspace/3.

        self.batch_size = 128
        self.chunk_size = 10
        self.chunk_repeat = 100
        self.imid, self.chunk_id = 0, 0

        self.val = False
        self.test = False
        if txt_file != "train.txt":
            self.val = True
            self.chunk_size = 1
            self.chunk_repeat = 1
            self.batch_size = 1000

            if test is not None:
                self.test = True

            
        else:
            np.random.shuffle(self.flist)

            
    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):

        if self.test:
            full_img = np.float32(imread(self.flist[idx]))/255.
            full_img = torch.from_numpy(full_img)        

        if self.chunk_id == 0:
            self.imgs = []
            for i in range(self.chunk_size):
                self.imgs.append(np.float32(imread(self.flist[self.imid]))/255.)
                self.imid = (self.imid+1) % len(self.flist)
            if self.val:
                shp = self.imgs[0].shape
                nump = (shp[0]-24)*(shp[1]-24)
                pidx = [int(i) for i in np.linspace(0, nump-1, self.chunk_repeat*self.batch_size)]
                self.xs = [i%(shp[0]-24) for i in pidx]
                self.ys = [int(i/(shp[0]-24)) for i in pidx]
                self.pidx = 0


        self.chunk_id = (self.chunk_id+1)%self.chunk_repeat

        network_in = torch.zeros(self.batch_size, 4, 24, 24)
        network_gt = torch.zeros(self.batch_size, 3, 8, 8)
        for i in range(self.batch_size):
            # Find random image and location
            if self.val:
                iid = 0; x = self.xs[self.pidx]; y = self.ys[self.pidx]
                self.pidx += 1
                
            else:
                iid = torch.randint(self.chunk_size, (1,1))
                x = torch.randint(self.imgs[iid].shape[0]-24, (1,1))
                y = torch.randint(self.imgs[iid].shape[1]-24, (1,1))

            # Permute array dims
            im = self.imgs[iid][x:(x+24), y:(y+24), :].transpose(2,0,1).copy()

            # Crop center patch as gt output
            gt = im[:, 8:16, 8:16].reshape(3,8,8)

            # Creat input to sensor, add noise
            im = im.reshape(3, 24*24); im = np.dot(self.cspace, im)
            im += np.random.normal(0, 0.01, im.shape)
            np.maximum(0.,im,im); np.minimum(1.,im,im);
            im = im.reshape(4, 24, 24)
            transformed = self.transform({'input': im, 'gt': gt})
            network_in[i] = transformed['input']
            network_gt[i] = transformed['gt']            

        if self.test:
            return {"input":network_in,
                    "gt":network_gt,
                    "full_img":full_img}
        else:
            return {"input":network_in, "gt":network_gt}

        
