import torch
import dcDataLayer
import networks
import aux
import psnr
import os, sys
import numpy as np
import argparse

PSZ=6 # Log of psize for PSNR (i.e., 64x64 patches)


p = argparse.ArgumentParser()
p.add_argument('--cfa',
               help="which color filter array to use",
               type=str,
               default="lcfa")
p.add_argument('--noise',
               help="additive noise",
               type=float,
               default=0.01)
args = p.parse_args()

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# initiate networks
LCFA = networks.LCFA()
Demosaic = networks.Demosaic()#Demosaic_runDM(device)
LCFA.to(device)
Demosaic.to(device)

Bayer = networks.Bayer()
Bayer.to(device)

CFZ = networks.CFZ()
CFZ.to(device)


LCFA.load_state_dict(torch.load("models/LCFA_3000.pt"))
Demosaic.load_state_dict(torch.load("models/Demosaic_3000.pt"))


data_test = torch.utils.data.DataLoader(dcDataLayer.dcDataLayer('test.txt', test=True),
                                         batch_size=1, shuffle=False, num_workers=0)


# optimization
lossFn = torch.nn.MSELoss()

test_losses = []

LCFA.eval(); Demosaic.eval()
for i, data in enumerate(data_test):
    with torch.no_grad():
        gt, patch =  data["gt"].to(device)[0], data["input"].to(device)[0]
        #gt = img[8:-8,8:-8,:].to(device)
        

        
        '''
        # sample with given cfa and crop out ground truth
        if args.cfa == "bayer":
            simg = aux.bayer(img, args.noise)
        if args.cfa == "cfz":
            simg = aux.cfz(img, args.noise)
        if args.cfa == "lcfa":
            cfa, code  = LCFA(patch)
            code, codef = aux.cfa2code(code)
            np.save("learned_sensor/"+ep+".npy", codef.cpu().numpy())
            sys.exit()
            simg = aux.lcfa(img, args.noise, code, device)
        '''

        if args.cfa == "LCFA":
            cfa, code = LCFA(patch)

        if args.cfa == "Bayer":
            cfa = Bayer(patch, device)

        if args.cfa == "CFZ":
            cfa = CFZ(patch, device)

        noise = torch.tensor(np.float32(np.random.normal(0,1,cfa.shape))* args.noise, device=device)
        cfa = cfa + noise
        
        y = Demosaic(cfa)

        #y = y.cpu().numpy()
        #gt = gt.cpu().numpy()

        val_loss = lossFn(y, gt)
        
        psnr = 10 * torch.log(torch.tensor(1. / val_loss))
        test_losses.append(psnr.data.item())
        
        #print(np.array(test_losses).mean())
print(np.array(test_losses).mean())



