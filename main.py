import torch
import dcDataLayer
import networks
import os, sys
#import wandb
import numpy as np
import aux

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# logging
#wandb.init(project='SIGMOID')

# initiate networks
LCFA = networks.LCFA()
Demosaic = networks.Demosaic()
LCFA.to(device)
Demosaic.to(device)

# create dataloaders
data = dcDataLayer.dcDataLayer('train.txt')
batch = len(data)
data_train = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=0)
data = dcDataLayer.dcDataLayer('val.txt')
batch = len(data)
data_val = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=False, num_workers=0)
data = dcDataLayer.dcDataLayer('test.txt')
batch = len(data)
data_test = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=False, num_workers=0)

'''
ep = "100"
LCFA.load_state_dict(torch.load("models/LCFA_"+ep+".pt"))
Demosaic.load_state_dict(torch.load("models/Demosaic_"+ep+".pt"))
'''

# optimization
params = []
params += list(LCFA.parameters())
params += list(Demosaic.parameters())
optim = torch.optim.SGD(params, lr=0.001, momentum=0.9)
lossFn = torch.nn.MSELoss()

iterations = [2500, 5000, 7500, 10000, 12500, 25000,
              100000, 200000, 300000, 400000, 500000, 600000,
              1000000, 1100000, 1200000, 1300000, 1400000, 1500000]


#print('epochs', int(1.5e6))
#print(len(data_train), len(data_val))

global_train, global_val = [], []
train_losses, val_losses = torch.zeros(1,1), torch.zeros(1,1)
for epoch in range(int(1.5e6)//len(data_train)):
    LCFA.train(); Demosaic.train()
    for i, data in enumerate(data_train):
        x, gt = data['input'].to(device)[0], data['gt'].to(device)[0]

        optim.zero_grad()
        CFA, code = LCFA(x)
        code, codef = aux.cfa2code(code)
        #np.save("learned_sensor/"+ep+".npy", codef.cpu().numpy())
        #print(i)
        y = Demosaic(CFA)

        loss = lossFn(y, gt)
        loss.backward()
        optim.step()
        train_losses = torch.cat((train_losses, loss.cpu().data.view(1,1)))
        #print("{0:8o}".format())
        #print(train_losses.mean(), (codef==3).sum())

    '''
    LCFA.eval(); Demosaic.eval()
    for i, data in enumerate(data_val):
        with torch.no_grad():
            x, gt = data['input'].to(device)[0], data['gt'].to(device)[0]
            CFA, _ = LCFA(x)
            y = Demosaic(CFA)

            loss = lossFn(y, gt)
            val_losses = torch.cat((val_losses, loss.cpu().data.view(1,1)))


    if not epoch % 20:

        global_train.append(train_losses[1:].mean())
        global_val.append(val_losses[1:].mean())
        train_loss = np.array(global_train).mean()
        val_loss = np.array(global_val).mean()
        val_psnr = 10 * torch.log(torch.tensor(1. / val_loss))
        
        print("{0:6o} | train {1:.6f} | val {2:.6f} | val PSNR {3:3.3f}".
              format(epoch,
                     train_loss,
                     val_loss,
                     val_psnr))


        i = torch.randint(0,120,(1,)).data.item()
        wandb.log({
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Val PSNR": val_psnr,
            "images/gt": [wandb.Image(gt[i].cpu().detach().numpy().transpose(1,2,0), caption="Label")],
            "images/pred": [wandb.Image(y[i].cpu().detach().numpy().transpose(1,2,0), caption="Label")]})

        torch.save(LCFA.state_dict(), 'models/LCFA_'+str(epoch)+'.pt')
        torch.save(Demosaic.state_dict(), 'models/Demosaic_'+str(epoch)+'.pt')
    '''
