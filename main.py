import torch
import dcDataLayer
import networks
import os

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initiate networks
LCFA = networks.LCFA()
Demosaic = networks.Demosaic()
LCFA.to(device)
Demosaic.to(device)

# create dataloader
data = dcDataLayer.dcDataLayer()
dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True, num_workers=0)

# optimization
params = []
params += list(LCFA.parameters())
params += list(Demosaic.parameters())
optim = torch.optim.SGD(params, lr=0.001, momentum=0.9)
lossFn = torch.nn.MSELoss()

# iterate through batches

for epoch in range(1000):
    losses = torch.zeros(1,1)
    for i, data in enumerate(dataloader):
        light, gt = data['light'].to(device), data['gt'].to(device)

        optim.zero_grad()
        CFA = LCFA(light)
        y = Demosaic(CFA)

        #print(CFA.max(), y.max())
        
        loss = lossFn(y, gt)
        loss.backward()
        optim.step()

        losses = torch.cat((losses, loss.cpu().data.view(1,1)))
    print(losses[1:].mean())
