import numpy as np

ep = "800"
# 0,100,600,800
x = np.load("learned_sensor/"+ep+".npy").reshape(8,8)
print(ep, (x==3).sum())

