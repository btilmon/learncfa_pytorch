import numpy as np
import matplotlib.pyplot as plt

plt.subplot(131)
x = np.load("0.npy").reshape(8,8)
print((x==3).sum())

xtmp = np.zeros((8,8,3))
mask = x == 0
xtmp[mask,0] = 255
mask = x == 1
xtmp[mask,1] = 255
mask = x == 2
xtmp[mask,2] = 255
mask = x == 3
xtmp[mask,:] = 255

plt.imshow(xtmp)
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel("0 Iterations")


plt.subplot(132)
x = np.load("20.npy").reshape(8,8)
print((x==0).sum())
xtmp = np.zeros((8,8,3))
mask = x == 0
xtmp[mask,0] = 255
mask = x == 1
xtmp[mask,1] = 255
mask = x == 2
xtmp[mask,2] = 255
mask = x == 3
xtmp[mask,:] = 255

plt.imshow(xtmp)
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel("9,000 Iterations")



plt.subplot(133)
x = np.load("100.npy").reshape(8,8)
print((x==0).sum())
xtmp = np.zeros((8,8,3))
mask = x == 0
xtmp[mask,0] = 255
mask = x == 1
xtmp[mask,1] = 255
mask = x == 2
xtmp[mask,2] = 255
mask = x == 3
xtmp[mask,:] = 255

plt.imshow(xtmp)
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel("46,000 Iterations")


plt.show()
