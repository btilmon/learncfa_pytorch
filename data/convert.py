# Process RAW Gehler-Shi data to produce normalized black-level
# corrected 8-bit pngs.

import numpy as np
from skimage.io import imread, imsave
from glob import glob
import sys

flist = sorted(glob(sys.argv[1] + '/*.png'))
for i in range(len(flist)):
    onm = "processed/%06d.png" % (i+1)
    print("Converting " + flist[i] + " to " + onm)

    im = imread(flist[i])
    im = np.float64(im)
    if i >= 86:
        im = np.maximum(0.,im-129.)
    im = im / np.max(im)
    imsave(onm,np.uint8(np.round(im*255.)),check_contrast=False)


