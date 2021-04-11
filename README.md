## Learning Sensor Multiplexing Design through Back-propagation [PyTorch]

PyTorch implementation of:

Ayan Chakrabarti, "[[Learning Sensor Multiplexing Design through Back-propagation](https://github.com/ayanc/learncfa)]," Advances in Neural Information Processing Systems (NIPS) 2016. 

This paper has two components that are jointly learned. First, a sensor network learns the optimal color pattern on a digital camera, instead of, for example, the conventional RGGB Bayer pattern. Second, with the predicted color pattern from the sensor network, a reconstruction network learns to reconstruct the RGB image instead of using traditional demosaicking algorithms. The photometric loss between the predicted RGB patch and ground truth RGB patch is then propagated to simultaneously update both the reconstruction network and sensor network.

## Requirements

Assuming you have [[Anaconda](https://www.anaconda.com/)]) on your machine: create a conda environment with included dependencies from `environment.yml` and activate the conda environment.

pytorch, numpy, skimage

## Prepare Data

While developing this code, the [[Gehler-Shi dataset](https://www2.cs.sfu.ca/~colour/data/shi_gehler/)]) was not available, so the data is provided in `data/raw/` (Thanks to Jon Barron for providing the Gehler-Shi dataset [[here](https://github.com/google/ffcc)]). Run `convert.py` in `data/` to convert the original RAW images into normalized 8-bit PNG files:

```bash
$ cd data
$ python convert.py raw
```

## Training








