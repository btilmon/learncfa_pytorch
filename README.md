## Learning Sensor Multiplexing Design through Back-propagation [PyTorch]

PyTorch implementation of:

Ayan Chakrabarti, "[Learning Sensor Multiplexing Design through Back-propagation](https://github.com/ayanc/learncfa)," Advances in Neural Information Processing Systems (NIPS) 2016. 

See the authors original code if you want Caffe instead of PyTorch. Also see the authors code for copyright details.

This paper has two components that are jointly learned. First, a sensor network learns the optimal color pattern on a digital camera, instead of, for example, the conventional RGGB Bayer pattern. Second, with the predicted color pattern from the sensor network, a reconstruction network learns to reconstruct the RGB image instead of using traditional demosaicking algorithms. The photometric loss between the predicted RGB patch and ground truth RGB patch is then propagated to simultaneously update both the reconstruction network and sensor network.

## Dependencies

Assuming you have [Anaconda](https://www.anaconda.com/products/individual#Downloads) :

```bash
$ conda create -n your_env python=3.7
$ conda activate your_env
```

Tested on Ubuntu 16.04.6 LTS with NVIDIA drivers:
  - pytorch 1.8.1 (omit `cudatoolkit=10.2` if you don't have/want GPU)
  
      `conda install pytorch==1.8.1 torchvision cudatoolkit=10.2 -c pytorch`
      
  - scitkit-image
  
      `conda install scikit-image`


## Prepare Data

While developing this code, the [Gehler-Shi dataset](https://www2.cs.sfu.ca/~colour/data/shi_gehler/) was not available, so the data is provided in `data/raw/` (Thanks to Jon Barron for providing the Gehler-Shi dataset [here](https://github.com/google/ffcc)). This step is already complete, but if you want to run from scratch, run `convert.py` in `data/` to convert the original RAW images into normalized 8-bit PNG files:

```bash
$ cd data
$ python convert.py raw
```

## Training
```bash
$ python main.py
```

## Testing

To test with the learned color filter array and noise level with pretrained weights from the final iteration, run:
```bash
$ python test.py --cfa LCFA --noise 0
```

--cfa options = LCFA, Bayer, CPZ

--noise options = 0, 1e-9, 1e-10, 1e-11

## Visualize Learned Sensors

```bash
$ cd learned_sensors
$ python plot.py
```







