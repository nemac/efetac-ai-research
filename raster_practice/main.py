import numpy as np
import torch
from torch import nn
import rasterio

# reading data
dataset = rasterio.open('C:/Users/CyborgOctopus/Downloads/ALCLAEA.20200819.1-yr-baseline.img')
band1 = dataset.read(1)

# thresholding
band1 = int(band1 > np.mean(band1))
new_dataset = rasterio.open('threshold.tiff', 'w', )
