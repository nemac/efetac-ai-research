import os
import numpy as np
import rasterio

# File paths
raster_path = 'C:/Users/CyborgOctopus/Downloads/ForWarn_Data'
mask_path = 'C:/Users/CyborgOctopus/Downloads/ForWarn_Masks'


# Gets satellite image rasters
def get_rasters():
    rasters = []
    for root, dirs, files in os.walk(raster_path):
        for file in files:
            if file[-3:] == 'tif':
                rasters.append(rasterio.open(os.path.join(root, file)))
    return rasters


# Get masks for satellite image rasters
def get_masks():
    for root, dirs, files in os.walk(mask_path):
        return [rasterio.open(os.path.join(root, file)) for file in files]
