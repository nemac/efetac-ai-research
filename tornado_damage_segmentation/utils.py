# Various useful functions that don't really belong anywhere else
import os
import rasterio
from config import raster_dir


# Gets satellite image rasters
def get_rasters():
    rasters = []
    for root, dirs, files in os.walk(raster_dir):
        for file in files:
            rasters.append(rasterio.open(os.path.join(root, file)))
    return rasters
