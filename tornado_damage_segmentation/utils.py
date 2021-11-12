# Various useful functions that don't really belong anywhere else
import os
import rasterio
import fiona
from config import shapefile_dir


# Gets satellite image rasters
def get_rasters(directory):
    rasters = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            rasters.append(rasterio.open(os.path.join(root, file)))
    return rasters


# Get forest disturbance shapefiles
def get_shapefiles():
    shapefiles = []
    for root, dirs, files in os.walk(shapefile_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.shp':
                shapefiles.append(fiona.open(os.path.join(root, file)))
    return shapefiles
