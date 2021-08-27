import os
import rasterio

# File paths
assets_dir = 'data'
raster_dir = os.path.join(assets_dir, 'ForWarn_Data')
mask_dir = os.path.join(assets_dir, 'ForWarn_Masks')
shapefile_dir = os.path.join(assets_dir, 'extractDamage2016_2021')
shapefile_path = os.path.join(shapefile_dir, 'polys_2163', 'polys_2163.shp')

# Create necessary directories
for directory in [assets_dir, raster_dir, mask_dir, shapefile_dir]:
    if not os.path.exists(directory):
        os.mkdir(directory)


# Gets satellite image rasters
def get_rasters():
    rasters = []
    for root, dirs, files in os.walk(raster_dir):
        for file in files:
            rasters.append(rasterio.open(os.path.join(root, file)))
    return rasters
