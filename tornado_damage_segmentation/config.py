import os
import rasterio

# File paths for data
assets_dir = 'data'
raster_dir = os.path.join(assets_dir, 'ForWarn_Data')
mask_dir = os.path.join(assets_dir, 'ForWarn_Masks')
shapefile_dir = os.path.join(assets_dir, 'shapefiles')
shapefile_path = os.path.join(shapefile_dir, 'polys_4326', 'nws_dat_damage_polys.shp')
models_dir = os.path.join(assets_dir, 'models')  # File paths for saved models

# Create necessary directories
for directory in [assets_dir, raster_dir, mask_dir, shapefile_dir, models_dir]:
    if not os.path.exists(directory):
        os.mkdir(directory)


# Gets satellite image rasters
def get_rasters():
    rasters = []
    for root, dirs, files in os.walk(raster_dir):
        for file in files:
            rasters.append(rasterio.open(os.path.join(root, file)))
    return rasters
