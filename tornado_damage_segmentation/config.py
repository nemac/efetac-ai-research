import os

# File paths for data
assets_dir = 'data'
raster_dir = os.path.join(assets_dir, 'ForWarn_Data')
mask_dir = os.path.join(assets_dir, 'ForWarn_Masks')
shapefile_dir = os.path.join(assets_dir, 'shapefiles')
shapefile_path = os.path.join(shapefile_dir, 'polys_4326', 'nws_dat_damage_polys.shp')
models_dir = os.path.join(assets_dir, 'models')  # File paths for saved models
training_data = os.path.join(assets_dir, 'training_data.txt')

# Determines the type of filename in ForWarn_Data and thus the proper function to use in 'parse_filenames.py'
filename_format = 0  # TODO: FIX

# Create necessary directories
for directory in [assets_dir, raster_dir, mask_dir, shapefile_dir, models_dir]:
    if not os.path.exists(directory):
        os.mkdir(directory)
