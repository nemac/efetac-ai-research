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


# Get raster locations from text file
# The format is as follows:
# Blocks are separated by two blank lines
# Each block is a list of file paths, each on its own line.
# Optionally, blocks may begin with a root directory for the file paths in that block
# This is separated from the rest of the block by a single blank line.
# Example:
# ________________________________
# C:/Users/CyborgOctopus/Documents
#
# my_stuff/thing
# my_stuff/thang
# thingy
#
#
# a_folder
#
# a_file
#
#
# C:/Users/CyborgOctopus/thingamajig
# widget
# gadget
def get_rasters_from_file(text_file):
    rasters = []
    with open(text_file) as f:
        contents = f.read()
    contents = [content.split('\n\n') for content in contents.split('\n\n\n')]
    for content in contents:
        if len(content) == 1:
            content = [''] + content
        rasters += [rasterio.open(os.path.join(content[0], file_name + '.tif')) for file_name in content[1].split('\n')]
    return rasters
