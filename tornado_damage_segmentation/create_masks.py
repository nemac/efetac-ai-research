import os
from datetime import datetime as dt, timedelta
import numpy as np
import rasterio
import rasterio.mask
import fiona
from config import mask_dir, shapefile_path
from utils import get_rasters

# Get shapefile, shapes, and storm dates
shapefile = fiona.open(shapefile_path)
shapes = np.array([feature['geometry'] for feature in shapefile])
stormdates = np.array([dt.strptime(feature['properties']['stormdate'], '%Y-%m-%d') for feature in shapefile])

# Create masks for ForWarn rasters
for raster in get_rasters():
    filename = os.path.basename(raster.name)
    date = dt.strptime(filename.split('.')[1], '%Y%m%d')
    shapes_to_mask = shapes[np.logical_and(date - timedelta(days=183) < stormdates, stormdates < date)]
    # Included to make sure there are actually some shapes that meet the criteria, because otherwise it throws an error
    if len(shapes_to_mask) == 0:
        continue
    out_meta = raster.meta
    blank = rasterio.open(os.path.join(mask_dir, filename), 'w', nbits=1, **out_meta)
    blank.write(np.zeros((1,) + raster.shape, dtype=np.uint8))
    blank.close()
    blank = rasterio.open(os.path.join(mask_dir, filename))
    out_img, out_trans = rasterio.mask.mask(blank, shapes_to_mask, invert=True, nodata=255)
    out_meta.update({'transform': out_trans})
    blank.close()
    with rasterio.open(os.path.join(mask_dir, filename), 'w', nbits=1, **out_meta) as dest:
        dest.write(out_img)
        dest.close()
