import os
import numpy as np
from tornado_damage_segmentation.data import get_rasters, array_path

# Save the raw raster data as numpy arrays
for raster in get_rasters():
    name = raster.name.split('/')[-1]
    # name[14:16] gets the last two digits of the year, while name[:-4] gets everything except the '.tif' at the end
    np.save(os.path.join(array_path, 'ForWarn.20' + name[14:16] + '_X_LC_5YEAR', name[:-4]), raster.read(1))
