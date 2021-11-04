# TODO Lots of duplication from rasterio_dataset and create_mask, there's a better and more concise way to do all this.
from datetime import datetime as dt
import numpy as np
import rasterio
import rasterio.mask
import fiona
from fiona.crs import from_epsg
import geopandas as gpd
from shapely.geometry import shape
from config import shapefile_path, get_rasters


# Takes the filename of a raster and retrieves the date.
def get_date_from_raster(raster):
    filename = raster.name.split('/')[-1]
    return dt.strptime(filename.split('.')[1], '%Y%m%d')


# Reads a specified raster within a specified bounding box
def read_raster_within_bounding_box(raster, bounding_box):
    return raster.read(1, window=rasterio.windows.from_bounds(*bounding_box, transform=raster.transform))


# Load data
rasters = get_rasters()
raster_dates = [get_date_from_raster(raster) for raster in rasters]
shapefile = fiona.open(shapefile_path)
geoms = np.array([feature['geometry'] for feature in shapefile])
stormdates = np.array([dt.strptime(feature['properties']['stormdate'], '%Y-%m-%d') for feature in shapefile])

# Generate before-and-after images for each tornado within shapefile bounding box
before = None
after = None
count = 0
for geom, date in zip(geoms, stormdates):
    bbox = shape(geom).bounds
    #geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=rasters[0].crs)
    for raster_date, raster in zip(raster_dates, rasters):
        if raster_date < date:
            before = raster
        elif raster_date > date:
            after = raster
            break
    if before and after:
        out_meta = before.meta
        before = read_raster_within_bounding_box(before, bbox)
        after = read_raster_within_bounding_box(after, bbox)
        diff = after - before
        filename = 'data\\diff' + str(count)

        # Write diff
        with rasterio.open(filename + '.tif', 'w', **out_meta) as dst:
            print('here goes')
            dst.write(diff, 1)
            dst.close()

        # Write version with tornado path mask
        diff = rasterio.open(filename + '.tif')
        out_meta = diff.meta
        mask = rasterio.open(filename + '_mask.tif', 'w', **out_meta)
        mask.write(np.zeros((1,) + diff.shape, dtype=np.uint8))
        diff.close()
        mask.close()
        mask = rasterio.open(filename + '_mask.tif')
        print(geom)
        out_img, out_trans = rasterio.mask.mask(mask, np.array([geom]), invert=True, nodata=255)
        out_meta.update({'transform': out_trans})
        mask.close()
        with rasterio.open(filename + '_mask.tif', 'w', **out_meta) as dst:
            dst.write(out_img)
            dst.close()
    count += 1
