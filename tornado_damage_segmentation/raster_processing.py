# TODO Lots of duplication from rasterio_dataset and create_mask, there's a better and more concise way to do all this.
from datetime import datetime as dt
import numpy as np
import rasterio
import fiona
from shapely.geometry import shape
from config import shapefile_path
from utils import get_rasters


# Takes the filename of a raster and retrieves the date.
def get_date_from_raster(raster):
    filename = raster.name.split('/')[-1]
    return dt.strptime(filename.split('.')[1], '%Y%m%d')


# Load data
rasters = get_rasters()
print(len(rasters))
raster_dates = [get_date_from_raster(raster) for raster in rasters]
shapefile = fiona.open(shapefile_path)
geoms = np.array([feature['geometry'] for feature in shapefile])
stormdates = np.array([dt.strptime(feature['properties']['stormdate'], '%Y-%m-%d') for feature in shapefile])
assert len(geoms) == len(stormdates), "The lengths are not equal? Oh no, the Daleks are here!"

#mask = stormdates == dt(2020, 4, 12)
#geoms = geoms[mask]
#stormdates = stormdates[mask]

# Generate before-and-after images for each tornado within shapefile bounding box
count = 0
for geom, date in zip(geoms, stormdates):
    before = None
    after = None
    bbox = shape(geom).bounds
    for raster_date, raster in zip(raster_dates, rasters):
        if raster_date < date:
            before = raster
        elif raster_date > date:
            after = raster
            break

    if before and after:
        assert before.meta == after.meta, "Metas are not the same! The whole universe will come crashing down!!!"
        out_meta = before.meta.copy()
        window = rasterio.windows.from_bounds(*bbox, transform=before.transform)
        out_trans = before.window_transform(window)
        assert out_trans != before.transform, "What, they shouldn't be equal! The universe is exploding."
        out_meta.update({'width': window.width, 'height': window.height, 'transform': out_trans})
        before = before.read(1, window=window)
        after = after.read(1, window=window)
        diff = after - before
        filename = 'data\\diffs\\diff' + str(count) + '_' + date.isoformat().split('T')[0].replace('-', '')

        # Write diff
        if int(window.width) and int(window.height):
            with rasterio.open(filename + '.tif', 'w', **out_meta) as dst:
                print('here goes')
                dst.write(diff, 1)
                dst.close()

            count += 1
        else:
            print('oh shit')

        # Write version with tornado path mask
        """diff = rasterio.open(filename + '.tif')
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
            dst.close()"""
