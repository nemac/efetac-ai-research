# TODO Lots of duplication from rasterio_dataset and create_mask, there's a better and more concise way to do all this.
import os
from datetime import datetime as dt
import rasterio
import rasterio.warp
import fiona
from shapely.geometry import shape
from utils import get_rasters
from config import raster_dir, shapefile_path


# Takes the filename of a raster and retrieves the date.
def get_date_from_raster(raster):
    filename = raster.name.split('/')[-1]
    return dt.strptime(filename.split('.')[1], '%Y%m%d')


# Load data
shapefile = fiona.open(shapefile_path)
geoms = [feature['geometry'] for feature in shapefile]
stormdates = [dt.strptime(feature['properties']['stormdate'], '%Y-%m-%d') for feature in shapefile]

assert len(geoms) == len(stormdates), "The lengths are not equal? Oh no, the Daleks are here!"

#mask = stormdates == dt(2020, 4, 12)
#geoms = geoms[mask]
#stormdates = stormdates[mask]

# Generate before-and-after images for each tornado within shapefile bounding box
count = 0
for directory in os.listdir(raster_dir):
    rasters = get_rasters(os.path.join(raster_dir, directory))
    raster_dates = [get_date_from_raster(raster) for raster in rasters]
    for geom, date in zip(geoms, stormdates):
        before = None
        afters = []

        for raster_date, raster in zip(raster_dates, rasters):
            if raster_date < date:
                before = raster
            elif raster_date > date:
                afters.append(raster)
                if len(afters) == 2:
                    break

        if before:
            for after in afters:
                assert before.meta == after.meta, "Metas are not the same! The universe will come crashing down!!!"
                out_meta = before.meta.copy()
                print(shapefile.crs, before.crs)
                print('before: ' + str(shape(geom).bounds))
                bbox = rasterio.warp.transform_bounds(shapefile.crs, before.crs, *shape(geom).bounds)
                print('after: ' + str(bbox))
                window = rasterio.windows.from_bounds(*bbox, transform=before.transform)
                out_trans = before.window_transform(window)
                assert out_trans != before.transform, "What, they shouldn't be equal! The universe is exploding."
                out_meta.update({'width': window.width, 'height': window.height, 'transform': out_trans})
                before_name = os.path.basename(os.path.splitext(before.name)[0])
                after_name = os.path.basename(os.path.splitext(after.name)[0])
                before_arr = before.read(1, window=window)
                after = after.read(1, window=window)
                diff = after - before_arr
                filename = os.path.join('data', 'diffs', 'diff') + str(count) + '.' \
                    + date.isoformat().split('T')[0].replace('-', '') + '.' + before_name + '_' + after_name

                # Write diff
                if int(window.width) and int(window.height):
                    with rasterio.open(filename + '.tif', 'w', **out_meta) as dst:
                        print('here goes')
                        dst.write(diff, 1)
                        dst.close()

                    count += 1
                else:
                    print('aww shucks')
