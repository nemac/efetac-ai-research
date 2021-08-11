import os
import rasterio
import rasterio.mask
import fiona


# Masks shapefiles and write to new files
def create_masks(rasters, new=False):
    with fiona.open('C:/Users/CyborgOctopus/Downloads/extractDamage2016_2021/new10_2163/new10_2163.shp') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]
    for raster in rasters:
        out_img, out_trans = rasterio.mask.mask(raster, shapes)
        out_meta = raster.meta
        out_meta.update({'driver': 'GTiff', 'height': out_img.shape[1], 'width': out_img.shape[2],
                         'transform': out_trans})
        added = '_new' if new else ''
        with rasterio.open('C:/Users/CyborgOctopus/Downloads/ForWarn_2021' + added + '_masks' + '/' +
                           raster.name.split('/')[-1][:-3] + 'tif', 'w', **out_meta) as dest:
            dest.write(out_img)
            dest.close()


# Loads rasters
def load_rasters(directory):
    return [rasterio.open(directory + '/' + filename) for filename in os.listdir(directory)]


old_rasters = load_rasters('C:/Users/CyborgOctopus/Downloads/ForWarn.2021_X_LC_5YEAR.tar')
new_rasters = load_rasters('C:/Users/CyborgOctopus/Downloads/ForWarn_2021_new')
create_masks(old_rasters)
create_masks(new_rasters, new=True)
