import os
import numpy as np
from torch.utils.data import Dataset
import rasterio


class RasterioDataset(Dataset):
    """
    Defines a dataset for reading rasterio objects. Relies on corresponding images and masks having the same
    filename and the mask directory being flat (no subfolders). If there is no corresponding mask for an image, or
    vice versa, that image will not be included. Assumes rasters have only one band. \n

    Parameters: \n
    image_path - the path to the folder containing raster images\n
    mask_path - the path to the folder containing raster masks\n
    bbox_coords - the coordinates of a rectangular geospatial bounding box defining what part of the raster data will
    be read. They should be expressed as a 4-dimensional vector like (top_left_corner_x, top_left_corner_y,
    bottom_right_corner_x, bottom_right_corner_y). If None, the entire data will be read (default: None)\n
    transform - the transform to use on each raster when it's read (default: None)
    """

    def __init__(self, image_path, mask_path, bbox_coords=None, transform=None):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.bbox_coords = bbox_coords
        self.transform = transform
        self.rasters = []
        self.load_rasters()

    def __len__(self):
        return len(self.rasters)

    def __getitem__(self, index):
        # expand_dims used to create a single color channel in the channels-last style
        item = [np.expand_dims(self.read_within_bounding_box(raster), axis=2) for raster in self.rasters[index]]
        if self.transform:
            item = [self.transform(arr) for arr in item]
        return tuple(item)

    # Reads a raster, but only returns the values within the rectangular bounding box specified by bbox_coords, if
    # provided. Otherwise, returns all values.
    def read_within_bounding_box(self, raster):
        bbox = raster.bounds
        if self.bbox_coords:
            bbox = self.bbox_coords
        return raster.read(1, window=rasterio.windows.from_bounds(*bbox, transform=raster.transform))

    # Generates tuples of data, mask pairs
    def load_rasters(self):
        for path, dirs, files in os.walk(self.image_path):
            for file in files:
                try:
                    self.rasters.append((rasterio.open(os.path.join(path, file)),
                                         rasterio.open(os.path.join(self.mask_path, file))))
                except rasterio.errors.RasterioIOError:
                    pass  # If there is no corresponding mask file, skip this image
