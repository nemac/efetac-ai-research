import os
import numpy as np
import torch
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
    be read. They should be expressed as a 4-dimensional vector like (left, bottom, right, top). If None, the entire
    # data will be read (default: None)\n
    transform - the transform to use on each raster when it's read (default: None) \n
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
        item = [self.read_within_bounding_box(raster) for raster in self.rasters[index]]
        # expand_dims used to create a single color channel in the channels-last style
        item[0] = np.expand_dims(item[0], axis=2)
        if self.transform:
            item[0] = self.transform(item[0])
        item[1] = torch.from_numpy(item[1]).type(torch.LongTensor)  # Cast to long
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
