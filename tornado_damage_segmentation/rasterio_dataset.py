import os
import numpy as np
from torch.utils.data import Dataset
import rasterio


class RasterioDataset(Dataset):
    """Defines a dataset for reading rasterio objects. Relies on corresponding images and masks having the same
    filename and the mask directory being flat (no subfolders). If there is no corresponding mask for an image, or
    vice versa, that image will not be included."""

    def __init__(self, image_path, mask_path, transform=None):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform
        self.rasters = []
        self.load_rasters()

    def __len__(self):
        return len(self.rasters)

    def __getitem__(self, index):
        # expand_dims used for raster.read(1) to create a single color channel in the channels-last style
        item = [np.expand_dims(raster.read(1), axis=2) for raster in self.rasters[index]]
        if self.transform:
            item = [self.transform(arr) for arr in item]
        return tuple(item)

    # Generates tuples of data, mask pairs
    def load_rasters(self):
        for path, dirs, files in os.walk(self.image_path):
            for file in files:
                try:
                    self.rasters.append((rasterio.open(os.path.join(path, file)),
                                         rasterio.open(os.path.join(self.mask_path, file))))
                except rasterio.errors.RasterioIOError:
                    pass  # If there is no corresponding mask file, skip this image
