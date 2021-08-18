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
        # raster.read(1) enclosed in brackets to represent the single color channel dimension for the image
        return tuple([np.array([raster.read(1)]) for raster in self.rasters[index]])

    # Generates tuples of data, mask pairs
    def load_rasters(self):
        for path, dirs, files in os.walk(self.image_path):
            for file in files:
                try:
                    self.rasters.append((rasterio.open(os.path.join(path, file)),
                                         rasterio.open(os.path.join(self.mask_path, file))))
                except rasterio.errors.RasterioIOError:
                    pass  # If there is no corresponding mask file, skip this image
